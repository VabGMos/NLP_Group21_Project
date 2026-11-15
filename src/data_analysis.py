import re
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel

from src.tokenizer import tokenize_texts


def load_input_embeddings(model_name: str) -> torch.Tensor:
    """(vocab_size, dim) input-embedding table of a HuggingFace checkpoint."""
    model = AutoModel.from_pretrained(model_name)
    weights = model.get_input_embeddings().weight.detach().cpu().clone()
    del model
    return weights


def word_vector(word: str, weights: torch.Tensor, tokenizer):
    """Vector of a word (mean of its subword pieces) and its number of pieces."""
    ids = tokenizer(word, add_special_tokens=False)["input_ids"]
    return weights[ids].mean(0), len(ids)


def nearest_neighbours(word: str, weights: torch.Tensor, tokenizer, k: int = 8):
    """Top-k cosine neighbours of ``word`` among non-special vocabulary tokens."""
    vec, _ = word_vector(word, weights, tokenizer)
    sims = F.cosine_similarity(vec[None], weights, dim=1)
    sims[list(tokenizer.all_special_ids)] = -1.0
    own = set(tokenizer(word, add_special_tokens=False)["input_ids"])
    top = [i for i in sims.argsort(descending=True).tolist() if i not in own][:k]
    return [(tokenizer.convert_ids_to_tokens(i), float(sims[i])) for i in top]


def pair_similarities(pairs, weights: torch.Tensor, tokenizer) -> list[float]:
    """Cosine similarity of each word pair; pieces are averaged for rare words."""
    sims = []
    for a, b in pairs:
        va, _ = word_vector(a, weights, tokenizer)
        vb, _ = word_vector(b, weights, tokenizer)
        sims.append(float(F.cosine_similarity(va, vb, dim=0)))
    return sims


def mean_pair_similarity(pairs, weights: torch.Tensor, tokenizer) -> float:
    return float(np.mean(pair_similarities(pairs, weights, tokenizer)))


def token_stats(texts, tokenizer, vocab_size: int, max_length: int = 512):
    """Token lengths per text and per-vocabulary-id corpus counts."""
    ids = tokenize_texts(list(texts), tokenizer, max_length)
    lengths = np.fromiter((len(seq) for seq in ids), dtype=np.int64, count=len(ids))
    flat = np.fromiter((i for seq in ids for i in seq), dtype=np.int64)
    return lengths, np.bincount(flat, minlength=vocab_size)


def pieces_per_word(texts, tokenizer):
    """Subword pieces of every unique lower-cased word, plus each word's frequency."""
    freq = Counter(w for text in texts for w in re.findall(r"[^\W_]+", text.lower()))
    words = sorted(freq)
    ids = tokenizer(words, add_special_tokens=False)["input_ids"]
    return np.array([len(seq) for seq in ids]), np.array([freq[w] for w in words])


def lexical_overlap(premises, hypotheses) -> np.ndarray:
    """Fraction of distinct hypothesis words that also occur in the premise."""
    out = []
    for p, h in zip(premises, hypotheses):
        p_words = set(re.findall(r"[a-z']+", p.lower()))
        h_words = set(re.findall(r"[a-z']+", h.lower()))
        out.append(len(h_words & p_words) / max(len(h_words), 1))
    return np.array(out)


NEGATION_WORDS = {"no", "not", "nobody", "nothing", "never", "none", "nowhere"}


def has_negation(texts) -> np.ndarray:
    """True for texts with a negation word or an n't contraction."""
    flags = []
    for text in texts:
        words = re.findall(r"[a-z']+", text.lower())
        flags.append(any(w in NEGATION_WORDS or w.endswith("n't") for w in words))
    return np.array(flags)


def shallow_features(df) -> np.ndarray:
    """Surface cues: lexical overlap, word counts and negation in premise/hypothesis."""
    p, h = df["premise"], df["hypothesis"]
    return np.c_[
        lexical_overlap(p, h),
        p.str.split().str.len(),
        h.str.split().str.len(),
        has_negation(h),
        has_negation(p),
    ]


def shallow_baseline(train_df, eval_df, seed: int = 0) -> float:
    """Accuracy of a gradient-boosting model that sees only the surface cues."""
    clf = HistGradientBoostingClassifier(random_state=seed)
    clf.fit(shallow_features(train_df), train_df["label"].to_numpy())
    return float(clf.score(shallow_features(eval_df), eval_df["label"].to_numpy()))


def pair_cosines(premises, hypotheses, weights, tokenizer, max_length=128):
    """Cosine between mean-pooled premise and hypothesis embeddings."""
    u = mean_pool(premises, weights, tokenizer, max_length)
    v = mean_pool(hypotheses, weights, tokenizer, max_length)
    num = (u * v).sum(1)
    return num / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))


def pca_2d(vectors: np.ndarray) -> np.ndarray:
    centred = vectors - vectors.mean(0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return centred @ vt[:2].T


def random_pair_similarity(
    weights: torch.Tensor, token_ids, n_pairs: int = 5000, seed: int = 0
) -> float:
    """Baseline: mean cosine of random token pairs drawn from ``token_ids``."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.asarray(token_ids), size=(n_pairs, 2))
    a, b = weights[idx[:, 0]], weights[idx[:, 1]]
    return float(F.cosine_similarity(a, b, dim=1).mean())


def mean_pool(texts, weights: torch.Tensor, tokenizer, max_length: int = 128):
    """Embed-and-pool: mean of token embeddings per text (special tokens included)."""
    ids = tokenize_texts(list(texts), tokenizer, max_length)
    flat = torch.tensor([i for seq in ids for i in seq])
    offsets = torch.tensor([0, *np.cumsum([len(s) for s in ids])[:-1]])
    return F.embedding_bag(flat, weights, offsets, mode="mean").numpy()


def pair_features(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.concatenate([u, v, np.abs(u - v), u * v], axis=1)


def probe_nli(weights, tokenizer, train_df, eval_df, max_length=128, seed=0) -> float:
    """Accuracy of a logistic regression on frozen mean-pooled embeddings."""

    def features(df):
        u = mean_pool(df["premise"], weights, tokenizer, max_length)
        v = mean_pool(df["hypothesis"], weights, tokenizer, max_length)
        return pair_features(u, v)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(features(train_df))
    clf = LogisticRegression(max_iter=300, random_state=seed)
    clf.fit(x_train, train_df["label"].to_numpy())
    x_eval = scaler.transform(features(eval_df))
    return float(clf.score(x_eval, eval_df["label"].to_numpy()))


def report_tokenizer_coverage(
    texts: list[str], tokenizer, max_length: int, MODEL_NAME: str
) -> dict:
    """Report coverage in the tokenizer's actual subword-token units.

    MPNet uses subword pieces, not whitespace-delimited words. Consequently,
    low [UNK] rates and the average number of pieces per whitespace word are
    more informative than comparing raw words directly with the vocabulary.
    """
    unk_id = tokenizer.unk_token_id
    pieces_total = 0
    pieces_unknown = 0
    over_limit = 0
    length_total = 0
    batch_size = 256

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, add_special_tokens=True, truncation=False)
        for ids in encoded["input_ids"]:
            pieces_total += len(ids)
            length_total += len(ids)
            if unk_id is not None:
                pieces_unknown += sum(token_id == unk_id for token_id in ids)
            if len(ids) > max_length:
                over_limit += 1

    # Approximate lexical fragmentation for unique whitespace-delimited forms.
    # This is diagnostic only: whitespace words are not MPNet's token units.
    words = set()
    for text in texts:
        words.update(re.findall(r"\S+", text))
    unique_words = list(words)
    word_unknown = 0
    subword_counts = []
    for word in unique_words:
        ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        subword_counts.append(len(ids))
        if unk_id is not None and unk_id in ids:
            word_unknown += 1

    vocab = tokenizer.get_vocab()
    return {
        "model": MODEL_NAME,
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocabulary_size": len(vocab),
        "corpus_subword_pieces": pieces_total,
        "unknown_subword_rate": pieces_unknown / max(pieces_total, 1),
        "unique_whitespace_forms": len(unique_words),
        "unique_whitespace_forms_containing_unk_rate": word_unknown
        / max(len(unique_words), 1),
        "mean_subword_pieces_per_whitespace_form": float(np.mean(subword_counts))
        if subword_counts
        else 0.0,
        "documents_longer_than_max_length_rate": over_limit / max(len(texts), 1),
        "max_length": max_length,
        "note": "MPNet vocabulary coverage is subword-based; whitespace-form rates are approximate diagnostics.",
    }
