"""Custom word embeddings trained on SNLI: one word back, one word ahead.

Model ("directional skip-gram", window 1):
    token id -> nn.Embedding(V, d) -> head_prev: logits over the previous token
                                   -> head_next: logits over the next token
    loss = CE(prev) + CE(next)   (full softmax; fine for a vocab of ~1k-10k)

Two separate heads, so the model knows which side a neighbour is on.
After training, `model.embedding_weights()` is a (V, d) matrix whose row i is the
vector for token id i of the tokenizer it was trained with (row 0 = [PAD] = zeros).

Quick use (from a notebook, after the repo root is on sys.path):
    from src.embedding_models import train_embeddings, nearest
    model, tok, history = train_embeddings(train_sents, vocab_size=1000, val_sentences=val_sents)
    nearest("man", model, tok)

Or run everything (loads SNLI, trains tokenizer + embeddings, prints neighbours):
    python -m src.embedding_models
"""

import time

import torch
from torch import nn

IGNORE = -100  # CrossEntropyLoss skips targets with this value (no previous / next word)


# ---------------------------------------------------------------- data
def make_xy(ids):
    """One example per token: x = token, y_prev = token before it, y_next = token after it."""
    y_prev = [IGNORE] + ids[:-1]  # first token has no previous word
    y_next = ids[1:] + [IGNORE]  # last token has no next word
    return ids, y_prev, y_next


def build_training_set(sentences, tok, batch_size=10_000):
    """Tokenize all sentences and stack (x, y_prev, y_next) into three long tensors."""
    X, Yp, Yn = [], [], []
    for start in range(0, len(sentences), batch_size):
        for enc in tok.encode_batch(sentences[start : start + batch_size]):
            x, yp, yn = make_xy(enc.ids)
            X += x
            Yp += yp
            Yn += yn
    as_t = lambda v: torch.tensor(v, dtype=torch.long)
    return as_t(X), as_t(Yp), as_t(Yn)


# ---------------------------------------------------------------- model
class NeighbourEmbedding(nn.Module):
    """Embedding table + one output head for the previous and one for the next token."""

    def __init__(self, vocab_size, d=100, pad_id=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d, padding_idx=pad_id)
        self.head_prev = nn.Linear(d, vocab_size)
        self.head_next = nn.Linear(d, vocab_size)

    def forward(self, x):
        e = self.emb(x)  # (B, d)
        return self.head_prev(e), self.head_next(e)  # two (B, V) logit tensors

    def embedding_weights(self):
        """(V, d) matrix to plug into the classifiers (nn.Embedding.from_pretrained)."""
        return self.emb.weight.detach().cpu().clone()


def default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, X, Yp, Yn, batch_size=8192, device=None, ks=(1, 5)):
    """Loss and top-k accuracy of both heads on held-out tensors.

    top-k accuracy = share of real neighbours that are among the head's k highest scores.
    Expect low top-1 numbers: after "a" many words are valid, so even a perfect model can't
    always name the one that occurred. Top-5 is the fairer view.
    """
    device = device or next(model.parameters()).device
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE, reduction="sum")
    loss_sum = {"prev": 0.0, "next": 0.0}
    hits = {(h, k): 0 for h in ("prev", "next") for k in ks}
    count = {"prev": 0, "next": 0}
    for start in range(0, len(X), batch_size):
        x = X[start : start + batch_size].to(device)
        logits = dict(zip(("prev", "next"), model(x)))
        targets = {"prev": Yp[start : start + batch_size].to(device), "next": Yn[start : start + batch_size].to(device)}
        for h in ("prev", "next"):
            t, lg = targets[h], logits[h]
            valid = t != IGNORE
            loss_sum[h] += loss_fn(lg, t).item()
            count[h] += valid.sum().item()
            topk = lg.topk(max(ks), dim=1).indices  # (B, max k)
            for k in ks:
                hits[(h, k)] += ((topk[:, :k] == t[:, None]).any(dim=1) & valid).sum().item()
    out = {"loss": sum(loss_sum[h] / count[h] for h in ("prev", "next"))}  # same scale as the training loss
    for h in ("prev", "next"):
        for k in ks:
            out[f"{h}_top{k}"] = 100 * hits[(h, k)] / count[h]
    return out


# ---------------------------------------------------------------- training
def train_model(
    X,
    Yp,
    Yn,
    vocab_size,
    d=100,
    epochs=3,
    batch_size=4096,
    lr=1e-3,
    pad_id=0,
    seed=42,
    device=None,
    log_every=200,
    val=None,
):
    """Train a NeighbourEmbedding on prepared tensors.

    val: optional (X, Yp, Yn) tensors from held-out sentences; evaluated after every epoch.
    Returns (model, history): one dict per epoch with the train loss and, if val is given,
    val loss and top-1 / top-5 accuracy of both heads (in %).
    """
    torch.manual_seed(seed)
    device = device or default_device()
    model = NeighbourEmbedding(vocab_size, d, pad_id).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE)

    n = len(X)
    print(
        f"{n:,} training tokens | V={vocab_size} d={d} | "
        f"{sum(p.numel() for p in model.parameters()):,} params | device={device}"
    )
    print(f"loss at random init should be about 2*ln(V) = {2 * torch.log(torch.tensor(vocab_size)).item():.2f}")

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)  # shuffle every epoch
        total, steps, t0 = 0.0, 0, time.time()
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            x, yp, yn = X[idx].to(device), Yp[idx].to(device), Yn[idx].to(device)

            logits_prev, logits_next = model(x)
            loss = loss_fn(logits_prev, yp) + loss_fn(logits_next, yn)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()
            steps += 1
            if log_every and steps % log_every == 0:
                print(f"  epoch {epoch} step {steps:>6}  loss {total / steps:.3f}")
        row = {"epoch": epoch, "train_loss": total / steps}
        msg = f"epoch {epoch}: train loss {row['train_loss']:.3f}"
        if val is not None:
            v = evaluate(model, *val, device=device)
            row.update({f"val_{k}": val_v for k, val_v in v.items()})
            msg += (
                f" | val loss {v['loss']:.3f}"
                f" | prev top1 {v['prev_top1']:.1f}% top5 {v['prev_top5']:.1f}%"
                f" | next top1 {v['next_top1']:.1f}% top5 {v['next_top5']:.1f}%"
            )
        history.append(row)
        print(f"{msg}  ({time.time() - t0:.0f}s)")
    return model, history


def train_embeddings(
    sentences,
    vocab_size=1000,
    d=100,
    epochs=3,
    min_frequency=2,
    lowercase=True,
    val_sentences=None,
    tokenizer=None,
    **kw,
):
    """Full pipeline: train a BPE tokenizer on `sentences`, then the embedding model.

    `sentences` should be the deduplicated SNLI training sentences only.
    `val_sentences` (optional, e.g. deduplicated validation sentences) are used only to
    report val loss / accuracy after each epoch, never for training.
    `tokenizer` (optional): reuse an already trained tokenizer instead of training one
    (vocab_size / min_frequency / lowercase are then ignored).
    Returns (model, tokenizer, history).
    """
    from src.tokenizer import PAD_TOKEN, train_tokenizer

    tok = tokenizer or train_tokenizer(sentences, vocab_size, min_frequency=min_frequency, lowercase=lowercase)
    X, Yp, Yn = build_training_set(sentences, tok)
    val = build_training_set(val_sentences, tok) if val_sentences is not None else None
    model, history = train_model(
        X, Yp, Yn, tok.get_vocab_size(), d=d, epochs=epochs, pad_id=tok.token_to_id(PAD_TOKEN), val=val, **kw
    )
    return model, tok, history


# ---------------------------------------------------------------- inspection
def nearest(word, model, tok, k=10):
    """Tokens with the highest cosine similarity to `word` (whole words carry the ▁ prefix)."""
    W = model.embedding_weights()
    W = W / W.norm(dim=1, keepdim=True).clamp_min(1e-8)
    token = word if word.startswith("▁") else "▁" + word
    idx = tok.token_to_id(token)
    if idx is None:
        raise KeyError(f"{token!r} is not a single token in this vocab: {tok.encode(word).tokens}")
    sims = W @ W[idx]
    sims[idx] = -1  # skip the word itself
    top = torch.topk(sims, k)
    return [(tok.id_to_token(i), round(s, 3)) for s, i in zip(top.values.tolist(), top.indices.tolist())]


def unique_train_sentences():
    """Deduplicated premises + hypotheses from SNLI's training split."""
    import pandas as pd

    from src.dataset import convert_to_dataframe, load_snli_dataset

    train_df, _, _ = convert_to_dataframe(load_snli_dataset())
    sents = pd.concat([train_df["premise"], train_df["hypothesis"]]).dropna().astype(str)
    return list(dict.fromkeys(sents))  # premises repeat ~3x in SNLI


if __name__ == "__main__":
    sents = unique_train_sentences()
    print(f"{len(sents):,} unique training sentences")
    model, tok, history = train_embeddings(sents, vocab_size=1000, d=100, epochs=3)
    for w in ["man", "woman", "dog", "red", "sitting", "running", "beach", "not"]:
        try:
            print(f"{w:>8}: {nearest(w, model, tok, k=8)}")
        except KeyError as e:
            print(f"{w:>8}: {e}")
