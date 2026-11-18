from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.trainers import BpeTrainer
from transformers import AutoTokenizer

# Every embedding in src.embeddings uses this vocabulary, so it is the single tokenizer.
MPNET_TOKENIZER_NAME = "sentence-transformers/all-mpnet-base-v2"


def load_mpnet_tokenizer(name: str = MPNET_TOKENIZER_NAME):
    return AutoTokenizer.from_pretrained(name)


def tokenize_texts(
    texts: list[str], tokenizer, max_length: int, batch_size: int = 1024
) -> list[list[int]]:
    """Token ids (with special tokens, truncated, not padded) for each text."""
    ids = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start : start + batch_size],
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )
        ids.extend(batch["input_ids"])
    return ids

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"


def train_tokenizer(
    corpus,
    vocab_size,
    min_frequency,
    pretokenizer=Metaspace(),
    unk_token=UNK_TOKEN,
    pad_token=PAD_TOKEN,
    lowercase=False,
):
    """
    corpus: a list of texts (each a string)
    vocab_size: number of distinct BPE tokens (special tokens included)
    min_frequency: we only merge token pairs that are frequent enough
    pretokenizer: a simple tokenizer that we use before BPE tokenization
    pad_token: filler id used to make sequences in a batch equally long (id 0)
    lowercase: if True, "A man" and "a man" become the same tokens
    """
    tokenizer = Tokenizer(BPE(unk_token=unk_token))
    if lowercase:
        tokenizer.normalizer = Lowercase()
    tokenizer.pre_tokenizer = pretokenizer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[pad_token, unk_token],  # pad -> id 0, unk -> id 1
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    return tokenizer
