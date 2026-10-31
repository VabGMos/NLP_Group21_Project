from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Sequence
from sentence_transformers import SentenceTransformer


# ---------- Thomas: tokenizer wrappers ----------

class BaseTokenizer(ABC):
    """Minimal interface every tokenizer must implement."""
    pad_token_id: int
    vocab_size: int

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> List[List[int]]:
        """Return a list of token-id lists (unpadded)."""


class HFTokenizerWrapper(BaseTokenizer):
    """Wraps any HuggingFace tokenizer (AutoTokenizer, etc.)."""

    def __init__(self, tokenizer, max_length: int = 64):
        self.tok = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.vocab_size = len(tokenizer)

    def encode(self, texts):
        return self.tok(list(texts), truncation=True, max_length=self.max_length,
                        add_special_tokens=True)["input_ids"]


class SentenceTransformerTokenizer(HFTokenizerWrapper):
    """Base tokenizer: the tokenizer that ships with a sentence-transformers model."""

    def __init__(self, model_name: str = "sentence-transformers/nli-mpnet-base-v2",
                 max_length: int = 64):
        st_model = SentenceTransformer(model_name, device="cpu")
        super().__init__(st_model.tokenizer, max_length)


# ---------- BPE tokenizer (copied from part1/experiment.py, NLP1 tutorial) ----------

def train_tokenizer(corpus, vocab_size, min_frequency, pretokenizer=Metaspace(), unk_token="[UNK]"):
    """
    corpus: a list of texts (each a string)
    vocab_size: number of distinct BPE tokens
    min_frequency: we only merge token pairs that are frequent enough
    pretokenizer: a simple tokenizer that we use before BPE tokenization
    """
    tokenizer = Tokenizer(BPE(unk_token=unk_token))
    tokenizer.pre_tokenizer = pretokenizer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[unk_token]
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    return tokenizer
