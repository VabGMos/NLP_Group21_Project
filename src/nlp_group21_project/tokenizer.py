from tokenizers import Tokenizer
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Sequence
from sentence_transformers import SentenceTransformer

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


class TokenIndicator():

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        return len(self.tokenizer.get_vocab())

    def __call__(self, x: str) -> 'np.array':
        """Represent x as a D-dimensional vector of real numbers"""
        unique_token_ids = list(set(self.tokenizer.encode(x).ids))

        # A 0 vector with the same length as the vocabulary except in positions corresponding
        # to tokens appearing in the document
        vector = np.zeros(len(self))
        vector[unique_token_ids] = 1
        return vector


class BagofWords():

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        return len(self.tokenizer.get_vocab())

    def __call__(self, x: str) -> np.array:
        """Represent x as a D-dimensional vector of real numbers"""
        raise NotImplementedError(
            "Concrete implementations will be done later")
