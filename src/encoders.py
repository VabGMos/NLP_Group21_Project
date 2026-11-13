import numpy as np
from tokenizers import Tokenizer


class Encoder:
    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        raise NotImplementedError("Concrete implementations will be discussed later")

    def __call__(self, x: str) -> "np.array":
        """Represent x as a D-dimensional vector of real numbers"""
        raise NotImplementedError("Implement me in a sub-class")


class TokenIndicator(Encoder):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        return len(self.tokenizer.get_vocab())

    def __call__(self, x: str) -> "np.array":
        """Represent x as a D-dimensional vector of real numbers"""
        unique_token_ids = list(set(self.tokenizer.encode(x).ids))

        # A 0 vector with the same length as the vocabulary except in positions corresponding
        # to tokens appearing in the document
        vector = np.zeros(len(self))
        vector[unique_token_ids] = 1
        return vector


class BagofWords(Encoder):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        return len(self.tokenizer.get_vocab())

    def __call__(self, x: str) -> np.array:
        """Represent x as a D-dimensional vector of real numbers"""
        token_ids = list(self.tokenizer.encode(x).ids)

        # A 0 vector with the same length as the vocabulary except in positions corresponding
        # to tokens appearing in the document, where the value is the counter of appearances
        vector = np.zeros(len(self))
        np.add.at(vector, token_ids, 1)
        return vector


class MPNetEncoder(Encoder):
    def __init__(self, sentence_model):
        self.model = sentence_model

    def __len__(self):
        return self.model.get_sentence_embedding_dimension()

    def __call__(self, text):
        return self.model.encode(text, convert_to_numpy=True)


# pair encoder for all encoders
class PairEncoder(Encoder):
    def __init__(
        self, base_encoder: Encoder, overlap=False
    ):  # whetehr or not to count whether certain tokens appear in both premise and hypothesis
        self.base_encoder = base_encoder
        self.overlap = overlap

    def __len__(self):
        if self.overlap:
            return 3 * len(self.base_encoder)  # hypothesis, premise, overlap
        return 2 * len(self.base_encoder)  # hypoithesis, premise

    def __call__(self, x: tuple):  # taking in tuple of (premise, hypothesis)
        premise_encoding = self.base_encoder(x[0])
        hypothesis_encoding = self.base_encoder(x[1])
        paired = np.concatenate([premise_encoding, hypothesis_encoding])

        if self.overlap:
            overlaps = ((premise_encoding > 0) & (hypothesis_encoding > 0)).astype(
                float
            )
            paired = np.concatenate([paired, overlaps])
        return paired
