import numpy as np

from src.encoders import Encoder


class PooledEmbeddingEncoder(Encoder):
    """Encode text with a fixed table, matching unpadded IDs and a pooler."""

    def __init__(self, embedding_matrix, encode_ids, pooling):
        self.vectors = np.asarray(embedding_matrix, dtype=np.float32)
        self.encode_ids = encode_ids
        self.pooling = pooling

    def __len__(self):
        return self.vectors.shape[1]

    def __call__(self, text):
        ids = self.encode_ids(text)
        if not ids:
            return np.zeros(len(self), dtype=np.float32)
        return self.pooling(self.vectors[ids])
