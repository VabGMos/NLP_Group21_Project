import numpy as np
from scipy.sparse import csr_matrix, vstack


def encode_sparse(texts, encoder, batch_size=1000):
    """Encode one small dense batch at a time, keeping completed batches sparse."""
    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors = np.asarray(
            [encoder(text) for text in batch],
            dtype=np.float32,
        )
        chunks.append(csr_matrix(vectors))
    return vstack(chunks, format="csr")
