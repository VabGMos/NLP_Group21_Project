import torch
from torch import nn

POOLING_TYPES = ("mean", "sum", "max")


class PoolingSentenceEncoder(nn.Module):
    """Token ids -> embeddings -> masked pooling -> one vector per sentence."""

    def __init__(self, embedding_weights, pad_id, pooling="mean",
                 freeze_embeddings=True, embedding_dropout=0.0):
        super().__init__()
        if pooling not in POOLING_TYPES:
            raise ValueError(f"pooling must be one of {POOLING_TYPES}")
        self.embedding = nn.Embedding.from_pretrained(
            embedding_weights.clone(), freeze=freeze_embeddings, padding_idx=pad_id
        )
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.pooling = pooling
        self.output_dim = self.embedding.embedding_dim

    def forward(self, input_ids, attention_mask):
        vectors = self.embedding_dropout(self.embedding(input_ids))
        valid = attention_mask.bool().unsqueeze(-1)
        if self.pooling == "max":
            # Padding must not win when every real value is negative.
            pooled = vectors.masked_fill(~valid, -torch.inf).max(dim=1).values
            return torch.where(valid.any(dim=1), pooled, torch.zeros_like(pooled))

        pooled = vectors.masked_fill(~valid, 0.0).sum(dim=1)
        if self.pooling == "mean":
            pooled = pooled / valid.sum(dim=1).clamp_min(1)
        return pooled
