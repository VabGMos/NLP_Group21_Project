import torch
import torch.nn as nn
from transformers import AutoModel


def pair_features(u, v):
    """Given two tensors u and v, returns the concatenation of [u, v, |u-v|, u*v],
    which is a common feature representation for sentence-pair classification tasks. Because it takes into account
    the similarity and difference between the two vectors, it is often used in natural language inference (NLI) tasks."""
    return torch.cat([u, v, (u - v).abs(), u * v], dim=-1)

class MLPNLI(nn.Module):
    """
    MLP for sentence-pair classification:
      token ids -> embedding -> masked pooling (per sentence) -> [u, v, |u-v|, u*v] -> MLP -> logits
    """

    def __init__(
        self,
        vocab_size,
        pad_token_id,
        emb_dim=768,
        hidden_dims=(128, 64),
        num_classes=3,
        dropout=0.3,
        pooling="mean",
        pretrained=True
    ):
        super().__init__()
        assert pooling in {"mean", "max"}
        self.pooling = pooling
        if pretrained:
            # start from MPNet's pretrained token embeddings (fine-tuned further during training)
            mpnet = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
            weights = mpnet.get_input_embeddings().weight.detach()
            assert weights.shape == (vocab_size, emb_dim), (
                f"pretrained embeddings are {tuple(weights.shape)}, but vocab_size={vocab_size}, emb_dim={emb_dim}: "
                "use the MPNet tokenizer and emb_dim=768"
            )
            self.emb = nn.Embedding.from_pretrained(weights, freeze=False, padding_idx=pad_token_id)
        else:
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_token_id)
        layers, in_dim = [], 4 * emb_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def encode(self, ids, mask):
        x = self.emb(ids)  # [B, T, E]
        m = mask.unsqueeze(-1)  # [B, T, 1]
        if self.pooling == "mean":
            return (x * m).sum(1) / m.sum(1).clamp(min=1)
        return x.masked_fill(m == 0, -1e9).max(1).values

    def forward(self, batch):
        u = self.encode(batch["premise_ids"], batch["premise_mask"])
        v = self.encode(batch["hypothesis_ids"], batch["hypothesis_mask"])
        return self.mlp(pair_features(u, v))


class MLPSimple(MLPNLI):
    """
    Simpler MLP: same embedding + pooling, but the features are just [u, v]
    and there is a single hidden layer.
    """

    def __init__(
        self,
        vocab_size,
        pad_token_id,
        pretrained=True,
        emb_dim=768,
        hidden_dims=(128,),
        num_classes=3,
        dropout=0.3,
        pooling="mean",
    ):
        super().__init__(
            vocab_size,
            pad_token_id,
            emb_dim=emb_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
            pooling=pooling,
            pretrained=pretrained
        )
        layers, in_dim = [], 2 * emb_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, batch):
        u = self.encode(batch["premise_ids"], batch["premise_mask"])
        v = self.encode(batch["hypothesis_ids"], batch["hypothesis_mask"])
        return self.mlp(torch.cat([u, v], dim=-1))


class VectorMLPNLI(nn.Module):
    """
    MLPNLI for already-encoded sentence pairs (any representation, e.g. PairEncoder output):
      [u ; v] -> [u, v, |u-v|, u*v] -> MLP -> logits
    Same classifier as MLPNLI, without the embedding / pooling step.
    The input must be exactly [premise ; hypothesis] (PairEncoder without the overlap block).
    """

    def __init__(self, input_dim, hidden_dims=(128, 64), num_classes=3, dropout=0.3):
        super().__init__()
        assert input_dim % 2 == 0, "expects [u ; v]: premise and hypothesis vectors of equal size"
        self.d = input_dim // 2
        layers, in_dim = [], 4 * self.d
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        u, v = x[:, : self.d], x[:, self.d :]
        return self.mlp(pair_features(u, v))


class VectorMLPSimple(nn.Module):
    """
    MLPSimple for already-encoded sentence pairs (any representation, e.g. PairEncoder output):
      [u ; v] -> MLP with a single hidden layer -> logits
    Same classifier as MLPSimple, without the embedding / pooling step.
    """

    def __init__(self, input_dim, hidden_dims=(128,), num_classes=3, dropout=0.3):
        super().__init__()
        layers, in_dim = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
