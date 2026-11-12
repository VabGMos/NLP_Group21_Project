import torch
import torch.nn as nn


def pair_features(u, v):
    """Concatenate the sentence vectors, their absolute difference and product."""
    return torch.cat([u, v, (u - v).abs(), u * v], dim=-1)


class VectorMLPNLI(nn.Module):
    """
    MLPNLI for already-encoded sentence pairs (any representation, e.g. PairEncoder output):
      [u ; v] -> [u, v, |u-v|, u*v] -> MLP -> logits
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
