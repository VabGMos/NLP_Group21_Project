import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


def pair_features(u, v):
    """Concatenate the sentence vectors, their absolute difference and product."""
    return torch.cat([u, v, (u - v).abs(), u * v], dim=-1)


def make_mlp(input_dim, hidden_dims, num_classes, dropout):
    """Linear-ReLU-Dropout blocks followed by a linear output layer producing logits."""
    layers, in_dim = [], input_dim
    for h in hidden_dims:
        layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
        in_dim = h
    layers.append(nn.Linear(in_dim, num_classes))
    return nn.Sequential(*layers)


class VectorMLPNLI(nn.Module):
    """
    MLPNLI for already-encoded sentence pairs (any representation, e.g. PairEncoder output):
      [u ; v] -> [u, v, |u-v|, u*v] -> MLP -> logits
    The input must be exactly [premise ; hypothesis] (PairEncoder without the overlap block).
    """

    def __init__(self, input_dim, hidden_dims=(128, 64), num_classes=3, dropout=0.3):
        super().__init__()
        assert input_dim % 2 == 0, (
            "expects [u ; v]: premise and hypothesis vectors of equal size"
        )
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


RNN_TYPES = {"gru": nn.GRU, "lstm": nn.LSTM, "rnn": nn.RNN}


class RecurrentSentenceEncoder(nn.Module):
    """Token ids -> (pretrained) embeddings -> recurrent net -> one vector per sentence.

    rnn_type selects the recurrent cell ("gru", "lstm" or "rnn"). The sentence vector is
    the final hidden state of the last layer (both directions concatenated if bidirectional).
    """

    def __init__(
        self,
        embedding_weights,
        pad_id,
        rnn_type="gru",
        hidden_size=128,
        num_layers=1,
        bidirectional=True,
        freeze_embeddings=True,
        embedding_dropout=0.0,
        rnn_dropout=0.2,
    ):
        super().__init__()
        if rnn_type not in RNN_TYPES:
            raise ValueError(f"rnn_type must be one of {sorted(RNN_TYPES)}")
        self.embedding = nn.Embedding.from_pretrained(
            embedding_weights.clone(), freeze=freeze_embeddings, padding_idx=pad_id
        )
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.rnn = RNN_TYPES[rnn_type](
            input_size=embedding_weights.shape[1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=rnn_dropout if num_layers > 1 else 0.0,
        )
        self.bidirectional = bidirectional
        self.output_dim = hidden_size * (2 if bidirectional else 1)

    def forward(self, input_ids, attention_mask):
        x = self.embedding_dropout(self.embedding(input_ids))
        lengths = attention_mask.sum(dim=1).clamp_min(1).cpu()
        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed)
        if isinstance(hidden, tuple):  # LSTM returns (h_n, c_n)
            hidden = hidden[0]
        if self.bidirectional:
            return torch.cat([hidden[-2], hidden[-1]], dim=1)
        return hidden[-1]


class PairClassifier(nn.Module):
    """Any sentence encoder (shared for both sentences) + MLP over the pair features.

    encoder must expose `output_dim` and map (input_ids, attention_mask) to (B, output_dim).
    """

    def __init__(self, encoder, num_classes=3, mlp_hidden_dims=(256,), dropout=0.2):
        super().__init__()
        self.encoder = encoder
        self.mlp = make_mlp(
            4 * encoder.output_dim, mlp_hidden_dims, num_classes, dropout
        )

    def forward(self, premise_ids, premise_mask, hypothesis_ids, hypothesis_mask):
        premise = self.encoder(premise_ids, premise_mask)
        hypothesis = self.encoder(hypothesis_ids, hypothesis_mask)
        return self.mlp(pair_features(premise, hypothesis))


def build_pair_model(
    embedding_weights,
    pad_id,
    num_classes=3,
    rnn_type="gru",
    hidden_size=128,
    num_layers=1,
    bidirectional=True,
    freeze_embeddings=True,
    embedding_dropout=0.0,
    rnn_dropout=0.2,
    mlp_hidden_dims=(256,),
    mlp_dropout=0.2,
):
    """Embedding -> recurrent encoder (gru/lstm/rnn) -> MLP -> logits."""
    encoder = RecurrentSentenceEncoder(
        embedding_weights,
        pad_id,
        rnn_type=rnn_type,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
        freeze_embeddings=freeze_embeddings,
        embedding_dropout=embedding_dropout,
        rnn_dropout=rnn_dropout,
    )
    return PairClassifier(encoder, num_classes, mlp_hidden_dims, mlp_dropout)
