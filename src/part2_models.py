"""Part 2: custom word embeddings -> mean-pooled sentence vectors -> Part 1 MLPs.

  sentence -> BPE tokens -> embedding of each token (custom (V, d) matrix, frozen)
           -> average over the tokens -> one d-dim sentence vector
  pair     -> [premise vector ; hypothesis vector] (2d) -> Part 1 MLP -> 3 logits
              "simple": VectorMLPSimple, 1 hidden layer (128) on [u ; v]
              "nli":    VectorMLPNLI, 2 hidden layers (128, 64) on [u, v, |u-v|, u*v]

The embeddings are frozen, so every sentence vector is computed once up front and the MLP
trains on fixed vectors, exactly like the MPNet sentence embeddings in Part 1.
"""

import copy
import time

import numpy as np
import torch
from torch import nn

from src.models import VectorMLPNLI, VectorMLPSimple

# Part 1 models and their training settings
MODELS = {
    "simple": VectorMLPSimple,  # [u ; v] -> 1 hidden layer (128) -> logits
    "nli": VectorMLPNLI,  # [u, v, |u-v|, u*v] -> 2 hidden layers (128, 64) -> logits
}
SETTINGS = {
    "simple": dict(hidden_dims=(128,), dropout=0.1, lr=1e-4, weight_decay=1e-4, batch_size=64, clip=None),
    "nli": dict(hidden_dims=(128, 64), dropout=0.3, lr=3e-4, weight_decay=1e-6, batch_size=128, clip=1.0),
}


# ---------------------------------------------------------------- features
def mean_pool_embed(sentences, tok, W, batch_size=10_000):
    """(N, d) tensor: the average embedding of each sentence's tokens (zeros for an empty sentence)."""
    W = torch.as_tensor(W, dtype=torch.float32)
    out = torch.zeros(len(sentences), W.shape[1])
    sentences = [str(s) for s in sentences]
    for start in range(0, len(sentences), batch_size):
        for i, enc in enumerate(tok.encode_batch(sentences[start : start + batch_size])):
            if enc.ids:
                out[start + i] = W[enc.ids].mean(dim=0)
    return out


def pair_features(df, tok, W):
    """X = [premise vector ; hypothesis vector] (N, 2d), y = labels (N,)."""
    u = mean_pool_embed(df["premise"].tolist(), tok, W)
    v = mean_pool_embed(df["hypothesis"].tolist(), tok, W)
    return torch.cat([u, v], dim=1), torch.tensor(df["label"].astype(int).to_numpy())


# ---------------------------------------------------------------- training
@torch.no_grad()
def accuracy(model, X, y, batch_size=8192):
    """Accuracy in % of model on (X, y)."""
    device = next(model.parameters()).device
    model.eval()
    correct = 0
    for s in range(0, len(X), batch_size):
        pred = model(X[s : s + batch_size].to(device)).argmax(dim=-1).cpu()
        correct += (pred == y[s : s + batch_size]).sum().item()
    return 100 * correct / len(X)


def train_mlp(X_train, y_train, X_val, y_val, model="simple", epochs=20, seed=42, device=None, **overrides):
    """Train a Part 1 MLP with its Part 1 settings (see SETTINGS; any of them can be overridden).

    model: "simple" (1 hidden layer) or "nli" (2 hidden layers on [u, v, |u-v|, u*v]).
    Trains for `epochs`, checks val accuracy after every epoch and returns
    (model with the best-val weights, history).
    """
    cfg = {**SETTINGS[model], **overrides}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, clip = cfg["batch_size"], cfg["clip"]
    model = MODELS[model](X_train.shape[1], hidden_dims=cfg["hidden_dims"], dropout=cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    history, best_acc, best_state = [], -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        for idx in np.array_split(rng.permutation(len(X_train)), max(1, len(X_train) // batch_size)):
            x, y = X_train[idx].to(device), y_train[idx].to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        val_acc = accuracy(model, X_val, y_val)
        history.append({"epoch": epoch, "val_acc": val_acc})
        if val_acc > best_acc:
            best_acc, best_state = val_acc, copy.deepcopy(model.state_dict())
        print(f"epoch {epoch:>2}: val acc {val_acc:.2f}%  ({time.time() - t0:.0f}s)")
    model.load_state_dict(best_state)
    return model, history


# ================================================================ LSTM
# Unlike the MLPs above, the LSTM reads the sequence of token vectors, so sentences stay as
# token ids (padded to the same length) and the embedding table lives inside the model.
#
#   premise ids    -> embedding -> LSTM -> last hidden state u ┐
#   hypothesis ids -> embedding -> LSTM -> last hidden state v ┴-> [u ; v] -> Part 1 MLP -> logits
# The same embedding + LSTM (same weights) encode both sentences.


def token_tensors(df, tok, max_len=64, pad_id=0):
    """Padded token ids + lengths for premises and hypotheses, plus labels.

    Ids are stored as int16 (vocab < 32k) to keep the full training set small in memory.
    Returns dict with p_ids (N, L), p_len (N,), h_ids, h_len, y.
    """

    def encode(texts):
        encs = tok.encode_batch([str(t) for t in texts])
        lengths = torch.tensor([max(1, min(len(e.ids), max_len)) for e in encs])
        ids = torch.full((len(encs), int(lengths.max())), pad_id, dtype=torch.int16)
        for i, e in enumerate(encs):
            seq = e.ids[:max_len]
            ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.int16)
        return ids, lengths

    p_ids, p_len = encode(df["premise"].tolist())
    h_ids, h_len = encode(df["hypothesis"].tolist())
    y = torch.tensor(df["label"].astype(int).to_numpy())
    return {"p_ids": p_ids, "p_len": p_len, "h_ids": h_ids, "h_len": h_len, "y": y}


def _batch(data, idx, device):
    """Select rows idx, trim padding to the longest sentence in the batch, move to device."""
    p_len, h_len = data["p_len"][idx], data["h_len"][idx]
    p_ids = data["p_ids"][idx, : int(p_len.max())].long().to(device)
    h_ids = data["h_ids"][idx, : int(h_len.max())].long().to(device)
    return p_ids, p_len, h_ids, h_len, data["y"][idx].to(device)


class LSTMEncoder(nn.Module):
    """Token ids -> embeddings -> single-layer LSTM -> final hidden state (one vector per sentence)."""

    def __init__(self, W, hidden_size=128, pad_id=0, freeze=True):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.as_tensor(W, dtype=torch.float32).clone(), freeze=freeze, padding_idx=pad_id
        )
        self.lstm = nn.LSTM(self.embedding.embedding_dim, hidden_size, batch_first=True)
        self.output_dim = hidden_size

    def forward(self, ids, lengths):
        x = self.embedding(ids)  # (B, L, d)
        # packing makes the LSTM stop at each sentence's real length instead of reading padding
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)  # h_n: (1, B, hidden) = state after the last real token
        return h_n[-1]


class LSTMPairClassifier(nn.Module):
    """Shared LSTM encoder for premise and hypothesis + a Part 1 MLP ("simple" or "nli") on [u ; v]."""

    def __init__(self, W, hidden_size=128, pad_id=0, freeze=True, head="nli", **mlp_kwargs):
        super().__init__()
        self.encoder = LSTMEncoder(W, hidden_size, pad_id, freeze)
        self.mlp = MODELS[head](2 * hidden_size, **mlp_kwargs)

    def forward(self, p_ids, p_len, h_ids, h_len):
        u = self.encoder(p_ids, p_len)
        v = self.encoder(h_ids, h_len)
        return self.mlp(torch.cat([u, v], dim=1))


def default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def lstm_accuracy(model, data, batch_size=1024):
    """Accuracy in % of an LSTMPairClassifier on token_tensors(...) data."""
    device = next(model.parameters()).device
    model.eval()
    correct, n = 0, len(data["y"])
    for s in range(0, n, batch_size):
        p_ids, p_len, h_ids, h_len, y = _batch(data, torch.arange(s, min(s + batch_size, n)), device)
        correct += (model(p_ids, p_len, h_ids, h_len).argmax(dim=-1) == y).sum().item()
    return 100 * correct / n


def train_lstm(
    train,
    val,
    W,
    pad_id=0,
    hidden_size=128,
    freeze=True,
    head="nli",
    lr=1e-3,
    batch_size=128,
    clip=1.0,
    epochs=5,
    seed=42,
    device=None,
):
    """Adam + cross-entropy on token_tensors(...) data; keeps the best-val epoch.

    Returns (model with the best-val weights, history).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = device or default_device()
    model = LSTMPairClassifier(W, hidden_size, pad_id, freeze, head).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    n = len(train["y"])
    print(f"{n:,} training pairs | {sum(p.numel() for p in params):,} trainable params | device={device}")

    history, best_acc, best_state = [], -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        t0, loss_sum, steps = time.time(), 0.0, 0
        for idx in np.array_split(rng.permutation(n), max(1, n // batch_size)):
            p_ids, p_len, h_ids, h_len, y = _batch(train, torch.from_numpy(idx), device)
            loss = loss_fn(model(p_ids, p_len, h_ids, h_len), y)
            opt.zero_grad()
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(params, clip)
            opt.step()
            loss_sum += loss.item()
            steps += 1
        val_acc = lstm_accuracy(model, val)
        history.append({"epoch": epoch, "train_loss": loss_sum / steps, "val_acc": val_acc})
        if val_acc > best_acc:
            best_acc, best_state = val_acc, copy.deepcopy(model.state_dict())
        print(f"epoch {epoch}: train loss {loss_sum / steps:.3f} | val acc {val_acc:.2f}%  ({time.time() - t0:.0f}s)")
    model.load_state_dict(best_state)
    return model, history
