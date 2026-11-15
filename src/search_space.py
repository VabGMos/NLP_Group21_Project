"""Search spaces for hyperparameter tuning (Optuna).

`suggest_params` returns a flat dict that `src.runner.train` accepts directly.
Add or edit entries here to change what is tuned; nothing else needs to change.
"""

import optuna

from src.embeddings import EMBEDDINGS


def suggest_params(
    trial: optuna.Trial,
    embeddings: list[str] | None = None,
    rnns: list[str] | None = None,
) -> dict:
    embeddings = embeddings or sorted(EMBEDDINGS)
    rnns = rnns or ["gru"]
    params = {
        # embedding
        "embedding": trial.suggest_categorical("embedding", embeddings),
        "train_embeddings": trial.suggest_categorical(
            "train_embeddings", [False, True]
        ),
        "embedding_dropout": trial.suggest_float("embedding_dropout", 0.0, 0.5),
        # recurrent encoder
        "rnn": trial.suggest_categorical("rnn", rnns),
        "hidden_size": trial.suggest_categorical(
            "hidden_size", [64, 128, 256, 384, 512]
        ),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "bidirectional": trial.suggest_categorical("bidirectional", [True, False]),
        "rnn_dropout": trial.suggest_float("rnn_dropout", 0.0, 0.5),
        # MLP head
        "mlp_layers": trial.suggest_int("mlp_layers", 1, 3),
        "mlp_width": trial.suggest_categorical("mlp_width", [128, 256, 512, 1024]),
        "mlp_dropout": trial.suggest_float("mlp_dropout", 0.0, 0.5),
        # optimisation
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
    }
    if params["train_embeddings"]:
        params["embedding_lr_scale"] = trial.suggest_float(
            "embedding_lr_scale", 0.01, 1.0, log=True
        )
    return params
