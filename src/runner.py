"""One place that turns a flat hyperparameter dict into a trained model.

Shared by the training CLI and the tuner so both use identical code paths.
"""

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.embeddings import load_embedding_weights
from src.lit_module import NLIModule

# Keys of `params` that are not NLIModule arguments.
TRAINER_KEYS = ("embedding", "batch_size", "clip_norm")


def make_wandb_logger(project, name=None, entity=None, group=None, mode="online", **kw):
    """WandbLogger; mode is "online", "offline" or "disabled"."""
    return WandbLogger(
        project=project, name=name, entity=entity, group=group, mode=mode, **kw
    )


DEFAULT_PARAMS = {
    "embedding": "all-mpnet-base-v2",
    "batch_size": 128,
    "clip_norm": 1.0,
    "rnn": "gru",
    "pooling": None,  # None keeps the recurrent encoder; otherwise mean/sum/max.
    "hidden_size": 128,
    "num_layers": 1,
    "bidirectional": True,
    "train_embeddings": False,
    "embedding_lr_scale": 1.0,
    "embedding_dropout": 0.0,
    "rnn_dropout": 0.2,
    "mlp_layers": 1,
    "mlp_width": 256,
    "mlp_dropout": 0.2,
    "lr": 1e-3,
    "weight_decay": 1e-4,
}


def train(
    params: dict,
    datamodule,
    *,
    max_epochs: int = 10,
    patience: int = 3,
    monitor: str = "val/accuracy",
    logger=None,
    callbacks=(),
    checkpoint_dir: str | None = None,
    enable_progress_bar: bool = True,
):
    """Fit one model for `params` (missing keys fall back to DEFAULT_PARAMS).

    Returns (trainer, module). Early-stops on `monitor`; if checkpoint_dir is given the
    best checkpoint is kept there (trainer.checkpoint_callback.best_model_path).
    """
    params = {**DEFAULT_PARAMS, **params}
    module_params = {k: v for k, v in params.items() if k not in TRAINER_KEYS}
    datamodule.batch_size = params["batch_size"]

    module = NLIModule(
        load_embedding_weights(params["embedding"]),
        datamodule.pad_id,
        **module_params,
    )
    if logger is not None:
        logger.log_hyperparams(
            {"embedding": params["embedding"], "batch_size": params["batch_size"]}
        )

    all_callbacks = [EarlyStopping(monitor, mode="max", patience=patience), *callbacks]
    if checkpoint_dir is not None:
        all_callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                monitor=monitor,
                mode="max",
                save_top_k=1,
                filename="best",
            )
        )
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        logger=logger if logger is not None else False,
        callbacks=all_callbacks,
        gradient_clip_val=params["clip_norm"],
        enable_checkpointing=checkpoint_dir is not None,
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=False,
        log_every_n_steps=10,
    )
    trainer.fit(module, datamodule=datamodule)
    return trainer, module
