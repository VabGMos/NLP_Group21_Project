"""LightningModule for NLI pair classification (embedding -> recurrent encoder -> MLP)."""

import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

import wandb
from src.dataset import LABEL_NAMES
from src.models import build_pair_model


class NLIModule(L.LightningModule):
    """All keyword arguments are logged as hyperparameters (the weights are not).

    embedding_lr_scale multiplies the learning rate of the embedding table when it is
    trainable. The MLP has `mlp_layers` hidden layers of `mlp_width` units each.
    To reload a checkpoint: NLIModule.load_from_checkpoint(path, embedding_weights=w,
    pad_id=pad_id) with any tensor w of the right shape (the saved weights override it).
    """

    def __init__(
        self,
        embedding_weights: torch.Tensor,
        pad_id: int,
        num_classes: int = len(LABEL_NAMES),
        rnn: str = "gru",
        hidden_size: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        train_embeddings: bool = False,
        embedding_lr_scale: float = 1.0,
        embedding_dropout: float = 0.0,
        rnn_dropout: float = 0.2,
        mlp_layers: int = 1,
        mlp_width: int = 256,
        mlp_dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["embedding_weights"])
        self.model = build_pair_model(
            embedding_weights,
            pad_id,
            num_classes=num_classes,
            rnn_type=rnn,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            freeze_embeddings=not train_embeddings,
            embedding_dropout=embedding_dropout,
            rnn_dropout=rnn_dropout,
            mlp_hidden_dims=(mlp_width,) * mlp_layers,
            mlp_dropout=mlp_dropout,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = nn.ModuleDict(
            {
                f"{stage}_{name}": cls(num_classes=num_classes, **kwargs)
                for stage in ("val", "test")
                for name, cls, kwargs in (
                    ("accuracy", MulticlassAccuracy, {}),
                    ("macro_f1", MulticlassF1Score, {"average": "macro"}),
                )
            }
        )
        self._test_preds, self._test_targets = [], []

    def forward(self, p_ids, p_mask, h_ids, h_mask):
        return self.model(p_ids, p_mask, h_ids, h_mask)

    def _step(self, batch, stage):
        p_ids, p_mask, h_ids, h_mask, labels = batch
        logits = self(p_ids, p_mask, h_ids, h_mask)
        loss = self.criterion(logits, labels)
        batch_size = len(labels)
        self.log(
            f"{stage}/loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        if stage != "train":
            preds = logits.argmax(dim=1)
            for name in ("accuracy", "macro_f1"):
                metric = self.metrics[f"{stage}_{name}"]
                metric(preds, labels)
                self.log(
                    f"{stage}/{name}",
                    metric,
                    on_epoch=True,
                    prog_bar=(name == "accuracy"),
                    batch_size=batch_size,
                )
        return loss, logits

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")[0]

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        _, logits = self._step(batch, "test")
        self._test_preds.append(logits.argmax(dim=1).cpu())
        self._test_targets.append(batch[-1].cpu())

    def on_test_epoch_end(self):
        preds = torch.cat(self._test_preds).tolist()
        targets = torch.cat(self._test_targets).tolist()
        self._test_preds, self._test_targets = [], []
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log(
                {
                    "test/confusion_matrix": wandb.plot.confusion_matrix(
                        y_true=targets, preds=preds, class_names=LABEL_NAMES
                    )
                }
            )

    def configure_optimizers(self):
        hp = self.hparams
        embedding = [
            p for p in self.model.encoder.embedding.parameters() if p.requires_grad
        ]
        embedding_ids = {id(p) for p in embedding}
        rest = [
            p
            for p in self.parameters()
            if p.requires_grad and id(p) not in embedding_ids
        ]
        groups = [{"params": rest, "lr": hp.lr}]
        if embedding:
            groups.append({"params": embedding, "lr": hp.lr * hp.embedding_lr_scale})
        return torch.optim.AdamW(groups, lr=hp.lr, weight_decay=hp.weight_decay)
