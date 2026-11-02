from attr import dataclass

import torch
import random
import numpy as np


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    model_name: str = "mlp"  # key in MODEL_REGISTRY
    tokenizer_name: str = "sentence-transformers/nli-mpnet-base-v2"
    max_length: int = 32
    batch_size: int = 2048
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-6
    max_train_samples: int | None = None  # set e.g. 50_000 for quick runs
    seed: int = 42
    cv_folds: int = 5  # set >1 for k-fold cross-validation on the training split
    num_workers: int = 16
    wandb_project: str = "nlp-group21-project"
    wandb_mode: str = "online"
    checkpoint_dir: str = "checkpoints"
