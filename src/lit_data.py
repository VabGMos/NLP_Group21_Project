import lightning as L

from src.dataset import (
    PairDataset,
    convert_to_dataframe,
    load_snli_dataset,
    make_pair_loader,
)
from src.tokenizer import load_mpnet_tokenizer


class SNLIDataModule(L.LightningDataModule):
    """SNLI pairs tokenized once with the MPNet tokenizer.

    `batch_size` is read every time a loader is built, so one instance can be reused
    across fits with different batch sizes (e.g. hyperparameter-tuning trials).
    """

    def __init__(
        self,
        max_length: int = 128,
        batch_size: int = 128,
        max_train_samples: int | None = None,
        max_eval_samples: int | None = None,
        num_workers: int = 0,
        seed: int = 42,
        tokenizer=None,
    ):
        super().__init__()
        self.max_length = max_length
        self.batch_size = batch_size
        self.max_train_samples = max_train_samples
        self.max_eval_samples = max_eval_samples
        self.num_workers = num_workers
        self.seed = seed
        self.tokenizer = tokenizer or load_mpnet_tokenizer()
        self.pad_id = self.tokenizer.pad_token_id
        self.datasets = {}

    def _subsample(self, df, n):
        return (
            df if n is None or n >= len(df) else df.sample(n=n, random_state=self.seed)
        )

    def setup(self, stage=None):
        if self.datasets:
            return
        train, val, test = convert_to_dataframe(load_snli_dataset())
        splits = {
            "train": self._subsample(train, self.max_train_samples),
            "val": self._subsample(val, self.max_eval_samples),
            "test": self._subsample(test, self.max_eval_samples),
        }
        self.datasets = {
            name: PairDataset.from_dataframe(df, self.tokenizer, self.max_length)
            for name, df in splits.items()
        }

    def _loader(self, split, shuffle=False):
        return make_pair_loader(
            self.datasets[split],
            self.pad_id,
            self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return self._loader("train", shuffle=True)

    def val_dataloader(self):
        return self._loader("val")

    def test_dataloader(self):
        return self._loader("test")
