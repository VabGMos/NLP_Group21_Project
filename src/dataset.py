import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from src.tokenizer import tokenize_texts

LABEL_NAMES = ["entailment", "neutral", "contradiction"]


def load_snli_dataset():
    return load_dataset("stanfordnlp/snli")


def convert_to_dataframe(dataset):
    # Convert each split to a Pandas DataFrame
    train_df = dataset["train"].to_pandas()
    validation_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()

    # Remove invalid labels (-1)
    train_df = train_df[train_df["label"] != -1]
    validation_df = validation_df[validation_df["label"] != -1]
    test_df = test_df[test_df["label"] != -1]

    return train_df, validation_df, test_df


class PairDataset(Dataset):
    """Tokenized (premise, hypothesis, label) triples; padding happens per batch."""

    def __init__(self, premises, hypotheses, labels, tokenizer, max_length=128):
        self.premise_ids = tokenize_texts(list(premises), tokenizer, max_length)
        self.hypothesis_ids = tokenize_texts(list(hypotheses), tokenizer, max_length)
        self.labels = [int(label) for label in labels]

    @classmethod
    def from_dataframe(cls, df, tokenizer, max_length=128):
        return cls(df["premise"], df["hypothesis"], df["label"], tokenizer, max_length)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.premise_ids[idx], self.hypothesis_ids[idx], self.labels[idx]


class PairCollator:
    """Pads a batch to its longest sequence; returns ids, masks and labels."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def _pad(self, sequences):
        length = max(len(seq) for seq in sequences)
        ids = torch.full((len(sequences), length), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(sequences), length), dtype=torch.long)
        for row, seq in enumerate(sequences):
            ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[row, : len(seq)] = 1
        return ids, mask

    def __call__(self, batch):
        premises, hypotheses, labels = zip(*batch)
        p_ids, p_mask = self._pad(premises)
        h_ids, h_mask = self._pad(hypotheses)
        return p_ids, p_mask, h_ids, h_mask, torch.tensor(labels, dtype=torch.long)


def make_pair_loader(dataset, pad_id, batch_size=128, shuffle=False, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=PairCollator(pad_id),
        num_workers=num_workers,
    )
