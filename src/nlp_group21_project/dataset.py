from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset

from tokenizer import BaseTokenizer
from typing import Dict, List

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


class SNLIDataset(Dataset):
    def __init__(self, split: str, max_samples: int | None = None):
        ds = load_dataset("stanfordnlp/snli", split=split)
        # drop pairs without gold label
        ds = ds.filter(lambda x: x["label"] != -1)
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        self.premises, self.hypotheses, self.labels = ds["premise"], ds["hypothesis"], ds["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"premise": self.premises[i], "hypothesis": self.hypotheses[i],
                "label": self.labels[i]}


def pad_sequences(seqs: List[List[int]], pad_id: int):
    max_len = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    return ids, mask


class NLICollator:
    """Tokenizes and pads a batch with whichever tokenizer it is given."""

    def __init__(self, tokenizer: BaseTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict]):
        p_ids, p_mask = pad_sequences(
            self.tokenizer.encode([b["premise"] for b in batch]), self.tokenizer.pad_token_id)
        h_ids, h_mask = pad_sequences(
            self.tokenizer.encode([b["hypothesis"] for b in batch]), self.tokenizer.pad_token_id)
        return {
            "premise_ids": p_ids, "premise_mask": p_mask,
            "hypothesis_ids": h_ids, "hypothesis_mask": h_mask,
            "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        }


def make_dataloader(dataset, tokenizer, batch_size=64, shuffle=False, num_workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=NLICollator(tokenizer))


if __name__ == "__main__":
    dataset = load_snli_dataset()
    train_df, validation_df, test_df = convert_to_dataframe(dataset)
    print(dataset)

    print("Train DataFrame:", train_df.shape)
    print("Validation DataFrame:", validation_df.shape)
    print("Test DataFrame:", test_df.shape)
