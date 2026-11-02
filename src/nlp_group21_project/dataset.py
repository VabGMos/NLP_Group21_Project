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
        self.premises, self.hypotheses, self.labels = (
            ds["premise"],
            ds["hypothesis"],
            ds["label"],
        )
        self.encoded_premises = None
        self.encoded_hypotheses = None
        self.premise_lengths = None
        self.hypothesis_lengths = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self.encoded_premises is not None:
            return {
                "premise_ids": self.encoded_premises[i],
                "hypothesis_ids": self.encoded_hypotheses[i],
                "premise_length": self.premise_lengths[i],
                "hypothesis_length": self.hypothesis_lengths[i],
                "label": self.labels[i],
            }
        return {
            "premise": self.premises[i],
            "hypothesis": self.hypotheses[i],
            "label": self.labels[i],
        }


def pad_sequences(
    seqs: List[List[int]], pad_id: int, max_length: int | None = None, dtype=torch.long
):
    max_len = max_length if max_length is not None else max(len(s) for s in seqs)
    padded = [s[:max_len] + [pad_id] * max(0, max_len - len(s)) for s in seqs]
    ids = torch.tensor(padded, dtype=dtype)
    lengths = torch.tensor([min(len(s), max_len) for s in seqs])
    mask = (torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)).long()
    return ids, mask


TOKENIZATION_BATCH_SIZE = 4096


class NLICollator:
    """Tokenizes and pads a batch with whichever tokenizer it is given."""

    def __init__(self, tokenizer: BaseTokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Dict]):
        if "premise_ids" in batch[0]:
            p_ids = torch.stack([b["premise_ids"] for b in batch]).long()
            h_ids = torch.stack([b["hypothesis_ids"] for b in batch]).long()
            p_lengths = torch.stack([b["premise_length"] for b in batch])
            h_lengths = torch.stack([b["hypothesis_length"] for b in batch])
            return {
                "premise_ids": p_ids,
                "premise_mask": (
                    torch.arange(p_ids.size(1)).unsqueeze(0) < p_lengths.unsqueeze(1)
                ).long(),
                "hypothesis_ids": h_ids,
                "hypothesis_mask": (
                    torch.arange(h_ids.size(1)).unsqueeze(0) < h_lengths.unsqueeze(1)
                ).long(),
                "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
            }

        p_ids, p_mask = pad_sequences(
            self.tokenizer.encode([b["premise"] for b in batch]),
            self.tokenizer.pad_token_id,
        )
        h_ids, h_mask = pad_sequences(
            self.tokenizer.encode([b["hypothesis"] for b in batch]),
            self.tokenizer.pad_token_id,
        )
        return {
            "premise_ids": p_ids,
            "premise_mask": p_mask,
            "hypothesis_ids": h_ids,
            "hypothesis_mask": h_mask,
            "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        }


def cache_tokenized_dataset(
    dataset: SNLIDataset,
    tokenizer: BaseTokenizer,
    batch_size: int = TOKENIZATION_BATCH_SIZE,
):
    max_length = tokenizer.max_length
    encoded_premises = torch.empty((len(dataset), max_length), dtype=torch.int32)
    encoded_hypotheses = torch.empty_like(encoded_premises)
    premise_lengths = torch.empty(len(dataset), dtype=torch.int32)
    hypothesis_lengths = torch.empty_like(premise_lengths)
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        premise_ids, premise_mask = pad_sequences(
            tokenizer.encode(dataset.premises[start:end]),
            tokenizer.pad_token_id,
            max_length=max_length,
            dtype=torch.int32,
        )
        hypothesis_ids, hypothesis_mask = pad_sequences(
            tokenizer.encode(dataset.hypotheses[start:end]),
            tokenizer.pad_token_id,
            max_length=max_length,
            dtype=torch.int32,
        )
        encoded_premises[start:end] = premise_ids
        encoded_hypotheses[start:end] = hypothesis_ids
        premise_lengths[start:end] = premise_mask.sum(1).to(torch.int32)
        hypothesis_lengths[start:end] = hypothesis_mask.sum(1).to(torch.int32)

    dataset.encoded_premises = encoded_premises
    dataset.encoded_hypotheses = encoded_hypotheses
    dataset.premise_lengths = premise_lengths
    dataset.hypothesis_lengths = hypothesis_lengths


def make_dataloader(dataset, tokenizer, batch_size=64, shuffle=False, num_workers=4):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=NLICollator(tokenizer),
    )


if __name__ == "__main__":
    dataset = load_snli_dataset()
    train_df, validation_df, test_df = convert_to_dataframe(dataset)
    print(dataset)

    print("Train DataFrame:", train_df.shape)
    print("Validation DataFrame:", validation_df.shape)
    print("Test DataFrame:", test_df.shape)
