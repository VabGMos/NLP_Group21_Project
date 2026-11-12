from datasets import load_dataset


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
