import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from .config import MODEL_NAME, PROCESSED_DIR

PRETRAINED_TOKENIZER = MODEL_NAME  

tokenizer = AutoTokenizer.from_pretrained(
    PRETRAINED_TOKENIZER,
    use_fast=False,
    add_prefix_space=True
)

def load_processed():
    return {
        "train": pd.read_csv(f"{PROCESSED_DIR}/train_processed.csv"),
        "validation": pd.read_csv(f"{PROCESSED_DIR}/validation_processed.csv"),
        "test": pd.read_csv(f"{PROCESSED_DIR}/test_processed.csv")
    }

def tokenize_datasets(max_length=256):

    dfs = load_processed()
    datasets = {}

    def tok(batch):
        texts = batch["text"]

        texts = [
            "" if t is None else str(t).strip()
            for t in texts
        ]

        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    for split, df in dfs.items():

        if "label" in df.columns:
            df["label"] = df["label"].fillna(0).astype(int)

        ds = Dataset.from_pandas(df)

        ds = ds.map(tok, batched=True)

        if "__index_level_0__" in ds.column_names:
            ds = ds.remove_columns(["__index_level_0__"])

        if "label" in ds.column_names:
            ds = ds.rename_column("label", "labels")

        ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )

        datasets[split] = ds

    return datasets["train"], datasets["validation"], datasets["test"]
