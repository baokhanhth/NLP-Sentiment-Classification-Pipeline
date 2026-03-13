# src/load_data.py
import os
import pandas as pd
from datasets import load_dataset

from .config import DATA_DIR

def load_data(dataset_name="tarudesu/ViCTSD"):
    train_path = os.path.join(DATA_DIR, "train.csv")
    valid_path = os.path.join(DATA_DIR, "valid.csv")
    test_path  = os.path.join(DATA_DIR, "test.csv")

    if os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(test_path):
        print("Loading data from local CSV files in", DATA_DIR)
        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)
        test_df  = pd.read_csv(test_path)
    else:
        print(f"Local CSV not found; downloading dataset '{dataset_name}' from Hugging Face...")
        ds = load_dataset(dataset_name)

        train_df = pd.DataFrame(ds["train"]) if "train" in ds else pd.DataFrame()

        if "validation" in ds:
            valid_df = pd.DataFrame(ds["validation"])
        elif "dev" in ds:
            valid_df = pd.DataFrame(ds["dev"])
        else:
            if not train_df.empty:
                valid_df = train_df.sample(frac=0.1, random_state=42).reset_index(drop=True)
                train_df = train_df.drop(valid_df.index).reset_index(drop=True)
            else:
                valid_df = pd.DataFrame()

        if "test" in ds:
            test_df = pd.DataFrame(ds["test"])
        else:
            test_df = pd.DataFrame()

    for k, df in [("train", train_df), ("validation", valid_df), ("test", test_df)]:
        if isinstance(df, pd.DataFrame):
            df.reset_index(drop=True, inplace=True)
        else:
            df = pd.DataFrame()

    dfs = {"train": train_df, "validation": valid_df, "test": test_df}
    print("Loaded split sizes:", {k: v.shape for k, v in dfs.items()})
    return dfs
