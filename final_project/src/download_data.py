from datasets import load_dataset
import pandas as pd
import os

DATA_DIR = "E:/AI IN BUSINESS/final_project/data"
os.makedirs(DATA_DIR, exist_ok=True)

dataset = load_dataset("tarudesu/ViCTSD")

dataset['train'].to_csv(f"{DATA_DIR}/train.csv", index=False)
dataset['validation'].to_csv(f"{DATA_DIR}/valid.csv", index=False)
dataset['test'].to_csv(f"{DATA_DIR}/test.csv", index=False)

print("Dataset đã được lưu vào folder data/")