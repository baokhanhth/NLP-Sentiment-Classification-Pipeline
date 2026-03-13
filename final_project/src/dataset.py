from datasets import load_dataset
import pandas as pd

dataset = load_dataset("tarudesu/ViCTSD")
dataset['train'].to_csv("E:/AI IN BUSINESS/final_project/data/train.csv", index=False)
dataset['validation'].to_csv("E:/AI IN BUSINESS/final_project/data/valid.csv", index=False)
dataset['test'].to_csv("E:/AI IN BUSINESS/final_project/data/test.csv", index=False)
