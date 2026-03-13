import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = r"E:\AI IN BUSINESS\final_project\processed"
MODEL_DIR = os.path.join(BASE_DIR, "models")

CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
FINAL_MODEL_DIR = os.path.join(MODEL_DIR, "phobert-sentiment")

MODEL_NAME = "vinai/phobert-base"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)