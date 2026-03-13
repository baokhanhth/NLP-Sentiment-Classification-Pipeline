from src.build_model import build_model
from src.config import MODEL_NAME
import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

model, compute_metrics = build_model(model_name=MODEL_NAME, device=device)

build_dir = os.path.join("models", "build_temp")
os.makedirs(build_dir, exist_ok=True)
model.save_pretrained(build_dir)

print(f"Model đã build xong và lưu tạm tại: {build_dir}")
