import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, TrainerCallback
)
from src.config import PROCESSED_DIR, MODEL_DIR
from tqdm.auto import tqdm

label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {0: "negative", 1: "neutral", 2: "positive"}

device = "cuda" if torch.cuda.is_available() else "cpu"

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class ProgressBarCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        total = state.max_steps if state.max_steps is not None else 0
        self.pbar = tqdm(total=total, desc="Training", dynamic_ncols=True)

    def on_step_end(self, args, state, control, **kwargs):
        if hasattr(self, "pbar"):
            self.pbar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        if hasattr(self, "pbar"):
            self.pbar.close()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])

    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)

    precision = np.mean(TP / (TP + FP + 1e-8))
    recall = np.mean(TP / (TP + FN + 1e-8))
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    accuracy = np.sum(TP) / np.sum(cm)
    specificity = np.mean(TN / (TN + FP + 1e-8))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
    }


def train_model(build_temp_dir):

    print("\n===== LOADING PROCESSED DATA =====\n")

    train_df = pd.read_csv(f"{PROCESSED_DIR}/train_processed.csv")
    valid_df = pd.read_csv(f"{PROCESSED_DIR}/validation_processed.csv")

    train_texts = train_df["text"].tolist()
    train_labels = train_df["label"].tolist()

    valid_texts = valid_df["text"].tolist()
    valid_labels = valid_df["label"].tolist()

    tokenizer = AutoTokenizer.from_pretrained(build_temp_dir, use_fast=False)

    train_enc = tokenizer(train_texts, truncation=True, padding="max_length", max_length=256)
    valid_enc = tokenizer(valid_texts, truncation=True, padding="max_length", max_length=256)

    train_ds = TextDataset(train_enc, train_labels)
    valid_ds = TextDataset(valid_enc, valid_labels)

    print("\n===== LOADING MODEL FROM BUILD_TEMP =====\n")

    model = AutoModelForSequenceClassification.from_pretrained(
        build_temp_dir,
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    ).to(device)

    args = TrainingArguments(
        output_dir=f"{MODEL_DIR}/checkpoints",
        overwrite_output_dir=False,

        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        num_train_epochs=4,

        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,

        fp16=True,

        lr_scheduler_type="cosine",

        eval_strategy="epoch", 
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        logging_steps=100,
        save_total_limit=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[ProgressBarCallback()]
    )

    print("\n===== START TRAINING =====\n")

    last_checkpoint = None
    checkpoints_dir = f"{MODEL_DIR}/checkpoints"

    if os.path.exists(checkpoints_dir):
        ckpts = [os.path.join(checkpoints_dir, x) for x in os.listdir(checkpoints_dir) if x.startswith("checkpoint")]
        if ckpts:
            last_checkpoint = sorted(ckpts)[-1]
            print(f"Resuming from checkpoint: {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    final_path = f"{MODEL_DIR}/sentiment-phobert"
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    print(f"\n=== TRAINING COMPLETE. MODEL SAVED TO: {final_path} ===\n")