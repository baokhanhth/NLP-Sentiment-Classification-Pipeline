import pandas as pd
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

from src.config import PROCESSED_DIR, MODEL_DIR

def evaluate_model(batch_size=32, model_dir=None, device=None):
    if model_dir is None:
        model_dir = os.path.join(MODEL_DIR, "sentiment-phobert")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from: {model_dir}")

    test_path = os.path.join(PROCESSED_DIR, "test_processed.csv")
    if not os.path.exists(test_path):
        print(f"Không tìm thấy file dữ liệu: {test_path}")
        return

    df = pd.read_csv(test_path)
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip().str.len() > 0]
    
    texts = df["text"].tolist()
    labels_true = df["label"].fillna(1).astype(int).tolist()

    print(f"Data loaded: {len(texts)} samples")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        model.eval()
    except Exception as e:
        print(f"Lỗi load model: {e}")
        print("Hãy chắc chắn bạn đã chạy train model thành công!")
        return

    preds = []
    print("Đang chạy dự đoán trên tập test...")
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(device)
            
            outputs = model(**enc)
            batch_preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
            preds.extend(batch_preds)

    target_names = ["Negative", "Neutral", "Positive"]
    
    print("\n" + "="*30)
    print("=== Classification Report ===")
    print("="*30)
    print(classification_report(labels_true, preds, target_names=target_names, digits=4))

    cm = confusion_matrix(labels_true, preds)

    TP = cm.diagonal()
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)
    
    specificity = TN / (TN + FP + 1e-9)

    print("\n=== Detailed Metrics (per class) ===")
    print(f"{'Class':<10} {'TP':<6} {'TN':<6} {'FP':<6} {'FN':<6} {'Specificity':<10}")
    print("-" * 60)
    for idx, cls in enumerate(target_names):
        print(f"{cls:<10} {TP[idx]:<6} {TN[idx]:<6} {FP[idx]:<6} {FN[idx]:<6} {specificity[idx]:.4f}")

    print("\nĐang vẽ bảng Detailed Metrics (Image)...")
    
    cell_text = []
    for idx, cls in enumerate(target_names):
        cell_text.append([
            cls, 
            str(TP[idx]), 
            str(TN[idx]), 
            str(FP[idx]), 
            str(FN[idx]), 
            f"{specificity[idx]:.4f}"
        ])
    
    columns = ["Class", "TP", "TN", "FP", "FN", "Specificity"]
    
    fig_table, ax_table = plt.subplots(figsize=(10, 3))
    ax_table.axis('tight')
    ax_table.axis('off')
    
    the_table = ax_table.table(
        cellText=cell_text,
        colLabels=columns,
        loc='center',
        cellLoc='center'
    )
    
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    the_table.scale(1.2, 2)
    
    for (row, col), cell in the_table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2c3e50')
        else:
            if row % 2 == 0:
                cell.set_facecolor('#ecf0f1')
            else:
                cell.set_facecolor('white')

    plt.title("Detailed Performance Metrics", fontsize=14, weight='bold', pad=10)
    
    save_table_path = os.path.join(MODEL_DIR, "detailed_metrics_table.png")
    plt.savefig(save_table_path, bbox_inches='tight', dpi=300)
    print(f"Đã lưu ảnh bảng metrics tại: {save_table_path}")

    print("\nĐang vẽ Confusion Matrix...")
    
    plt.figure(figsize=(8, 6))
    sns.set_context("notebook", font_scale=1.2)
    
    sns.heatmap(
        cm, 
        annot=True,         
        fmt='d',            
        cmap='Blues',       
        xticklabels=target_names, 
        yticklabels=target_names
    )
    
    plt.xlabel('Dự đoán (Predicted)', fontsize=13)
    plt.ylabel('Thực tế (Actual)', fontsize=13)
    plt.title('Confusion Matrix - PhoBERT Sentiment', fontsize=15, fontweight='bold')
    
    save_img_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(save_img_path)
    print(f"Đã lưu ảnh Confusion Matrix tại: {save_img_path}")
    
    plt.show()