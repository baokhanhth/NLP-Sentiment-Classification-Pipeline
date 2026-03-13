# src/preprocess.py
import os
import pandas as pd
from transformers import pipeline
from .config import PROCESSED_DIR
from .utils import clean_text   

os.makedirs(PROCESSED_DIR, exist_ok=True)



summarizer = pipeline("summarization", model="VietAI/vit5-base-vietnews-summarization", device=-1)

def summarize_conditional(texts, word_threshold=30):
    """
    Chỉ tóm tắt các văn bản dài hơn word_threshold (30 từ).
    Giúp tránh lỗi 'lon lon lon...' khi ép model tóm tắt câu quá ngắn.
    """
    long_text_indices = [i for i, t in enumerate(texts) if len(t.split()) > word_threshold]
    long_texts = [texts[i] for i in long_text_indices]

    if not long_texts:
        return texts

    try:
        summaries = summarizer(
            long_texts,
            max_length=64,
            min_length=10,
            truncation=True,
            batch_size=8
        )
        summary_values = [s["summary_text"] for s in summaries]
    except Exception as e:
        print(f"Lỗi summarize: {e}")
        return texts

    final_texts = texts[:]
    for idx, summary_text in zip(long_text_indices, summary_values):
        final_texts[idx] = summary_text
    
    return final_texts

sentiment_pipeline = pipeline("sentiment-analysis", model="wonrax/phobert-base-vietnamese-sentiment", device=-1)

def get_sentiment_batch(texts):
    """
    Dự đoán cảm xúc và map về số:
    NEG -> 0
    NEU -> 1
    POS -> 2
    """
    outputs = sentiment_pipeline(texts, truncation=True, max_length=256)
    
    labels_mapped = []
    for o in outputs:
        lbl = o["label"]
        if lbl == "NEG":
            labels_mapped.append(0) # Negative
        elif lbl == "POS":
            labels_mapped.append(2) # Positive
        else:
            labels_mapped.append(1) # Neutral (NEU)
            
    return labels_mapped

def preprocess_data(dfs, summarize=True, sentiment_label=True, batch_size=32):
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    for split in dfs:
        df = dfs[split]
        print(f"\n=== PROCESSING {split.upper()} ===")

        out_path = os.path.join(PROCESSED_DIR, f"{split}_processed.csv")
        checkpoint_file = os.path.join(PROCESSED_DIR, f"{split}_checkpoint.txt")

        start = 0
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "r") as f:
                content = f.read().strip()
                if content:
                    start = int(content)
            print(f"→ Resume từ dòng {start}")

        if not os.path.exists(out_path) or start == 0:
            pd.DataFrame(columns=["text", "label"]).to_csv(out_path, index=False)

        total_rows = len(df)
        for i in range(start, total_rows, batch_size):
            batch = df.iloc[i : min(i + batch_size, total_rows)].copy()
            
            batch["Comment"] = batch["Comment"].fillna("")
            batch["text"] = batch["Comment"].apply(clean_text)
            current_texts = batch["text"].tolist()

            if summarize:
                print(f"→ Summarizing batch {i} (Conditional)")
                batch["text"] = summarize_conditional(current_texts, word_threshold=30)

            if sentiment_label:
                print(f"→ Sentiment batch {i}")
                final_input_texts = batch["text"].tolist()
                
                safe_inputs = [t if len(str(t).strip()) > 0 else "bình thường" for t in final_input_texts]
                
                batch["label"] = get_sentiment_batch(safe_inputs)

            batch[["text", "label"]].to_csv(out_path, mode="a", header=False, index=False)

            with open(checkpoint_file, "w") as f:
                f.write(str(i + len(batch)))

            print(f" Đã lưu batch {i}")

    print("=== DONE ALL SPLITS ===")
    return dfs