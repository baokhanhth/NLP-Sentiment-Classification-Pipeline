import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import MODEL_DIR
from src.utils import clean_text

MODEL_PATH = "models/sentiment-phobert"
device = "cuda" if torch.cuda.is_available() else "cpu"

LABELS = ["Tiêu cực", "Bình thường", "Tích cực"]
COLORS = ["#FF4D4D", "#FFD700", "#32CD32"] 

print(f"Đang load model từ: {MODEL_PATH}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    print("Load model thành công!")
except Exception as e:
    print(f"Lỗi load model: {e}")
    model = None
    tokenizer = None

# 2. Hàm Dự đoán
def predict_sentiment(text):
    if model is None:
        return "Lỗi: Model chưa được load.", None

    cleaned_text = clean_text(text)
    if not cleaned_text:
        return "Vui lòng nhập nội dung hợp lệ.", None

    enc = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().cpu().numpy()

    scores = {LABELS[i]: float(probs[i]) for i in range(3)}
    predicted_index = np.argmax(probs)
    conclusion = LABELS[predicted_index]
    
    text_output = f"Nội dung gốc: {text}\n"
    text_output += f"Nội dung sau xử lý: {cleaned_text}\n\n"
    text_output += "-"*20 + "\n"
    for label, score in scores.items():
        text_output += f"{label}: {score*100:.2f}%\n"
    text_output += "-"*20 + "\n"
    text_output += f"➡ KẾT LUẬN: {conclusion.upper()}"

    fig, ax = plt.subplots(figsize=(8, 5))
    wedges, texts, autotexts = ax.pie(
        probs,
        autopct="%1.2f%%", # Hiển thị 2 số thập phân
        startangle=90,
        colors=COLORS,
        # Chữ % màu trắng, to rõ (size 16)
        textprops=dict(color="white", fontsize=16), 
        wedgeprops=dict(width=0.6, edgecolor='w'),
        pctdistance=0.75 
    )
    ax.legend(
        wedges, LABELS, 
        title="Phân loại",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    
    plt.setp(autotexts, size=16, weight="bold")
    ax.set_title(f"Kết quả: {conclusion}", fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    return text_output, fig

custom_css = """
.gradio-container {font-family: 'Roboto', sans-serif;}
h1 {
    text-align: center; 
    color: #2c3e50; 
    font-weight: bold; 
    margin-bottom: 10px;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="PhoBERT Sentiment Analysis") as demo:

    gr.Markdown("# ỨNG DỤNG AI TRONG PHÂN TÍCH VÀ ĐÁNH GIÁ THÁI ĐỘ NGƯỜI DÙNG QUA VĂN BẢN")
    
    gr.Markdown("<div style='text-align: center;'>Nhập nội dung cần phân tích vào bên dưới để hệ thống dự đoán mức độ: <b>Tích cực - Tiêu cực - Trung tính</b></div>")
    
    with gr.Row():
        input_box = gr.Textbox(
            label="Nhập nội dung cần phân tích", 
            placeholder="Ví dụ: Sản phẩm dùng rất tốt, giao hàng nhanh...",
            lines=3
        )
    
    with gr.Row():
        submit_btn = gr.Button("Phân tích ngay", variant="primary", scale=1)
        clear_btn = gr.Button("Xóa", variant="secondary", scale=0)

    with gr.Row():
        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Chi tiết phân tích", 
                lines=12, 
                interactive=False
            )
        
        with gr.Column(scale=1):
            output_plot = gr.Plot(label="Biểu đồ trực quan")

    submit_btn.click(
        fn=predict_sentiment, 
        inputs=input_box, 
        outputs=[output_text, output_plot]
    )
    
    clear_btn.click(
        lambda: (None, None, None),
        inputs=None,
        outputs=[input_box, output_text, output_plot]
    )

iface = demo