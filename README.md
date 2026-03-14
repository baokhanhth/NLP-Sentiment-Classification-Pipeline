# NLP-Sentiment-Classification-Pipeline
---

## Project Overview
**NLP-Sentiment-Classification-Pipeline** is a comprehensive Natural Language Processing (NLP) system designed to classify Vietnamese text into three emotional categories: **Positive, Negative, and Neutral**.

This project goes beyond simple model training by focusing on building an optimized **End-to-End Pipeline**. It combines the abstractive summarization capabilities of **ViT5** with the deep contextual understanding of **PhoBERT**.

## Core Pipeline Technologies
The system is built on modern technical standards to ensure both performance and reliability:

* **Transfer Learning & Fine-tuning:** Fine-tuned **PhoBERT-Base** (State-of-the-art for Vietnamese) to accurately capture local linguistic nuances and slang.
* **Data Engineering & Optimization:**
    * **Text Cleaning:** Standardized text using Regex to remove noise (URLs, special characters).
    * **Conditional Summarization:** Utilized **ViT5-Base** to "refine" long comments (>30 words), ensuring input consistency and enhancing feature extraction efficiency.
* **Deep Learning Engineering:** Optimized hardware resources using **Mixed Precision Training (fp16)**, **Gradient Accumulation**, and **Cosine Annealing Scheduler**.
* **Deployment:** Implemented a real-time inference interface via **Gradio** for instant testing and feedback.

## Experimental Results (Quality Assurance Focus)
With a career focus on **Intern Test/QA** roles, this project emphasizes error analysis and model specificity evaluated on the **ViCTSD** dataset:

### General Performance Metrics:
* **Overall Accuracy:** 76.75%
* **F1-Score (Macro):** 0.76

### Tester Mindset & Quality Control:
* **Critical Error Rate:** Achieved a low of **3.9%**. This is the most vital metric, representing the system's ability to minimize harmful misclassifications between opposing polarities (Positive ↔ Negative).
* **Positive Precision (82.55%):** Ensures extremely high reliability when recording satisfied customer feedback.
* **Specificity Optimization:** Focused on reducing "false alarms," protecting the system from mislabeling negative or toxic comments as neutral/positive.

## Future Roadmap (Application Orientations)
The project is designed with a vision for integration into real-world systems:

* **Auto-Moderation:** Developing an automated content moderation system for social groups (e.g., Pet Adoption groups or Gaming forums).
* **Mental Health Tracking:** Providing a supportive tool to monitor user psychological trends through text-based analysis.
* **API Integration:** Building a centralized API to allow external systems to call and utilize the sentiment processing model.

## Source Code Structure
```
├── data/                    # Raw datasets (ViCTSD)
├── processed/               # Pipeline output (Cleaned & Summarized data)
├── models/                  # Model configurations (Weights excluded due to size)
├── src/                     # Core logic (Preprocess, Train, Evaluate)
├── run_*.py                 # Automated workflow execution scripts
└── README.md                # Project documentation and reports
```
## Live Demo & Deployment

This project is deployed on **Hugging Face Spaces** using **Gradio** for a real-time user interface. You can test the model's performance directly without any local installation.

### [Live Demo](https://huggingface.co/spaces/Arus152/NLP-Sentiment-Classification-Pipeline)

### Deployment Technical Stack:
* **Host:** Hugging Face Spaces (CPU Basic).
* **Framework:** Gradio for the UI.
* **Storage:** Managed via **Git LFS** to handle large model weights.
* 
### 📖 How to use:
1. Enter any Vietnamese text (comment, review, or feedback) into the textbox.
2. Click **Submit**.
3. The system will classify the text into: **Positive, Negative, or Neutral** along with the confidence score.
