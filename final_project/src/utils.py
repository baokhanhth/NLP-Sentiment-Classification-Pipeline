import re

def clean_text(text):
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9à-ỹ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
