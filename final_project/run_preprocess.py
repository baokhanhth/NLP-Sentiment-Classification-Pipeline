from src.preprocess import preprocess_data
from src.load_data import load_data  
dfs = load_data()
dfs = preprocess_data(dfs, summarize=True, sentiment_label=True)
