
import pandas as pd

def load_data(file_path):
  
    try:
        df = pd.read_csv(file_path)
        print(f" Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f" Failed to load data: {e}")
        raise
