import pandas as pd

from src.pathway_index import build_vector_index

def load_dataset(csv_path):
    return pd.read_csv(csv_path)
