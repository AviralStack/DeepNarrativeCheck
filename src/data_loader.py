import pandas as pd
# CHANGED: Import the correct function name
from src.pathway_index import build_vector_index

def load_dataset(csv_path):
    return pd.read_csv(csv_path)

# We can remove the old manual loader functions since Pathway handles it now