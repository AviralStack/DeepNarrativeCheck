import pandas as pd
from src.pathway_index import build_index


def load_dataset(csv_path):
    return pd.read_csv(csv_path)

def load_book(book_name):
    path = f"Books/{book_name}.txt"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_and_index_books():
    books = {}
    for name in ["In Search of the Castaways", "The Count of Monte Cristo"]:
        with open(f"Books/{name}.txt", "r", encoding="utf-8") as f:
            books[name] = f.read()
    index = build_index(books)
    return index, books