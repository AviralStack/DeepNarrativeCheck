import pandas as pd
from src.predict import predict_one
from src.data_loader import load_dataset
from src.pathway_index import build_vector_index

def run_and_save(input_csv, output_csv):
    # 1. Load Data
    df = load_dataset(input_csv)
    results = []

    # 2. Build Index ONCE (Saves time)
    print("Building Vector Index... (This might take 1-2 mins)")
    vector_store = build_vector_index("./Books")
    
    # 3. Loop through rows
    print(f"Processing {len(df)} rows...")
    for index, row in df.iterrows():
        print(f"\n--- Processing Row {index} ---")
        # Ensure we pass the vector_store here!
        pred, rationale = predict_one(row, vector_store)
        
        results.append({
            "Story ID": row.get("id", index),
            "Prediction": 1 if pred == "consistent" else 0,
            "Rationale": rationale
        })

    # 4. Save
    out = pd.DataFrame(results)
    out.to_csv(output_csv, index=False)
    print(f"\nSaved predictions to {output_csv}")

if __name__ == "__main__":
    run_and_save("test.csv", "results.csv")