import pandas as pd
from src.predict import predict_one
from src.data_loader import load_dataset

def run_and_save(input_csv, output_csv):
    df = load_dataset(input_csv)
    results = []

    for _, row in df.iterrows():
        pred = predict_one(row)
        results.append({
            "id": row["id"],
            "prediction": pred
        })

    out = pd.DataFrame(results)
    out.to_csv(output_csv, index=False)
    print(f"\nSaved predictions to {output_csv}")

if __name__ == "__main__":
    run_and_save("test.csv", "results.csv")
