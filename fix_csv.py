import pandas as pd

# 1. Load your results
df = pd.read_csv("results.csv")

# 2. Check if values are strings and convert them
# (If they are already 0 and 1, this won't hurt anything)
def clean_prediction(val):
    s = str(val).lower().strip()
    if "consistent" in s:
        return 1
    elif "contradict" in s:
        return 0
    elif s == "1" or s == "1.0":
        return 1
    elif s == "0" or s == "0.0":
        return 0
    else:
        return 0 # Default fallback

df["Prediction"] = df["Prediction"].apply(clean_prediction)

# 3. Ensure columns are exactly what the judges want
# The rules say: "Story ID", "Prediction", "Rationale"
required_cols = ["Story ID", "Prediction", "Rationale"]

# Rename columns if they don't match exactly
# (e.g. if you have "id" instead of "Story ID")
rename_map = {
    "id": "Story ID",
    "story_id": "Story ID",
    "pred": "Prediction",
    "prediction": "Prediction",
    "rationale": "Rationale"
}
df.rename(columns=rename_map, inplace=True)

# Keep only the required columns
df = df[required_cols]

# 4. Save the fixed file
df.to_csv("results_final.csv", index=False)
print("✅ Fixed! Saved as 'results_final.csv'. You can submit this file.")
print(df.head())