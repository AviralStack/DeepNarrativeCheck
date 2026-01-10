from src.data_loader import load_dataset, load_book
from src.claims import extract_claims
from src.reasoning import find_relevant_chunks, check_contradiction

def predict_one(row):
    book = load_book(row["book_name"])
    backstory = row["content"]

    print("\nBOOK NAME:", row["book_name"])
    print("CHARACTER:", row["char"])
    print("\nBACKSTORY (first 300 chars):\n", backstory[:300])
    print("\nBOOK (first 300 chars):\n", book[:300])

    claims = extract_claims(backstory)
    print("\nEXTRACTED CLAIMS:", claims)

    for claim in claims:
        evidence = find_relevant_chunks(book, claim)
        print("\nCLAIM:", claim)
        print("EVIDENCE:", evidence[:2])

        if check_contradiction(claim, evidence):
            print("\n❌ CONTRADICTION FOUND")
            return "contradict"

    print("\n✅ NO CONTRADICTIONS FOUND")
    return "consistent"

if __name__ == "__main__":
    df = load_dataset("train.csv")
    row = df.iloc[0]

    result = predict_one(row)
    print("\nFINAL PREDICTION:", result)
    