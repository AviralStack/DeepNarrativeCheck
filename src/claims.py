def extract_claims(backstory):
    # Simple local heuristic: take first 5 meaningful sentences as "claims"
    sentences = [s.strip() for s in backstory.split(".") if len(s.strip()) > 30]
    return sentences[:5]
