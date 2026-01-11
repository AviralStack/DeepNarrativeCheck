from src.claims import extract_claims
from src.reasoning import retrieve_context_for_claims, check_all_claims_batch

def predict_one(row, vector_store):
    backstory = row["content"]
    
    # 1. Extract Claims
    claims = extract_claims(backstory)
    if not claims:
        return "consistent", "No claims extracted."
    
    # 2. Retrieve Evidence (Using the passed vector_store)
    context = retrieve_context_for_claims(vector_store, claims)
    
    if not context:
        return "consistent", "No relevant text found."

    # 3. Ask Local AI
    is_consistent, rationale = check_all_claims_batch(claims, context)
    
    result_label = "consistent" if is_consistent else "contradict"
    
    return result_label, rationale