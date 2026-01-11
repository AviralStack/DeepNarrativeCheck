import ollama
import os

# Get the host from the environment, or default to localhost if not set
target_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
print(f"   --> Connecting to Ollama at: {target_host}")

# Create a manual client pointing to Windows
client = ollama.Client(host=target_host)

def retrieve_context_for_claims(vector_store, claims, top_k=3):
    combined_context = set()
    print(f"   --> Searching book for {len(claims)} claims...")
    for claim in claims:
        results = vector_store.query(query=claim, k=top_k)
        for row in results:
            combined_context.add(row["chunks"])
    return "\n---\n".join(list(combined_context))

def check_all_claims_batch(claims, context):
    claims_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(claims)])
    
    prompt = f"""
    You are a strictly logical AI. verify if the CLAIMS contradict the BOOK TEXT.
    
    BOOK TEXT:
    {context[:12000]} 
    
    CLAIMS:
    {claims_text}
    
    INSTRUCTIONS:
    - Check for DIRECT CONTRADICTIONS only.
    - If a claim is just "new information" (not mentioned in text), it is CONSISTENT.
    - Only flag if it is impossible given the text.

    ANSWER FORMAT:
    Start with "CONSISTENT" or "CONTRADICT". Then 1 sentence rationale.
    """

    print("   --> Asking local AI (Mistral)...")
    
    try:
        # USE THE CLIENT WE CREATED ABOVE
        response = client.chat(model='mistral', messages=[
            {'role': 'user', 'content': prompt},
        ])
        
        answer = response['message']['content']
        print(f"   --> AI Answer: {answer[:50]}...")
        
        is_consistent = True
        rationale = answer
        
        if "CONTRADICT" in answer.upper():
            is_consistent = False
            
        return is_consistent, rationale

    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return True, f"Error: {e}"