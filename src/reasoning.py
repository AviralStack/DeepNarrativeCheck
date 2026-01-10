from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def find_relevant_chunks(book_text, claim, top_k=5):
    sentences = book_text.split(". ")
    emb_book = model.encode(sentences, convert_to_tensor=True)
    emb_claim = model.encode(claim, convert_to_tensor=True)

    scores = util.cos_sim(emb_claim, emb_book)[0]
    best = scores.topk(top_k).indices

    return [sentences[i] for i in best]

def check_contradiction(claim, evidence):
    text = " ".join(evidence).lower()
    claim = claim.lower()

    negations = [" not ", " never ", " no ", " none ", " cannot ", " can't "]
    for n in negations:
        if n in text and any(word in text for word in claim.split()[:3]):
            return True

    return False
