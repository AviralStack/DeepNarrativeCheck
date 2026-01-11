import pathway as pw
from pathway.xpacks.llm.splitters import TokenCountSplitter
from sentence_transformers import SentenceTransformer, util
import torch

class LocalVectorStore:
    def __init__(self, chunks_text):
        # Filter out bad data/empty strings
        self.chunks = [str(c) for c in chunks_text if c and len(str(c).strip()) > 0]
        
        # --- CHECK THIS NUMBER IN YOUR LOGS ---
        print(f"   --> Encoding {len(self.chunks)} chunks... (this might take 30s)")
        
        if not self.chunks:
            print("WARNING: No chunks found. The book might be empty or failed to parse.")
            self.embeddings = None
            return

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.model.encode(self.chunks, convert_to_tensor=True)

    def query(self, query, k=3):
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        # Safe limit for k (prevents "index out of range" crash)
        safe_k = min(k, len(self.chunks))
        
        query_emb = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, self.embeddings)[0]
        top_results = torch.topk(scores, k=safe_k)
        
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            results.append({
                "chunks": self.chunks[idx],
                "score": score.item()
            })
        return results

def build_vector_index(book_folder="./Books"):
    # 1. READ
    data_source = pw.io.fs.read(
        book_folder, 
        format="plaintext", 
        mode="static", 
        with_metadata=True
    )

    # 2. CHUNK
    splitter = TokenCountSplitter(max_tokens=500)
    
    documents = data_source.select(doc=pw.this.data)
    
    # Flatten splits the list of chunks into separate rows
    chunks_table = documents.select(
        chunks=splitter(pw.this.doc)
    ).flatten(pw.this.chunks)

    # 3. EXECUTE
    print("   --> Running Pathway Ingestion Pipeline...")
    data_list = pw.debug.table_to_dicts(chunks_table)
    
    # 4. BULLETPROOF EXTRACTION (The Fix)
    text_chunks = []
    
    if data_list:
        for row in data_list:
            try:
                # OPTION 1: It's a Dictionary (Standard)
                if isinstance(row, dict) and 'chunks' in row:
                    text_chunks.append(row['chunks'])
                
                # OPTION 2: It's a List/Tuple (CRITICAL FIX HERE)
                elif isinstance(row, (list, tuple)) and len(row) > 0:
                    # If the first item is a string, assume the whole list is chunks
                    if isinstance(row[0], str):
                        text_chunks.extend(row) # <--- THIS GRABS ALL 59,000 CHUNKS
                    else:
                        text_chunks.append(row[0]) # Fallback
                
                # OPTION 3: It's a plain string
                elif isinstance(row, str):
                    text_chunks.append(row)
                    
                # OPTION 4: Fallback
                else:
                    text_chunks.append(str(row))
            
            except Exception:
                text_chunks.append(str(row))

    return LocalVectorStore(text_chunks)