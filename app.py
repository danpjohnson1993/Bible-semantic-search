import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load Bible data with embeddings
with open("data/kjv_with_embeddings.json", "r") as f:
    bible_data = json.load(f)

# Extract embeddings and metadata
embeddings = [np.array(entry["embedding"], dtype=np.float32) for entry in bible_data]
texts = [entry["text"] for entry in bible_data]
references = [entry["reference"] for entry in bible_data]

# Build FAISS index
dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.vstack(embeddings))

# Load the sentence-transformers model for generating query embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # or whichever model you prefer

def search_bible(query, top_k=5):
    # Embed the query using sentence-transformers
    query_embedding = model.encode([query], convert_to_numpy=True).astype(np.float32)
    
    # Search in the FAISS index
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "text": texts[idx],
            "reference": references[idx],
            "distance": float(dist)
        })
    return results

# Example: test a search
if __name__ == "__main__":
    test_query = "In the beginning God created the heavens and the earth."
    results = search_bible(test_query)
    for r in results:
        print(f"{r['reference']}: {r['text']} (score: {r['distance']})")
