import os
import json
import numpy as np
import faiss
import requests
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# URL to your big JSON file hosted on your website
JSON_URL = "https://romansten.org/wp-content/kjv_with_embeddings.json"

print("Fetching Bible embeddings JSON from remote URL (this may take a while)...")
response = requests.get(JSON_URL)
response.raise_for_status()  # fail fast if URL unreachable
bible_data = response.json()
print(f"Loaded {len(bible_data)} scripture entries.")

# Prepare data for FAISS index
embeddings = []
texts = []
references = []

for entry in bible_data:
    embeddings.append(np.array(entry["embedding"], dtype=np.float32))
    texts.append(entry["text"])
    references.append(entry["reference"])

dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.vstack(embeddings))

# Load sentence-transformers model once
model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_search(query, top_k=5):
    query_vec = model.encode([query])[0].astype(np.float32)
    distances, indices = index.search(np.array([query_vec]), top_k)
    results = []
    for i in indices[0]:
        results.append({
            "reference": references[i],
            "text": texts[i]
        })
    return results

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_question = data.get("question", "")

    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    # Run semantic search on cached data
    results = semantic_search(user_question)

    # Return results directly without calling OpenAI
    return jsonify({
        "results": results
    })

@app.route('/')
def index():
    return "Bible Semantic Search API is running."

if __name__ == '__main__':
    app.run(debug=True)
