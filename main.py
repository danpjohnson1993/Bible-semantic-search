from flask import Flask, request, jsonify
import faiss
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import os

app = Flask(__name__)

# Use your absolute paths to the index files
index_path = "faiss_index/Bible_faiss.index"
refs_path = "faiss_index/bible_refs.json"

index = faiss.read_index(index_path)

with open(refs_path, "r") as f:
    refs = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get("question", "").strip()

    if not query:
        return jsonify({"error": "No question provided"}), 400

    try:
        query_vec = model.encode([query]).astype("float32")
        D, I = index.search(query_vec, k=5)

        results = []
        for idx in I[0]:
            if 0 <= idx < len(refs):
                verse = refs[idx]
                results.append({
                    "reference": verse.get("reference", ""),
                    "text": verse.get("text", "")
                })

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

# Required for Deta to recognize the app
if __name__ == "__main__":
    app.run()
