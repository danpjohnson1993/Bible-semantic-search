import os
import json
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify

app = Flask(__name__)

JSON_URL = "https://romansten.org/wp-content/kjv_with_embeddings.json"
LOCAL_CACHE_PATH = "/tmp/kjv_with_embeddings.json"  # Render instance's temp storage

def load_bible_data():
    if os.path.exists(LOCAL_CACHE_PATH):
        print("Loading Bible data from local cache...")
        with open(LOCAL_CACHE_PATH, "r") as f:
            bible_data = json.load(f)
    else:
        print("Downloading Bible data from remote URL (this may take a while)...")
        response = requests.get(JSON_URL, stream=True)
        response.raise_for_status()
        # Save to local cache file as it downloads
        with open(LOCAL_CACHE_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete. Loading JSON...")
        with open(LOCAL_CACHE_PATH, "r") as f:
            bible_data = json.load(f)
    return bible_data

bible_data = load_bible_data()

# Continue with your existing code to build FAISS index and semantic search...
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

@app.route("/")
def index():
    return "Bible Semantic Search API is running."

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_question = data.get("question", "")

    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    results = semantic_search(user_question)
    return jsonify({"results": results})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
# ...rest of your Flask app routes, etc.
