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

# ...rest of your Flask app routes, etc.
