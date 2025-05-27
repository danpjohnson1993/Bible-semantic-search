import os
import json
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Configuration
JSON_URL = "https://romansten.org/wp-content/kjv_with_embeddings.json"
LOCAL_CACHE_PATH = "/tmp/kjv_with_embeddings.json"

# Load Bible Data
def load_bible_data():
    if os.path.exists(LOCAL_CACHE_PATH):
        print("Loading Bible data from local cache...")
        with open(LOCAL_CACHE_PATH, "r") as f:
            return json.load(f)
    else:
        print("Downloading Bible data from remote URL...")
        try:
            response = requests.get(JSON_URL, stream=True, timeout=30)
            response.raise_for_status()
            with open(LOCAL_CACHE_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            with open(LOCAL_CACHE_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            print("Failed to load Bible data:", str(e))
            return []

bible_data = load_bible_data()

# Prepare FAISS index
embeddings = []
texts = []
references = []

for entry in bible_data:
    embeddings.append(np.array(entry["embedding"], dtype=np.float32))
    texts.append(entry["text"])
    references.append(entry["reference"])

if embeddings:
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.vstack(embeddings))
else:
    index = None
    print("Warning: No embeddings loaded. Search will not work.")

model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_search(query, top_k=5):
    if not index:
        return [{"text": "Search index not loaded.", "reference": ""}]
    query_vec = model.encode([query])[0].astype(np.float32)
    distances, indices = index.search(np.array([query_vec]), top_k)
    return [
        {"reference": references[i], "text": texts[i]}
        for i in indices[0]
    ]

@app.route("/")
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Bible Chat</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background: #f4f4f4;
                display: flex;
                flex-direction: column;
                min-height: 100vh;
            }

            #loading {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                background-color: #ffffff;
                flex-direction: column;
                color: #555;
            }

            .spinner {
                border: 6px solid #f3f3f3;
                border-top: 6px solid #007BFF;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin-bottom: 15px;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            #chat-interface {
                display: none;
                padding: 1rem;
                flex: 1;
                max-width: 600px;
                margin: auto;
            }

            #chat-box {
                background: white;
                border-radius: 10px;
                padding: 1rem;
                height: 60vh;
                overflow-y: auto;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                margin-bottom: 1rem;
            }

            .bubble {
                max-width: 80%;
                margin-bottom: 10px;
                padding: 10px 15px;
                border-radius: 15px;
                clear: both;
                word-wrap: break-word;
            }

            .user {
                background: #007BFF;
                color: white;
                float: right;
                text-align: right;
            }

            .bot {
                background: #e2e2e2;
                float: left;
            }

            #question {
                width: 75%;
                padding: 10px;
                font-size: 1em;
                border: 1px solid #ccc;
                border-radius: 8px;
            }

            button {
                padding: 10px 15px;
                font-size: 1em;
                border: none;
                border-radius: 8px;
                background: #007BFF;
                color: white;
                margin-left: 10px;
                cursor: pointer;
            }

            @media (max-width: 600px) {
                #question {
                    width: 100%;
                    margin-bottom: 10px;
                }

                button {
                    width: 100%;
                }
            }
        </style>
    </head>
    <body>
        <div id="loading">
            <div class="spinner"></div>
            <div>Loading Bible embeddings and search index...</div>
        </div>

        <div id="chat-interface">
            <div id="chat-box"></div>
            <div style="display: flex; flex-wrap: wrap;">
                <input type="text" id="question" placeholder="Find any answer in the Bible here..." />
                <button onclick="askQuestion()">Ask</button>
            </div>
        </div>

        <script>
            function addMessage(content, type) {
                const chatBox = document.getElementById("chat-box");
                const bubble = document.createElement("div");
                bubble.className = "bubble " + type;
                bubble.textContent = content;
                chatBox.appendChild(bubble);
                chatBox.scrollTop = chatBox.scrollHeight;
            }

            function checkAPIReady() {
                fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ question: "test" })
                })
                .then(res => res.json())
                .then(() => {
                    document.getElementById("loading").style.display = "none";
                    document.getElementById("chat-interface").style.display = "block";
                })
                .catch(err => {
                    document.getElementById("loading").innerHTML = '<div class="spinner"></div><div>Still loading... please wait.</div>';
                    setTimeout(checkAPIReady, 2000);
                });
            }

            function askQuestion() {
                const input = document.getElementById("question");
                const q = input.value.trim();
                if (!q) return;
                addMessage(q, "user");
                input.value = "";
                fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ question: q })
                })
                .then(res => res.json())
                .then(data => {
                    const results = data.results?.map(r => r.text).join("\\n\\n") || "No results.";
                    addMessage(results, "bot");
                });
            }

            checkAPIReady();
        </script>
    </body>
    </html>
    """)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_question = data.get("question", "").strip()

    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    results = semantic_search(user_question)
    return jsonify({"results": results})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
