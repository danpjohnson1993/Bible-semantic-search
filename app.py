import os
import json
import numpy as np
import faiss
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# Load Bible data with embeddings
with open("data/kjv_with_embeddings.json", "r") as f:
    bible_data = json.load(f)

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

def build_openai_prompt(user_question, results):
    scripture_texts = "\n\n".join([f"{r['reference']}: {r['text']}" for r in results])
    prompt = (
        f"You are a helpful assistant referencing the King James Bible.\n"
        f"User question: {user_question}\n\n"
        f"Relevant scriptures:\n{scripture_texts}\n\n"
        "Use these scriptures to answer the user's question accurately and kindly."
    )
    return prompt

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_question = data.get("question", "")

    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    # Perform semantic search to get relevant scriptures
    results = semantic_search(user_question)

    # Build prompt for OpenAI including relevant scriptures
    prompt = build_openai_prompt(user_question, results)

    # Call OpenAI ChatCompletion API
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant referencing the King James Bible."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.7,
        )
        answer = response['choices'][0]['message']['content']
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "answer": answer,
        "scriptures": results
    })

if __name__ == '__main__':
    app.run(debug=True)
