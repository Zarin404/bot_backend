from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAQ data
with open("faq_data.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

questions = [item["question"].lower().strip() for item in faq_data]  # lowercase for better matching
answers = [item["answer"] for item in faq_data]

# Precompute embeddings
question_embeddings = model.encode(questions)

SIMILARITY_THRESHOLD = 0.5

# Keep track of clarification state per session
user_sessions = {}  # {session_id: last_low_confidence_question}

@app.route("/ask", methods=["POST"])
def chat():
    session_id = request.json.get("session_id", "default")
    user_input = request.json.get("message", "").lower().strip()

    # Combine with previous low-confidence question if user is clarifying
    if session_id in user_sessions:
        combined_input = user_sessions[session_id] + " " + user_input
    else:
        combined_input = user_input

    # Encode and compare
    user_embedding = model.encode([combined_input])
    similarities = cosine_similarity(user_embedding, question_embeddings)[0]
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score < SIMILARITY_THRESHOLD:
        # Already asked for clarification once? Then fallback
        if session_id in user_sessions:
            del user_sessions[session_id]
            return jsonify({
                "reply": "I’m not fully sure about that. Please check the official FAQ or contact a staff on Discord."
            })
        else:
            # Ask user to elaborate
            user_sessions[session_id] = user_input
            return jsonify({"reply": "Could you please elaborate what you mean?"})

    # High-confidence answer found, clear session
    if session_id in user_sessions:
        del user_sessions[session_id]

    return jsonify({
        "reply": answers[best_idx]
    })

if __name__ == "__main__":
    app.run(debug=True)


