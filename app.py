import gradio as gr
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Load FAQ data
with open("faq_data.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

questions = [item["question"].lower().strip() for item in faq_data]
answers = [item["answer"] for item in faq_data]

question_embeddings = model.encode(questions)

SIMILARITY_THRESHOLD = 0.5
user_sessions = {}

def chat(user_input, session_id="default"):
    user_input = user_input.lower().strip()

    if session_id in user_sessions:
        combined_input = user_sessions[session_id] + " " + user_input
    else:
        combined_input = user_input

    user_embedding = model.encode([combined_input])
    similarities = cosine_similarity(user_embedding, question_embeddings)[0]
    best_idx = int(np.argmax(similarities))
    best_score = similarities[best_idx]

    if best_score < SIMILARITY_THRESHOLD:
        if session_id in user_sessions:
            del user_sessions[session_id]
            return "I’m not fully sure about that. Please check the official FAQ or contact a staff on Discord."
        else:
            user_sessions[session_id] = user_input
            return "Could you please elaborate what you mean?"

    if session_id in user_sessions:
        del user_sessions[session_id]

    return answers[best_idx]

# Gradio UI
iface = gr.Interface(
    fn=chat,
    inputs=[gr.Textbox(label="Ask a question"), gr.Textbox(value="default", visible=False, label="Session ID")],
    outputs="text",
    title="Eduvance FAQ Bot"
)

iface.launch()

