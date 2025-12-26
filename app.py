from sentence_transformers import SentenceTransformer
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAQ data
with open("faq_data.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

questions = [item["question"] for item in faq_data]
answers = [item["answer"] for item in faq_data]

# Precompute embeddings
question_embeddings = model.encode(questions)

SIMILARITY_THRESHOLD = 0.5

def ask_bot(user_input):
    user_embedding = model.encode([user_input])
    similarities = cosine_similarity(user_embedding, question_embeddings)[0]

    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score < SIMILARITY_THRESHOLD:
        return "I’m not fully sure about that. Please check the official FAQ or contact a staff on Discord."
    return answers[best_idx]

# Gradio interface
iface = gr.Interface(
    fn=ask_bot,
    inputs=gr.Textbox(label="Ask a question"),
    outputs=gr.Textbox(label="Answer"),
    title="Eduvance FAQ Bot"
)

iface.launch()

