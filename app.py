from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

app = FastAPI()

# Allow same-origin requests (safe here)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend folder
app.mount("/static", StaticFiles(directory="frontend"), name="static")

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

# Serve frontend UI
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()

# Chat endpoint
@app.post("/ask")
async def ask_bot(request: Request):
    data = await request.json()
    user_input = data.get("message", "").lower().strip()
    session_id = data.get("session_id", "default")

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
            return JSONResponse({
                "reply": "I’m not fully sure about that. Please check the official FAQ or contact a staff on Discord."
            })
        else:
            user_sessions[session_id] = user_input
            return JSONResponse({"reply": "Could you please elaborate what you mean?"})

    if session_id in user_sessions:
        del user_sessions[session_id]

    return JSONResponse({"reply": answers[best_idx]})