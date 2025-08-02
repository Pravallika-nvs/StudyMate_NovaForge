import os
import json
import random
import numpy as np
import fitz  # PyMuPDF
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from deep_translator import GoogleTranslator
import google.generativeai as genai

# ==============================
# 🔹 Secure API Key Handling
# ==============================
HF_API_KEY = st.secrets.get("HF_API_KEY", os.environ.get("HF_API_KEY"))
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))

if not HF_API_KEY or not GEMINI_API_KEY:
    st.error("❌ Missing API keys! Please set HF_API_KEY and GEMINI_API_KEY in secrets.")
    st.stop()

# Hugging Face & Gemini setup
client = InferenceClient(model="mistralai/Mixtral-8x7B-Instruct-v0.1", token=HF_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ==============================
# 🔹 Streamlit Page Setup
# ==============================
st.set_page_config(page_title="StudyMate - PDF Q&A", layout="wide")
st.title("📘 StudyMate - AI-powered PDF Q&A & Quiz Generator")

lang = st.selectbox("Select Output Language", ["en", "hi", "te", "ta", "fr", "de"])

# ==============================
# 🔹 Session State
# ==============================
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "chunks" not in st.session_state: st.session_state.chunks = []
if "chunk_metadata" not in st.session_state: st.session_state.chunk_metadata = []
if "index" not in st.session_state: st.session_state.index = None
if "quiz_questions" not in st.session_state: st.session_state.quiz_questions = []
if "quiz_answers" not in st.session_state: st.session_state.quiz_answers = {}
if "quiz_submitted" not in st.session_state: st.session_state.quiz_submitted = False
if "embed_model" not in st.session_state:
    st.session_state.embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# ==============================
# 🔹 PDF Processing & Chunking
# ==============================
def extract_chunks_with_metadata(file, chunk_size=1000, overlap=200):
    chunks, metadata = [], []
    doc = fitz.open(stream=file.read(), filetype="pdf")
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                metadata.append({"page": page_num})
            start += chunk_size - overlap
    return chunks, metadata

# ==============================
# 🔹 Chunk Retrieval & Answering
# ==============================
def retrieve_chunks(query, top_k=3):
    query_emb = st.session_state.embed_model.encode([query], convert_to_numpy=True)
    distances, indices = st.session_state.index.search(query_emb, top_k)
    results = []
    for i in range(top_k):
        idx = indices[0][i]
        results.append({
            "text": st.session_state.chunks[idx],
            "page": st.session_state.chunk_metadata[idx]["page"],
            "distance": distances[0][i]
        })
    return results

def ask_gemini(question):
    prompt = f"""
    You are a helpful educational AI assistant for students.

    Instructions:
    - If the question is math or requires calculations, give a step-by-step solution.
    - If the question is theoretical, explain it simply.
    - Provide mnemonics or memory aids whenever possible.
    - End with a short summary if needed.

    Question: {question}
    """
    return gemini_model.generate_content(prompt).text

def generate_grounded_answer(query):
    results = retrieve_chunks(query)

    # Out-of-PDF fallback if top chunk is irrelevant
    if results[0]["distance"] > 50:
        gemini_answer = ask_gemini(query)
        return f"*(Not from the PDF)*\n\n{gemini_answer}", "Gemini Fallback"

    # Use PDF context with Mixtral
    context = "\n\n".join([f"[Page {r['page']}]\n{r['text']}" for r in results])
    messages = [
        {"role": "system", "content":
         "You are an academic assistant. Answer ONLY using the PDF context. "
         "If the PDF doesn't answer, clearly say 'Not from the PDF' and then answer."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"}
    ]
    response = client.chat_completion(messages=messages, max_tokens=300)
    return response.choices[0].message["content"], results

# ==============================
# 🔹 MCQ Generation
# ==============================
def generate_mcqs(num_mcqs=10):
    selected_chunks = random.sample(st.session_state.chunks, min(10, len(st.session_state.chunks)))
    combined_text = "\n\n".join(selected_chunks)

    messages = [
        {"role": "system", "content":
         """You are an exam MCQ generator. Create multiple-choice questions
         from the given PDF content. Return JSON in this format:

            [
                {
                    "question": "...",
                    "options": ["A...", "B...", "C...", "D..."],
                    "answer": "A",
                    "concept": "...",
                    "explanation": "..."
                }
            ]

         - 4 options per question
         - 10 questions total
         """},
        {"role": "user", "content": f"PDF Content:\n{combined_text}\n\nGenerate {num_mcqs} MCQs."}
    ]
    response = client.chat_completion(messages=messages, max_tokens=2000)
    json_str = response.choices[0].message.content.strip()
    json_str = json_str[json_str.find("[") : json_str.rfind("]") + 1]
    try:
        return json.loads(json_str)
    except:
        return []

# ==============================
# 🔹 Multi-PDF Upload
# ==============================
uploaded_files = st.file_uploader("📤 Upload your PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files and st.session_state.index is None:
    all_chunks, all_metadata = [], []
    for file in uploaded_files:
        chunks, metadata = extract_chunks_with_metadata(file)
        all_chunks.extend(chunks)
        all_metadata.extend(metadata)
    
    st.session_state.chunks = all_chunks
    st.session_state.chunk_metadata = all_metadata

    embeddings = st.session_state.embed_model.encode(st.session_state.chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    st.session_state.index = index
    st.success(f"✅ Processed {len(st.session_state.chunks)} chunks from {len(uploaded_files)} PDFs.")

# ==============================
# 🔹 Chat Interface
# ==============================
prompt = st.chat_input("Ask a question from your PDFs or anything...")
if prompt:
    translated_q = GoogleTranslator(source='auto', target='en').translate(prompt)
    answer, source = generate_grounded_answer(translated_q)
    answer_final = GoogleTranslator(source='en', target=lang).translate(answer)
    st.chat_message("user").markdown(prompt)
    st.chat_message("assistant").markdown(answer_final)
    st.session_state.chat_history.append({"user": prompt, "bot": answer_final})

# ==============================
# 🔹 Quiz Section
# ==============================
def submit_quiz():
    """Evaluate the quiz and display the score."""
    if not st.session_state.quiz_questions:
        st.warning("⚠️ No quiz generated yet!")
        return

    score = 0
    total = len(st.session_state.quiz_questions)

    for i, item in enumerate(st.session_state.quiz_questions, start=1):
        qkey = f"q{i}"
        answer = item['answer'].strip().upper()
        options = item['options']
        index = ord(answer) - 65  # Convert 'A'->0

        if 0 <= index < len(options):
            correct_option = options[index]
            user_answer = st.session_state.quiz_answers[qkey]
            if user_answer == correct_option:
                score += 1
        else:
            st.warning(f"⚠️ Invalid answer '{item['answer']}' for Q{i}. Options: {options}")

    st.success(f"✅ Your Score: {score}/{total}")

if st.button("🧠 Generate Quiz"):
    st.session_state.quiz_questions = generate_mcqs()
    st.session_state.quiz_answers = {}
    st.session_state.quiz_submitted = False

if st.session_state.quiz_questions:
    with st.expander("🧠 Test Your Knowledge", expanded=True):
        for i, item in enumerate(st.session_state.quiz_questions, start=1):
            qkey = f"q{i}"
            st.markdown(f"**Q{i}:** {item['question']}")
            st.session_state.quiz_answers[qkey] = st.radio(
                label="", options=item['options'], index=0, key=qkey
            )
        if st.button("✅ Submit Quiz"):
            submit_quiz()
