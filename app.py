import streamlit as st
import os
import uuid
import tempfile
import json
from datetime import datetime

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from groq_llm import GroqLLM

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Legal Assistant", layout="wide")
st.title("⚖️ AI Legal Assistant")

DB_PATH = "vector_store/faiss_index"
CHAT_LOG_DIR = "chat_logs"
os.makedirs(CHAT_LOG_DIR, exist_ok=True)

# --- SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

history_file = os.path.join(CHAT_LOG_DIR, f"{st.session_state.user_id}.jsonl")

# --- EMBEDDINGS + VECTOR DB ---
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists(DB_PATH):
    vectordb = FAISS.load_local(DB_PATH, embedding_model)
else:
    vectordb = FAISS.from_texts([], embedding_model)
    vectordb.save_local(DB_PATH)

retriever = vectordb.as_retriever()

# --- FILE UPLOAD ---
uploaded_files = st.file_uploader("📄 Upload Legal Documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    docs = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        # Loader based on extension
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)

        docs.extend(loader.load())
        os.unlink(tmp_path)

    # Chunk and add to vector DB
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    vectordb.add_documents(split_docs)
    vectordb.save_local(DB_PATH)
    st.success(f"✅ {len(uploaded_files)} file(s) processed and indexed.")

# --- CHAT INPUT ---
user_input = st.chat_input("💬 Ask a legal question...")

if user_input:
    with st.spinner("Generating legal response..."):
        llm = GroqLLM(model="llama3", temperature=0.2)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        response = qa_chain.run(user_input)

        # Log interaction
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": user_input,
            "response": response
        }
        with open(history_file, "a") as f:
            json.dump(log_entry, f)
            f.write("\n")

        # Update UI history
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("ai", response))

# --- CHAT DISPLAY ---
for sender, msg in st.session_state.chat_history:
    with st.chat_message(sender):
        st.markdown(msg)

# --- CHAT DOWNLOAD ---
if os.path.exists(history_file):
    with open(history_file) as f:
        chat_log = f.read()
    st.download_button("📥 Download Chat History", data=chat_log, file_name=f"chat_{st.session_state.user_id}.jsonl", mime="application/json")
