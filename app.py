import streamlit as st
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import json
from datetime import datetime
import os

# Setup
st.set_page_config(page_title="AI Legal Assistant", layout="wide")
st.title("🧑‍⚖️ AI Legal Assistant")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize embedding and vectorstore
embedding_model = HuggingFaceEmbeddings()
db_path = "vector_store/faiss_index"

if os.path.exists(db_path):
    vectordb = FAISS.load_local(db_path, embedding_model)
else:
    vectordb = FAISS.from_texts([], embedding_model)
    vectordb.save_local(db_path)

retriever = vectordb.as_retriever()

# File uploader
uploaded_files = st.file_uploader("📄 Upload Legal Files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    docs = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        # Detect file type
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)
        docs.extend(loader.load())
        os.unlink(tmp_path)

    # Chunk and embed
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    vectordb.add_documents(split_docs)
    vectordb.save_local(db_path)
    st.success(f"✅ {len(uploaded_files)} file(s) processed and added to memory.")

# Chat input
user_input = st.chat_input("Ask a legal question...")

if user_input:
    with st.spinner("Generating answer..."):
        # LLM via Groq (using Langchain's generic OpenAI wrapper here)
        llm = OpenAI(model_name="llama3", temperature=0.2, openai_api_key=os.getenv("GROQ_API_KEY"))

        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        response = qa_chain.run(user_input)

        # Log interaction
        log = {
            "timestamp": datetime.now().isoformat(),
            "query": user_input,
            "response": response,
        }
        with open("chat_history.jsonl", "a") as f:
            json.dump(log, f)
            f.write("\n")

        # Update chat history
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("ai", response))

# Display chat
for sender, msg in st.session_state.chat_history:
    with st.chat_message(sender):
        st.markdown(msg)
