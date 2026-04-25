import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import ollama
import os
import json
from datetime import datetime

st.set_page_config(page_title="Knowledge Assistant", page_icon="🧠", layout="wide")

st.title("🧠 Enterprise Knowledge Assistant")
st.caption("Local AI Knowledge Platform")

CHAT_FILE = "chat_history.json"

# ---------------------------
# Load resources
# ---------------------------
@st.cache_resource
def load_resources():
    index = faiss.read_index("vector_store/docs.index")

    with open("vector_store/texts.pkl", "rb") as f:
        texts = pickle.load(f)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, texts, model

index, texts, model = load_resources()

# ---------------------------
# Chat persistence
# ---------------------------
if os.path.exists(CHAT_FILE):
    with open(CHAT_FILE, "r") as f:
        history = json.load(f)
else:
    history = []

if "messages" not in st.session_state:
    st.session_state.messages = history

# ---------------------------
# Sidebar
# ---------------------------
import os
import pickle
import faiss
import numpy as np
from pypdf import PdfReader

# Sidebar mode selector
mode = st.sidebar.selectbox(
    "Choose Assistant Mode",
    ["Onboarding", "Support", "Architect", "Leadership"]
)

st.sidebar.markdown("## 📂 File Management")

# Ensure folders exist
os.makedirs("data", exist_ok=True)

# -----------------------------
# Existing files dashboard
# -----------------------------
existing_files = os.listdir("data")

st.sidebar.markdown("### Stored Files")

if existing_files:
    for file in existing_files:
        col1, col2 = st.sidebar.columns([3,1])

        with col1:
            st.write(file)

        with col2:
            if st.button("🗑", key=file):
                os.remove(os.path.join("data", file))
                st.sidebar.success(f"{file} deleted")
                st.rerun()
else:
    st.sidebar.caption("No files stored")

st.sidebar.markdown("---")

# -----------------------------
# Upload Section
# -----------------------------
st.sidebar.markdown("### Upload New Files")

uploaded_files = st.sidebar.file_uploader(
    "Choose TXT or PDF files",
    type=["txt", "pdf"],
    accept_multiple_files=True,
    key="file_upload"
)

if uploaded_files:

    st.sidebar.write("Selected Files:")

    for f in uploaded_files:
        st.sidebar.write("• " + f.name)

    if st.sidebar.button("Submit Upload"):

        added_chunks = []

        for uploaded_file in uploaded_files:

            file_path = os.path.join("data", uploaded_file.name)

            # save file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # read content
            if uploaded_file.name.endswith(".txt"):
                content = uploaded_file.getvalue().decode("utf-8")

            else:
                reader = PdfReader(file_path)
                content = ""
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        content += text + "\n"

            # chunking
            chunks = [content[i:i+500] for i in range(0, len(content), 500)]
            added_chunks.extend(chunks)

        # embed and update vector db
        if added_chunks:
            new_embeddings = model.encode(added_chunks)
            index.add(np.array(new_embeddings))
            texts.extend(added_chunks)

            faiss.write_index(index, "vector_store/docs.index")

            with open("vector_store/texts.pkl", "wb") as f:
                pickle.dump(texts, f)

        st.sidebar.success("Files uploaded and indexed successfully.")

        st.rerun()
        
if uploaded_files:
    for uploaded_file in uploaded_files:

        file_path = os.path.join("data", uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Read content
        if uploaded_file.name.endswith(".txt"):
            content = uploaded_file.getvalue().decode("utf-8")

        else:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            content = ""
            for page in reader.pages:
                content += page.extract_text() + "\n"

        # Chunk simple
        chunks = [content[i:i+500] for i in range(0, len(content), 500)]

        # Embed and append
        new_embeddings = model.encode(chunks)
        index.add(np.array(new_embeddings))

        texts.extend(chunks)

    # Save updated index
    faiss.write_index(index, "vector_store/docs.index")

    with open("vector_store/texts.pkl", "wb") as f:
        pickle.dump(texts, f)

    st.sidebar.success("Files uploaded and indexed successfully.")

# ---------------------------
# Prompt modes
# ---------------------------
def get_mode_prompt(mode):
    prompts = {
        "Onboarding": "Explain simply for a new employee.",
        "Support": "Focus on troubleshooting and resolution steps.",
        "Architect": "Focus on system design and dependencies.",
        "Leadership": "Summarize business impact, risks, actions."
    }
    return prompts[mode]

# ---------------------------
# Display chat
# ---------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------------------
# Input
# ---------------------------
question = st.chat_input("Ask your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            q_embedding = model.encode([question])
            distances, indices = index.search(np.array(q_embedding), 5)

            chunks = [texts[i] for i in indices[0]]
            context = "\n\n".join(chunks)

            # confidence estimate
            avg_distance = float(np.mean(distances[0]))
            confidence = max(0, min(100, int(100 - avg_distance * 10)))

            prompt = f"""
You are an enterprise assistant.

Behavior:
{get_mode_prompt(mode)}

Use ONLY the context below.
If not found, clearly say information unavailable.

Context:
{context}

Question:
{question}
"""

            response = ollama.chat(
                model="mistral",
                messages=[{"role": "user", "content": prompt}]
            )

            answer = response["message"]["content"]

            st.write(answer)

            st.progress(confidence / 100)
            st.caption(f"Confidence Score: {confidence}%")

            with st.expander("📚 Sources Used"):
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(f"**Source {i}:**")
                    st.write(chunk[:1000])

    st.session_state.messages.append({"role": "assistant", "content": answer})

    # save chat history
    with open(CHAT_FILE, "w") as f:
        json.dump(st.session_state.messages, f)

# ---------------------------
# Footer metrics
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.metric("Documents Loaded", len(texts))
st.sidebar.metric("Messages", len(st.session_state.messages))
st.sidebar.metric("Mode", mode)