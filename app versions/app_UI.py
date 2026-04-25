import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import ollama
import os
import time
from datetime import datetime
from pypdf import PdfReader

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Cognivault AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_FOLDER = "data"
VECTOR_FOLDER = "vector_store"

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Chat"

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------------------------
# PREMIUM CSS
# -------------------------------------------------
st.markdown("""
<style>

/* Global */
.stApp {
    background: #f8fafc;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#081225 0%, #0f172a 100%);
    border-right: 1px solid #1e293b;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 14px;
    padding: 0.75rem 1rem;
    font-weight: 700;
    font-size: 15px;
    border: 1px solid #dbeafe;
    background: white !important;
    color: #111827 !important;
}

.stButton > button:hover {
    border-color: #2563eb;
    background: #eff6ff !important;
    color: #2563eb !important;
}

/* Hero */
.hero {
    background: linear-gradient(135deg,#1d4ed8 0%, #7c3aed 100%);
    padding: 1.8rem;
    border-radius: 24px;
    color: white;
    box-shadow: 0 20px 45px rgba(37,99,235,0.25);
}

/* Cards */
.card {
    background: white;
    border-radius: 18px;
    padding: 1rem;
    border: 1px solid #e5e7eb;
    box-shadow: 0 8px 24px rgba(0,0,0,0.04);
}

/* KPI */
.kpi {
    background: white;
    border-radius: 18px;
    padding: 1rem;
    text-align: center;
    border: 1px solid #e5e7eb;
    box-shadow: 0 8px 24px rgba(0,0,0,0.03);
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: white;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    padding: 0.8rem;
    box-shadow: 0 8px 20px rgba(0,0,0,0.03);
    margin-bottom: 0.8rem;
}

/* Hide footer/menu */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}


</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -------------------------------------------------
# VECTOR STORE
# -------------------------------------------------
@st.cache_resource
def load_vector_store():
    try:
        index = faiss.read_index(f"{VECTOR_FOLDER}/docs.index")

        with open(f"{VECTOR_FOLDER}/texts.pkl", "rb") as f:
            texts = pickle.load(f)

    except:
        index = faiss.IndexFlatL2(384)
        texts = []

    return index, texts


def save_vector(index, texts):
    faiss.write_index(index, f"{VECTOR_FOLDER}/docs.index")

    with open(f"{VECTOR_FOLDER}/texts.pkl", "wb") as f:
        pickle.dump(texts, f)

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def extract_text(file_path):
    content = ""

    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

    elif file_path.endswith(".pdf"):
        reader = PdfReader(file_path)

        for page in reader.pages:
            txt = page.extract_text()

            if txt:
                content += txt + "\n"

    return content


def chunk_text(content, size=1200):
    return [
        content[i:i+size]
        for i in range(0, len(content), size)
    ]


def get_latest_chunk_count():
    try:
        with open(f"{VECTOR_FOLDER}/texts.pkl", "rb") as f:
            texts = pickle.load(f)

        return len(texts)

    except:
        return 0


def rebuild_index(progress=None, status=None):

    files = os.listdir(DATA_FOLDER)
    all_chunks = []

    total_files = len(files)

    for i, file in enumerate(files):

        if status:
            status.text(f"Reading {file}")

        path = os.path.join(DATA_FOLDER, file)

        text = extract_text(path)

        if text.strip():
            chunks = chunk_text(text)
            all_chunks.extend(chunks)

        if progress and total_files > 0:
            progress.progress((i+1)/total_files * 0.3)

    if all_chunks:

        batch_size = 32
        embeddings = []

        total_chunks = len(all_chunks)

        for start in range(0, total_chunks, batch_size):

            end = min(start+batch_size, total_chunks)

            if status:
                status.text(
                    f"Embedding {start+1}-{end} of {total_chunks}"
                )

            batch = all_chunks[start:end]

            emb = model.encode(
                batch,
                show_progress_bar=False
            )

            embeddings.extend(emb)

            if progress:
                progress.progress(
                    0.3 + (end/total_chunks * 0.5)
                )

        if status:
            status.text("Building vector index...")

        index = faiss.IndexFlatL2(384)
        index.add(np.array(embeddings))

        save_vector(index, all_chunks)

    else:
        save_vector(faiss.IndexFlatL2(384), [])

    if progress:
        progress.progress(1.0)

    if status:
        status.text("Completed successfully.")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:

    st.markdown("""
    <div style="padding:1rem 0 1rem 0;">
        <div style="font-size:1.8rem;font-weight:800;">
            🧠 Cognivault AI
        </div>
        <div style="opacity:0.85;font-size:0.95rem;">
            Intelligent Knowledge. Trusted Decisions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("💬 Chat Workspace"):
        st.session_state.page = "Chat"

    if st.button("📊 Governance Center"):
        st.session_state.page = "Governance"

    st.markdown("---")

# -------------------------------------------------
# CHAT PAGE
# -------------------------------------------------
if st.session_state.page == "Chat":

    st.markdown("""
    <div class="hero">
        <div style="font-size:2.4rem;font-weight:800;">
            🧠 Cognivault AI
        </div>
        <div style="font-size:1.1rem;margin-top:0.3rem;">
            Your Enterprise Intelligence Platform
        </div>
        <div style="margin-top:0.8rem;opacity:0.9;">
            Ask documents • Retrieve trusted answers • Accelerate decisions
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.button("📘 Ask Policy")
    with c2:
        st.button("🏗 Explain Architecture")
    with c3:
        st.button("📄 Summarize File")
    with c4:
        st.button("📈 Executive View")

    st.write("")

    left, right = st.columns([3,1])

    with left:

        mode = st.selectbox(
            "Assistant Mode",
            [
                "Onboarding",
                "Support",
                "Architect",
                "Leadership"
            ]
        )

        def mode_prompt(mode):
            prompts = {
                "Onboarding":
                    "Explain simply for beginners.",
                "Support":
                    "Focus on troubleshooting and issue resolution.",
                "Architect":
                    "Focus on architecture and dependencies.",
                "Leadership":
                    "Summarize business insights and risks."
            }

            return prompts[mode]

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        question = st.chat_input("Ask Cognivault AI...")

        if question:

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):

                with st.spinner("Thinking..."):

                    t0 = time.time()

                    index, texts = load_vector_store()
                    t1 = time.time()

                    q_embedding = model.encode([question])
                    t2 = time.time()

                    distances, indices = index.search(
                        np.array(q_embedding),
                        3
                    )
                    t3 = time.time()

                    chunks = []

                    if len(texts) > 0:
                        chunks = [
                            texts[i]
                            for i in indices[0]
                        ]

                    context = "\n\n".join(chunks)

                    prompt = f"""
You are Cognivault AI.

Behavior:
{mode_prompt(mode)}

Use ONLY the context below.

Context:
{context}

Question:
{question}
"""

                    response = ollama.chat(
                        model="mistral",
                        messages=[
                            {
                                "role":"user",
                                "content":prompt
                            }
                        ]
                    )

                    answer = response["message"]["content"]

                    st.write(answer)

                    with st.expander("⚡ Performance Debug"):
                        st.write("Load:", round(t1-t0,2), "sec")
                        st.write("Embed:", round(t2-t1,2), "sec")
                        st.write("Search:", round(t3-t2,4), "sec")
                        st.write("Chunks:", len(texts))

                    with st.expander("📚 Sources Used"):
                        for i, c in enumerate(chunks,1):
                            st.markdown(f"**Source {i}:**")
                            st.write(c[:1000])

            st.session_state.messages.append(
                {
                    "role":"assistant",
                    "content":answer
                }
            )

    with right:

        files = len(os.listdir(DATA_FOLDER))
        chunks = get_latest_chunk_count()

        st.markdown(
            f"""
            <div class="kpi">
                <h3>{files}</h3>
                <div>Stored Files</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        st.markdown(
            f"""
            <div class="kpi">
                <h3>{chunks}</h3>
                <div>Indexed Chunks</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        st.markdown(
            """
            <div class="card">
                <b>System Status</b><br><br>
                🟢 Ollama Running<br>
                🟢 Vector DB Ready<br>
                🟢 Model Loaded
            </div>
            """,
            unsafe_allow_html=True
        )

# -------------------------------------------------
# GOVERNANCE PAGE
# -------------------------------------------------
if st.session_state.page == "Governance":

    st.markdown("""
    <div class="card">
        <div style="font-size:1.8rem;font-weight:800;">
            📊 Governance Center
        </div>
        <div style="color:#64748b;">
            Manage enterprise files, indexing and trust operations.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    files = os.listdir(DATA_FOLDER)

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Stored Files", len(files))

    with c2:
        st.metric("Indexed Chunks", get_latest_chunk_count())

    st.markdown("---")

    st.subheader("📂 Uploaded Files")

    if files:

        for file in files:

            path = os.path.join(DATA_FOLDER, file)

            with st.expander(file):

                st.write(
                    "Size:",
                    round(os.path.getsize(path)/1024,2),
                    "KB"
                )

                if st.button(
                    "Delete",
                    key=f"delete_{file}"
                ):

                    os.remove(path)

                    progress = st.progress(0)
                    status = st.empty()

                    rebuild_index(progress, status)

                    load_vector_store.clear()

                    st.success("Deleted successfully.")

                    st.rerun()

    st.markdown("---")

    st.subheader("⬆ Upload Files")

    uploaded = st.file_uploader(
        "Select files",
        type=["txt","pdf"],
        accept_multiple_files=True,
        key=f"upload_{st.session_state.uploader_key}"
    )

    if uploaded:

        if st.button("Submit Upload"):

            for file in uploaded:

                path = os.path.join(
                    DATA_FOLDER,
                    file.name
                )

                with open(path,"wb") as f:
                    f.write(file.getbuffer())

            progress = st.progress(0)
            status = st.empty()

            rebuild_index(progress, status)

            load_vector_store.clear()

            st.session_state.uploader_key += 1

            st.success("Uploaded successfully.")

            st.rerun()

    st.markdown("---")

    if st.button("🔄 Rebuild Full Index"):

        progress = st.progress(0)
        status = st.empty()

        rebuild_index(progress, status)

        load_vector_store.clear()

        st.success("Rebuilt successfully.")

        st.rerun()