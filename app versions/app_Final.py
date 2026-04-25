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

# --------------------------------
# CONFIG
# --------------------------------
st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Enterprise Knowledge Assistant")
st.caption("Local AI Knowledge Platform")

DATA_FOLDER = "data"
VECTOR_FOLDER = "vector_store"

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

# --------------------------------
# PAGE STATE
# --------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Chat"

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# --------------------------------
# SIDEBAR NAVIGATION
# --------------------------------
st.sidebar.title("🧭 Navigation")

if st.sidebar.button("💬 Chat Assistant"):
    st.session_state.page = "Chat"

if st.sidebar.button("📊 Governance Dashboard"):
    st.session_state.page = "Governance"

st.sidebar.markdown("---")

# --------------------------------
# LOAD MODEL
# --------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# --------------------------------
# LOAD VECTOR STORE
# --------------------------------
@st.cache_resource
def load_vector_store():
    try:
        index = faiss.read_index(
            f"{VECTOR_FOLDER}/docs.index"
        )

        with open(
            f"{VECTOR_FOLDER}/texts.pkl",
            "rb"
        ) as f:
            texts = pickle.load(f)

    except:
        index = faiss.IndexFlatL2(384)
        texts = []

    return index, texts

# --------------------------------
# SAVE VECTOR STORE
# --------------------------------
def save_vector(index, texts):
    faiss.write_index(
        index,
        f"{VECTOR_FOLDER}/docs.index"
    )

    with open(
        f"{VECTOR_FOLDER}/texts.pkl",
        "wb"
    ) as f:
        pickle.dump(texts, f)

# --------------------------------
# HELPERS
# --------------------------------
def extract_text(file_path):
    content = ""

    if file_path.endswith(".txt"):
        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:
            content = f.read()

    elif file_path.endswith(".pdf"):
        reader = PdfReader(file_path)

        for page in reader.pages:
            text = page.extract_text()

            if text:
                content += text + "\n"

    return content


def chunk_text(content, size=1200):
    return [
        content[i:i + size]
        for i in range(0, len(content), size)
    ]


def get_latest_chunk_count():
    try:
        with open(
            f"{VECTOR_FOLDER}/texts.pkl",
            "rb"
        ) as f:
            latest_texts = pickle.load(f)

        return len(latest_texts)

    except:
        return 0

# --------------------------------
# REBUILD INDEX
# --------------------------------
def rebuild_index(
    progress_bar=None,
    status_text=None
):
    all_chunks = []
    files = os.listdir(DATA_FOLDER)
    total_files = len(files)

    for i, file in enumerate(files):

        if status_text:
            status_text.text(
                f"Reading file {i+1}/{total_files}: {file}"
            )

        path = os.path.join(
            DATA_FOLDER,
            file
        )

        content = extract_text(path)

        if content.strip():
            chunks = chunk_text(content)
            all_chunks.extend(chunks)

        if progress_bar and total_files > 0:
            progress_bar.progress(
                (i + 1) / total_files * 0.30
            )

    if all_chunks:

        batch_size = 32
        total_chunks = len(all_chunks)
        all_embeddings = []

        for start in range(
            0,
            total_chunks,
            batch_size
        ):

            end = min(
                start + batch_size,
                total_chunks
            )

            if status_text:
                status_text.text(
                    f"Generating embeddings "
                    f"{start+1}-{end} of "
                    f"{total_chunks}"
                )

            batch = all_chunks[start:end]

            batch_embeddings = model.encode(
                batch,
                show_progress_bar=False
            )

            all_embeddings.extend(
                batch_embeddings
            )

            if progress_bar:
                progress_bar.progress(
                    0.30 +
                    (end / total_chunks * 0.50)
                )

        embeddings = np.array(
            all_embeddings
        )

        if status_text:
            status_text.text(
                "Building FAISS index..."
            )

        new_index = faiss.IndexFlatL2(384)
        new_index.add(embeddings)

        if progress_bar:
            progress_bar.progress(0.90)

        if status_text:
            status_text.text(
                "Saving vector database..."
            )

        save_vector(
            new_index,
            all_chunks
        )

    else:
        save_vector(
            faiss.IndexFlatL2(384),
            []
        )

    if progress_bar:
        progress_bar.progress(1.0)

    if status_text:
        status_text.text(
            "Completed successfully."
        )

# --------------------------------
# CHAT PAGE
# --------------------------------
if st.session_state.page == "Chat":

    mode = st.sidebar.selectbox(
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
                "Focus on system design, integrations, dependencies.",

            "Leadership":
                "Summarize business risks, blockers, actions."
        }

        return prompts[mode]

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input(
        "Ask your documents..."
    )

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

                q_embedding = model.encode(
                    [question]
                )
                t2 = time.time()

                distances, indices = index.search(
                    np.array(q_embedding),
                    3
                )
                t3 = time.time()

                if len(texts) > 0:

                    chunks = [
                        texts[i]
                        for i in indices[0]
                    ]

                    context = "\n\n".join(
                        chunks
                    )

                else:
                    chunks = []
                    context = ""

                prompt = f"""
You are an enterprise assistant.

Behavior:
{mode_prompt(mode)}

Use ONLY the context below.
If answer is unavailable say not found.

Context:
{context}

Question:
{question}
"""

                response = ollama.chat(
                    model="mistral",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                answer = response[
                    "message"
                ]["content"]

                st.write(answer)

                with st.expander(
                    "⚡ Performance Debug"
                ):
                    st.write(
                        "Load Vector Store:",
                        round(t1 - t0, 2),
                        "sec"
                    )
                    st.write(
                        "Question Embedding:",
                        round(t2 - t1, 2),
                        "sec"
                    )
                    st.write(
                        "FAISS Search:",
                        round(t3 - t2, 4),
                        "sec"
                    )
                    st.write(
                        "Total Chunks:",
                        len(texts)
                    )

                with st.expander(
                    "📚 Sources Used"
                ):

                    if chunks:
                        for i, chunk in enumerate(
                            chunks,
                            1
                        ):
                            st.markdown(
                                f"**Source {i}:**"
                            )
                            st.write(
                                chunk[:1000]
                            )
                    else:
                        st.write(
                            "No sources available."
                        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

# --------------------------------
# GOVERNANCE PAGE
# --------------------------------
if st.session_state.page == "Governance":

    st.title("📊 Governance Dashboard")

    files = os.listdir(DATA_FOLDER)
    chunk_count = get_latest_chunk_count()

    col1, col2 = st.columns(2)

    files_metric = col1.empty()
    chunks_metric = col2.empty()

    def refresh_metrics():
        latest_files = os.listdir(
            DATA_FOLDER
        )

        latest_chunks = get_latest_chunk_count()

        files_metric.metric(
            "Stored Files",
            len(latest_files)
        )

        chunks_metric.metric(
            "Indexed Chunks",
            latest_chunks
        )

    refresh_metrics()

    st.markdown("---")

    # FILES
    st.subheader("📂 Uploaded Files")

    if files:

        for file in files:

            path = os.path.join(
                DATA_FOLDER,
                file
            )

            size = round(
                os.path.getsize(path) / 1024,
                2
            )

            modified = datetime.fromtimestamp(
                os.path.getmtime(path)
            ).strftime(
                "%Y-%m-%d %H:%M"
            )

            with st.expander(file):

                st.write(
                    f"Size: {size} KB"
                )

                st.write(
                    f"Updated: {modified}"
                )

                if st.button(
                    "Delete",
                    key=f"delete_{file}"
                ):

                    start_time = time.time()

                    with st.spinner(
                        f"Deleting {file}..."
                    ):
                        os.remove(path)

                    progress = st.progress(0)
                    status = st.empty()

                    rebuild_index(
                        progress,
                        status
                    )

                    load_vector_store.clear()

                    elapsed = round(
                        time.time() -
                        start_time,
                        2
                    )

                    st.success(
                        f"{file} deleted "
                        f"successfully in "
                        f"{elapsed} sec."
                    )

                    st.rerun()

    else:
        st.info("No files uploaded.")

    st.markdown("---")

    # UPLOAD
    st.subheader("⬆ Upload Files")

    uploaded_files = st.file_uploader(
        "Select TXT / PDF files",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        key=f"upload_files_{st.session_state.uploader_key}"
    )

    if uploaded_files:

        st.write("Selected Files:")

        for f in uploaded_files:
            st.write("•", f.name)

        if st.button("Submit Upload"):

            start_time = time.time()

            with st.spinner(
                "Saving uploaded files..."
            ):

                for file in uploaded_files:

                    path = os.path.join(
                        DATA_FOLDER,
                        file.name
                    )

                    with open(
                        path,
                        "wb"
                    ) as out:
                        out.write(
                            file.getbuffer()
                        )

            progress = st.progress(0)
            status = st.empty()

            rebuild_index(
                progress,
                status
            )

            load_vector_store.clear()

            elapsed = round(
                time.time() -
                start_time,
                2
            )

            st.success(
                f"Files uploaded and indexed "
                f"in {elapsed} sec."
            )

            st.session_state.uploader_key += 1

            st.rerun()

    st.markdown("---")

    # FULL REBUILD
    if st.button(
        "🔄 Rebuild Full Index"
    ):

        start_time = time.time()

        progress = st.progress(0)
        status = st.empty()

        rebuild_index(
            progress,
            status
        )

        load_vector_store.clear()

        elapsed = round(
            time.time() - start_time,
            2
        )

        status.text(
            "Completed successfully."
        )

        progress.progress(1.0)

        st.success(
            f"Index rebuilt successfully "
            f"in {elapsed} sec."
        )

        time.sleep(1)

        st.rerun()