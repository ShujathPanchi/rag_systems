import faiss
import pickle
import numpy as np
import os
from pypdf import PdfReader
import streamlit as st

from utils.config import DATA_FOLDER, VECTOR_FOLDER
from services.model_service import load_model

model = load_model()


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

            txt = page.extract_text()

            if txt:
                content += txt + "\n"

    return content


def chunk_text(content, size=1200):

    return [
        content[i:i+size]
        for i in range(
            0,
            len(content),
            size
        )
    ]


def get_latest_chunk_count():

    try:
        with open(
            f"{VECTOR_FOLDER}/texts.pkl",
            "rb"
        ) as f:
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
            status.text(
                f"Reading {file}"
            )

        path = os.path.join(
            DATA_FOLDER,
            file
        )

        text = extract_text(path)

        if text.strip():

            chunks = chunk_text(text)

            all_chunks.extend(chunks)

        if progress and total_files > 0:

            progress.progress(
                (i + 1) / total_files * 0.3
            )

    if all_chunks:

        batch_size = 32
        embeddings = []

        total_chunks = len(all_chunks)

        for start in range(
            0,
            total_chunks,
            batch_size
        ):

            end = min(
                start + batch_size,
                total_chunks
            )

            if status:
                status.text(
                    f"Embedding "
                    f"{start+1}-{end} "
                    f"of {total_chunks}"
                )

            batch = all_chunks[start:end]

            emb = model.encode(
                batch,
                show_progress_bar=False
            )

            embeddings.extend(emb)

            if progress:
                progress.progress(
                    0.3 +
                    (
                        end / total_chunks * 0.5
                    )
                )

        if status:
            status.text(
                "Building vector index..."
            )

        index = faiss.IndexFlatL2(384)

        index.add(
            np.array(embeddings)
        )

        save_vector(
            index,
            all_chunks
        )

    else:

        save_vector(
            faiss.IndexFlatL2(384),
            []
        )

    if progress:
        progress.progress(1.0)

    if status:
        status.text(
            "Completed successfully."
        )