import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import ollama

st.set_page_config(page_title="Knowledge Assistant", page_icon="🧠", layout="wide")

st.title("🧠 Enterprise Knowledge Assistant")
st.caption("Improved Retrieval RAG")

if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def load_resources():
    index = faiss.read_index("vector_store/docs.index")

    with open("vector_store/texts.pkl", "rb") as f:
        texts = pickle.load(f)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, texts, model

index, texts, model = load_resources()

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

def rerank_chunks(question, chunks):
    q_words = set(question.lower().split())
    scored = []

    for chunk in chunks:
        c_words = set(chunk.lower().split())
        overlap = len(q_words.intersection(c_words))
        scored.append((overlap, chunk))

    scored.sort(reverse=True, key=lambda x: x[0])

    return [x[1] for x in scored[:3]]

question = st.chat_input("Ask your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            q_embedding = model.encode([question])

            # retrieve top 8 first
            distances, indices = index.search(np.array(q_embedding), 8)

            candidate_chunks = [texts[i] for i in indices[0]]

            # rerank to best 3
            best_chunks = rerank_chunks(question, candidate_chunks)

            context = "\n\n".join(best_chunks)

            prompt = f"""
You are an enterprise assistant.

Use ONLY the context below.
If answer is missing, say you could not find it.

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

            with st.expander("📚 Sources Used"):
                for i, chunk in enumerate(best_chunks, 1):
                    st.markdown(f"**Source {i}:**")
                    st.write(chunk[:1200])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )