import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import ollama

st.set_page_config(page_title="Knowledge Assistant", page_icon="🧠", layout="wide")

st.title("🧠 Enterprise Knowledge Assistant")
st.caption("Local RAG Assistant with Sources")

# Session history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load resources once
@st.cache_resource
def load_resources():
    index = faiss.read_index("vector_store/docs.index")

    with open("vector_store/texts.pkl", "rb") as f:
        texts = pickle.load(f)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    return index, texts, model

index, texts, model = load_resources()

# Display previous chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
question = st.chat_input("Ask anything from your documents...")

if question:
    # show user message
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            q_embedding = model.encode([question])
            distances, indices = index.search(np.array(q_embedding), 3)

            retrieved_chunks = [texts[i] for i in indices[0]]
            context = "\n\n".join(retrieved_chunks)

            prompt = f"""
You are a helpful enterprise assistant.

Use ONLY the provided context.
If answer is not available, say:
'I could not find that in the uploaded documents.'

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
                for i, chunk in enumerate(retrieved_chunks, 1):
                    st.markdown(f"**Source {i}:**")
                    st.write(chunk[:1000])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )