import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import ollama

# Page config
st.set_page_config(page_title="Knowledge Assistant", page_icon="🧠")

st.title("🧠 Enterprise Knowledge Assistant")
st.caption("Ask questions from your documents locally")

# Load vector store
@st.cache_resource
def load_resources():
    index = faiss.read_index("vector_store/docs.index")

    with open("vector_store/texts.pkl", "rb") as f:
        texts = pickle.load(f)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    return index, texts, model

index, texts, model = load_resources()

# Ask question
question = st.text_input("Enter your question:")

if st.button("Ask") and question:
    with st.spinner("Thinking..."):

        q_embedding = model.encode([question])
        distances, indices = index.search(np.array(q_embedding), 3)

        context = "\n\n".join([texts[i] for i in indices[0]])

        prompt = f"""
Use the context below to answer clearly and accurately.

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

    st.subheader("Answer")
    st.write(answer)