import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import ollama

st.set_page_config(page_title="Knowledge Assistant", page_icon="🧠", layout="wide")

st.title("🧠 Enterprise Knowledge Assistant")
st.caption("Role-Based AI Assistant")

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

# Sidebar mode selector
mode = st.sidebar.selectbox(
    "Choose Assistant Mode",
    ["Onboarding", "Support", "Architect", "Leadership"]
)

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

def get_mode_prompt(mode):
    prompts = {
        "Onboarding": "Explain simply for a new employee. Use beginner-friendly language.",
        "Support": "Focus on troubleshooting, root causes, steps to resolve.",
        "Architect": "Focus on system design, components, integrations, dependencies.",
        "Leadership": "Summarize strategic risks, blockers, business impact, recommended actions."
    }
    return prompts[mode]

question = st.chat_input("Ask your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": f'[{mode}] {question}'})

    with st.chat_message("user"):
        st.write(f"**{mode} Mode:** {question}")

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            q_embedding = model.encode([question])
            distances, indices = index.search(np.array(q_embedding), 5)

            chunks = [texts[i] for i in indices[0]]
            context = "\n\n".join(chunks)

            prompt = f"""
You are an enterprise AI assistant.

Behavior Instructions:
{get_mode_prompt(mode)}

Use ONLY the context below.
If missing, say information not found.

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
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(f"**Source {i}:**")
                    st.write(chunk[:1000])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )