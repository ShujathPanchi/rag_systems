from sentence_transformers import SentenceTransformer
import faiss
import pickle
import ollama
import numpy as np

VECTOR_PATH = "vector_store"

# Load vector DB
index = faiss.read_index(f"{VECTOR_PATH}/docs.index")

with open(f"{VECTOR_PATH}/texts.pkl", "rb") as f:
    texts = pickle.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

while True:
    question = input("\nAsk Question (type exit to quit): ")

    if question.lower() == "exit":
        break

    # Convert question to embedding
    q_embedding = model.encode([question])

    # Search top 3 chunks
    distances, indices = index.search(np.array(q_embedding), 3)

    context = "\n\n".join([texts[i] for i in indices[0]])

    prompt = f"""
Use the context below to answer the question clearly.

Context:
{context}

Question:
{question}
"""

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    print("\nAnswer:\n")
    print(response["message"]["content"])