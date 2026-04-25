import ollama
import numpy as np
import time

def ask_question(model,index,texts,question,mode_prompt):

    t0 = time.time()

    q_embedding = model.encode([question])

    t1 = time.time()

    distances, indices = index.search(
        np.array(q_embedding),3
    )

    t2 = time.time()

    chunks = []

    if len(texts) > 0:
        chunks = [texts[i] for i in indices[0]]

    context = "\n\n".join(chunks)

    prompt = f"""
You are Cognivault AI.

Behavior:
{mode_prompt}

Use ONLY the context below.

Context:
{context}

Question:
{question}
"""

    response = ollama.chat(
        model="mistral",
        messages=[{
            "role":"user",
            "content":prompt
        }]
    )

    answer = response["message"]["content"]

    return answer, chunks, t0, t1, t2