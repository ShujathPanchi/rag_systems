from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # ✅ fixed
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np  # ✅ added (needed for embeddings)
import os
import pickle

DATA_PATH = "data"
VECTOR_PATH = "vector_store"

docs = []
for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_PATH, file))
        docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)
texts = [chunk.page_content for chunk in chunks]

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts)
embeddings = np.array(embeddings).astype("float32")  # ✅ fixed (faiss needs float32)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

os.makedirs(VECTOR_PATH, exist_ok=True)
faiss.write_index(index, f"{VECTOR_PATH}/docs.index")
with open(f"{VECTOR_PATH}/texts.pkl", "wb") as f:
    pickle.dump(texts, f)

print("Knowledge base created successfully.")