from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError
import pickle
from dotenv import load_dotenv
import os

load_dotenv()

# Load chunks
with open("chunks.txt", "r", encoding="utf-8") as f:
    chunks = f.read().split("\n---\n")

# Generate embeddings
model_name = "all-MiniLM-L6-v2"
embed_model = SentenceTransformer(model_name)

print("Generating embeddings...")
embeddings = embed_model.encode(chunks, show_progress_bar=True)

# Optional: save embeddings locally
with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

# Initialize ChromaDB client
client = chromadb.Client(
    Settings(
        persist_directory="./chroma_db",  
        anonymized_telemetry=False
    )
)

# Create or get collection
collection_name = "pdf_embeddings"
try:
    collection = client.get_collection(name=collection_name)
except NotFoundError:
    collection = client.create_collection(name=collection_name)

# Batch insert embeddings
ids = [str(i) for i in range(len(chunks))]
metadatas = [{"text": chunk} for chunk in chunks]
documents = chunks
embeddings_list = [emb.tolist() for emb in embeddings]

collection.add(
    ids=ids,
    embeddings=embeddings_list,
    metadatas=metadatas,
    documents=documents
)
print("Embeddings stored in ChromaDB successfully!")
print("Collections in DB:", client.list_collections())