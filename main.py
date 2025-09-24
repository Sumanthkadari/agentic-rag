import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError

from src.utils.ingestion.chunker import chunk_pdfs
from src.utils.ingestion.retriever import HybridRetriever


PDF_FILES = ["data/lucid_owners_manual.pdf", "data/wells_fargo.pdf"]
CHUNKS_FILE = "chunks.txt"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "pdf_embeddings"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5


def prepare_chunks(file_path=CHUNKS_FILE, pdf_files=PDF_FILES):
    """Load chunks from file, or create from PDFs if missing."""
    if os.path.exists(file_path):
        print("[INFO] Chunks file found, skipping chunking.")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().split("\n---\n")
    print("[INFO] Chunking PDFs...")
    chunks = chunk_pdfs(pdf_files)
    with open(file_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n---\n")
    print(f"[INFO] Total chunks created: {len(chunks)}")
    return chunks


def prepare_embeddings(chunks, collection_name=COLLECTION_NAME, chroma_dir=CHROMA_DIR, model_name=MODEL_NAME):
    """Ensure embeddings exist in ChromaDB, create if missing."""
    client = chromadb.PersistentClient(path=chroma_dir, settings=Settings())
    try:
        return client.get_collection(name=collection_name)
    except NotFoundError:
        print(f"[INFO] Collection '{collection_name}' not found. Creating and embedding chunks...")

    embed_model = SentenceTransformer(model_name)
    embeddings = embed_model.encode(chunks, show_progress_bar=True)

    collection = client.create_collection(name=collection_name)
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        embeddings=[emb.tolist() for emb in embeddings],
        metadatas=[{"text": chunk} for chunk in chunks],
        documents=chunks,
    )
    print("[INFO] Embeddings stored in ChromaDB successfully!")
    return collection


def interactive_cli(retriever):
    """Simple interactive query loop."""
    print(f"\n[INFO] System ready. Retrieving top {TOP_K} documents from vector DB.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Enter your query: ")
        if query.lower() == "exit":
            break

        results = retriever.hybrid_search(query, top_k=TOP_K)
        if not results:
            print("No relevant documents found.\n")
            continue

        print("\n--- Top Retrieved Documents ---")
        for i, doc in enumerate(results, 1):
            print(f"\n[{i}] {doc}\n")
            print("-" * 50)


def main():
    chunks = prepare_chunks()
    prepare_embeddings(chunks)
    retriever = HybridRetriever(chunks_file=CHUNKS_FILE, top_k=TOP_K)
    interactive_cli(retriever)


if __name__ == "__main__":
    main()
