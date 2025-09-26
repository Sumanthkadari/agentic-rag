import os
from src.utils.ingestion.chunker import chunk_pdfs
from src.utils.ingestion.retriever import HybridRetriever
from src.agents.orchestrator import Orchestrator
import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer

PDF_FILES = ["data/lucid_owners_manual.pdf", "data/wells_fargo.pdf"]
CHUNKS_FILE = "chunks.txt"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "pdf_embeddings"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5


def prepare_chunks(file_path=CHUNKS_FILE, pdf_files=PDF_FILES):
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


def prepare_embeddings(chunks):
    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings())
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except NotFoundError:
        print(f"[INFO] Collection '{COLLECTION_NAME}' not found. Creating and embedding chunks...")

    embed_model = SentenceTransformer(MODEL_NAME)
    embeddings = embed_model.encode(chunks, show_progress_bar=True)

    collection = client.create_collection(name=COLLECTION_NAME)
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        embeddings=[emb.tolist() for emb in embeddings],
        metadatas=[{"text": chunk} for chunk in chunks],
        documents=chunks,
    )
    print("[INFO] Embeddings stored in ChromaDB successfully!")
    return collection


def interactive_cli(orchestrator: Orchestrator):
    print(f"\n[INFO] System ready. Ask anything about Lucid Motors or Wells Fargo.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Enter your query: ")
        if query.lower() == "exit":
            break

        response = orchestrator.run(query, max_docs=TOP_K)
        print("\n--- Agentic RAG Response ---")
        print("Answer:\n", response["answer"])
        print("\nCitations:", response["citations"])
        print("Follow-ups:", response["follow_ups"])
        print("-" * 80)


def main():
    chunks = prepare_chunks()
    prepare_embeddings(chunks)
    retriever = HybridRetriever(chunks_file=CHUNKS_FILE, top_k=TOP_K)
    orchestrator = Orchestrator(retriever)
    interactive_cli(orchestrator)


if __name__ == "__main__":
    main()
