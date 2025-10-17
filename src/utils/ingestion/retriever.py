from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from rank_bm25 import BM25Okapi


class HybridRetriever:
    def __init__(self, chunks_file="chunks.txt", chroma_dir="./chroma_db", collection_name="pdf_embeddings", model_name="all-MiniLM-L6-v2", top_k=5):
        self.chunks_file = chunks_file
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.model_name = model_name
        self.top_k = top_k

        with open(self.chunks_file, "r", encoding="utf-8") as f:
            self.chunks = f.read().split("\n---\n")

        tokenized_corpus = [doc.split() for doc in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        self.embed_model = SentenceTransformer(self.model_name)

        self.client = chromadb.PersistentClient(path=self.chroma_dir, settings=Settings())

        try:
            self.collection = self.client.get_collection(name=self.collection_name)
        except NotFoundError:
            raise ValueError(f"Collection '{self.collection_name}' not found. Run your embedding step first!")

    # Semantic Search
    def semantic_search(self, query, top_k=None):
        if top_k is None:
            top_k = self.top_k
        query_emb = self.embed_model.encode([query])[0].tolist()
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k
        )
        return results["documents"][0]

    # BM25 Search
    def bm25_search(self, query, top_k=None):
        if top_k is None:
            top_k = self.top_k
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[-top_k:][::-1]
        return [self.chunks[i] for i in top_indices]

    #Hybrid Search (Semantic + BM25)
    def hybrid_search(self, query, top_k=None):
        if top_k is None:
            top_k = self.top_k

        semantic_results = self.semantic_search(query, top_k)
        bm25_results = self.bm25_search(query, top_k)

        # Merge + deduplicate (preserve order)
        combined_results = list(dict.fromkeys(semantic_results + bm25_results))
        return combined_results[:top_k]