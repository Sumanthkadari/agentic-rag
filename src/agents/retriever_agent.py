import logging
from typing import List, Dict, Any
from src.utils.ingestion.retriever import HybridRetriever

logger = logging.getLogger("retriever_agent")
logging.basicConfig(level=logging.INFO)

class RetrieverAgent:
    """Autonomous retriever agent using hybrid, semantic, or BM25 search."""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def run(self, query: str, top_k: int = 5, mode: str = "hybrid") -> List[str]:
        logger.info(f"[RetrieverAgent] mode={mode}, top_k={top_k}, query={query}")

        try:
            if mode == "semantic":
                return self.retriever.semantic_search(query, top_k)
            elif mode == "bm25":
                return self.retriever.bm25_search(query, top_k)
            else:
                return self.retriever.hybrid_search(query, top_k)
        except Exception as e:
            logger.error(f"[RetrieverAgent] Retrieval failed: {e}")
            return []

