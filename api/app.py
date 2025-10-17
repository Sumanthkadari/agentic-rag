from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import traceback

from src.agents.orchestrator import Orchestrator
from src.agents.retriever_agent import RetrieverAgent
from src.utils.ingestion.retriever import HybridRetriever

# Initialize retriever and agent
retriever = HybridRetriever(
    chunks_file="chunks.txt",
    chroma_dir="./chroma_db",
    collection_name="pdf_embeddings",
    top_k=5,
)

retriever_agent = RetrieverAgent(retriever)
orchestrator = Orchestrator(retriever_agent)

# FastAPI app
app = FastAPI(title="Agentic RAG API", version="1.0.0")


class Query(BaseModel):
    query: str
    user_id: Optional[str] = None
    max_docs: Optional[int] = 5


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Agentic RAG API is running"}


@app.post("/rag")
def rag_endpoint(req: Query):
    """Main RAG endpoint"""
    try:
        print(f"\n[API] Received query: '{req.query}'")  # Debug log
        
        # Pass query as keyword argument
        response = orchestrator.run(
            query=req.query,
            user_id=req.user_id,
            max_docs=req.max_docs
        )
        
        print(f"[API] Response: {response}")
        return response
        
    except Exception as e:
        print(f"\n[API ERROR] {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))