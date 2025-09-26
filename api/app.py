from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.agents.orchestrator import Orchestrator
from src.utils.ingestion.retriever import HybridRetriever


# Init retriever + orchestrator
retriever = HybridRetriever(
    chunks_file="chunks.txt",
    chroma_dir="./chroma_db",
    collection_name="pdf_embeddings",
    top_k=5,
)
orchestrator = Orchestrator(retriever)

app = FastAPI(title="Agentic RAG API", version="1.0.0")


class Query(BaseModel):
    query: str
    user_id: Optional[str] = None
    max_docs: Optional[int] = 5


@app.post("/rag")
def rag_endpoint(req: Query):
    try:
        response = orchestrator.run(
            query=req.query,
            user_id=req.user_id,
            max_docs=req.max_docs or 5, 
        )
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
