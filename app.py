import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from pipelines.retriever import RegulatoryRetriever

app = FastAPI(
    title="BDA / BBMP Building Bye-Laws RAG Query API",
    description="Production-ready API for regulatory setback and building restriction queries.",
    version="1.0.0"
)

# Initialize Retriever globally
retriever = None

@app.on_event("startup")
def startup_event():
    global retriever
    retriever = RegulatoryRetriever()
    print("Regulatory Retriever loaded into API memory.")

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
    strategy: Optional[str] = "hybrid"  # "dense", "metadata", or "hybrid"

class ChunkResponse(BaseModel):
    chunk_id: str
    text: str
    metadata: dict
    score: float

class QueryResponse(BaseModel):
    query: str
    strategy_used: str
    retrieved_chunks: List[ChunkResponse]

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "bda-bbmp-rag-eval"}

@app.post("/api/v1/query", response_model=QueryResponse)
def query_regulatory_rag(payload: QueryRequest):
    if not retriever:
        raise HTTPException(status_code=500, detail="Retriever service not initialized")

    query = payload.query
    top_k = payload.top_k
    strategy = payload.strategy.lower()

    if strategy == "dense":
        results = retriever.dense_search(query, top_k=top_k)
    elif strategy == "metadata":
        results = retriever.metadata_filtered_search(query, chunk_type="table", top_k=top_k)
    elif strategy == "hybrid":
        results = retriever.hybrid_search(query, top_k=top_k)
    else:
        raise HTTPException(status_code=400, detail="Invalid strategy. Choose 'dense', 'metadata', or 'hybrid'.")

    formatted_results = [
        ChunkResponse(
            chunk_id=res["chunk_id"],
            text=res["text"],
            metadata=res["metadata"],
            score=float(res.get("score", 1.0))
        )
        for res in results
    ]

    return QueryResponse(
        query=query,
        strategy_used=strategy,
        retrieved_chunks=formatted_results
    )