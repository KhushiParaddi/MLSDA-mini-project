import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import the 2-Agent orchestration system
from pipelines.agents import RetrieverAgent, ComplianceAgent

app = FastAPI(
    title="BDA / BBMP Building Bye-Laws Multi-Agent RAG Query API",
    description="Production-ready API powered by RetrieverAgent and ComplianceAgent for regulatory setback queries.",
    version="1.1.0"
)

# Initialize Agent instances globally
retriever_agent = None
compliance_agent = None

@app.on_event("startup")
def startup_event():
    global retriever_agent, compliance_agent
    retriever_agent = RetrieverAgent()
    compliance_agent = ComplianceAgent(retriever_agent)
    print("Multi-Agent System (RetrieverAgent & ComplianceAgent) loaded into API memory.")

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
    compliance_verdict: str
    retrieved_chunks: List[ChunkResponse]

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "bda-bbmp-rag-eval",
        "agents": ["RetrieverAgent", "ComplianceAgent"]
    }

@app.post("/api/v1/query", response_model=QueryResponse)
def query_regulatory_rag(payload: QueryRequest):
    if not retriever_agent or not compliance_agent:
        raise HTTPException(status_code=500, detail="Multi-Agent system not initialized.")

    query = payload.query
    top_k = payload.top_k
    strategy = payload.strategy.lower()

    if strategy not in ["dense", "metadata", "hybrid"]:
        raise HTTPException(status_code=400, detail="Invalid strategy. Choose 'dense', 'metadata', or 'hybrid'.")

    # 1. Execute Retrieval via Agent 1 (RetrieverAgent)
    results = retriever_agent.run(query, strategy=strategy, top_k=top_k)

    # 2. Audit Compliance Context via Agent 2 (ComplianceAgent)
    compliance_audit = compliance_agent.evaluate_compliance(query)

    # 3. Format Chunks for API Output
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
        compliance_verdict=compliance_audit["verdict"],
        retrieved_chunks=formatted_results
    )