import os
import sys
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import the Multi-Agent orchestration system
from pipelines.agents import RetrieverAgent, ComplianceAgent

# Single FastAPI Application Instance
app = FastAPI(
    title="BDA / BBMP Building Bye-Laws Multi-Agent RAG Service",
    description="Production-ready API and Web Portal powered by RetrieverAgent and ComplianceAgent for building proposal audit and regulatory setback queries.",
    version="2.0.0"
)

# Mount static files directory for frontend UI
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Agent instances globally
retriever_agent = None
compliance_agent = None

@app.on_event("startup")
def startup_event():
    global retriever_agent, compliance_agent
    retriever_agent = RetrieverAgent()
    compliance_agent = ComplianceAgent(retriever_agent, model_name="llama3.1")
    print("Multi-Agent System (RetrieverAgent & ComplianceAgent with local Llama) loaded into memory.")

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class UserQueryRequest(BaseModel):
    prompt: str

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

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def read_root():
    """Serves the interactive frontend web portal."""
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>BDA / BBMP Multi-Agent Web Service is Running.</h1><p>Frontend file static/index.html not found.</p>")

@app.get("/health")
def health_check():
    """Service health check endpoint."""
    return {
        "status": "healthy",
        "service": "bda-bbmp-rag-eval",
        "agents": ["RetrieverAgent", "ComplianceAgent"]
    }

@app.post("/api/v1/query-audit")
def process_query_endpoint(payload: UserQueryRequest):
    """Web portal endpoint for Llama 3.1 audit reports and out-of-context handling."""
    if not compliance_agent:
        raise HTTPException(status_code=500, detail="ComplianceAgent not initialized.")
    if not payload.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    return compliance_agent.process_query(payload.prompt)

@app.post("/api/v1/query", response_model=QueryResponse)
def query_regulatory_rag(payload: QueryRequest):
    """Raw RAG query endpoint for benchmarking and low-level chunk retrieval."""
    if not retriever_agent or not compliance_agent:
        raise HTTPException(status_code=500, detail="Multi-Agent system not initialized.")

    query = payload.query
    top_k = payload.top_k or 3
    strategy = (payload.strategy or "hybrid").lower()

    if strategy not in ["dense", "metadata", "hybrid"]:
        raise HTTPException(status_code=400, detail="Invalid strategy. Choose 'dense', 'metadata', or 'hybrid'.")

    # 1. Execute Retrieval via Agent 1 (RetrieverAgent)
    results = retriever_agent.run(query, strategy=strategy, top_k=top_k)

    # 2. Audit Compliance Context via Agent 2 (ComplianceAgent)
    compliance_audit = compliance_agent.process_query(query)

    # Extract verdict string safely
    verdict = "UNKNOWN"
    if isinstance(compliance_audit, dict):
        if "llm_output" in compliance_audit and "overall_verdict" in compliance_audit["llm_output"]:
            verdict = compliance_audit["llm_output"]["overall_verdict"]
        elif "status" in compliance_audit:
            verdict = compliance_audit["status"]

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
        compliance_verdict=verdict,
        retrieved_chunks=formatted_results
    )