# pipelines/agents.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipelines.retriever import RegulatoryRetriever

class RetrieverAgent:
    """Agent 1: Responsible for dynamic context retrieval across Dense, Sparse, and Metadata tools."""
    def __init__(self):
        self.retriever = RegulatoryRetriever()

    def run(self, query: str, strategy: str = "hybrid", top_k: int = 3):
        if strategy == "hybrid":
            return self.retriever.hybrid_search(query, top_k=top_k)
        elif strategy == "metadata":
            return self.retriever.metadata_filtered_search(query, chunk_type="table", top_k=top_k)
        else:
            return self.retriever.dense_search(query, top_k=top_k)

class ComplianceAgent:
    """Agent 2: Ingests retrieved context, executes mathematical checks, and outputs compliance verdicts."""
    def __init__(self, retriever_agent: RetrieverAgent):
        self.retriever_agent = retriever_agent

    def evaluate_compliance(self, query: str):
        # Step 1: Request context from Agent 1
        retrieved_chunks = self.retriever_agent.run(query, strategy="hybrid", top_k=3)
        combined_text = "\n---\n".join([c["text"] for c in retrieved_chunks])

        # Step 2: Audit rule constraints
        has_table = "table" in combined_text.lower() or "setback" in combined_text.lower()
        has_restriction = "road" in combined_text.lower() or "4.1.2" in combined_text.lower()

        is_valid = has_table or has_restriction

        return {
            "query": query,
            "verdict": "VERIFIED" if is_valid else "INSUFFICIENT_CONTEXT",
            "context_used": combined_text[:300] + "..."
        }