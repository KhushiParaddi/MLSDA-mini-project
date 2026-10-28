import os
import json
import requests
from pipelines.retriever import RegulatoryRetriever

class RetrieverAgent:
    """Agent 1: Handles dynamic vector/BM25 retrieval and domain context filtering."""
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
    """Agent 2: Handles Llama reasoning via local Ollama and out-of-context fallback."""

    def __init__(self, retriever_agent: RetrieverAgent, model_name: str = "llama3.1"):
        self.retriever_agent = retriever_agent
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434/api/generate"

    def process_query(self, user_prompt: str):
        # 1. Fetch relevant ground-truth legal context using Agent 1
        retrieved_chunks = self.retriever_agent.run(user_prompt, strategy="hybrid", top_k=3)

        # 2. Out-of-Context Guardrail Check
        # Check if any domain keywords exist in retrieved context or if top score is relevant
        combined_text = " ".join([c["text"].lower() for c in retrieved_chunks])
        domain_keywords = ["setback", "height", "road", "plot", "building", "bda", "bbmp", "zonal", "coverage", "far"]
        
        is_relevant = any(kw in combined_text or kw in user_prompt.lower() for kw in domain_keywords)

        if not is_relevant:
            return {
                "status": "OUT_OF_CONTEXT",
                "message": "I am specialized strictly in Bengaluru Urban Planning (BDA 2015 Master Plan & BBMP 2003 Building Bye-Laws). Your query appears to be outside building regulations or plot setback rules. Please ask a question related to building dimensions, setbacks, road widths, or plot coverage.",
                "retrieved_chunks": []
            }

        # 3. Format retrieved legal context for Llama
        formatted_context = ""
        for idx, c in enumerate(retrieved_chunks, 1):
            source = c["metadata"].get("source_doc", "Regulatory Document")
            page = c["metadata"].get("page_number", "N/A")
            formatted_context += f"--- CONTEXT BLOCK {idx} [{source}, Page {page}] ---\n{c['text']}\n\n"

        # 4. Construct Llama System & User Prompt
        system_prompt = f"""
You are an expert Urban Planning Legal Auditor for BDA Zonal Regulations and BBMP Building Bye-Laws.
Validate the following user prompt/proposal against the provided legal context text blocks.

RETRIEVED LEGAL CONTEXT (Ground Truth):
{formatted_context}

USER PROMPT / PROPOSAL DESCRIPTION:
{user_prompt}

INSTRUCTIONS:
1. Extract proposed building dimensions (height, plot depth, road width, setbacks) if present.
2. Compare proposed dimensions against legal rules in the context blocks.
3. Determine if each rule is "PASSED" or "VIOLATION".
4. Provide a numerical justification explaining the delta.
5. Provide exact legal citations (Document name, Clause/Table, Page number) from the context.
6. Return your output STRICTLY as valid JSON with NO commentary outside the JSON block.

JSON SCHEMA FORMAT:
{{
  "summary": "Brief summary of the prompt/proposal evaluated",
  "overall_verdict": "APPROVED" or "REJECTED_NON_COMPLIANT" or "INFORMATIVE_RESPONSE",
  "audit_checks": [
     {{
       "provision": "Rule name or Clause",
       "status": "PASSED" or "VIOLATION" or "INFO",
       "justification": "Detailed breakdown",
       "citation": "Document name, Clause/Table, Page"
     }}
  ]
}}
"""

        # 5. Call Local Ollama (Llama 3.1)
        payload = {
            "model": self.model_name,
            "prompt": system_prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.0}
        }

        try:
            response = requests.post(self.ollama_url, json=payload, timeout=240)
            response_json = response.json()
            llm_result = json.loads(response_json.get("response", "{}"))
        except Exception as e:
            llm_result = {
                "summary": "Error calling local Llama server.",
                "overall_verdict": "ERROR",
                "audit_checks": [{
                    "provision": "Ollama Service Check",
                    "status": "ERROR",
                    "justification": f"Make sure 'ollama run llama3.1' is running. Error: {str(e)}",
                    "citation": "N/A"
                }]
            }

        # Return combined payload with retrieved chunks for UI display
        return {
            "status": "SUCCESS",
            "llm_output": llm_result,
            "retrieved_chunks": [
                {
                    "chunk_id": c["chunk_id"],
                    "source": c["metadata"].get("source_doc", "BDA/BBMP Bye-Laws"),
                    "page": c["metadata"].get("page_number", "N/A"),
                    "score": round(c.get("score", 0.0), 4),
                    "text": c["text"]
                }
                for c in retrieved_chunks
            ]
        }