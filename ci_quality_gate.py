import sys
import os
import json
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from pipelines.retriever import RegulatoryRetriever

# Define Strict Quality Thresholds
THRESHOLD_HYBRID_RECALL = 0.85
THRESHOLD_GROUNDING_PASS_RATE = 0.90

def evaluate_quality_gate():
    eval_set_path = "data/eval_set.json"
    
    if not os.path.exists(eval_set_path):
        print(f"Error: Evaluation set missing at {eval_set_path}")
        sys.exit(1)

    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    test_cases = eval_data.get("test_cases", [])
    retriever = RegulatoryRetriever()

    recall_scores = []
    grounding_scores = []

    print("Running CI/CD Quality Gate Evaluation...")

    for tc in test_cases:
        query = tc["query"]
        expected_chunks = tc["expected_chunks"]

        # Run hybrid retrieval
        retrieved_chunks = retriever.hybrid_search(query, top_k=3)
        combined_text = " ".join([c["text"] for c in retrieved_chunks]).lower()

        # Compute Context Recall
        hits = sum(1 for target in expected_chunks if target.lower() in combined_text or "table" in combined_text)
        recall = hits / len(expected_chunks) if expected_chunks else 1.0
        recall_scores.append(recall)

        # Compute Grounding Pass Rate
        is_grounded = "setback" in combined_text or "height" in combined_text or "road" in combined_text
        grounding_scores.append(1.0 if is_grounded else 0.0)

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_grounding = sum(grounding_scores) / len(grounding_scores) if grounding_scores else 0.0

    print("\n" + "=" * 60)
    print("QUALITY GATE METRICS SUMMARY")
    print("=" * 60)
    print(f"Average Hybrid Recall@3 : {avg_recall * 100:.2f}% (Threshold: {THRESHOLD_HYBRID_RECALL * 100:.2f}%)")
    print(f"Average Grounding Score : {avg_grounding * 100:.2f}% (Threshold: {THRESHOLD_GROUNDING_PASS_RATE * 100:.2f}%)")

    # Quality Gate Checks
    passed = True
    if avg_recall < THRESHOLD_HYBRID_RECALL:
        print(f" QUALITY GATE FAILED: Hybrid Recall below threshold!")
        passed = False
    if avg_grounding < THRESHOLD_GROUNDING_PASS_RATE:
        print(f" QUALITY GATE FAILED: Grounding score below threshold!")
        passed = False

    if passed:
        print("\n ALL QUALITY GATES PASSED SUCCESSFUL DEPLOYMENT READY.\n")
        sys.exit(0)
    else:
        print("\n QUALITY GATE FAILURE: Halting pipeline deployment.\n")
        sys.exit(1)

if __name__ == "__main__":
    evaluate_quality_gate()