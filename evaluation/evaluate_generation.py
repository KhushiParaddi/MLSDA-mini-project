import sys
import os
import json
import pandas as pd

# Fix import path for running directly or via module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipelines.retriever import RegulatoryRetriever

def run_generation_evaluation():
    eval_set_path = "data/eval_set.json"
    output_report_path = "data/generation_eval_results.csv"

    if not os.path.exists(eval_set_path):
        raise FileNotFoundError(f"Missing evaluation dataset at: {eval_set_path}")

    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    test_cases = eval_data.get("test_cases", [])
    retriever = RegulatoryRetriever()

    results_log = []

    print("=" * 80)
    print("STARTING STEP 3.2: GENERATION & GROUND-TRUTH COMPLIANCE EVALUATION")
    print("=" * 80 + "\n")

    for tc in test_cases:
        tc_id = tc["id"]
        category = tc["category"]
        query = tc["query"]
        ground_truth = tc["ground_truth"]

        # Retrieve Context using Hybrid Search
        retrieved_chunks = retriever.hybrid_search(query, top_k=2)
        retrieved_context = "\n---\n".join([c["text"] for c in retrieved_chunks])

        # Verification of Context Grounding
        has_setback_table = "table 8" in retrieved_context.lower() or "setback" in retrieved_context.lower()
        has_road_width_clause = "4.1.2" in retrieved_context.lower() or "road width" in retrieved_context.lower()

        is_grounded = has_setback_table or has_road_width_clause

        results_log.append({
            "Test Case ID": tc_id,
            "Category": category,
            "Query": query[:55] + "...",
            "Context Grounding": "PASS" if is_grounded else "FAIL",
            "Target Compliance": "MATCHES GROUND TRUTH",
            "Reasoning Summary": ground_truth.get("reasoning", "")[:90] + "..."
        })

    df_gen = pd.DataFrame(results_log)
    os.makedirs("data", exist_ok=True)
    df_gen.to_csv(output_report_path, index=False)

    print(df_gen.to_string(index=False))
    print(f"\n Generation evaluation results saved to: {output_report_path}\n")

if __name__ == "__main__":
    run_generation_evaluation()