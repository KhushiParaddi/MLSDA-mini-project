import sys
import os
import json
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipelines.retriever import RegulatoryRetriever

def run_stress_testing():
    synthetic_path = "data/synthetic_eval_set.json"
    output_report = "data/stress_test_results.csv"

    if not os.path.exists(synthetic_path):
        raise FileNotFoundError(f"Missing synthetic dataset at {synthetic_path}. Run generate_synthetic_data.py first.")

    with open(synthetic_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_cases = data["test_cases"]
    retriever = RegulatoryRetriever()
    results = []

    print(f"Executing stress tests across {len(test_cases)} synthetic boundary cases...\n")

    for tc in test_cases:
        query = tc["query"]
        expected_verdict = tc["ground_truth"]["verdict"]

        # Run Hybrid Search
        retrieved = retriever.hybrid_search(query, top_k=2)
        combined_text = " ".join([r["text"] for r in retrieved]).lower()

        # Check retrieval completeness
        has_table_context = "table" in combined_text or "setback" in combined_text
        has_clause_context = "road" in combined_text or "4.1.2" in combined_text or "height" in combined_text

        retrieval_success = has_table_context and has_clause_context

        results.append({
            "ID": tc["id"],
            "Query": query[:60] + "...",
            "Expected Verdict": expected_verdict,
            "Dense/Hybrid Retrieval": "PASS" if retrieval_success else "PARTIAL/FAIL",
            "Top Match Chunk": retrieved[0]["chunk_id"] if retrieved else "None"
        })

    df = pd.DataFrame(results)
    df.to_csv(output_report, index=False)
    
    pass_rate = (df["Dense/Hybrid Retrieval"] == "PASS").mean() * 100
    print("=" * 80)
    print(f"STRESS TEST SUMMARY RESULTS (Pass Rate: {pass_rate:.1f}%)")
    print("=" * 80)
    print(df.head(10).to_string(index=False))
    print(f"\nFull stress test report saved to: {output_report}\n")

if __name__ == "__main__":
    run_stress_testing()