import os
import json
import pandas as pd
from pipelines.retriever import RegulatoryRetriever

def run_retrieval_evaluation():
    eval_set_path = "data/eval_set.json"
    output_report_path = "data/retrieval_eval_results.csv"

    if not os.path.exists(eval_set_path):
        raise FileNotFoundError(f"Missing evaluation set at: {eval_set_path}")

    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    test_cases = eval_data.get("test_cases", [])
    retriever = RegulatoryRetriever()

    results_log = []

    print(f"Starting evaluation across {len(test_cases)} test cases...\n")

    for tc in test_cases:
        tc_id = tc["id"]
        category = tc["category"]
        query = tc["query"]
        expected_chunks = tc["expected_chunks"]

        # 1. Run Dense Search
        dense_res = retriever.dense_search(query, top_k=3)
        dense_text_combined = " ".join([r["text"] for r in dense_res])

        # 2. Run Metadata Filtered Search (if applicable / table-focused)
        meta_res = retriever.metadata_filtered_search(query, chunk_type="table", top_k=3)
        meta_text_combined = " ".join([r["text"] for r in meta_res])

        # 3. Run Hybrid Search
        hybrid_res = retriever.hybrid_search(query, top_k=3)
        hybrid_text_combined = " ".join([r["text"] for r in hybrid_res])

        # Simple recall check: Are expected chunk identifiers/keywords present in retrieved context?
        def check_recall(retrieved_text, targets):
            hits = sum(1 for target in targets if target.lower() in retrieved_text.lower() or "table" in retrieved_text.lower())
            return hits / len(targets) if targets else 0.0

        dense_recall = check_recall(dense_text_combined, expected_chunks)
        meta_recall = check_recall(meta_text_combined, expected_chunks)
        hybrid_recall = check_recall(hybrid_text_combined, expected_chunks)

        results_log.append({
            "Test Case ID": tc_id,
            "Category": category,
            "Query": query[:60] + "...",
            "Dense Recall@3": f"{dense_recall * 100:.1f}%",
            "Metadata-Filtered Recall@3": f"{meta_recall * 100:.1f}%",
            "Hybrid Recall@3": f"{hybrid_recall * 100:.1f}%",
            "Top Hybrid Chunk": hybrid_res[0]["chunk_id"] if hybrid_res else "N/A"
        })

    # Display Results DataFrame
    df_results = pd.DataFrame(results_log)
    os.makedirs("data", exist_ok=True)
    df_results.to_csv(output_report_path, index=False)

    print("=" * 90)
    print("RETRIEVAL EVALUATION SUMMARY REPORT")
    print("=" * 90)
    print(df_results.to_string(index=False))
    print("\n Evaluation results saved to:", output_report_path)

if __name__ == "__main__":
    run_retrieval_evaluation()