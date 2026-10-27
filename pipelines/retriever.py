import os
import json
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

class RegulatoryRetriever:
    def __init__(self, chroma_dir="chroma_db", collection_name="bda_bbmp_bye_laws", chunks_file="data/processed_chunks.json"):
        # 1. Initialize Chroma Vector Store
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_collection(
            name=collection_name, 
            embedding_function=self.embedding_func
        )

        # 2. Initialize BM25 Index for Keyword Search
        with open(chunks_file, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        
        self.corpus = [chunk["text"] for chunk in self.chunks]
        tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def dense_search(self, query: str, top_k: int = 3):
        """Standard Vector Similarity Search"""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        return self._format_chroma_results(results)

    def metadata_filtered_search(self, query: str, chunk_type: str = "table", top_k: int = 3):
        """Vector Search filtered by Metadata (e.g., searching only tables)"""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"chunk_type": chunk_type}
        )
        return self._format_chroma_results(results)

    def hybrid_search(self, query: str, top_k: int = 3, alpha: float = 0.5):
        """
        Hybrid Search combining BM25 keyword matching with Dense Vector Similarity.
        Alpha balances Dense (1.0) vs BM25 (0.0).
        """
        # Get Dense scores from Chroma
        dense_res = self.collection.query(
            query_texts=[query],
            n_results=len(self.chunks)
        )
        
        dense_ids = dense_res["ids"][0]
        dense_distances = dense_res["distances"][0]
        # Convert cosine distance to similarity score
        dense_scores = {cid: (1.0 - dist) for cid, dist in zip(dense_ids, dense_distances)}

        # Get BM25 scores
        tokenized_query = query.lower().split()
        bm25_scores_raw = self.bm25.get_scores(tokenized_query)
        max_bm25 = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1.0
        
        combined_results = []
        for idx, chunk in enumerate(self.chunks):
            cid = chunk["chunk_id"]
            d_score = dense_scores.get(cid, 0.0)
            b_score = bm25_scores_raw[idx] / max_bm25  # Normalize BM25
            
            hybrid_score = (alpha * d_score) + ((1 - alpha) * b_score)
            combined_results.append({
                "chunk_id": cid,
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "score": hybrid_score
            })

        # Sort by hybrid score descending
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        return combined_results[:top_k]

    def _format_chroma_results(self, results):
        formatted = []
        for idx in range(len(results["ids"][0])):
            formatted.append({
                "chunk_id": results["ids"][0][idx],
                "text": results["documents"][0][idx],
                "metadata": results["metadatas"][0][idx],
                "score": 1.0 - results["distances"][0][idx] if "distances" in results else 1.0
            })
        return formatted


if __name__ == "__main__":
    retriever = RegulatoryRetriever()

    test_query = "What is the front setback required for a 10m height building?"
    
    print("=" * 60)
    print(f"QUERY: '{test_query}'")
    print("=" * 60)

    print("\n--- 1. Dense Semantic Search ---")
    for res in retriever.dense_search(test_query, top_k=2):
        print(f"[{res['chunk_id']}] (Score: {res['score']:.4f})\n{res['text'][:150]}...\n")

    print("\n--- 2. Metadata Filtered Search (chunk_type='table') ---")
    for res in retriever.metadata_filtered_search(test_query, chunk_type="table", top_k=2):
        print(f"[{res['chunk_id']}] Type: {res['metadata']['chunk_type']}\n{res['text'][:150]}...\n")

    print("\n--- 3. Hybrid Search (Dense + BM25) ---")
    for res in retriever.hybrid_search(test_query, top_k=2):
        print(f"[{res['chunk_id']}] (Hybrid Score: {res['score']:.4f})\n{res['text'][:150]}...\n")