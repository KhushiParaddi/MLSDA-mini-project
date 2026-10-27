import os
import json
import chromadb
from chromadb.utils import embedding_functions

def index_chunks_to_chroma():
    chunks_file = "data/processed_chunks.json"
    chroma_db_dir = "chroma_db"
    collection_name = "bda_bbmp_bye_laws"

    if not os.path.exists(chunks_file):
        raise FileNotFoundError(f"Missing processed chunks file at: {chunks_file}. Run pipelines/chunker.py first.")

    # 1. Load Chunks
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks from {chunks_file}.")

    # 2. Initialize Chroma Client (Persistent Storage on Disk)
    client = chromadb.PersistentClient(path=chroma_db_dir)

    # 3. Use SentenceTransformer embedding function
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Create or replace existing collection
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"}
    )

    # Prepare batch insertion payload
    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        ids.append(chunk["chunk_id"])
        documents.append(chunk["text"])
        
        # Ensure metadata values are non-null and valid types for ChromaDB
        meta = {
            "source_doc": str(chunk["metadata"].get("source_doc", "")),
            "page_number": int(chunk["metadata"]["page_number"]) if chunk["metadata"].get("page_number") is not None else -1,
            "chunk_type": str(chunk["metadata"].get("chunk_type", "")),
            "char_length": int(chunk["metadata"].get("char_length", 0))
        }
        metadatas.append(meta)

    # 4. Upsert into Vector Store
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    print(f" Successfully indexed {len(ids)} chunks into ChromaDB collection '{collection_name}'.")
    print(f" Database saved locally to directory: ./{chroma_db_dir}/")

    # 5. Quick Verification Retrieval Test
    test_query = "What is the front setback for a 10m height building?"
    results = collection.query(
        query_texts=[test_query],
        n_results=2
    )

    print("\n--- Verification Query Test ---")
    print(f"Query: '{test_query}'")
    for idx, doc in enumerate(results["documents"][0]):
        matched_id = results["ids"][0][idx]
        matched_meta = results["metadatas"][0][idx]
        print(f"\nResult #{idx + 1} [ID: {matched_id} | Doc: {matched_meta['source_doc']} | Page: {matched_meta['page_number']}]:")
        print(doc[:250] + "..." if len(doc) > 250 else doc)

if __name__ == "__main__":
    index_chunks_to_chroma()