# Building Plan Compliance Pre-Check Service

**Course:** ML System Design and AgentOps (UE24AM342AA5)  
**Mini Project:** 05 — Building Plan Compliance Pre-Check Service

## Overview

This project is a retrieval-augmented search prototype for selected Bengaluru building regulations. It retrieves passages that may be relevant to a building-related question and returns the passage text, source metadata, and relevance score.

The current version focuses on document retrieval. It is **not a complete building-plan compliance system** and does not issue an authoritative compliance approval. Any real building proposal must be reviewed by a qualified professional and the relevant authority.

## Corpus

The repository includes these source documents:

- `corpus/BBMP_Building_Byelaws_2003.pdf`
- `corpus/BDA_RMP_2015_Zonal_Regulations.pdf`

Selected passages are stored in `data/extracted_passages/`. This is a limited extract, not a complete index of every provision in either document.

## Features implemented

- Text extraction from selected PDF pages using `pdfplumber`
- Splitting extracted Markdown into chunks with source and page metadata
- Local embeddings using `all-MiniLM-L6-v2`
- Persistent vector storage using ChromaDB
- Dense vector retrieval
- Metadata-filtered retrieval for table chunks
- Hybrid retrieval combining dense similarity with BM25 keyword scores
- FastAPI endpoints for health checking and querying
- Scripts for retrieval evaluation, heuristic grounding checks, and synthetic retrieval stress tests
- A GitHub Actions workflow for running the quality-gate script

## Tech stack

- Python 3.11
- FastAPI and Uvicorn
- ChromaDB
- Sentence Transformers (`all-MiniLM-L6-v2`)
- BM25 (`rank-bm25`)
- pdfplumber / pypdf
- pandas
- GitHub Actions

## Repository structure

```text
.
├── app.py
├── ci_quality_gate.py
├── Dockerfile
├── requirements.txt
├── corpus/
│   ├── BBMP_Building_Byelaws_2003.pdf
│   └── BDA_RMP_2015_Zonal_Regulations.pdf
├── data/
│   ├── extracted_passages/
│   ├── processed_chunks.json
│   ├── eval_set.json
│   ├── synthetic_eval_set.json
│   └── *_results.csv
├── pipelines/
│   ├── chunker.py
│   ├── indexer.py
│   └── retriever.py
├── scripts/
│   ├── extract_passages.py
│   └── build_eval_dataset.py
├── evaluation/
│   ├── evaluate_retrieval.py
│   ├── evaluate_generation.py
│   ├── generate_synthetic_data.py
│   └── stress_test_runner.py
└── .github/workflows/
    └── rag_eval_ci.yml
```

## Setup

Use Python 3.11. Run commands from the repository root (the folder containing `app.py`).

### 1. Create and activate a virtual environment

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**macOS / Linux**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install dependencies

The current `requirements.txt` is empty, so install the packages used by the application and evaluation scripts explicitly:

```bash
python -m pip install fastapi "uvicorn[standard]" chromadb sentence-transformers rank-bm25 pandas pypdf pdfplumber tabulate
```

The first use of Sentence Transformers may download the `all-MiniLM-L6-v2` model.

### 3. Run the API

The repository includes a local ChromaDB store and processed chunks. From the repository root, run:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open the interactive API documentation at:

- `http://127.0.0.1:8000/docs`
- Health endpoint: `http://127.0.0.1:8000/health`

If the vector collection or processed-chunk file is missing, rebuild the chunks and index as described below before starting the API.

## API usage

### Health check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "service": "bda-bbmp-rag-eval"
}
```

### Query the regulations

```http
POST /api/v1/query
Content-Type: application/json
```

Example request:

```json
{
  "query": "What setback rules may apply to a residential building on a narrow road?",
  "top_k": 3,
  "strategy": "hybrid"
}
```

Supported `strategy` values:

| Strategy | Description |
|---|---|
| `dense` | Retrieves passages using vector similarity |
| `metadata` | Searches table-type chunks using a metadata filter |
| `hybrid` | Combines vector similarity and BM25 keyword scores |

Example PowerShell request:

```powershell
$body = @{
  query = "What is the front setback for a 10m height building?"
  top_k = 3
  strategy = "hybrid"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/query" -Method Post -ContentType "application/json" -Body $body
```

The response includes the original query, strategy used, and retrieved chunks with chunk IDs, text, metadata, and scores.

## Rebuild the retrieval data

Run these commands from the repository root.

### 1. Extract the selected passages

```bash
python scripts/extract_passages.py
```

This extracts the selected pages defined by the script into Markdown files under `data/extracted_passages/`.

### 2. Create chunks

```bash
python pipelines/chunker.py
```

This creates `data/processed_chunks.json`.

### 3. Build/update the ChromaDB index

```bash
python pipelines/indexer.py
```

This stores the chunks in the local `chroma_db/` directory. The embedding model may be downloaded on first use.

## Run the evaluation scripts

### Retrieval evaluation

```bash
python evaluation/evaluate_retrieval.py
```

Writes `data/retrieval_eval_results.csv`.

### Generation/grounding checks

```bash
python evaluation/evaluate_generation.py
```

Writes `data/generation_eval_results.csv`. Despite the script name, the current implementation does not call a generative language model; it applies simple checks to retrieved context and records the stored ground-truth reasoning.

### Generate synthetic cases

```bash
python evaluation/generate_synthetic_data.py
```

Writes or updates `data/synthetic_eval_set.json`.

### Synthetic stress test

```bash
python evaluation/stress_test_runner.py
```

Writes `data/stress_test_results.csv`.

### Quality gate

```bash
python ci_quality_gate.py
```

The GitHub Actions workflow installs dependencies and runs this script on pushes and pull requests to `main` or `master`.

## Evaluation notes

The saved retrieval evaluation uses three manually written cases, while the synthetic stress-test file contains 25 cases. The evaluation scripts use broad keyword-based checks. As a result, a reported `PASS` or `100% Recall@3` should be interpreted as a preliminary script result, not proof that the correct legal clauses were retrieved or that a building proposal is compliant.

For a stronger evaluation, the expected evidence should be checked manually against the source PDFs, expected chunk/clause identifiers should be exact, and the test set should include missing-corpus and conflicting-rule cases.

## Current limitations

- The corpus contains selected passages rather than every applicable regulation.
- The API retrieves evidence but does not make a verified compliance determination.
- The current code does not demonstrate the required multi-agent workflow, self-verification loop, or escalation path.
- The evaluation set is small and the checks are heuristic.
- The submitted repository does not include a multi-service Docker Compose setup, asynchronous job queue, monitoring dashboard, or measured load-test report.
- `requirements.txt` is currently empty; dependencies are listed in the setup instructions above.

## Safety and intended use

This project is for coursework and experimentation. Retrieved text can be incomplete or irrelevant, and the system must not be used as a substitute for the complete applicable regulations, professional review, or municipal approval.

## Authors

KR Tanishkaa
Khushi Paraddi
