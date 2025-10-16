# Rag-Financial (advanced_rag)

This repository contains `advanced_rag.py` — a refactored, non-Colab version of a Retrieval-Augmented Generation (RAG) script originally developed in Colab.

## Quick start

1. Create a virtual environment and activate it (macOS / zsh):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Create a `.env` file from the example and fill your keys (optional but convenient):

```bash
cp .env.example .env
# edit .env and paste your keys
```

3. Run the script (it expects PDF files in `./data`):

```bash
python advanced_rag.py
```

## Environment variables

- `OPENAI_API_KEY` (required)
- `LLAMA_CLOUD_API_KEY` (required)

The script optionally reads `.env` if you install `python-dotenv` (not required). The file `.env.example` is included as a template. DO NOT commit real `.env` files.

## Production readiness checklist (short)

- Persistent vector store (FAISS/Chroma/Weaviate) with on-disk persistence and versioning — ETA: 1-2 days
- Better ingestion pipeline: normalization, deduplication, metadata enrichment, error retries — ETA: 1-2 days
- Config-driven parameters (via config file or env) for chunk sizes, retriever top_k, reranker model — ETA: 0.5 day
- CI that runs unit tests and lint on push — ETA: 0.5 day
- Integration tests (optional) that run against real LLM/embedding providers — ETA: 1-2 days (requires API creds for CI)
- Monitoring/telemetry for query latency, errors, and RAG hallucination metrics — ETA: 2-3 days

If you want I can implement the above items incrementally. Tell me which you'd like next and I'll create a plan and PR-style edits.

## Chroma (optional)

- This project includes an optional Chroma-backed index. Install `chromadb` (already in requirements) to enable it. The code falls back to the in-memory VectorStoreIndex if Chroma isn't available.
- Use `build_chroma_index(documents)` to create or use the Chroma-backed retriever.

## CI

- A basic GitHub Actions workflow is included at `.github/workflows/ci.yml` that installs dependencies and runs tests on push/PR. If you want linting (ruff/flake8) or coverage, I can add them.
