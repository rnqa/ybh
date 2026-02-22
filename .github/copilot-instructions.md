# Copilot instructions for HelperYoristBot

This file gives actionable, project-specific guidance for AI coding agents working in this repository.

1) Big-picture architecture
- Purpose: a RAG-style legal assistant pipeline that ingests legal texts (JSONL), creates semantic/embedding indexes (Chroma, FAISS, Qdrant), and serves retrieval + LLM reasoning (local LM endpoints or LM Studio).
- Major components:
  - Data sources: `Laws/`, `NPA3001/`, other `*.jsonl` files in repo root.
  - Indexing pipelines: `build_all.py` (Chroma DB ./db/ALL), `build_index.py` (Chroma collection), `indexFAISS.py` (FAISS index in `faiss_indexes/`).
  - Semantic builder: `rag_builder.py` (Qdrant-based semantic chunks + SentenceTransformer embedding generation).
  - Serving / RAG usage: `rag_local.py` (telegram bot + Chroma retriever + local LLM), `model.py` (ad-hoc query against Chroma + LM Studio endpoint).

2) Critical workflows and commands
- Create Chroma DB for bot usage (default output `./db/ALL`):
  ```powershell
  python build_all.py
  ```
- Build a named Chroma collection (adjust `DATA_PATH` and `CHROMA_DB_PATH` in `build_index.py` or use env vars):
  ```powershell
  python build_index.py
  ```
- Build FAISS index from JSONL folder:
  ```powershell
  python indexFAISS.py
  ```
- Rebuild semantic Qdrant DB (semantic chunks + embeddings):
  ```powershell
  python -m rag_builder
  ```
  Note: `rag_builder.py` expects Qdrant client with a local `path` (default `./semantic_laws_db`).
- Run the Telegram assistant (requires local LM endpoint at `http://127.0.0.1:1234/v1` by default):
  ```powershell
  python rag_local.py
  ```

3) Project-specific conventions and patterns
- JSONL schema expected by scripts (fields commonly read): `text`, `chunk_id`, `hierarchy`, `hierarchy_str`, `law_id`, `source_title`, `source_url`, `id`/`original_id`.
- Dedup / id handling: `build_index.py` appends `_1`, `_2` to duplicate `chunk_id` values instead of discarding — preserve `chunk_id` semantics when changing ingestion.
- Chunking: `rag_builder.py` and `build_all.py` create chunks differently: `rag_builder.py` creates semantic chunks via sentence grouping and heuristics; `build_all.py` treats top-level text chunks directly. Changes to chunking affect downstream embedding shape and collection sizes.
- Multiple vector backends are used interchangeably: Chroma (`./db/ALL` or `chroma_db`), FAISS (`faiss_indexes/*.faiss`), Qdrant (`semantic_laws_db`). Keep metadata consistent across backends.

4) Integration points and external dependencies
- Local LLM endpoints observed in code:
  - `rag_local.py` expects `http://127.0.0.1:1234/v1` (via `OpenAI`-compatible client configuration).
  - `model.py` shows an LM Studio URL `http://10.8.1.33:1234/v1` — environment-specific.
  Ensure a compatible inference API is running and configure endpoints via files or environment variables before running services.
- Key Python packages used (install before editing runtime code):
  - `chromadb`, `sentence-transformers`, `qdrant-client`, `langchain`/`langchain_community`, `aiogram`, `faiss-cpu` or `faiss`, `numpy`, `requests`, `beautifulsoup4`.

5) Safety and operational notes
- Secrets: `config.env` exists in repo root and currently contains tokens. Do not commit real secrets to the repo; prefer local `.env` or secret store. When running locally, source `config.env` or set `TELEGRAM_TOKEN`, `CHROMA_PATH`, `COLLECTION_NAME` in your environment.
- Paths: several scripts use hardcoded Windows paths (e.g. `CHROMA_DB_PATH` in `build_index.py` and `laws_folder` in `rag_builder.py`). Prefer to set environment variables or update script constants when running in another environment.

6) Useful code locations to inspect when changing behavior
- RAG / semantic pipeline: `rag_builder.py`, `build_all.py`, `build_index.py` — modify chunking or embedding model here.
- Serving and prompts: `rag_local.py` and `model.py` — LLM invocation, prompt templates, reranker (`CrossEncoder`) usage.
- Index formats and storage: `indexFAISS.py`, `faiss_indexes/`, `db/`, `chroma_db/` — metadata expectations.

7) Quick examples for edits
- To change embedding model used for all pipelines, update `EMBEDDING_MODEL` in `build_all.py`, `EMBEDDING_MODEL_NAME` in `indexFAISS.py`, and the model name in `rag_builder.py`.
- To change the LM endpoint for `rag_local.py`, edit the `base_url` in `_init_llm()` or use a configuration wrapper to read an env var.

If any section is unclear or you want me to expand a particular area (e.g. exact env var names to standardize, or generate a `requirements.txt`), tell me which part to iterate on.
