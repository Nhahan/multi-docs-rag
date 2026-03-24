# AGENTS.md

## Project Purpose
This repository is a local multi-document RAG MVP built with Next.js + TypeScript + Node.js.
Primary corpus for local testing:
- `data/CFR - Code of Federal Regulations.pdf`
- `data/Agent_Memory_Below_the_Prompt.pdf`

These two files are fixture inputs only (legacy test corpus). The same runtime is intended to work with any local PDF corpus configured in `CORPUS_SOURCE_FILES`/`CORPUS_CONFIG_JSON`.

## Non-negotiable stack
- Next.js App Router
- TypeScript
- Node.js runtime
- LangChain.js for document loading/splitting/vector/search/build blocks
- LangGraph.js for workflow orchestration
- Local LLMs via Ollama-compatible runtime
  - Chat: `OLLAMA_CHAT_MODEL` (default `unsloth_Qwen3.5-9B-UD-Q4_K_XL:latest`)
  - Embedding: `OLLAMA_EMBEDDING_MODEL` (default `qwen3-embedding:4b`)
  - Rerank: `OLLAMA_RERANK_MODEL` (default `qwen3-embedding:4b`)

## Runtime boundaries
- All ingestion, chunking, embedding, retrieval, reranking, and generation must run on the server.
- No PDF parsing/embedding/retrieval should run in client components.
- No external cloud LLM APIs.

## Directory layout (expected)
- `app/` : Next.js routes and API handlers
- `src/ingestion/` : PDF parsing and page extraction
- `src/chunking/` : document-aware chunking strategy
- `src/retrieval/` : dense, lexical, fusion, rerank logic
- `src/graph/` : LangGraph workflow
- `src/llm/` or `src/lib/` : model and config helpers
- `data/` : PDFs and local indexes
- `scripts/` : ingest/smoke scripts, automation helpers

## Metadata requirements
Every stored chunk must carry at least:
- `source_file`
- `document_id`
- `page`
- `chunk_id`
- `section_title`

## Quality and behavior rules
- Retrieval must be hybrid: dense + lexical, with fusion.
- Generate answers only from retrieved evidence and include citations.
- Cite format should be deterministic and page-based (for example `[<source label> p.12]`).
- If evidence is insufficient, answer must explicitly state uncertainty.
- Retrieval and generation must stay document-agnostic. Do not branch on corpus family labels.

## Hard requirements: no hardcoding / no plugin / no domain heuristics
- Do not add hardcoded domain-specific behavior in shared pipeline modules (for example, legal/finance/control heuristics or fixed section maps).
- Do not add plugin/profile systems for core behavior switching by corpus type. There is one default, uniform path for all documents.
- Do not use domain-specific heuristics in retrieval/query rewriting/ranking/classification/evidence checks. Use general retrieval and metadata-aware logic only.
- Do not tie routing, retrieval, or citations to document titles, document names, or fixed domain enums.
- Fixture corpus labels may appear in smoke scripts/docs and seed/test artifacts only. Core runtime files under `src/` must remain document-agnostic.

## Key scripts
- `npm install` -> dependency install
- `npm run dev` -> run Next.js
- `npm run ingest` -> ingest local PDFs
- `npm run smoke` -> smoke queries

## Development principles for agents
- Prefer explicit, readable, and modular TypeScript.
- Keep changes minimal and aligned with the MVP goal.
- Do not add new dependencies without necessity.
- Document tradeoffs when deviating from requested defaults.
- Prioritize reliability of retrieval/generation over complex orchestration.

## Environment defaults
Use `.env.example` values and allow override:
- `OLLAMA_BASE_URL`
- `OLLAMA_CHAT_MODEL`
- `OLLAMA_EMBEDDING_MODEL`
- `OLLAMA_RERANK_MODEL`
- `OLLAMA_CHAT_THINK`
- `OLLAMA_CHAT_TEMPERATURE`
- `OLLAMA_REQUEST_TIMEOUT_MS`
- `INGEST_EMBED_BATCH_SIZE`
- `INGEST_CHUNK_SIZE`
- `INGEST_CHUNK_OVERLAP`
- retrieval tuning env vars used by retrieval module:
  - `RETRIEVAL_TOP_K`
  - `RETRIEVAL_CANDIDATE_K`
- `RETRIEVAL_ALLOW_DENSE_FALLBACK`
  - `RETRIEVAL_ENABLE_RERANK`
  - `RETRIEVAL_RERANK_TOP_K`
- `CORPUS_SOURCE_FILES` (preferred)
- `CORPUS_CONFIG_JSON` (full explicit config, highest priority)
- `CORPUS_PATH`
- `CHUNKS_PATH`
- `VECTOR_STORE_PATH`
- `LEXICAL_INDEX_PATH`
