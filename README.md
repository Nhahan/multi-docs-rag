# multi-docs-rag

Local multi-document RAG built with Next.js, TypeScript, LangChain, LangGraph, and Ollama-compatible local models.

The project is designed to ingest local PDF corpora, build local indexes, retrieve evidence with hybrid search, and generate citation-grounded answers without external cloud LLM APIs.

## Highlights

- Server-side PDF ingestion, chunking, embedding, retrieval, reranking, and answer generation
- Hybrid retrieval with lexical and dense search
- Page-based inline citations
- Grounded refusal when retrieved evidence is insufficient
- Debug-friendly API and UI surfaces for retrieval inspection
- Local-first runtime using Ollama-compatible models

## Scope

This repository is intended to work with arbitrary local PDF corpora.

The PDFs under `data/` are fixture inputs for local smoke and regression checks. They are not special runtime profiles and should not drive corpus-specific branching in shared pipeline code.

## Tech stack

- Next.js App Router
- TypeScript
- Node.js
- LangChain.js
- LangGraph.js
- Ollama-compatible local chat, embedding, and rerank models

## Architecture

End-to-end flow:

1. Load configured local PDFs
2. Extract page text and metadata
3. Chunk documents into retrieval units
4. Build local vector and lexical indexes
5. Retrieve evidence with hybrid search
6. Rerank retrieved chunks
7. Generate an answer from retrieved evidence only
8. Verify grounding against cited evidence
9. Return page-based citations

All of this runs on the server.

## Repository layout

- `app/`: Next.js routes and UI
- `app/api/ingest/route.ts`: ingest endpoint
- `app/api/query/route.ts`: query endpoint
- `src/ingestion/`: PDF loading and ingest pipeline
- `src/chunking/`: chunk construction and metadata extraction
- `src/retrieval/`: lexical, dense, fusion, and rerank logic
- `src/graph/`: workflow orchestration nodes
- `src/prompts/`: generation prompts
- `src/citations/`: citation extraction and formatting
- `src/synthesis/`: cross-document synthesis helpers
- `src/lib/`: config, model, and local store helpers
- `data/`: source PDFs and generated indexes
- `scripts/`: ingest, smoke, and support scripts

## Chunk metadata

Every stored chunk includes:

- `source_file`
- `document_id`
- `page`
- `chunk_id`
- `section_title`

## Getting started

Install dependencies:

```bash
npm install
```

Copy environment variables if needed:

```bash
cp .env.example .env
```

Start the app:

```bash
npm run dev
```

## Ingest local documents

Build the local indexes:

```bash
npm run ingest
```

Generated index files are written under `data/index/`.

## Run smoke checks

Fixture smoke scripts:

```bash
npm run smoke
npm run smoke:primary
npm run smoke:secondary
npm run smoke:hard-multidoc
```

These are useful regression checks, but they are not a substitute for manual semantic review on hard multi-document questions.

## Manual review harness

For manual semantic review, use:

```bash
npm run review:manual
```

Optional filters:

```bash
npm run review:manual -- --case hard-multidoc
npm run review:manual -- --file scripts/manual-review/cases.local.json
npm run review:manual -- --question "What does document-a say about ..." --id adhoc-1 --category adhoc
```

Generic template for other corpora:

```bash
cp scripts/manual-review/cases.template.json scripts/manual-review/my-cases.json
npm run review:manual -- --file scripts/manual-review/my-cases.json
```

This harness does not auto-pass or auto-fail cases. It prints:

- the full answer
- inline citations
- evidence availability summary
- document mix
- top retrieved chunks
- pipeline trace

Files under `scripts/manual-review/` are review inputs only.

- `cases.local.json`: fixture review set for the local test corpus
- `cases.template.json`: generic template for arbitrary corpora

These files are not runtime policy.

## Query API

### `POST /api/query`

Request:

```json
{
  "question": "What does the document say about ...?",
  "debug": true
}
```

Response shape:

```json
{
  "question": "...",
  "answer": "...",
  "citations": ["[document-a p.12]"],
  "citation_entries": ["[document-a p.12] ..."],
  "citation_footer": "Sources: ...",
  "has_citations": true,
  "quality": {
    "passed": true,
    "candidate_count": 6,
    "top_score": 0.12,
    "avg_score": 0.08,
    "reasons": []
  },
  "trace": [],
  "retrieval": {
    "reranked": true,
    "dense_enabled": true,
    "lexical_enabled": true,
    "degradation_reasons": [],
    "chunks": []
  }
}
```

Notes:

- `quality` currently represents evidence availability, not calibrated semantic confidence
- `retrieval.chunks` is primarily for debugging and inspection
- citations are deterministic and page-based

### `POST /api/ingest`

Triggers corpus ingestion and index rebuild.

## Debug mode

The UI exposes a debug view with:

- retrieved chunks
- chunk scores
- document-level retrieval distribution
- pipeline trace entries

`Document mix` summarizes how many final retrieved chunks came from each document and the best score observed for that document. It is useful for spotting single-document collapse in multi-document queries.

## Configuration

Key environment variables from `.env.example`:

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
- `CORPUS_SOURCE_FILES`
- `CORPUS_CONFIG_JSON`
- `CORPUS_PATH`
- `CHUNKS_PATH`
- `VECTOR_STORE_PATH`
- `LEXICAL_INDEX_PATH`
- `RETRIEVAL_TOP_K`
- `RETRIEVAL_CANDIDATE_K`
- `RETRIEVAL_ALLOW_DENSE_FALLBACK`
- `RETRIEVAL_ENABLE_RERANK`
- `RETRIEVAL_RERANK_TOP_K`
- `OLLAMA_INSUFFICIENT_EVIDENCE_MESSAGE`

### Corpus configuration

Simple configuration:

```env
CORPUS_SOURCE_FILES=document-a.pdf,document-b.pdf
```

Explicit configuration with stable document ids:

```env
CORPUS_CONFIG_JSON='[
  {"source_file":"document-a.pdf","document_id":"document-a"},
  {"source_file":"document-b.pdf","document_id":"document-b"}
]'
```

`CORPUS_CONFIG_JSON` takes precedence over `CORPUS_SOURCE_FILES`.

## Current quality bar

This repository should be treated as a generic local RAG prototype, not a finished production retrieval core.

Strengths:

- local multi-document ingestion
- server-side hybrid retrieval
- citation-grounded answers
- grounded refusal when evidence is insufficient
- useful debug visibility for retrieval behavior

Areas that still need careful manual review:

- hard cross-document synthesis
- identifier-heavy or regulation-heavy retrieval
- complex metric-to-control mapping across documents

## Development principles

- Keep server responsibilities on the server
- Avoid domain-specific hardcoding in shared runtime modules
- Prefer generic retrieval and metadata-aware logic over corpus-specific branching
- Keep changes explicit, minimal, and debuggable
