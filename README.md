# multi-docs-rag

Local multi-document RAG built with Next.js, TypeScript, LangChain, LangGraph, and Ollama-compatible local models.

The project ingests local PDF corpora, builds local indexes, retrieves evidence with hybrid search, and generates grounded answers without external cloud LLM APIs.

## Highlights

- Server-side PDF ingestion, chunking, embedding, retrieval, reranking, and answer generation
- Hybrid retrieval with lexical and dense search
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
8. Verify grounding against retrieved evidence

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

The Next.js app is expected to run on `http://localhost:3333`.

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
- evidence availability summary
- document mix
- top retrieved chunks
- pipeline trace

Files under `scripts/manual-review/` are review inputs only.

- `cases.local.json`: fixture review set for the local test corpus
- `cases.template.json`: generic template for arbitrary corpora

These files are not runtime policy.

## Evaluation difficulty ladder

The project uses a three-level difficulty model when discussing evaluation quality:

### 1. Baseline

Used to validate basic retrieval and grounded answering.

- single-doc factual
- single-doc summary
- identifier-heavy retrieval

### 2. Intermediate

Used to validate generic multi-document behavior without overloading a single question.

- multi-doc synthesis
- unsupported handling
- multilingual query

### 3. Hard

Used to validate cross-document reasoning under realistic local-LLM constraints.

- cross-doc relation
- partial support plus partial unsupported
- each question should test only one or two core goals

### Active testing policy

In practice, this repository should use only one active `hard` test question for routine evaluation.

Reason:

- local LLMs have tighter reasoning and latency budgets
- overly dense test questions mix too many failure modes at once
- one professional-grade hard case is easier to interpret and maintain than a large set of unstable stress questions

Stress-style questions can still be useful for debugging, but they should be treated as manual investigation prompts rather than the main quality gate.

The active hard gate question is:

```text
Report the 8K agent counts for FP16 and Q4 from agent-memory-below-the-prompt. Then say whether the retrieved CFR evidence directly supports matching compliance controls for that system. If not, say the CFR part is not supported by the retrieved evidence.
```

This gate is judged by manual answer review, not by automatic scoring.

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
  "evidence": {
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

- `evidence` represents evidence availability, not calibrated semantic confidence
- `retrieval.chunks` is primarily for debugging and inspection
- final answers are plain grounded text by default

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

Chat model is fixed in code to `unsloth_Qwen3.5-9B-UD-Q4_K_XL`.
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
