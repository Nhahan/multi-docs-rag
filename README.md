# Multi-Docs RAG (Next.js + LangChain.js + LangGraph.js, local-only)

Experimental local multi-document RAG for local PDFs.

Default repository data includes:
- `CFR - Code of Federal Regulations.pdf`
- `Agent_Memory_Below_the_Prompt.pdf`

These two PDFs are fixture inputs only. The runtime is designed to operate on any local PDF corpus configured in `CORPUS_SOURCE_FILES` / `CORPUS_CONFIG_JSON`.

The project is intentionally small and practical: local PDF ingestion, document-aware chunking, hybrid retrieval, LangGraph orchestration, and citation-grounded answers in a Next.js UI.

## Architecture

1. **Ingestion pipeline**
   - Parse local PDFs page-by-page (`pdfjs-dist`).
   - Apply generic, document-agnostic chunking:
     - paragraph boundary splitting + deterministic text splitting.
     - No domain-specific split rules are baked in.
   - Save chunk metadata for each chunk:
     - `source_file`
     - `document_id`
     - `page`
     - `chunk_id`
     - `section_title`
2. **Storage/indexing**
   - Dense vectors are produced with local Ollama embeddings and stored in `data/index/vector-store.json`.
   - Lexical index (tf-idf-like BM25-style term score implementation) in `data/index/lexical-index.json`.
3. **Retrieval (`src/retrieval/hybrid.ts`)**
   - Dense retrieval from local vector store.
   - Lexical retrieval from local term index.
   - Reciprocal-style fusion by weighted combination.
   - Optional reranking using local embedding model again (local fallback rerank).
4. **LangGraph workflow (`src/graph`)**
  - `retrieve` → `evidence_gate` → `generate` → `verify`.
  - This is intentionally structured as an explicit graph pipeline: retrieval, evidence availability check, grounded generation, and verification before final answer emission.
   - Produces grounded answer + computed citation list.
5. **Next.js UI (`app/page.tsx`)**
   - Query form, ingest button, answer rendering, citation list.
   - Optional debug view of retrieved chunks and scores.

## Why these choices

- **Next.js + Node runtime**
  - Fast iteration for a local MVP.
  - Simple API routes to keep heavy PDF parsing + embedding + retrieval on the server.
- **LangChain.js**
  - Provides stable building blocks for model and text splitting integration.
- **LangGraph.js**
  - Keeps the retrieval/classification/generation pipeline explicit and inspectable.
  - Avoids ad-hoc orchestration logic and makes routing behavior visible.
- **Local runtime via Ollama**
  - Ensures everything works without cloud model APIs.
- **Qwen3.5 9B**
  - Strong local reasoning/depth for mixed multi-document QA.
- **Qwen3 Embedding 4B**
  - Compact enough for local embedding workload while providing stable vector quality.
  - Also reused as fallback reranker when no separate local reranker API is available.

## Defaults

Defaults are configured in `.env.example` and resolved in `src/lib/config.ts`:

- `OLLAMA_BASE_URL=http://localhost:11434`
- `OLLAMA_CHAT_MODEL=unsloth_Qwen3.5-9B-UD-Q4_K_XL:latest`
- `OLLAMA_EMBEDDING_MODEL=qwen3-embedding:4b`
- `OLLAMA_RERANK_MODEL=qwen3-embedding:4b`
- `OLLAMA_CHAT_THINK=false`
- `OLLAMA_CHAT_TEMPERATURE=0`
- `OLLAMA_REQUEST_TIMEOUT_MS=120000`
- `CORPUS_SOURCE_FILES` (optional, comma/semicolon/newline list)
- `CORPUS_CONFIG_JSON` (optional, explicit per-doc config array)
- `CORPUS_PATH` (default `data`)
- `RETRIEVAL_ALLOW_DENSE_FALLBACK=true`
- Cross-document controls:
  - `CROSS_DOC_MIN_FRACTION=0.2`
  - `CROSS_DOC_MIN_CHUNKS=1`
  - `CROSS_DOC_MAX_BROAD_QUERIES=24`
  - `CROSS_DOC_MAX_PER_DOC_QUERIES=12`

These values are used unless environment variables override them.

## Repository structure

- `app/` – Next.js App Router, API routes, and UI.
- `src/ingestion/` – PDF loaders and ingestion pipeline.
- `src/chunking/` – document-aware chunking logic.
- `src/retrieval/` – lexical index, local vector store, hybrid rerank pipeline.
- `src/graph/` – LangGraph workflow nodes and runner.
- `src/lib/` – runtime/model configuration helpers.
- `src/prompts/` – reusable prompts.
- `scripts/` – ingestion and smoke test scripts.
- `data/` – local artifacts and PDF folder.

## Setup

1. Install dependencies

```bash
npm install
```

2. Put local PDF files in `data` (default).
By default, ingestion uses `CORPUS_SOURCE_FILES` if set, otherwise it auto-discovers PDFs in `data` and `data/docs`.

```bash
CORPUS_SOURCE_FILES=CFR - Code of Federal Regulations.pdf,Agent_Memory_Below_the_Prompt.pdf
```

If your files are not named explicitly, remove this variable and rely on auto-discovery.

3. Copy environment defaults:

```bash
cp .env.example .env
```

4. Ingest documents

```bash
npm run ingest
```
If local embedding is unavailable, ingestion still completes by building the lexical index and records a warning in the output.
For slower local embedding runtimes, lower `INGEST_EMBED_BATCH_SIZE` in `.env` (default `16`) to trade throughput for stability and clearer progress during indexing.

or via UI using **Ingest PDFs**.

5. Run app

```bash
npm run dev
```

6. Ask sample questions at `http://localhost:3000`

Example queries:

- Single-document question:
  - “What sections discuss information collection and submissions?”
- Single-document question:
  - “What is the abstract and what limitation is highlighted?”
- Cross-document comparison:
  - “Compare the treatment of memory and control requirements across documents.”

7. Smoke test

```bash
npm run smoke
npm run smoke:primary
npm run smoke:secondary
npm run smoke:hard-multidoc
```

`smoke` now validates:

- One fixture-backed single-document question for the first configured document
- One fixture-backed single-document question for the second configured document
- One fixture-backed cross-document comparison question

`smoke:hard-multidoc` exercises a stricter multi-source scenario:

- one document must supply technical performance evidence
- another document must supply regulatory evidence
- the answer must separate supported claims from unsupported claims instead of filling gaps with nearby sections or related examples
- the answer must not invent multiple unsupported Part 11 section numbers

This means the hard fixture is intentionally allowed to conclude that some requested controls are not supported by the retrieved evidence. For this project, that conservative behavior is preferred over speculative completion.

## Ouroboros 실제 설치/세팅 (이 프로젝트용)

`Q00/ouroboros`는 기본적으로 Claude Code용 플러그인 시스템을 가진 사양 정제 도구입니다. 이 레포에서는 **로컬 CLI 실행형 인터페이스**를 바로 사용하기 위해 `uvx` 기반 진입점을 붙였습니다.

요구사항:
- Python 런타임
- `uv` / `uvx` 설치

설치/세팅:

```bash
chmod +x scripts/ouroboros/*.sh
npm run ouroboros:doctor
npm run ouroboros:install
npm run ouroboros:setup
```

`npm run ouroboros:setup`은 내부적으로 `npm run ouroboros -- setup`을 실행합니다.  
이 버전의 설치 CLI 동작은 `setup/config` 경로가 현재 **초기화/확인용 정보 출력** 성격이 강해, 실제 설정 값은 환경에 맞게 `npm run ouroboros -- status`에서 Providers 상태를 점검하고 필요 시 별도 `config set`으로 점진적으로 구성합니다.

실행 예시:

```bash
npm run ouroboros -- --help
npm run ouroboros:setup
npm run ouroboros -- config show
npm run ouroboros -- status health
npm run ouroboros -- run workflow scripts/ouroboros/seeds/multidocs_seed.yaml --dry-run --no-orchestrator
npm run ouroboros -- tui monitor
```

참고: `ooo setup`은 Claude Code 세션에서 동작하는 별도 사용 방식입니다. Upstream 기준의 명령은 `README`의 Quick Start(claude plugin install, `ooo setup`)를 참고하세요.

## API

- `POST /api/query`

```json
{ "question": "..." , "debug": true }
```

Response includes:
- `answer`
- `citations` (deduped source tags like `[<document-id> p.12]`)
- `quality` (`passed`, `candidate_count`, top/avg scores)
- `trace` (per-stage metadata: retrieve, evidence_gate, generate, verify)
- `retrieval.chunks` (for debugging)

When `debug` is enabled in the UI, the retrieved-chunks panel also shows a `Document mix` summary. This is a per-document aggregation of the current retrieval set:

- `document_id`
- `source_file`
- number of retrieved chunks contributed by that document
- best chunk score seen for that document

This is only a debug aid. It helps distinguish:

- true multi-document retrieval
- single-document dominance
- document distribution shifts after retrieval changes

- `POST /api/ingest` – run ingestion from local PDFs.
- `GET /api/ingest` – route status/info message.

## How local model configuration works

All model/model-runtime settings are environment-driven and read from `src/lib/config.ts`.
To switch:

- chat: set `OLLAMA_CHAT_MODEL`
- embedding: set `OLLAMA_EMBEDDING_MODEL`
- rerank: set `OLLAMA_RERANK_MODEL` (defaults to embedding model)
- think behaviour: set `OLLAMA_CHAT_THINK` (`true` / `false`) to control chain-of-thought suppression (default `false`)
- generation temperature: set `OLLAMA_CHAT_TEMPERATURE` (default `0`)
- request timeout: set `OLLAMA_REQUEST_TIMEOUT_MS` (milliseconds, default `120000`)
- ingest embedding batch size: set `INGEST_EMBED_BATCH_SIZE` (default `16`) if local embedding rebuilds are too slow or unstable at larger batch sizes

No cloud APIs are used.

## Reranking note

The retrieval stage uses hybrid dense+lexical fusion first.
If dense embeddings are unavailable (for example Ollama embedding endpoint timeout), the pipeline automatically degrades to lexical-only retrieval and marks this in the trace.

For reranking, the MVP uses local embedding model scoring as a practical fallback because local dedicated rerankers are not always available in every Ollama setup. This is explicit in code and config:

- set `RETRIEVAL_ENABLE_RERANK=false` if you want pure hybrid ranking only.
- `RETRIEVAL_DENSE_WEIGHT` and `RETRIEVAL_LEXICAL_WEIGHT` control fusion.

## Current limitations

- Retrieval and reranking remain local-model bound, so full re-ingestion can take a long time on slower embedding runtimes.
- The hard multi-document fixture currently succeeds by correctly separating supported evidence from unsupported Part 11 claims, not by recovering a full set of Part 11 control sections from the provided CFR excerpt.
- When the retrieved CFR evidence only contains exclusion/applicability text, the system will now explicitly refuse to invent missing control sections or requirements.
- In other words, a hard-fixture `PASS` currently means "the system stayed grounded and refused unsupported claims," not "the system reconstructed every requested control mapping from the corpus."

`RETRIEVAL_ALLOW_DENSE_FALLBACK=false` disables dense lookup and forces lexical-first behavior even when dense vectors are unavailable.

## Ouroboros-style orchestration in this MVP

This project adopts a simplified version of Ouroboros’ “workflows with gates” pattern:

1. **Retrieve**: do dense+lexical hybrid retrieval and optional reranking while recording a document-agnostic route marker (`unified`) for telemetry.
2. **Quality gate**: require minimal retrieval evidence before answer generation.
3. **Generate**: produce a grounded answer only from retrieved chunks.
4. **Verify**: enforce citation-availability rules and force an insufficiency fallback when verification fails.

Each stage writes a trace event so you can inspect why an answer was accepted, degraded, or refused.

## Limitations

- PDF structure extraction is best-effort; malformed layouts can reduce section recovery.
- Reranking is currently local cosine re-scoring using embedding vectors (practical local fallback).
- Citation insertion is constrained by model behavior and prompt adherence; retrieved chunks are still surfaced in debug for traceability.

## Future improvements

- Add explicit cross-encoder reranker if available locally.
- Swap in a persistent vector DB (Chroma/pgvector) with dedicated ANN search.
- Add answer citations with per-claim alignment and exact claim-to-chunk mapping.
- Add ingestion progress UI and upload endpoint for new PDFs.
- Add regression test fixtures and snapshot checks for query outputs.
