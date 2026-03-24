import { appConfig } from "../lib/config";
import { getChatModel } from "../lib/llm";
import { RetrievalResult, ScoredChunk } from "../types/rag";
import { LexicalStore } from "./lexical";
import { LocalVectorStore } from "./vectorStore";
import { retrieveHybrid } from "./hybrid";

interface CrossDocInputs {
  query: string;
  candidateK?: number;
  topK?: number;
  overrideRerank?: boolean;
}

interface ScopedQueryPlan {
  document_id: string;
  queries: string[];
}

interface ScopedChunkSelection {
  document_id: string;
  chunk_ids: string[];
}


const chunkExcerpt = (chunk: ScoredChunk): string =>
  `[${chunk.chunk.metadata.document_id} p.${chunk.chunk.metadata.page}] ${chunk.chunk.text
    .replace(/\s+/gu, " ")
    .trim()
    .slice(0, 220)}`;

const sanitizeScoredChunkList = (chunks: ScoredChunk[]): ScoredChunk[] =>
  chunks.filter(
    (chunk) =>
      typeof chunk?.chunk?.metadata?.document_id === "string",
  );

const compareScoredChunks = (left: ScoredChunk, right: ScoredChunk): number => {
  const leftRerank =
    typeof left.rerankScore === "number" && Number.isFinite(left.rerankScore)
      ? left.rerankScore
      : Number.NEGATIVE_INFINITY;
  const rightRerank =
    typeof right.rerankScore === "number" && Number.isFinite(right.rerankScore)
      ? right.rerankScore
      : Number.NEGATIVE_INFINITY;
  if (leftRerank !== rightRerank) {
    return rightRerank - leftRerank;
  }
  return right.score - left.score;
};

const dedupeById = (chunks: ScoredChunk[]): ScoredChunk[] => {
  const bestById = new Map<string, ScoredChunk>();
  for (const chunk of sanitizeScoredChunkList(chunks)) {
    const existing = bestById.get(chunk.chunk.id);
    if (!existing || compareScoredChunks(chunk, existing) < 0) {
      bestById.set(chunk.chunk.id, chunk);
    }
  }
  return [...bestById.values()];
};

const extractExplicitDocumentIds = (query: string): string[] => {
  const normalizedQuery = query.toLowerCase();
  const ordered: string[] = [];
  const seen = new Set<string>();

  for (const doc of appConfig.corpus) {
    const matchesDocumentId = normalizedQuery.includes(doc.document_id.toLowerCase());
    const matchesSourceFile = normalizedQuery.includes(doc.source_file.toLowerCase());
    if (!matchesDocumentId && !matchesSourceFile) continue;
    if (seen.has(doc.document_id)) continue;
    seen.add(doc.document_id);
    ordered.push(doc.document_id);
  }

  return ordered;
};

const parseScopedQueryPlans = (raw: string): ScopedQueryPlan[] => {
  const trimmed = raw.trim();
  if (!trimmed) return [];

  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/u);
  const candidate = fenced?.[1]?.trim() ?? trimmed;

  try {
    const parsed = JSON.parse(candidate) as unknown;
    const plans =
      Array.isArray(parsed)
        ? parsed
        : parsed && typeof parsed === "object" && "plans" in parsed
          ? (parsed as { plans?: unknown }).plans
          : null;

    if (!Array.isArray(plans)) {
      return [];
    }

    return plans
      .filter(
        (entry): entry is ScopedQueryPlan =>
          Boolean(entry) &&
          typeof entry === "object" &&
          typeof (entry as ScopedQueryPlan).document_id === "string" &&
          (typeof (entry as { query?: unknown }).query === "string" ||
            Array.isArray((entry as { queries?: unknown }).queries)),
      )
      .map((entry) => ({
        document_id: entry.document_id.trim(),
        queries: (
          Array.isArray((entry as { queries?: unknown }).queries)
            ? (entry as { queries: unknown[] }).queries
            : [(entry as { query: string }).query]
        )
          .filter((value): value is string => typeof value === "string")
          .map((value) => value.trim())
          .filter(Boolean),
      }))
      .filter((entry) => entry.document_id.length > 0 && entry.queries.length > 0);
  } catch {
    return [];
  }
};

const extractTextContent = (content: unknown): string => {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") return part;
        if (part && typeof part === "object" && "text" in part) {
          const text = (part as { text?: unknown }).text;
          return typeof text === "string" ? text : "";
        }
        return "";
      })
      .join("\n");
  }
  return "";
};

const parseScopedChunkSelection = (raw: string): ScopedChunkSelection[] => {
  const trimmed = raw.trim();
  if (!trimmed) return [];

  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/u);
  const candidate = fenced?.[1]?.trim() ?? trimmed;

  try {
    const parsed = JSON.parse(candidate) as unknown;
    const selections =
      Array.isArray(parsed)
        ? parsed
        : parsed && typeof parsed === "object" && "selections" in parsed
          ? (parsed as { selections?: unknown }).selections
          : null;

    if (!Array.isArray(selections)) {
      return [];
    }

    return selections
      .filter(
        (entry): entry is ScopedChunkSelection =>
          Boolean(entry) &&
          typeof entry === "object" &&
          typeof (entry as ScopedChunkSelection).document_id === "string" &&
          Array.isArray((entry as ScopedChunkSelection).chunk_ids),
      )
      .map((entry) => ({
        document_id: entry.document_id.trim(),
        chunk_ids: entry.chunk_ids
          .filter((value): value is string => typeof value === "string")
          .map((value) => value.trim())
          .filter(Boolean),
      }))
      .filter((entry) => entry.document_id.length > 0);
  } catch {
    return [];
  }
};


const planScopedQueries = async (
  query: string,
  explicitDocumentIds: string[],
): Promise<Map<string, string>> => {
  if (explicitDocumentIds.length === 0) {
    return new Map();
  }

  const fallback = new Map(explicitDocumentIds.map((documentId) => [documentId, [query]]));

  try {
    const model = getChatModel();
    const response = await model.invoke(`
Given a user question and explicit document ids, produce up to 3 retrieval queries per document id.

Requirements:
- Keep each document_id unchanged.
- Each query must be a concise retrieval query likely to match the actual language of that document.
- Preserve exact identifiers, filenames, quoted strings, and numbers when relevant.
- Queries may differ across documents if the user asks for different evidence from each document.
- Prefer a small set of complementary evidence-seeking queries rather than near-duplicates.
- Do not invent facts or narrow to a specific domain beyond the user question.
- Return JSON only in this shape: {"plans":[{"document_id":"...","queries":["..."]}]}.

Document ids:
${explicitDocumentIds.map((documentId) => `- ${documentId}`).join("\n")}

Question:
${query}
`);

    const rawContent =
      typeof response.content === "string"
        ? response.content
        : Array.isArray(response.content)
          ? response.content
              .map((part) =>
                part && typeof part === "object" && "text" in part && typeof (part as { text?: unknown }).text === "string"
                  ? String((part as { text?: unknown }).text)
                  : "",
              )
              .join("\n")
          : "";

    const plans = parseScopedQueryPlans(rawContent);
    if (!plans.length) {
      return fallback;
    }

    const mapped = new Map<string, string[]>();
    for (const documentId of explicitDocumentIds) {
      const planned = plans.find((entry) => entry.document_id === documentId);
      mapped.set(documentId, planned?.queries ?? [query]);
    }
    return mapped;
  } catch {
    return fallback;
  }
};

const planRefinedScopedQueries = async (
  question: string,
  explicitDocumentIds: string[],
  scopedResults: Map<string, ScoredChunk[]>,
): Promise<Map<string, string[]>> => {
  if (explicitDocumentIds.length === 0) {
    return new Map();
  }

  const fallback = new Map<string, string[]>();
  for (const documentId of explicitDocumentIds) {
    const topChunks = (scopedResults.get(documentId) ?? []).slice(0, 2);
    if (topChunks.length === 0) continue;
    fallback.set(
      documentId,
      [
        `${documentId} ${topChunks
          .map((chunk) => chunk.chunk.text.replace(/\s+/gu, " ").trim().slice(0, 80))
          .join(" ")}`.trim(),
      ],
    );
  }

  try {
    const model = getChatModel();
    const response = await model.invoke(`
You are refining retrieval queries for explicitly referenced documents in a generic multi-document RAG system.

Given a user question and the currently retrieved evidence snippets per document_id, produce up to 2 better follow-up retrieval queries per document_id.

Requirements:
- Use only concepts, identifiers, and terminology that already appear in the user question or the evidence snippets.
- Do not invent section numbers, control names, framework names, or domain-specific labels not already present.
- Prefer concrete noun phrases and wording likely to appear verbatim in source text.
- If the current snippets are already the best available evidence, you may keep the query close to the existing wording.
- Return JSON only in this shape: {"plans":[{"document_id":"...","queries":["..."]}]}.

Question:
${question}

Evidence by document:
${explicitDocumentIds
  .map((documentId) => {
    const chunks = (scopedResults.get(documentId) ?? []).slice(0, 2);
    return `## ${documentId}\n${chunks.map(chunkExcerpt).join("\n")}`;
  })
  .join("\n\n")}
`);

    const rawContent =
      typeof response.content === "string"
        ? response.content
        : Array.isArray(response.content)
          ? response.content
              .map((part) =>
                part && typeof part === "object" && "text" in part && typeof (part as { text?: unknown }).text === "string"
                  ? String((part as { text?: unknown }).text)
                  : "",
              )
              .join("\n")
          : "";

    const plans = parseScopedQueryPlans(rawContent);
    if (!plans.length) {
      return fallback;
    }

    const mapped = new Map<string, string[]>();
    for (const documentId of explicitDocumentIds) {
      const planned = plans.find((entry) => entry.document_id === documentId);
      if (planned?.queries?.length) {
        mapped.set(documentId, planned.queries);
      } else if (fallback.has(documentId)) {
        mapped.set(documentId, fallback.get(documentId)!);
      }
    }
    return mapped;
  } catch {
    return fallback;
  }
};

const selectScopedEvidence = async (
  question: string,
  explicitDocumentIds: string[],
  candidatesByDocument: Map<string, ScoredChunk[]>,
): Promise<Map<string, string[]>> => {
  if (explicitDocumentIds.length === 0) {
    return new Map();
  }

  try {
    const model = getChatModel();
    const response = await model.invoke(`
You are selecting evidence chunks for explicitly referenced documents in a generic multi-document RAG system.

Given a user question and candidate chunks for each document_id, select the chunk ids that are most directly useful for answering the question.

Requirements:
- Use only the provided candidate chunks.
- Prefer chunks with direct factual support for the question.
- Reject chunks that are only taxonomy, index, table-of-contents, or loosely related listings unless they directly answer the question.
- Return at most 2 chunk ids per document_id.
- If a document has no directly useful chunk, return an empty list for that document_id.
- Return JSON only in this shape: {"selections":[{"document_id":"...","chunk_ids":["..."]}]}.

Question:
${question}

Candidates by document:
${explicitDocumentIds
  .map((documentId) => {
    const chunks = (candidatesByDocument.get(documentId) ?? []).slice(0, 8);
    return `## ${documentId}\n${chunks
      .map(
        (chunk) =>
          `- chunk_id=${chunk.chunk.id} page=${chunk.chunk.metadata.page} excerpt=${JSON.stringify(
            chunk.chunk.text.replace(/\s+/gu, " ").trim().slice(0, 260),
          )}`,
      )
      .join("\n")}`;
  })
  .join("\n\n")}
`);

    const selections = parseScopedChunkSelection(extractTextContent(response.content));
    const mapped = new Map<string, string[]>();
    for (const documentId of explicitDocumentIds) {
      const selected = selections.find((entry) => entry.document_id === documentId);
      mapped.set(documentId, selected?.chunk_ids ?? []);
    }
    return mapped;
  } catch {
    return new Map();
  }
};


export async function retrieveCrossDocument(
  lexicalStore: LexicalStore,
  vectorStore: LocalVectorStore | null,
  inputs: CrossDocInputs,
): Promise<RetrievalResult> {
  const {
    query,
    candidateK = appConfig.retrieval.candidateK,
    topK = appConfig.retrieval.topK,
    overrideRerank,
  } = inputs;

  const explicitDocumentIds = extractExplicitDocumentIds(query);
  const scopedQueryPlan = await planScopedQueries(query, explicitDocumentIds);

  const globalRetrieval = await retrieveHybrid(lexicalStore, vectorStore, {
    query,
    candidateK,
    topK: Math.max(candidateK, topK),
    overrideRerank,
    expandQueries: true,
  });

  const scopedRetrievals =
    explicitDocumentIds.length === 0
      ? []
      : await Promise.all(
          explicitDocumentIds.flatMap((documentId) =>
            (scopedQueryPlan.get(documentId) ?? [query]).map((plannedQuery) =>
              retrieveHybrid(lexicalStore, vectorStore, {
                query: plannedQuery,
                candidateK,
                topK: Math.max(candidateK, topK),
                overrideRerank,
                expandQueries: true,
                filterDocumentIds: [documentId],
              }),
            ),
          ),
        );

  const scopedResultMap = new Map<string, ScoredChunk[]>();
  let scopedIndex = 0;
  for (const documentId of explicitDocumentIds) {
    const plannedQueries = scopedQueryPlan.get(documentId) ?? [query];
    const collected: ScoredChunk[] = [];
    for (let index = 0; index < plannedQueries.length; index += 1) {
      collected.push(...(scopedRetrievals[scopedIndex]?.chunks ?? []));
      scopedIndex += 1;
    }
    scopedResultMap.set(documentId, dedupeById(collected).sort(compareScoredChunks));
  }

  const refinedQueryPlan = await planRefinedScopedQueries(
    query,
    explicitDocumentIds,
    scopedResultMap,
  );

  const refinedScopedRetrievals =
    refinedQueryPlan.size === 0
      ? []
      : await Promise.all(
          explicitDocumentIds
            .filter((documentId) => refinedQueryPlan.has(documentId))
            .flatMap((documentId) =>
              (refinedQueryPlan.get(documentId) ?? [query]).map((plannedQuery) =>
                retrieveHybrid(lexicalStore, vectorStore, {
                  query: plannedQuery,
                  candidateK,
                  topK: Math.max(candidateK, topK),
                  overrideRerank,
                  expandQueries: false,
                  filterDocumentIds: [documentId],
                }),
              ),
            ),
        );

  const scopedLeaders = [...scopedRetrievals, ...refinedScopedRetrievals]
    .map((result) => result.chunks[0])
    .filter((chunk): chunk is ScoredChunk => Boolean(chunk));

  const merged = dedupeById([
    ...scopedLeaders,
    ...[globalRetrieval, ...scopedRetrievals, ...refinedScopedRetrievals].flatMap((result) => result.chunks),
  ]).sort(compareScoredChunks);
  const candidatesByDocument = new Map<string, ScoredChunk[]>();
  for (const chunk of merged) {
    const documentId = chunk.chunk.metadata.document_id;
    const existing = candidatesByDocument.get(documentId) ?? [];
    existing.push(chunk);
    candidatesByDocument.set(documentId, existing);
  }

  const scopedSelections = await selectScopedEvidence(
    query,
    explicitDocumentIds,
    candidatesByDocument,
  );

  const selectedScopedChunks = explicitDocumentIds.flatMap((documentId) => {
    const selectedIds = scopedSelections.get(documentId) ?? [];
    const candidates = candidatesByDocument.get(documentId) ?? [];
    return selectedIds
      .map((chunkId) => candidates.find((chunk) => chunk.chunk.id === chunkId))
      .filter((chunk): chunk is ScoredChunk => Boolean(chunk));
  });

  const preferred = dedupeById([
    ...selectedScopedChunks,
    ...scopedLeaders,
  ]).sort(compareScoredChunks);

  const lockedIds = new Set(preferred.map((chunk) => chunk.chunk.id));
  const chunks = [
    ...preferred,
    ...merged.filter((chunk) => !lockedIds.has(chunk.chunk.id)),
  ].slice(0, topK);

  return {
    question: query,
    chunks,
    reranked: [globalRetrieval, ...scopedRetrievals, ...refinedScopedRetrievals].some((result) => result.reranked),
    degradation_reasons: [globalRetrieval, ...scopedRetrievals, ...refinedScopedRetrievals].flatMap((result) => result.degradation_reasons ?? []),
    dense_enabled: [globalRetrieval, ...scopedRetrievals, ...refinedScopedRetrievals].every((result) => result.dense_enabled !== false),
    lexical_enabled: [globalRetrieval, ...scopedRetrievals, ...refinedScopedRetrievals].every((result) => result.lexical_enabled !== false),
  };
}

export function hasMultiDocumentCoverage(chunks: ScoredChunk[]): boolean {
  const validChunks = sanitizeScoredChunkList(chunks);
  const docIds = new Set(validChunks.map((chunk) => chunk.chunk.metadata.document_id));
  return docIds.size > 1;
}

export function getDocumentCoverageSummary(
  chunks: ScoredChunk[],
): Record<string, { count: number; topScore: number }> {
  const summary: Record<string, { count: number; topScore: number }> = {};
  for (const sc of sanitizeScoredChunkList(chunks)) {
    const docId = sc.chunk.metadata.document_id;
    const existing = summary[docId];
    if (existing) {
      existing.count += 1;
      existing.topScore = Math.max(existing.topScore, sc.score);
    } else {
      summary[docId] = { count: 1, topScore: sc.score };
    }
  }
  return summary;
}
