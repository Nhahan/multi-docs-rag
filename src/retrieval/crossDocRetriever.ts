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
  query: string;
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
          typeof (entry as ScopedQueryPlan).query === "string",
      )
      .map((entry) => ({
        document_id: entry.document_id.trim(),
        query: entry.query.trim(),
      }))
      .filter((entry) => entry.document_id.length > 0 && entry.query.length > 0);
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

  const fallback = new Map(explicitDocumentIds.map((documentId) => [documentId, query]));

  try {
    const model = getChatModel();
    const response = await model.invoke(`
Given a user question and explicit document ids, produce one retrieval query per document id.

Requirements:
- Keep each document_id unchanged.
- Each query must be a concise retrieval query likely to match the actual language of that document.
- Preserve exact identifiers, filenames, quoted strings, and numbers when relevant.
- Queries may differ across documents if the user asks for different evidence from each document.
- Do not invent facts or narrow to a specific domain beyond the user question.
- Return JSON only in this shape: {"plans":[{"document_id":"...","query":"..."}]}.

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

    const mapped = new Map<string, string>();
    for (const documentId of explicitDocumentIds) {
      const planned = plans.find((entry) => entry.document_id === documentId);
      mapped.set(documentId, planned?.query ?? query);
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
): Promise<Map<string, string>> => {
  if (explicitDocumentIds.length === 0) {
    return new Map();
  }

  const fallback = new Map<string, string>();
  for (const documentId of explicitDocumentIds) {
    const topChunks = (scopedResults.get(documentId) ?? []).slice(0, 2);
    if (topChunks.length === 0) continue;
    fallback.set(
      documentId,
      `${documentId} ${topChunks
        .map((chunk) => chunk.chunk.text.replace(/\s+/gu, " ").trim().slice(0, 80))
        .join(" ")}`.trim(),
    );
  }

  try {
    const model = getChatModel();
    const response = await model.invoke(`
You are refining retrieval queries for explicitly referenced documents in a generic multi-document RAG system.

Given a user question and the currently retrieved evidence snippets per document_id, produce one better follow-up retrieval query per document_id.

Requirements:
- Use only concepts, identifiers, and terminology that already appear in the user question or the evidence snippets.
- Do not invent section numbers, control names, framework names, or domain-specific labels not already present.
- Prefer concrete noun phrases and wording likely to appear verbatim in source text.
- If the current snippets are already the best available evidence, you may keep the query close to the existing wording.
- Return JSON only in this shape: {"plans":[{"document_id":"...","query":"..."}]}.

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

    const mapped = new Map<string, string>();
    for (const documentId of explicitDocumentIds) {
      const planned = plans.find((entry) => entry.document_id === documentId);
      if (planned?.query) {
        mapped.set(documentId, planned.query);
      } else if (fallback.has(documentId)) {
        mapped.set(documentId, fallback.get(documentId)!);
      }
    }
    return mapped;
  } catch {
    return fallback;
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
          explicitDocumentIds.map((documentId) =>
            retrieveHybrid(lexicalStore, vectorStore, {
              query: scopedQueryPlan.get(documentId) ?? query,
              candidateK,
              topK: Math.max(candidateK, topK),
              overrideRerank,
              expandQueries: true,
              filterDocumentIds: [documentId],
            }),
          ),
        );

  const scopedResultMap = new Map<string, ScoredChunk[]>();
  for (let index = 0; index < explicitDocumentIds.length; index += 1) {
    scopedResultMap.set(
      explicitDocumentIds[index],
      scopedRetrievals[index]?.chunks ?? [],
    );
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
            .map((documentId) =>
              retrieveHybrid(lexicalStore, vectorStore, {
                query: refinedQueryPlan.get(documentId) ?? query,
                candidateK,
                topK: Math.max(candidateK, topK),
                overrideRerank,
                expandQueries: false,
                filterDocumentIds: [documentId],
              }),
            ),
        );

  const scopedLeaders = [...scopedRetrievals, ...refinedScopedRetrievals]
    .map((result) => result.chunks[0])
    .filter((chunk): chunk is ScoredChunk => Boolean(chunk));

  const merged = dedupeById([
    ...scopedLeaders,
    ...[globalRetrieval, ...scopedRetrievals, ...refinedScopedRetrievals].flatMap((result) => result.chunks),
  ]).sort(compareScoredChunks);

  const lockedIds = new Set(scopedLeaders.map((chunk) => chunk.chunk.id));
  const chunks = [
    ...scopedLeaders,
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
