import { appConfig } from "../lib/config";
import { RetrievalResult, ScoredChunk } from "../types/rag";
import { LexicalStore } from "./lexical";
import { LocalVectorStore } from "./vectorStore";
import { retrieveHybrid } from "./hybrid";

interface CrossDocInputs {
  query: string;
  additionalQueries?: string[];
  candidateK?: number;
  topK?: number;
  overrideRerank?: boolean;
  filterDocumentIds?: string[];
}

const sanitizeScoredChunkList = (chunks: ScoredChunk[]): ScoredChunk[] =>
  chunks.filter((chunk) => typeof chunk?.chunk?.metadata?.document_id === "string");

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

const dedupeQueries = (queries: string[]): string[] => {
  const seen = new Set<string>();
  const unique: string[] = [];
  for (const value of queries) {
    const normalized = value.replace(/\s+/gu, " ").trim();
    if (!normalized) continue;
    const key = normalized.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    unique.push(normalized);
  }
  return unique;
};

const QUERY_RRF_K = 60;

const tokenizeReferenceText = (value: string): string[] =>
  value
    .toLowerCase()
    .split(/[^a-z0-9]+/u)
    .map((token) => token.trim())
    .filter((token) => token.length >= 3);

const extractReferencedDocumentIds = (text: string): string[] => {
  const normalized = text.toLowerCase();
  const queryTokens = new Set(tokenizeReferenceText(text));
  const tokenFrequency = new Map<string, number>();
  const corpusTokens = appConfig.corpus.map((entry) => {
    const tokens = Array.from(
      new Set([
        ...tokenizeReferenceText(entry.document_id),
        ...tokenizeReferenceText(entry.source_file),
      ]),
    );
    for (const token of tokens) {
      tokenFrequency.set(token, (tokenFrequency.get(token) ?? 0) + 1);
    }
    return { entry, tokens };
  });
  const matches: string[] = [];

  for (const { entry, tokens } of corpusTokens) {
    if (
      normalized.includes(entry.document_id.toLowerCase()) ||
      normalized.includes(entry.source_file.toLowerCase())
    ) {
      matches.push(entry.document_id);
      continue;
    }

    const hasDistinctiveTokenMatch = tokens.some(
      (token) => queryTokens.has(token) && tokenFrequency.get(token) === 1,
    );
    if (hasDistinctiveTokenMatch) {
      matches.push(entry.document_id);
    }
  }

  return Array.from(new Set(matches));
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

export async function retrieveCrossDocument(
  lexicalStore: LexicalStore,
  vectorStore: LocalVectorStore | null,
  inputs: CrossDocInputs,
): Promise<RetrievalResult> {
  const {
    query,
    additionalQueries = [],
    candidateK = appConfig.retrieval.candidateK,
    topK = appConfig.retrieval.topK,
    overrideRerank,
    filterDocumentIds = [],
  } = inputs;

  const queries = dedupeQueries([query, ...additionalQueries]);
  const effectiveCandidateK =
    filterDocumentIds.length > 0
      ? Math.max(candidateK, topK * 5)
      : candidateK;
  const effectiveTopK =
    filterDocumentIds.length > 0
      ? Math.max(topK, topK * 2)
      : topK;
  const perQueryResults = await Promise.all(
    queries.map((retrievalQuery) => {
      const queryReferencedDocumentIds = extractReferencedDocumentIds(retrievalQuery);
      const intersectedFilterDocumentIds =
        filterDocumentIds.length > 0
          ? filterDocumentIds.filter((documentId) => queryReferencedDocumentIds.includes(documentId))
          : [];
      const effectiveFilterDocumentIds =
        queryReferencedDocumentIds.length > 0
          ? filterDocumentIds.length > 0
            ? intersectedFilterDocumentIds.length > 0
              ? intersectedFilterDocumentIds
              : filterDocumentIds
            : queryReferencedDocumentIds
          : filterDocumentIds;

      return (
      retrieveHybrid(lexicalStore, vectorStore, {
        query: retrievalQuery,
        queries: [retrievalQuery],
        filterDocumentIds: effectiveFilterDocumentIds,
        candidateK: effectiveCandidateK,
        topK: Math.max(effectiveCandidateK, effectiveTopK),
        overrideRerank,
        expandQueries: false,
      }),
      );
    }),
  );

  const merged = new Map<string, ScoredChunk>();
  for (const result of perQueryResults) {
    result.chunks.forEach((chunk, index) => {
      const existing = merged.get(chunk.chunk.id);
      const rrfScore = 1 / (QUERY_RRF_K + index + 1);
      const next: ScoredChunk = {
        ...chunk,
        score: (existing?.score ?? 0) + rrfScore,
        rerankScore:
          typeof chunk.rerankScore === "number"
            ? Math.max(chunk.rerankScore, existing?.rerankScore ?? Number.NEGATIVE_INFINITY)
            : existing?.rerankScore,
      };
      if (!existing) {
        merged.set(chunk.chunk.id, next);
        return;
      }
      merged.set(chunk.chunk.id, {
        ...next,
        denseScore: Math.max(chunk.denseScore ?? Number.NEGATIVE_INFINITY, existing.denseScore ?? Number.NEGATIVE_INFINITY),
        lexicalScore: Math.max(chunk.lexicalScore ?? Number.NEGATIVE_INFINITY, existing.lexicalScore ?? Number.NEGATIVE_INFINITY),
      });
    });
  }

  const retrieval = {
    question: query,
    chunks: Array.from(merged.values()).sort(compareScoredChunks),
    reranked: perQueryResults.some((result) => result.reranked),
    degradation_reasons: Array.from(
      new Set(perQueryResults.flatMap((result) => result.degradation_reasons ?? [])),
    ),
    dense_enabled: perQueryResults.every((result) => result.dense_enabled !== false),
    lexical_enabled: perQueryResults.every((result) => result.lexical_enabled !== false),
  } satisfies RetrievalResult;

  return {
    ...retrieval,
    question: query,
    chunks: dedupeById(retrieval.chunks).sort(compareScoredChunks).slice(0, effectiveTopK),
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
