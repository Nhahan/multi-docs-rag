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
  } = inputs;

  const queries = dedupeQueries([query, ...additionalQueries]);
  const retrieval = await retrieveHybrid(lexicalStore, vectorStore, {
    query,
    queries,
    candidateK,
    topK: Math.max(candidateK, topK),
    overrideRerank,
    expandQueries: false,
  });

  const explicitDocumentIds = extractExplicitDocumentIds(query);
  const scopedRetrievals =
    explicitDocumentIds.length === 0
      ? []
      : await Promise.all(
          explicitDocumentIds.map((documentId) =>
            retrieveHybrid(lexicalStore, vectorStore, {
              query,
              queries,
              candidateK,
              topK: Math.max(candidateK, topK),
              overrideRerank,
              expandQueries: false,
              filterDocumentIds: [documentId],
            }),
          ),
        );

  const lockedChunks = scopedRetrievals
    .map((result) => result.chunks.slice().sort(compareScoredChunks)[0])
    .filter((chunk): chunk is ScoredChunk => Boolean(chunk));
  const lockedIds = new Set(lockedChunks.map((chunk) => chunk.chunk.id));

  const mergedChunks = dedupeById([
    ...lockedChunks,
    ...retrieval.chunks,
    ...scopedRetrievals.flatMap((result) => result.chunks.slice(0, 2)),
  ]).sort(compareScoredChunks);

  const finalChunks = [
    ...lockedChunks.sort(compareScoredChunks),
    ...mergedChunks.filter((chunk) => !lockedIds.has(chunk.chunk.id)),
  ].slice(0, topK);

  return {
    ...retrieval,
    question: query,
    chunks: finalChunks,
    reranked: retrieval.reranked || scopedRetrievals.some((result) => result.reranked),
    degradation_reasons: [
      ...(retrieval.degradation_reasons ?? []),
      ...scopedRetrievals.flatMap((result) => result.degradation_reasons ?? []),
    ],
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
