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

  return {
    ...retrieval,
    question: query,
    chunks: dedupeById(retrieval.chunks).sort(compareScoredChunks).slice(0, topK),
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
