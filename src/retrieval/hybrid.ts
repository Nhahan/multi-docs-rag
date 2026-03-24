import { appConfig } from "../lib/config";
import { getEmbeddingModel, getRerankEmbeddingModel } from "../lib/llm";
import { DocumentType, RetrievalResult, ScoredChunk } from "../types/rag";
import { LexicalStore } from "./lexical";
import { LocalVectorStore } from "./vectorStore";
import { expandRetrievalQueries } from "./queryExpander";

interface HybridRetrieveOptions {
  query: string;
  filterDocumentTypes?: DocumentType[];
  filterDocumentIds?: string[];
  candidateK?: number;
  topK?: number;
  overrideRerank?: boolean;
  expandQueries?: boolean;
}

interface CandidateChunk {
  chunk: ScoredChunk["chunk"];
  denseScore?: number;
  lexicalScore?: number;
  denseRrf?: number;
  lexicalRrf?: number;
}

const toNumber = (value: unknown): number | undefined => {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.length > 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
};

const QUERY_RRF_K = 60;

const compareScoredChunks = (left: ScoredChunk, right: ScoredChunk): number => {
  const leftRerank = toNumber(left.rerankScore) ?? Number.NEGATIVE_INFINITY;
  const rightRerank = toNumber(right.rerankScore) ?? Number.NEGATIVE_INFINITY;
  if (leftRerank !== rightRerank) {
    return rightRerank - leftRerank;
  }
  return right.score - left.score;
};

const uniqueScored = (value: unknown): ScoredChunk | null => {
  if (!value || typeof value !== "object") return null;
  if (Array.isArray(value)) {
    const tupleSecond = value[1];
    if (!tupleSecond || typeof tupleSecond !== "object") return null;
    return uniqueScored(tupleSecond);
  }
  const candidate = value as Partial<ScoredChunk>;
  if (!candidate || typeof candidate.chunk !== "object" || !candidate.chunk) return null;
  if (
    candidate.chunk.metadata === undefined ||
    typeof candidate.chunk.metadata.document_id !== "string" ||
    typeof candidate.chunk.metadata.source_file !== "string"
  ) return null;
  if (typeof candidate.chunk.id !== "string") return null;

  return {
    chunk: candidate.chunk as ScoredChunk["chunk"],
    score: toNumber(candidate.score) ?? 0,
    denseScore: toNumber(candidate.denseScore),
    lexicalScore: toNumber(candidate.lexicalScore),
    rerankScore: toNumber(candidate.rerankScore),
  };
};

const rerankByEmbedding = async (
  candidates: ScoredChunk[],
  queries: string[],
  vectorStore: LocalVectorStore | null,
): Promise<{ chunks: ScoredChunk[]; used: boolean }> => {
  if (!vectorStore || !candidates.length) {
    return { chunks: candidates, used: false };
  }

  try {
    const reranker = getRerankEmbeddingModel();
    const uniqueQueries = dedupe(queries);
    if (!uniqueQueries.length) {
      return { chunks: candidates, used: false };
    }
    const queryVectors = await reranker.embedDocuments(uniqueQueries);
    const rerankLimit = Math.min(appConfig.retrieval.rerankTopK, candidates.length);
    const limited = [...candidates]
      .sort((a, b) => b.score - a.score)
      .slice(0, rerankLimit)
      .map((entry) => {
        const vector = vectorStore.getVectorForChunk(entry.chunk.id);
        if (!vector) return entry;
        const cNorm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
        let sim = Number.NEGATIVE_INFINITY;
        for (const queryVector of queryVectors) {
          if (!Array.isArray(queryVector) || queryVector.length === 0) continue;
          const dot = queryVector.reduce((sum, _, index) => sum + queryVector[index] * (vector[index] ?? 0), 0);
          const qNorm = Math.sqrt(queryVector.reduce((sum, value) => sum + value * value, 0));
          const candidateSim = qNorm && cNorm ? dot / (qNorm * cNorm) : 0;
          if (candidateSim > sim) {
            sim = candidateSim;
          }
        }
        if (!Number.isFinite(sim)) {
          sim = 0;
        }
        return { ...entry, rerankScore: sim };
      });

    const reranked = limited
      .sort((a, b) => (b.rerankScore ?? Number.NEGATIVE_INFINITY) - (a.rerankScore ?? Number.NEGATIVE_INFINITY));

    const untouched = candidates.slice(rerankLimit);
    return { chunks: [...reranked, ...untouched], used: true };
  } catch {
    return { chunks: candidates, used: false };
  }
};

const gatherCandidates = async (
  lexicalStore: LexicalStore,
  vectorStore: LocalVectorStore | null,
  queries: string[],
  candidateK: number,
  filterTypes: DocumentType[],
  filterDocumentIds: string[] = [],
): Promise<Map<string, CandidateChunk>> => {
  const map = new Map<string, CandidateChunk>();
  const upsert = (id: string, candidate: Partial<CandidateChunk>) => {
    const existing = map.get(id);
    if (!existing) {
      if (candidate.chunk) {
        map.set(id, {
          chunk: candidate.chunk,
          denseScore: candidate.denseScore,
          lexicalScore: candidate.lexicalScore,
          denseRrf: candidate.denseRrf,
          lexicalRrf: candidate.lexicalRrf,
        });
      }
      return;
    }

    if (candidate.denseScore !== undefined && (existing.denseScore === undefined || candidate.denseScore > existing.denseScore)) {
      existing.denseScore = candidate.denseScore;
    }
    if (candidate.lexicalScore !== undefined && (existing.lexicalScore === undefined || candidate.lexicalScore > existing.lexicalScore)) {
      existing.lexicalScore = candidate.lexicalScore;
    }
    if (candidate.denseRrf !== undefined) {
      existing.denseRrf = (existing.denseRrf ?? 0) + candidate.denseRrf;
    }
    if (candidate.lexicalRrf !== undefined) {
      existing.lexicalRrf = (existing.lexicalRrf ?? 0) + candidate.lexicalRrf;
    }
  };

  const embeddingModel = getEmbeddingModel();
  const enableDense =
    Boolean(vectorStore && vectorStore.isUsable() && appConfig.retrieval.allowDenseFallback);
  const denseQueryVectors = new Map<string, number[]>();

  if (enableDense && vectorStore && queries.length > 0) {
    try {
      const uniqueQueries = dedupe(queries);
      const vectors = await embeddingModel.embedDocuments(uniqueQueries);
      for (let index = 0; index < uniqueQueries.length; index += 1) {
        const query = uniqueQueries[index];
        const vector = vectors[index];
        if (Array.isArray(vector) && vector.length > 0) {
          denseQueryVectors.set(query, vector);
        }
      }
    } catch {
      // Dense retrieval failure is surfaced by degraded scoring coverage.
    }
  }

  for (const retrievalQuery of queries) {
    const lexicalResults = await lexicalStore.search(
      retrievalQuery,
      candidateK,
      filterTypes,
      filterDocumentIds,
    );
    for (const [index, result] of lexicalResults.entries()) {
      if (!result.chunk) continue;
      upsert(result.chunk.id, {
        chunk: result.chunk,
        lexicalScore: result.score,
        lexicalRrf: 1 / (QUERY_RRF_K + index + 1),
      });
    }

    if (!enableDense || !vectorStore) {
      continue;
    }

    try {
      const queryVector = denseQueryVectors.get(retrievalQuery);
      if (!queryVector) {
        continue;
      }
      const denseResults = await vectorStore.searchByEmbedding(
        queryVector,
        candidateK,
        filterTypes,
        filterDocumentIds,
      );
      for (const [index, result] of denseResults.entries()) {
        if (!result.chunk) continue;
        upsert(result.chunk.id, {
          chunk: result.chunk,
          denseScore: result.score,
          denseRrf: 1 / (QUERY_RRF_K + index + 1),
        });
      }
    } catch {
      // Dense retrieval failure is surfaced by degraded scoring coverage.
    }
  }

  return map;
};

export const retrieveHybrid = async (
  lexicalStore: LexicalStore,
  vectorStore: LocalVectorStore | null,
  options: HybridRetrieveOptions,
): Promise<RetrievalResult> => {
  const {
    query,
    filterDocumentTypes = [],
    filterDocumentIds = [],
    candidateK = appConfig.retrieval.candidateK,
    topK = appConfig.retrieval.topK,
    overrideRerank,
    expandQueries = true,
  } = options;

  const retrievalQueries = expandQueries ? await expandRetrievalQueries(query, undefined) : [query];
  const denseEnabled = Boolean(vectorStore && vectorStore.isUsable());
  const lexicalEnabled = true;
  const degradationReasons: string[] = [];
  const normalizedCandidateK =
    Number.isFinite(candidateK) && candidateK > 0 ? Math.floor(candidateK) : appConfig.retrieval.candidateK;
  const normalizedTopK =
    Number.isFinite(topK) && topK > 0 ? Math.floor(topK) : appConfig.retrieval.topK;

  if (!denseEnabled) {
    degradationReasons.push("dense retrieval unavailable; using lexical retrieval only");
  }

  const filterTypes = filterDocumentTypes;

  const chunkCandidates = await gatherCandidates(
    lexicalStore,
    vectorStore,
    retrievalQueries,
    normalizedCandidateK,
    filterTypes,
    filterDocumentIds,
  );

  const records = [...chunkCandidates.values()];
  if (!records.length) {
    return {
      question: query,
      chunks: [],
      reranked: false,
      degradation_reasons: [
        ...degradationReasons,
        "no lexical/dense candidates found for expanded queries",
      ],
      dense_enabled: denseEnabled,
      lexical_enabled: lexicalEnabled,
    };
  }

  const scored = records.map((entry) => ({
    chunk: entry.chunk,
    score: (entry.denseRrf ?? 0) + (entry.lexicalRrf ?? 0),
    denseScore: entry.denseScore,
    lexicalScore: entry.lexicalScore,
    rerankScore: undefined as number | undefined,
  } satisfies ScoredChunk));

  const normalizedScored = scored
    .map((entry) => uniqueScored(entry))
    .filter((entry): entry is ScoredChunk => Boolean(entry));
  if (!normalizedScored.length) {
    return {
      question: query,
      chunks: [],
      reranked: false,
      degradation_reasons: [...degradationReasons, "all candidates were malformed"],
      dense_enabled: denseEnabled,
      lexical_enabled: lexicalEnabled,
    };
  }

  const shouldRerank =
    (overrideRerank ?? appConfig.retrieval.enableRerank) &&
    appConfig.models.rerankModel.length > 0 &&
    (denseEnabled || vectorStore !== null);

  const rerankResult = shouldRerank
    ? await rerankByEmbedding(normalizedScored, retrievalQueries, vectorStore)
    : { chunks: normalizedScored, used: false };

  const final = Array.from(
    new Map(rerankResult.chunks.map((item) => [item.chunk.id, item])).values(),
  )
    .sort(compareScoredChunks)
    .slice(0, normalizedTopK);

  return {
    question: query,
    chunks: final,
    reranked: rerankResult.used,
    degradation_reasons: degradationReasons.length > 0 ? degradationReasons : undefined,
    dense_enabled: denseEnabled,
    lexical_enabled: lexicalEnabled,
  };
};
