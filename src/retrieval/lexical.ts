import { writeFile } from "node:fs/promises";
import { readFile, mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import { CorpusChunk, DocumentType } from "../types/rag";

type TermFreq = Record<string, number>;
type TermDocFreq = Record<string, number>;
type ChunkTermFreq = Record<string, TermFreq>;

const normalizeWord = (value: string) => value.trim().toLowerCase();

const DEFAULT_BM25_K1 = 1.2;
const DEFAULT_BM25_B = 0.75;
const LEXICAL_INDEX_VERSION = 6;

const tokenizeText = (text: string): string[] => {
  const rawTokens =
    text
      .toLowerCase()
      .match(/[\p{L}\p{N}][\p{L}\p{N}._-]*|\d+(?:\.\d+)*(?:[a-z0-9_-]+)?/gu) ?? [];

  return rawTokens
    .map((token) => normalizeWord(token))
    .filter((token) => {
      if (!token || token.length === 0) return false;
      const normalized = token;
      if (!normalized.length) return false;
      if (/^\d+$/.test(normalized)) {
        return normalized.length >= 2;
      }
      return true;
    });
};

export interface LexicalStoreData {
  version?: number;
  chunks: CorpusChunk[];
  termDocFreq: TermDocFreq;
  chunkTermFreq: ChunkTermFreq;
  chunkLengths: Record<string, number>;
  averageChunkLength?: number;
}

export interface LexicalResult {
  chunk: CorpusChunk;
  score: number;
}

export class LexicalStore {
  private constructor(
    private data: LexicalStoreData,
    private averageChunkLength: number,
  ) {}

  static async build(chunks: CorpusChunk[], outputPath: string): Promise<LexicalStore> {
    const termDocFreq: TermDocFreq = {};
    const chunkTermFreq: ChunkTermFreq = {};
    const chunkLengths: Record<string, number> = {};
    let totalLength = 0;

    for (const chunk of chunks) {
      const searchable = [
        chunk.metadata.section_title,
        chunk.text,
      ]
        .filter(Boolean)
        .join("\n");
      const tokens = tokenizeText(searchable);
      const tf: TermFreq = {};

      const tokenCount = tokens.length || 1;
      chunkLengths[chunk.id] = tokenCount;
      totalLength += tokenCount;

      for (const token of tokens) {
        tf[token] = (tf[token] ?? 0) + 1;
      }
      chunkTermFreq[chunk.id] = tf;

      for (const token of Object.keys(tf)) {
        termDocFreq[token] = (termDocFreq[token] ?? 0) + 1;
      }
    }

    const averageChunkLength = Math.max(totalLength / Math.max(chunks.length, 1), 1);
    const instance = new LexicalStore(
      {
        chunks,
        version: LEXICAL_INDEX_VERSION,
        termDocFreq,
        chunkTermFreq,
        chunkLengths,
        averageChunkLength,
      },
      averageChunkLength,
    );

    await instance.save(outputPath);
    return instance;
  }

  static async load(filePath: string): Promise<LexicalStore> {
    const raw = await readFile(filePath, "utf8");
    const parsed = JSON.parse(raw) as LexicalStoreData;
    if (parsed.version !== LEXICAL_INDEX_VERSION) {
      throw new Error("Lexical index version mismatch. Run ingestion first.");
    }
    const averageChunkLength = parsed.averageChunkLength ?? 1;
    return new LexicalStore(
      {
        ...parsed,
        chunkLengths: parsed.chunkLengths ?? {},
        termDocFreq: parsed.termDocFreq ?? {},
        chunkTermFreq: parsed.chunkTermFreq ?? {},
      },
      averageChunkLength,
    );
  }

  private getTermCount(chunkId: string, term: string): number {
    return this.data.chunkTermFreq[chunkId]?.[term] ?? 0;
  }

  async search(
    query: string,
    candidates: number,
    filterTypes: DocumentType[] = [],
    filterDocumentIds: string[] = [],
  ): Promise<LexicalResult[]> {
    const queryTokens = tokenizeText(query);
    if (!queryTokens.length) return [];

    const querySet = new Map<string, number>();
    for (const token of queryTokens) {
      querySet.set(token, (querySet.get(token) ?? 0) + 1);
    }

    const scores = new Map<string, number>();
    void filterTypes;
    const allChunks = this.data.chunks.filter(
      (chunk) =>
        filterDocumentIds.length === 0 || filterDocumentIds.includes(chunk.metadata.document_id),
    );

    const n = Math.max(allChunks.length, 1);
    const k1 = DEFAULT_BM25_K1;
    const b = DEFAULT_BM25_B;

    for (const chunk of allChunks) {
      const tokenCount = this.data.chunkLengths[chunk.id] ?? 1;
      let score = 0;
      for (const [term, queryCount] of querySet.entries()) {
        const df = this.data.termDocFreq[term] ?? 1;
        const tf = this.getTermCount(chunk.id, term);
        if (!tf) continue;
        const idf = Math.log((n - df + 0.5) / (df + 0.5) + 1);
        const norm = k1 * (1 - b + b * (tokenCount / this.averageChunkLength));
        const tfNormalized = (tf * (k1 + 1)) / (tf + norm || 1);
        score += queryCount * idf * tfNormalized;
      }

      if (score <= 0) continue;
      if (score > 0) {
        scores.set(chunk.id, score);
      }
    }

    const scored = [...scores.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, candidates)
      .map(([chunkId, score]) => {
        const chunk = this.data.chunks.find((entry) => entry.id === chunkId);
        if (!chunk) return null;
        return { chunk, score };
      })
      .filter((entry): entry is LexicalResult => entry !== null);

    return scored.map((entry) => ({
      ...entry,
      score: Number.isFinite(entry.score) ? entry.score : 0,
    }));
  }

  toJSON(): LexicalStoreData {
    return {
      ...this.data,
      version: this.data.version ?? LEXICAL_INDEX_VERSION,
      averageChunkLength: this.averageChunkLength,
    };
  }

  async save(filePath: string): Promise<void> {
    await mkdir(dirname(filePath), { recursive: true });
    await writeFile(filePath, JSON.stringify(this.data, null, 2), "utf8");
  }
}

export const buildLexicalStore = async (chunks: CorpusChunk[], outputPath: string) =>
  LexicalStore.build(chunks, outputPath);
