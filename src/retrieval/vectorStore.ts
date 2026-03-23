import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { existsSync } from "node:fs";
import { DocumentType } from "../types/rag";
import { CorpusChunk, ScoredChunk } from "../types/rag";

interface SerializedVectorData {
  chunks: CorpusChunk[];
  vectors: number[][];
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  let dot = 0;
  let magA = 0;
  let magB = 0;

  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }

  if (magA === 0 || magB === 0) return 0;
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

const readPositiveInt = (raw: string | undefined, fallback: number): number => {
  const parsed = Number(raw ?? NaN);
  return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : fallback;
};

export class LocalVectorStore {
  private chunkById: Map<string, CorpusChunk>;
  private vectorById: Map<string, number[]>;

  private constructor(private chunks: CorpusChunk[], private vectors: number[][]) {
    this.chunkById = new Map(chunks.map((chunk, index) => [chunk.id, chunk]));
    this.vectorById = new Map(chunks.map((chunk, index) => [chunk.id, vectors[index]]));
  }

  static async build(outputPath: string, chunks: CorpusChunk[], embedding: { embedDocuments: (text: string[]) => Promise<number[][]> }): Promise<LocalVectorStore> {
    const texts = chunks.map((chunk) => chunk.text);
    const vectors: number[][] = [];
    const batchSize = readPositiveInt(process.env.INGEST_EMBED_BATCH_SIZE, 16);

    for (let start = 0; start < texts.length; start += batchSize) {
      const batch = texts.slice(start, start + batchSize);
      const end = Math.min(start + batch.length, texts.length);
      console.log(`[vector-store] embedding batch ${Math.floor(start / batchSize) + 1}/${Math.ceil(texts.length / batchSize)} (${start + 1}-${end} of ${texts.length})`);
      const batchVectors = await embedding.embedDocuments(batch);
      vectors.push(...batchVectors);
    }

    const store = new LocalVectorStore(chunks, vectors);
    await store.save(outputPath);
    return store;
  }

  static async load(path: string): Promise<LocalVectorStore> {
    if (!existsSync(path)) {
      throw new Error(`Vector store missing at ${path}. Run ingestion first.`);
    }
    const raw = await readFile(path, "utf8");
    const parsed = JSON.parse(raw) as SerializedVectorData;
    if (!parsed.chunks?.length || !parsed.vectors?.length) {
      throw new Error("Invalid vector store file.");
    }
    return new LocalVectorStore(parsed.chunks, parsed.vectors);
  }

  async save(path?: string): Promise<void> {
    if (!path) return;
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, JSON.stringify({ chunks: this.chunks, vectors: this.vectors }, null, 2), "utf8");
  }

  async searchByEmbedding(
    queryVector: number[],
    topK: number,
    filterTypes: DocumentType[] = [],
    filterDocumentIds: string[] = [],
  ): Promise<ScoredChunk[]> {
    const scored: ScoredChunk[] = [];
    void filterTypes;
    for (let index = 0; index < this.chunks.length; index += 1) {
      const chunk = this.chunks[index];
      const vector = this.vectors[index];
      if (!chunk || !vector) continue;
      if (filterDocumentIds.length > 0 && !filterDocumentIds.includes(chunk.metadata.document_id)) continue;
      scored.push({
        chunk,
        score: cosineSimilarity(queryVector, vector),
      });
    }
    return scored.sort((a, b) => b.score - a.score).slice(0, topK);
  }

  getChunksByType(filterTypes: DocumentType[]): CorpusChunk[] {
    void filterTypes;
    return this.chunks;
  }

  getAllChunks(): CorpusChunk[] {
    return this.chunks;
  }

  getVectorForChunk(chunkId: string): number[] | undefined {
    return this.vectorById.get(chunkId);
  }

  isUsable(): boolean {
    return this.chunks.length > 0 && this.vectors.length === this.chunks.length;
  }

  size(): number {
    return this.chunks.length;
  }
}
