import { readFile, writeFile, mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname } from "node:path";
import { CorpusChunk } from "../types/rag";

/**
 * Persisted chunk storage with indexed retrieval by document_id and chunk id.
 *
 * On-disk format (JSON):
 *   { chunks: CorpusChunk[], createdAt: string }
 *
 * In-memory indexes are built lazily on first access after load.
 */
export class ChunkStore {
  private chunks: CorpusChunk[] = [];
  private byId: Map<string, CorpusChunk> = new Map();
  private byDocumentId: Map<string, CorpusChunk[]> = new Map();
  private createdAt: string | null = null;

  /* ------------------------------------------------------------------ */
  /*  Construction helpers                                               */
  /* ------------------------------------------------------------------ */

  /** Create a store from an array of chunks (e.g. after ingestion). */
  static fromChunks(chunks: CorpusChunk[]): ChunkStore {
    const store = new ChunkStore();
    store.addChunks(chunks);
    return store;
  }

  /** Load a previously-saved store from disk. */
  static async load(filePath: string): Promise<ChunkStore> {
    if (!existsSync(filePath)) {
      throw new Error(`Chunk store file not found: ${filePath}`);
    }
    const raw = await readFile(filePath, "utf8");
    const data = JSON.parse(raw) as { chunks: CorpusChunk[]; createdAt?: string };

    const store = new ChunkStore();
    store.createdAt = data.createdAt ?? null;
    store.addChunks(data.chunks ?? []);
    return store;
  }

  /* ------------------------------------------------------------------ */
  /*  Mutation                                                           */
  /* ------------------------------------------------------------------ */

  /** Add chunks and update in-memory indexes. */
  addChunks(chunks: CorpusChunk[]): void {
    for (const chunk of chunks) {
      // Avoid duplicate ids
      if (this.byId.has(chunk.id)) continue;

      this.chunks.push(chunk);
      this.byId.set(chunk.id, chunk);

      const docId = chunk.metadata.document_id;
      const bucket = this.byDocumentId.get(docId);
      if (bucket) {
        bucket.push(chunk);
      } else {
        this.byDocumentId.set(docId, [chunk]);
      }
    }
  }

  /** Remove all chunks and clear indexes. */
  clear(): void {
    this.chunks = [];
    this.byId.clear();
    this.byDocumentId.clear();
    this.createdAt = null;
  }

  /* ------------------------------------------------------------------ */
  /*  Persistence                                                        */
  /* ------------------------------------------------------------------ */

  /** Save to disk as JSON. */
  async save(filePath: string): Promise<void> {
    await mkdir(dirname(filePath), { recursive: true });
    const payload = {
      chunks: this.chunks,
      createdAt: this.createdAt ?? new Date().toISOString(),
    };
    await writeFile(filePath, JSON.stringify(payload, null, 2), "utf8");
  }

  /* ------------------------------------------------------------------ */
  /*  Retrieval                                                          */
  /* ------------------------------------------------------------------ */

  /** Get a single chunk by its unique id (for example, "<document-id>-p<page>-s<index>"). */
  getById(chunkId: string): CorpusChunk | undefined {
    return this.byId.get(chunkId);
  }

  /** Get all chunks belonging to a document_id. */
  getByDocumentId(documentId: string): CorpusChunk[] {
    return this.byDocumentId.get(documentId) ?? [];
  }

  /** Return all stored chunks. */
  getAll(): CorpusChunk[] {
    return this.chunks;
  }

  /** Total number of stored chunks. */
  get size(): number {
    return this.chunks.length;
  }

  /** List all unique document_ids in the store. */
  getDocumentIds(): string[] {
    return Array.from(this.byDocumentId.keys());
  }

  /** Get chunk count per document_id. */
  getSummary(): Record<string, number> {
    const summary: Record<string, number> = {};
    for (const [docId, docChunks] of this.byDocumentId) {
      summary[docId] = docChunks.length;
    }
    return summary;
  }
}
