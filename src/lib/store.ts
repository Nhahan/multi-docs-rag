import { existsSync } from "node:fs";
import { appConfig } from "./config";
import { LexicalStore } from "../retrieval/lexical";
import { LocalVectorStore } from "../retrieval/vectorStore";
import { ChunkStore } from "../retrieval/chunkStore";

let initialized: Promise<{
  lexical: LexicalStore;
  vector: LocalVectorStore | null;
  chunks: ChunkStore;
  vectorAvailable: boolean;
}> | null = null;

export const getIndexStore = async ({
  force = false,
  allowDenseMissing = false,
}: {
  force?: boolean;
  allowDenseMissing?: boolean;
} = {}) => {
  if (!initialized || force) {
    initialized = (async () => {
      if (!existsSync(appConfig.data.lexicalStorePath)) {
        throw new Error("Lexical index missing. Run ingestion first.");
      }

      const vectorExists = existsSync(appConfig.data.vectorStorePath);
      const lexical = await LexicalStore.load(appConfig.data.lexicalStorePath);

      let vector: LocalVectorStore | null = null;
      if (vectorExists) {
        try {
          vector = await LocalVectorStore.load(appConfig.data.vectorStorePath);
          if (!vector.isUsable()) {
            if (!allowDenseMissing) {
              throw new Error("Vector index appears invalid or empty.");
            }
            vector = null;
            console.warn("Dense vector store loaded but failed validation. Falling back to lexical-only retrieval.");
          }
        } catch {
          if (!allowDenseMissing) {
            throw new Error("Vector index exists but failed to load. Re-run ingestion.");
          }
          console.warn("Dense vector store unavailable. Falling back to lexical-only retrieval.");
        }
      } else if (!allowDenseMissing) {
        throw new Error("Vector index missing. Run ingestion first.");
      } else {
        console.warn("Vector index missing. Falling back to lexical-only retrieval.");
      }

      const vectorAvailable = vector !== null && vector.isUsable();

      let chunks: ChunkStore;
      if (existsSync(appConfig.data.chunksPath)) {
        chunks = await ChunkStore.load(appConfig.data.chunksPath);
      } else {
        // Fallback: build chunk store from available lexical chunks if chunks.json is missing
        chunks = ChunkStore.fromChunks(lexical.toJSON().chunks);
      }

      return {
        lexical,
        vector,
        chunks,
        vectorAvailable,
      };
    })().catch((error) => {
      initialized = null;
      throw error;
    });
  }
  return initialized;
};

export const clearIndexStore = () => {
  initialized = null;
};
