import { mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { basename, dirname, isAbsolute, resolve } from "node:path";
import { appConfig } from "../lib/config";
import { getEmbeddingModel } from "../lib/llm";
import { CorpusChunk } from "../types/rag";
import { buildChunksFromPdf } from "./pdfLoader";
import { buildLexicalStore } from "../retrieval/lexical";
import { LocalVectorStore } from "../retrieval/vectorStore";
import { ChunkStore } from "../retrieval/chunkStore";

export interface IngestResult {
  sourceFiles: string[];
  totalChunks: number;
  totalDocuments: number;
  vectorStorePath: string;
  lexicalStorePath: string;
  chunksPath: string;
  vectorStoreBuilt: boolean;
  vectorStoreError?: string;
}

const resolveCorpusFile = (rawSourcePath: string, configuredCorpusPath: string): string | null => {
  const trimmed = rawSourcePath.trim();
  if (!trimmed) return null;

  const candidate = isAbsolute(trimmed) ? trimmed : resolve(configuredCorpusPath, trimmed);
  return existsSync(candidate) ? candidate : null;
};

export const ingestCorpus = async (): Promise<IngestResult> => {
  const allChunks: CorpusChunk[] = [];
  const sourceFiles: string[] = [];
  const configuredChunkSize = Number(process.env.INGEST_CHUNK_SIZE ?? NaN);
  const configuredChunkOverlap = Number(process.env.INGEST_CHUNK_OVERLAP ?? NaN);

  await mkdir(dirname(appConfig.data.vectorStorePath), { recursive: true });
  await mkdir(dirname(appConfig.data.lexicalStorePath), { recursive: true });
  await mkdir(dirname(appConfig.data.chunksPath), { recursive: true });

  for (const doc of appConfig.corpus) {
    const candidatePath = resolveCorpusFile(doc.source_file, appConfig.data.corpusPath);
    const filePath = candidatePath;
    if (!filePath) {
      throw new Error(`Missing document: ${doc.source_file}. Expected file in corpus path: ${appConfig.data.corpusPath}.`);
    }

    const chunks = await buildChunksFromPdf({
      filePath,
      sourceFile: doc.source_file,
      documentId: doc.document_id,
      sectionAwareOptions: {
        ...(Number.isFinite(configuredChunkSize) && configuredChunkSize > 50
          ? { chunkSize: configuredChunkSize }
          : {}),
        ...(Number.isFinite(configuredChunkOverlap) && configuredChunkOverlap >= 0
          ? { chunkOverlap: configuredChunkOverlap }
          : {}),
      },
    });

    allChunks.push(...chunks);
    sourceFiles.push(doc.source_file);
  }

  if (!allChunks.length) {
    throw new Error("No chunks were produced. Check PDFs and parsing quality.");
  }

  const embedder = getEmbeddingModel();
  let vectorStoreBuilt = false;
  let vectorStoreError: string | undefined;
  try {
    await LocalVectorStore.build(appConfig.data.vectorStorePath, allChunks, embedder);
    vectorStoreBuilt = true;
  } catch (error) {
    vectorStoreError = error instanceof Error ? error.message : String(error);
    console.warn(`Vector embedding/build step failed: ${vectorStoreError}`);
    console.warn("Continuing with lexical index only. Querying will fallback to lexical retrieval.");
  }

  await buildLexicalStore(allChunks, appConfig.data.lexicalStorePath);

  const chunkStore = ChunkStore.fromChunks(allChunks);
  await chunkStore.save(appConfig.data.chunksPath);

  return {
    sourceFiles,
    totalChunks: allChunks.length,
    totalDocuments: appConfig.corpus.length,
    vectorStorePath: appConfig.data.vectorStorePath,
    lexicalStorePath: appConfig.data.lexicalStorePath,
    chunksPath: appConfig.data.chunksPath,
    vectorStoreBuilt,
    vectorStoreError,
  };
};
