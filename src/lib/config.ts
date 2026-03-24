if (typeof process.loadEnvFile === "function") {
  process.loadEnvFile();
}

import { basename, resolve } from "node:path";
import { existsSync, readdirSync } from "node:fs";
import { CorpusDocConfig } from "../types/rag";

const clampToPositiveInt = (raw: string | undefined, fallback: number): number => {
  const parsed = Number(raw ?? NaN);
  return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : fallback;
};

const clampToPositiveFloatRange = (
  raw: string | undefined,
  fallback: number,
  min: number,
  max: number,
): number => {
  const parsed = Number(raw ?? NaN);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, parsed));
};

const clampToZeroOne = (raw: string | undefined, fallback: number): number =>
  clampToPositiveFloatRange(raw, fallback, 0, 1);

const parseBooleanEnv = (value: string | undefined, defaultValue: boolean) => {
  if (value === undefined) return defaultValue;
  const normalized = value.trim().toLowerCase();
  if (["1", "true", "yes", "on", "y"].includes(normalized)) return true;
  if (["0", "false", "no", "off", "n"].includes(normalized)) return false;
  return defaultValue;
};

const deriveDocumentId = (sourceFile: string, seen: Set<string>): string => {
  const base = basename(sourceFile)
    .replace(/\.[^.]+$/u, "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/gu, "-")
    .replace(/^-+|-+$/gu, "");

  const sanitized = base.length > 0 ? base : "document";
  if (!seen.has(sanitized)) {
    seen.add(sanitized);
    return sanitized;
  }

  let suffix = 2;
  while (seen.has(`${sanitized}-${suffix}`)) {
    suffix += 1;
  }
  const uniqueId = `${sanitized}-${suffix}`;
  seen.add(uniqueId);
  return uniqueId;
};

const parseCorpusSourceFiles = (): string[] => {
  const raw =
    process.env.CORPUS_SOURCE_FILES ??
    process.env.CORPUS_FILES ??
    "";

  if (!raw.trim()) {
    return [];
  }

  return raw
    .split(/[,\n;]+/u)
    .map((value) => value.trim())
    .filter((value) => value.length > 0);
};

const discoverPdfFiles = (searchRoots: string[]): string[] => {
  const discovered: string[] = [];
  for (const root of searchRoots) {
    if (!existsSync(root)) {
      continue;
    }

    const files = readdirSync(root)
      .map((name) => name.trim())
      .filter((name) => name.toLowerCase().endsWith(".pdf"));
    for (const name of files) {
      if (!discovered.includes(name)) {
        discovered.push(name);
      }
    }
  }

  discovered.sort((a, b) => a.localeCompare(b));
  return discovered;
};

const parseCorpusConfigFromEnv = (): CorpusDocConfig[] | null => {
  const raw = process.env.CORPUS_CONFIG_JSON;
  if (!raw) return null;

  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return null;

    const documents: CorpusDocConfig[] = [];
    const seenIds = new Set<string>();
    for (const item of parsed) {
      if (!item || typeof item !== "object") continue;
      const candidate = item as Partial<CorpusDocConfig>;
      if (typeof candidate.source_file !== "string" || candidate.source_file.trim().length === 0) {
        continue;
      }

      const requestedId =
        typeof candidate.document_id === "string" && candidate.document_id.trim().length > 0
          ? candidate.document_id.trim()
          : candidate.source_file.trim();
      documents.push({
        source_file: candidate.source_file.trim(),
        document_id: deriveDocumentId(requestedId, seenIds),
      });
    }

    return documents.length > 0 ? documents : null;
  } catch {
    return null;
  }
};

const buildCorpusFromSourceFiles = (files: string[]): CorpusDocConfig[] => {
  const uniqueSourceFiles = [...new Set(files.map((file) => file.trim()).filter(Boolean))];
  const seenIds = new Set<string>();

  return uniqueSourceFiles.map((source_file) => ({
    source_file,
    document_id: deriveDocumentId(source_file, seenIds),
  }));
};

const resolveCorpusFileList = (): CorpusDocConfig[] | null => {
  const envFileList = parseCorpusSourceFiles();
  if (envFileList.length > 0) {
    return buildCorpusFromSourceFiles(envFileList);
  }

  const corpusPath = resolve(process.cwd(), process.env.CORPUS_PATH ?? "data");
  const candidates = discoverPdfFiles([corpusPath, resolve(corpusPath, "docs")]);
  if (candidates.length === 0) {
    return null;
  }

  return buildCorpusFromSourceFiles(candidates);
};

  const defaultCorpus: CorpusDocConfig[] | null = resolveCorpusFileList();
const corpus = parseCorpusConfigFromEnv() ?? defaultCorpus ?? [];

export const appConfig = {
  models: {
    baseUrl: process.env.OLLAMA_BASE_URL ?? "http://localhost:11434",
    chatModel:
      process.env.OLLAMA_CHAT_MODEL ??
      "unsloth_Qwen3.5-9B-UD-Q4_K_XL:latest",
    embeddingModel: process.env.OLLAMA_EMBEDDING_MODEL ?? "qwen3-embedding:4b",
    rerankModel: process.env.OLLAMA_RERANK_MODEL ?? process.env.OLLAMA_EMBEDDING_MODEL ?? "qwen3-embedding:4b",
  },
  retrieval: {
    topK: Number(process.env.RETRIEVAL_TOP_K ?? 6),
    candidateK: Number(process.env.RETRIEVAL_CANDIDATE_K ?? 20),
    allowDenseFallback:
      (process.env.RETRIEVAL_ALLOW_DENSE_FALLBACK ?? "true").toLowerCase() !== "false",
    rerankTopK: Number(process.env.RETRIEVAL_RERANK_TOP_K ?? 8),
    enableRerank: (process.env.RETRIEVAL_ENABLE_RERANK ?? "true").toLowerCase() !== "false",
  },
  runtime: {
    ollamaRequestTimeoutMs: Number(process.env.OLLAMA_REQUEST_TIMEOUT_MS ?? 120000),
    ollamaChatThink: parseBooleanEnv(
      process.env.OLLAMA_CHAT_THINK ?? process.env.OLLAMA_THINK,
      false,
    ),
    ollamaChatTemperature: clampToPositiveFloatRange(
      process.env.OLLAMA_CHAT_TEMPERATURE,
      0.0,
      0,
      2,
    ),
  },
  messages: {
    insufficientEvidence:
      process.env.OLLAMA_INSUFFICIENT_EVIDENCE_MESSAGE ??
      "I cannot answer from the indexed evidence.",
  },
  data: {
    corpusPath: resolve(process.cwd(), process.env.CORPUS_PATH ?? "data"),
    chunksPath: resolve(process.cwd(), process.env.CHUNKS_PATH ?? "data/index/chunks.json"),
    vectorStorePath: resolve(process.cwd(), process.env.VECTOR_STORE_PATH ?? "data/index/vector-store.json"),
    lexicalStorePath: resolve(process.cwd(), process.env.LEXICAL_INDEX_PATH ?? "data/index/lexical-index.json"),
  },
  corpus,
} as const;
