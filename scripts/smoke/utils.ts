import { appConfig } from "../../src/lib/config";
import { CorpusDocConfig, ScoredChunk } from "../../src/types/rag";

export interface ResolvedCorpusDoc extends CorpusDocConfig {}

export const normalizeCitationPrefix = (value: string): string => {
  const trimmed = value.trim().toLowerCase();
  const compact = trimmed
    .replace(/\.[^.]+$/u, "")
    .replace(/[\s/\\]+/gu, "-")
    .replace(/[^a-z0-9._-]+/gu, "-")
    .replace(/-+/gu, "-")
    .replace(/^-+|-+$/gu, "");
  return compact.length > 0 ? compact : "doc";
};

export const resolveCitationPrefix = (doc: { document_id: string; source_file: string }): string => {
  return normalizeCitationPrefix(doc.document_id || doc.source_file);
};

export const buildCitationRegex = (prefix: string): RegExp => {
  const escaped = prefix.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`\\[${escaped} p\\.\\d+\\]`, "g");
};

export const getCorpusDocByIndex = (index: number): ResolvedCorpusDoc | undefined =>
  appConfig.corpus[index];

export const findCorpusDocBySourceFile = (
  sourceFile: string,
  fallbackToIndex = 0,
): ResolvedCorpusDoc | undefined => {
  const normalizedTarget = sourceFile.trim().toLowerCase();
  const exact = appConfig.corpus.find(
    (doc) => doc.source_file.trim().toLowerCase() === normalizedTarget,
  );
  if (exact) return exact;
  return appConfig.corpus[fallbackToIndex];
};

export const getDistinctDocPrefixes = (chunks: ScoredChunk[]): string[] => {
  const ordered: string[] = [];
  const seen = new Set<string>();

  for (const item of chunks) {
    const metadata = item?.chunk?.metadata;
    if (!metadata) continue;
    const prefix = resolveCitationPrefix(metadata);
    if (!seen.has(prefix)) {
      seen.add(prefix);
      ordered.push(prefix);
    }
  }

  return ordered;
};

export const findChunkByDocPrefix = (chunks: ScoredChunk[], prefix: string): ScoredChunk | undefined =>
  chunks.find((entry) => {
    const metadata = entry?.chunk?.metadata;
    if (!metadata) return false;
    return resolveCitationPrefix(metadata) === prefix;
  });
