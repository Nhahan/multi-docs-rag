/**
 * Citation extraction and formatting for RAG retrieval results.
 *
 * Parses source document metadata (filename, document id, page number)
 * from retrieval chunks and formats them into standardised citation strings:
 *
 * - [<doc-id-or-file> p.<page>]
 *
 * The document prefix is derived from source metadata and is stable across
 * documents while remaining domain-agnostic.
 *
 * Also provides utilities to:
 *   - Extract cited references from generated answer text
 *   - Validate citations against the set of retrieved evidence
 *   - Deduplicate and sort citation lists
 */

import { ScoredChunk, SourceMetadata } from "../types/rag";

/* ------------------------------------------------------------------ */
/*  Citation helpers                                                   */
/* ------------------------------------------------------------------ */

const isScoredChunk = (entry: unknown): entry is ScoredChunk =>
  !!entry &&
  typeof entry === "object" &&
  !Array.isArray(entry) &&
  typeof (entry as ScoredChunk).chunk === "object" &&
  (entry as ScoredChunk).chunk !== null &&
  typeof (entry as ScoredChunk).chunk.id === "string" &&
  (entry as ScoredChunk).chunk.metadata !== undefined &&
  typeof (entry as ScoredChunk).chunk.metadata === "object";

const getChunkMetadata = (entry: ScoredChunk): SourceMetadata | null => {
  const metadata = entry.chunk?.metadata;
  if (!metadata) return null;
  return metadata;
};

const stripFileExtension = (value: string): string => value.replace(/\.[^.]+$/u, "").trim();

const normalizeCitationPrefix = (value: string): string => {
  const normalized = value.trim().toLowerCase();
  const compact = stripFileExtension(normalized)
    .replace(/[\s/\\]+/gu, "-")
    .replace(/[^a-z0-9._-]+/gu, "-")
    .replace(/-+/gu, "-")
    .replace(/^-+|-+$/gu, "");
  return compact;
};

const resolveCitationPrefix = (metadata: Pick<SourceMetadata, "document_id" | "source_file">): string => {
  const fromDocumentId = normalizeCitationPrefix(metadata.document_id);
  const fromSourceFile = normalizeCitationPrefix(metadata.source_file);
  const fallback =
    fromDocumentId.length > 0 ? fromDocumentId : fromSourceFile;
  if (!fallback || fallback.length === 0) {
    throw new Error("Cannot resolve citation prefix: document_id and source_file are empty.");
  }
  return fallback;
};

/** Regex that matches any well-formed citation tag in answer text. */
export const CITATION_REGEX = /\[([^\[\]\r\n]+?) p\.(\d+)\]/g;
const BRACKETED_TEXT_REGEX = /\[([^\[\]\r\n]+)\]/g;

/* ------------------------------------------------------------------ */
/*  Core formatting helpers                                            */
/* ------------------------------------------------------------------ */

/**
 * Build the inner citation text (without brackets) from chunk metadata.
 *
 * @example citationForMetadata({ document_id: "doc-alpha", page: 5, ... }) // "doc-alpha p.5"
 * @example citationForMetadata({ document_id: "doc-beta", page: 12, ... }) // "doc-beta p.12"
 */
export const citationForMetadata = (metadata: SourceMetadata): string => {
  const prefix = resolveCitationPrefix(metadata);
  return `${prefix} p.${metadata.page}`;
};

/**
 * Build the full bracketed citation label from chunk metadata.
 *
 * @example citationLabel({ document_id: "doc-alpha", page: 5, ... }) // "[doc-alpha p.5]"
 */
export const citationLabel = (metadata: SourceMetadata): string =>
  `[${citationForMetadata(metadata)}]`;

const formatChunkContextLine = (entry: ScoredChunk): string => {
  const label = citationLabel(entry.chunk.metadata);
  const sectionTitle = entry.chunk.metadata.section_title?.trim();
  return sectionTitle ? `${label} ${sectionTitle} — ${entry.chunk.text}` : `${label} ${entry.chunk.text}`;
};

const formatPlainContextLine = (entry: ScoredChunk): string => {
  const sectionTitle = entry.chunk.metadata.section_title?.trim();
  const documentId = entry.chunk.metadata.document_id?.trim();
  if (sectionTitle && documentId) {
    return `${documentId} ${sectionTitle} — ${entry.chunk.text}`.trim();
  }
  if (sectionTitle) {
    return `${sectionTitle} — ${entry.chunk.text}`.trim();
  }
  if (documentId) {
    return `${documentId} — ${entry.chunk.text}`.trim();
  }
  return entry.chunk.text.trim();
};

/* ------------------------------------------------------------------ */
/*  Extraction from retrieval results                                  */
/* ------------------------------------------------------------------ */

/**
 * Extract unique, sorted citation labels from an array of scored chunks.
 *
 * Returns deduplicated citation strings in the order they first appear,
 * preserving the ranking order from retrieval.
 */
export const extractCitationsFromChunks = (
  chunks: ScoredChunk[],
): string[] => {
  const seen = new Set<string>();
  const citations: string[] = [];

  for (const rawEntry of chunks) {
    if (!isScoredChunk(rawEntry)) continue;
    const metadata = getChunkMetadata(rawEntry);
    if (!metadata) continue;

    const label = citationLabel(metadata);
    if (!seen.has(label)) {
      seen.add(label);
      citations.push(label);
    }
  }

  return citations;
};

/**
 * Build a context string suitable for prompt injection from scored chunks.
 *
 * Each block is prefixed with its citation label so the LLM can reference
 * the correct source in its generated answer.
 *
 * @returns Formatted context with citation-prefixed text blocks.
 */
export const buildCitedContext = (chunks: ScoredChunk[]): string =>
  chunks
    .filter(isScoredChunk)
    .filter((entry) => typeof entry.chunk.text === "string")
    .map((entry) => formatChunkContextLine(entry))
    .join("\n\n");

export const buildEvidenceContext = (chunks: ScoredChunk[]): string =>
  chunks
    .filter(isScoredChunk)
    .filter((entry) => typeof entry.chunk.text === "string")
    .map((entry) => formatPlainContextLine(entry))
    .join("\n\n");

/* ------------------------------------------------------------------ */
/*  Extraction from generated answer text                              */
/* ------------------------------------------------------------------ */

/** A parsed citation reference found in answer text. */
export interface ParsedCitation {
  /** The full match including brackets, e.g. "[doc-alpha p.5]" */
  full: string;
  /** The citation prefix, e.g. "doc-alpha" or "doc-beta" */
  prefix: string;
  /** The page number */
  page: number;
}

/**
 * Parse all citation references from a generated answer string.
 *
 * @returns Array of parsed citations in the order they appear.
 */
export const parseCitationsFromAnswer = (answer: string): ParsedCitation[] => {
  const results: ParsedCitation[] = [];
  // Reset global regex state
  const regex = new RegExp(CITATION_REGEX.source, "g");
  let match: RegExpExecArray | null;

  while ((match = regex.exec(answer)) !== null) {
    results.push({
      full: match[0],
      prefix: match[1],
      page: parseInt(match[2], 10),
    });
  }

  return results;
};

/**
 * Extract unique citation strings from a generated answer.
 *
 * @returns Deduplicated citation strings in order of first appearance.
 */
export const extractCitationsFromAnswer = (answer: string): string[] => {
  const parsed = parseCitationsFromAnswer(answer);
  const seen = new Set<string>();
  const unique: string[] = [];

  for (const { full } of parsed) {
    if (!seen.has(full)) {
      seen.add(full);
      unique.push(full);
    }
  }

  return unique;
};

/* ------------------------------------------------------------------ */
/*  Validation                                                         */
/* ------------------------------------------------------------------ */

/**
 * Check whether all citations in the answer are supported by the
 * set of allowed (retrieved evidence) citations.
 *
 * @param answer  - The generated answer text
 * @param allowed - Set of allowed citation strings (e.g. from extractCitationsFromChunks)
 * @returns true if every cited reference is in the allowed set (and at least one exists)
 */
export const allCitationsSupported = (
  answer: string,
  allowed: Set<string>,
): boolean => {
  const cited = extractCitationsFromAnswer(answer);
  if (cited.length === 0) return false;
  return cited.every((c) => allowed.has(c));
};

/**
 * Find any citations in the answer that are NOT in the allowed set.
 *
 * @returns Array of unsupported citation strings (empty if all are valid).
 */
export const findUnsupportedCitations = (
  answer: string,
  allowed: Set<string>,
): string[] => {
  const cited = extractCitationsFromAnswer(answer);
  return cited.filter((c) => !allowed.has(c));
};

/**
 * Find bracketed tags that are not valid supported citations.
 *
 * Square brackets are reserved for exact citation labels in this project, so
 * raw filenames, titles, shorthand page references, and malformed citation
 * variants are all rejected here.
 */
export const findInvalidCitationLikeTags = (
  answer: string,
  allowed: Set<string>,
): string[] => {
  const matches = answer.match(new RegExp(BRACKETED_TEXT_REGEX.source, "g")) ?? [];
  return matches.filter((tag) => !allowed.has(tag));
};

/**
 * Check if the answer contains at least one citation tag.
 */
export const hasCitations = (answer: string): boolean => {
  const regex = new RegExp(CITATION_REGEX.source);
  return regex.test(answer);
};

/* ------------------------------------------------------------------ */
/*  Summary helpers                                                    */
/* ------------------------------------------------------------------ */

/**
 * Produce a compact summary of top retrieval results with scores and citations.
 *
 * Used for trace / debug output.
 *
 * @param chunks - Scored chunks (usually the top-k slice)
 * @param limit  - Max entries to include (default 3)
 */
export const summarizeCitations = (
  chunks: ScoredChunk[],
  limit = 3,
): {
  uniqueSources: string[];
  sourceCount: number;
  top: string;
  context: string;
} => {
  const validChunks = chunks.filter(isScoredChunk).slice(0, Math.max(limit, 0));
  const topChunks = validChunks.slice(0, limit);
  const uniqueSources = extractCitationsFromChunks(topChunks);
  const top = topChunks
    .map(
      (entry, i) => `${i + 1}. [${Number(entry.score ?? 0).toFixed(3)}] ${citationLabel(entry.chunk.metadata)}`,
    )
    .join("\n");
  const context = buildCitedContext(topChunks);

  return {
    uniqueSources,
    sourceCount: chunks.length,
    top,
    context,
  };
};
