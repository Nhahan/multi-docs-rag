/**
 * Cross-document answer synthesis.
 *
 * Groups retrieved chunks by source document, builds structured context
 * that highlights document boundaries, and generates synthesis prompts
 * that instruct the LLM to combine and compare information across documents.
 */

import { citationLabel } from "../citations/citationExtractor";
import { ScoredChunk } from "../types/rag";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

/** A group of chunks from a single source document. */
export interface DocumentGroup {
  documentId: string;
  sourceFile: string;
  chunks: ScoredChunk[];
  /** Human-readable label for the document, derived from document metadata. */
  label: string;
}

/** Result of grouping and analysing chunks across documents. */
export interface CrossDocAnalysis {
  /** Chunk groups keyed by document_id */
  groups: DocumentGroup[];
  /** Total number of unique documents represented */
  documentCount: number;
  /** Whether chunks span multiple documents */
  isMultiDocument: boolean;
  /** Structured context string for prompt injection */
  structuredContext: string;
  /** Synthesis instruction tailored to the document mix */
  synthesisInstruction: string;
}

const formatDocumentLabel = (docId: string, sourceFile: string): string =>
  sourceFile || docId;

const isValidScoredChunk = (sc: ScoredChunk | undefined | null): sc is ScoredChunk =>
  !!sc &&
  !!sc.chunk &&
  !!sc.chunk.metadata &&
  typeof sc.chunk.metadata.document_id === "string" &&
  typeof sc.chunk.metadata.source_file === "string";

/* ------------------------------------------------------------------ */
/*  Grouping                                                           */
/* ------------------------------------------------------------------ */

/**
 * Group scored chunks by their source document, preserving retrieval order
 * within each group.
 */
export const groupChunksByDocument = (
  chunks: ScoredChunk[],
): DocumentGroup[] => {
  const groupMap = new Map<string, DocumentGroup>();

  for (const sc of chunks.filter(isValidScoredChunk)) {
    const { document_id, source_file } = sc.chunk.metadata;
    let group = groupMap.get(document_id);
    if (!group) {
      group = {
        documentId: document_id,
        sourceFile: source_file,
        chunks: [],
        label: formatDocumentLabel(document_id, source_file),
      };
      groupMap.set(document_id, group);
    }
    group.chunks.push(sc);
  }

  // Sort groups by highest-scoring chunk (best evidence first)
  return [...groupMap.values()].sort(
    (a, b) => (b.chunks[0]?.score ?? 0) - (a.chunks[0]?.score ?? 0),
  );
};

/* ------------------------------------------------------------------ */
/*  Structured context                                                 */
/* ------------------------------------------------------------------ */

/**
 * Build a structured context string that clearly separates evidence by
 * source document, making it easier for the LLM to attribute and compare.
 */
export const buildStructuredContext = (groups: DocumentGroup[]): string => {
  const sections: string[] = [];

  for (const group of groups) {
    const header = `=== Source: ${group.label} ===`;
    const chunkTexts = group.chunks.map(
      (sc) =>
        isValidScoredChunk(sc) && typeof sc.chunk.text === "string"
          ? `${citationLabel(sc.chunk.metadata)} ${sc.chunk.metadata.section_title ?? ""} ${sc.chunk.text}`.trim()
          : "[missing metadata]",
    );
    sections.push([header, ...chunkTexts].join("\n\n"));
  }

  return sections.join("\n\n---\n\n");
};

/* ------------------------------------------------------------------ */
/*  Synthesis instructions                                             */
/* ------------------------------------------------------------------ */

/**
 * Generate a synthesis instruction based on the document mix.
 *
 * For multi-document queries, the instruction guides the LLM to:
 * - Identify overlapping/complementary information
 * - Note areas of agreement and divergence
 * - Attribute each claim to its source
 * - Produce a coherent merged answer
 */
const buildEvidenceInstruction = (groups: DocumentGroup[]): string => {
  const docNames = groups.map((g) => g.label).join(" and ");
  const docCount = groups.length;

  return [
    `You are answering from ${docCount} source document group(s): ${docNames}.`,
    "",
    "Synthesis guidelines:",
    "1. **Compare**: Where documents address the same topic, note agreements and any differences.",
    "2. **Combine**: Merge complementary information into a unified answer — do not just list each document separately.",
    "3. **Attribute**: Every factual claim must cite its source using the citation labels shown in context.",
    "4. **Structure**: Organize the answer by topic or theme with explicit source references.",
    "5. **Stay on asked fields**: Answer the user's requested sub-questions only; do not add adjacent metrics or controls unless the user asked for them.",
    "6. **No metric substitution**: Do not answer a latency request with capacity evidence, or a capacity request with latency evidence. If the requested field is not supported, say so.",
  ].join("\n");
};

/* ------------------------------------------------------------------ */
/*  Main analysis entry point                                          */
/* ------------------------------------------------------------------ */

/**
 * Analyse a set of scored chunks for cross-document synthesis.
 *
 * Returns grouped context, structured prompt fragments, and metadata
 * about the document coverage.
 */
export const analyseCrossDocEvidence = (
  chunks: ScoredChunk[],
): CrossDocAnalysis => {
  const validChunks = chunks.filter(isValidScoredChunk);
  const groups = groupChunksByDocument(validChunks);
  const isMultiDocument = groups.length > 1;

  const structuredContext = buildStructuredContext(groups);
  const synthesisInstruction = groups.length > 0
    ? buildEvidenceInstruction(groups)
    : "No evidence documents available.";

  return {
    groups,
    documentCount: groups.length,
    isMultiDocument,
    structuredContext,
    synthesisInstruction,
  };
};

/* ------------------------------------------------------------------ */
/*  Coverage summary for tracing                                       */
/* ------------------------------------------------------------------ */

/**
 * Produce a concise summary of document coverage for trace/debug output.
 */
export const crossDocCoverageSummary = (
  analysis: CrossDocAnalysis,
): Record<string, unknown> => ({
  documentCount: analysis.documentCount,
  isMultiDocument: analysis.isMultiDocument,
  documents: analysis.groups.map((g) => ({
    id: g.documentId,
    type: g.documentType,
    chunkCount: g.chunks.length,
    topScore: g.chunks[0]?.score?.toFixed(4) ?? "N/A",
  })),
});
