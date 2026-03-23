import { describe, it, expect } from "vitest";
import {
  groupChunksByDocument,
  buildStructuredContext,
  analyseCrossDocEvidence,
  crossDocCoverageSummary,
  type DocumentGroup,
} from "../crossDocSynthesizer";
import { ScoredChunk } from "../../types/rag";

/* ------------------------------------------------------------------ */
/*  Fixtures                                                           */
/* ------------------------------------------------------------------ */

const makeScoredChunk = (
  docId: string,
  sourceFile: string,
  page: number,
  score: number,
  text: string,
): ScoredChunk => ({
  chunk: {
    id: `${docId}-p${page}-s0`,
    text,
    metadata: {
      source_file: sourceFile,
      document_id: docId,
      page,
      chunk_id: 0,
      section_title: null,
    },
  },
  score,
});

const primaryChunk1 = makeScoredChunk("doc-alpha", "Alpha_Document.pdf", 5, 0.9, "Section 91.103 requires...");
const primaryChunk2 = makeScoredChunk("doc-alpha", "Alpha_Document.pdf", 12, 0.75, "Appendix A specifies...");
const secondaryChunk1 = makeScoredChunk("doc-beta", "Beta_Document.pdf", 3, 0.85, "Persistent memory systems store...");
const secondaryChunk2 = makeScoredChunk("doc-beta", "Beta_Document.pdf", 7, 0.6, "Embedding-based retrieval...");

/* ------------------------------------------------------------------ */
/*  groupChunksByDocument                                              */
/* ------------------------------------------------------------------ */

describe("groupChunksByDocument", () => {
  it("groups chunks by document_id", () => {
    const groups = groupChunksByDocument([primaryChunk1, secondaryChunk1, primaryChunk2, secondaryChunk2]);
    expect(groups).toHaveLength(2);

    const docIds = groups.map((g) => g.documentId);
    expect(docIds).toContain("doc-alpha");
    expect(docIds).toContain("doc-beta");
  });

  it("preserves retrieval order within each group", () => {
    const groups = groupChunksByDocument([primaryChunk1, secondaryChunk1, primaryChunk2, secondaryChunk2]);
    const primaryGroup = groups.find((g) => g.documentId === "doc-alpha")!;
    expect(primaryGroup.chunks[0].chunk.metadata.page).toBe(5);
    expect(primaryGroup.chunks[1].chunk.metadata.page).toBe(12);
  });

  it("sorts groups by top score (highest first)", () => {
    const groups = groupChunksByDocument([secondaryChunk1, primaryChunk1]);
    // doc-alpha has 0.9, paper has 0.85 → doc-alpha first
    expect(groups[0].documentId).toBe("doc-alpha");
    expect(groups[1].documentId).toBe("doc-beta");
  });

  it("returns single group for single-document chunks", () => {
    const groups = groupChunksByDocument([primaryChunk1, primaryChunk2]);
    expect(groups).toHaveLength(1);
    expect(groups[0].documentId).toBe("doc-alpha");
    expect(groups[0].chunks).toHaveLength(2);
  });

  it("returns empty array for empty input", () => {
    expect(groupChunksByDocument([])).toEqual([]);
  });
});

/* ------------------------------------------------------------------ */
/*  buildStructuredContext                                              */
/* ------------------------------------------------------------------ */

describe("buildStructuredContext", () => {
  it("separates documents with headers and dividers", () => {
    const groups = groupChunksByDocument([primaryChunk1, secondaryChunk1]);
    const ctx = buildStructuredContext(groups);

    expect(ctx).toContain("=== Source:");
    expect(ctx).toContain("---");
    expect(ctx).toContain("[doc-alpha p.5]");
    expect(ctx).toContain("[doc-beta p.3]");
  });

  it("includes citation labels inline with chunk text", () => {
    const groups = groupChunksByDocument([primaryChunk1]);
    const ctx = buildStructuredContext(groups);

    expect(ctx).toContain("[doc-alpha p.5]  Section 91.103 requires...");
  });
});

/* ------------------------------------------------------------------ */
/*  analyseCrossDocEvidence                                            */
/* ------------------------------------------------------------------ */

describe("analyseCrossDocEvidence", () => {
  it("detects multi-document evidence", () => {
    const analysis = analyseCrossDocEvidence(
      [primaryChunk1, secondaryChunk1, primaryChunk2, secondaryChunk2],
    );

    expect(analysis.isMultiDocument).toBe(true);
    expect(analysis.documentCount).toBe(2);
    expect(analysis.groups).toHaveLength(2);
  });

  it("detects single-document evidence", () => {
    const analysis = analyseCrossDocEvidence([primaryChunk1, primaryChunk2]);

    expect(analysis.isMultiDocument).toBe(false);
    expect(analysis.documentCount).toBe(1);
  });

  it("generates multi-doc synthesis instruction for cross queries", () => {
    const analysis = analyseCrossDocEvidence(
      [primaryChunk1, secondaryChunk1],
    );

    expect(analysis.synthesisInstruction).toContain("You are answering from 2 source document group(s)");
    expect(analysis.synthesisInstruction).toContain("Compare");
    expect(analysis.synthesisInstruction).toContain("Combine");
    expect(analysis.synthesisInstruction).toContain("Attribute");
  });

  it("generates single-doc instruction for single-source evidence", () => {
    const analysis = analyseCrossDocEvidence([secondaryChunk1, secondaryChunk2]);

    expect(analysis.synthesisInstruction).toContain("You are answering from 1 source document group(s)");
  });

  it("includes structured context with document separation", () => {
    const analysis = analyseCrossDocEvidence(
      [primaryChunk1, secondaryChunk1],
    );

    expect(analysis.structuredContext).toContain("=== Source:");
    expect(analysis.structuredContext).toContain("[doc-alpha p.5]");
    expect(analysis.structuredContext).toContain("[doc-beta p.3]");
  });

  it("handles empty chunks gracefully", () => {
    const analysis = analyseCrossDocEvidence([]);

    expect(analysis.isMultiDocument).toBe(false);
    expect(analysis.documentCount).toBe(0);
    expect(analysis.synthesisInstruction).toBe("No evidence documents available.");
  });
});

/* ------------------------------------------------------------------ */
/*  crossDocCoverageSummary                                            */
/* ------------------------------------------------------------------ */

describe("crossDocCoverageSummary", () => {
  it("produces a summary with document counts and scores", () => {
    const analysis = analyseCrossDocEvidence(
      [primaryChunk1, secondaryChunk1, primaryChunk2],
    );
    const summary = crossDocCoverageSummary(analysis);

    expect(summary.documentCount).toBe(2);
    expect(summary.isMultiDocument).toBe(true);
    expect(Array.isArray(summary.documents)).toBe(true);
    const docs = summary.documents as Array<{ id: string; chunkCount: number }>;
    expect(docs).toHaveLength(2);

    const primaryDoc = docs.find((d) => d.id === "doc-alpha")!;
    expect(primaryDoc.chunkCount).toBe(2);
  });
});
