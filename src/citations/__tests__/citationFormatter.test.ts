/**
 * Tests for the citation formatting module (citationFormatter.ts).
 *
 * Validates:
 * - CitationEntry construction from metadata for multiple generic documents
 * - Building citation entries from scored chunks with deduplication
 * - Filtering entries to only those actually cited in an answer
 * - Citation footer generation
 * - Inline citation injection (ensureInlineCitations)
 * - Full formatCitationResponse integration for multiple documents
 * - Edge cases (empty chunks, refusal answers, mixed documents)
 */

import {
  buildCitationEntry,
  buildCitationEntries,
  filterCitedEntries,
  buildCitationFooter,
  ensureInlineCitations,
  formatCitationResponse,
  CitationEntry,
} from "../citationFormatter";
import { ScoredChunk, SourceMetadata, CorpusChunk } from "../../types/rag";

/* ------------------------------------------------------------------ */
/*  Test helpers                                                       */
/* ------------------------------------------------------------------ */

const makeMetadata = (overrides: Partial<SourceMetadata> = {}): SourceMetadata => ({
  source_file: "Alpha_Document.pdf",
  document_id: "doc-alpha",
  page: 1,
  chunk_id: 0,
  section_title: null,
  ...overrides,
});

const makeChunk = (
  id: string,
  text: string,
  meta: Partial<SourceMetadata> = {},
): CorpusChunk => ({
  id,
  text,
  metadata: makeMetadata(meta),
});

const makeScoredChunk = (
  id: string,
  text: string,
  score: number,
  meta: Partial<SourceMetadata> = {},
): ScoredChunk => ({
  chunk: makeChunk(id, text, meta),
  score,
});

/** Shorthand for a primary-document chunk */
const alphaChunk = (page: number, score: number, text = "Document alpha text") =>
  makeScoredChunk(`doc-alpha-p${page}`, text, score, {
    document_id: "doc-alpha",
    source_file: "Alpha_Document.pdf",
    page,
  });

/** Shorthand for a secondary-document chunk */
const betaChunk = (page: number, score: number, text = "Document beta text") =>
  makeScoredChunk(`doc-beta-p${page}`, text, score, {
    document_id: "doc-beta",
    source_file: "Beta_Document.pdf",
    page,
  });

/* ================================================================== */
/*  Tests: buildCitationEntry                                          */
/* ================================================================== */

describe("buildCitationEntry", () => {
  it("builds a primary-document citation entry from metadata", () => {
    const meta = makeMetadata({
      document_id: "doc-alpha",
      source_file: "Alpha_Document.pdf",
      page: 5,
    });
    const entry = buildCitationEntry(meta);
    expect(entry).toEqual({
      label: "[doc-alpha p.5]",
      document_id: "doc-alpha",
      page: 5,
      source_file: "Alpha_Document.pdf",
    });
  });

  it("builds a secondary-document citation entry from metadata", () => {
    const meta = makeMetadata({
      document_id: "doc-beta",
      source_file: "Beta_Document.pdf",
      page: 12,
    });
    const entry = buildCitationEntry(meta);
    expect(entry).toEqual({
      label: "[doc-beta p.12]",
      document_id: "doc-beta",
      page: 12,
      source_file: "Beta_Document.pdf",
    });
  });

  it("handles page 1", () => {
    const entry = buildCitationEntry(makeMetadata({ page: 1 }));
    expect(entry.label).toBe("[doc-alpha p.1]");
    expect(entry.page).toBe(1);
  });

  it("handles large page numbers", () => {
    const entry = buildCitationEntry(makeMetadata({ page: 999 }));
    expect(entry.label).toBe("[doc-alpha p.999]");
    expect(entry.page).toBe(999);
  });
});

/* ================================================================== */
/*  Tests: buildCitationEntries                                        */
/* ================================================================== */

describe("buildCitationEntries", () => {
  it("returns empty array for no chunks", () => {
    expect(buildCitationEntries([])).toEqual([]);
  });

  it("builds entries from primary-document chunks only", () => {
    const chunks = [alphaChunk(1, 0.9), alphaChunk(5, 0.8)];
    const entries = buildCitationEntries(chunks);
    expect(entries).toHaveLength(2);
    expect(entries[0].label).toBe("[doc-alpha p.1]");
    expect(entries[1].label).toBe("[doc-alpha p.5]");
  });

  it("builds entries from secondary-document chunks", () => {
    const chunks = [betaChunk(3, 0.9), betaChunk(8, 0.7)];
    const entries = buildCitationEntries(chunks);
    expect(entries).toHaveLength(2);
    expect(entries[0].label).toBe("[doc-beta p.3]");
    expect(entries[1].label).toBe("[doc-beta p.8]");
  });

  it("deduplicates entries from same page", () => {
    const chunks = [
      alphaChunk(1, 0.9),
      alphaChunk(1, 0.8), // same page
      alphaChunk(2, 0.7),
    ];
    const entries = buildCitationEntries(chunks);
    expect(entries).toHaveLength(2);
    expect(entries.map((e) => e.label)).toEqual(["[doc-alpha p.1]", "[doc-alpha p.2]"]);
  });

  it("builds mixed primary and secondary document entries preserving order", () => {
    const chunks = [
      alphaChunk(5, 0.95),
      betaChunk(3, 0.90),
      alphaChunk(12, 0.85),
      betaChunk(7, 0.80),
    ];
    const entries = buildCitationEntries(chunks);
    expect(entries).toHaveLength(4);
    expect(entries.map((e) => e.label)).toEqual([
      "[doc-alpha p.5]",
      "[doc-beta p.3]",
      "[doc-alpha p.12]",
      "[doc-beta p.7]",
    ]);
  });
});

/* ================================================================== */
/*  Tests: filterCitedEntries                                          */
/* ================================================================== */

describe("filterCitedEntries", () => {
  const allEntries: CitationEntry[] = [
    { label: "[doc-alpha p.1]", document_id: "doc-alpha", page: 1, source_file: "doc-alpha.pdf" },
    { label: "[doc-alpha p.5]", document_id: "doc-alpha", page: 5, source_file: "doc-alpha.pdf" },
    { label: "[doc-beta p.3]", document_id: "doc-beta", page: 3, source_file: "am.pdf" },
  ];

  it("filters to only cited entries", () => {
    const answer = "The regulation requires X [doc-alpha p.1] and Y.";
    const result = filterCitedEntries(answer, allEntries);
    expect(result).toHaveLength(1);
    expect(result[0].label).toBe("[doc-alpha p.1]");
  });

  it("returns all entries when all are cited", () => {
    const answer = "Rule [doc-alpha p.1] plus [doc-alpha p.5] and paper [doc-beta p.3].";
    const result = filterCitedEntries(answer, allEntries);
    expect(result).toHaveLength(3);
  });

  it("returns empty when no citations in answer", () => {
    const answer = "No citations present in this text.";
    const result = filterCitedEntries(answer, allEntries);
    expect(result).toHaveLength(0);
  });

  it("handles secondary-document citations correctly", () => {
    const answer = "According to [doc-beta p.3], the approach works.";
    const result = filterCitedEntries(answer, allEntries);
    expect(result).toHaveLength(1);
    expect(result[0].document_id).toBe("doc-beta");
  });
});

/* ================================================================== */
/*  Tests: buildCitationFooter                                         */
/* ================================================================== */

describe("buildCitationFooter", () => {
  it("returns empty string for no entries", () => {
    expect(buildCitationFooter([])).toBe("");
  });

  it("formats primary-document citation footer", () => {
    const entries: CitationEntry[] = [
      { label: "[doc-alpha p.5]", document_id: "doc-alpha", page: 5, source_file: "Alpha_Document.pdf" },
    ];
    const footer = buildCitationFooter(entries);
    expect(footer).toBe("Sources:\n[1] [doc-alpha p.5] — Alpha_Document.pdf, p.5");
  });

  it("formats secondary-document citation footer", () => {
    const entries: CitationEntry[] = [
      { label: "[doc-beta p.3]", document_id: "doc-beta", page: 3, source_file: "Beta_Document.pdf" },
    ];
    const footer = buildCitationFooter(entries);
    expect(footer).toBe(
      "Sources:\n[1] [doc-beta p.3] — Beta_Document.pdf, p.3",
    );
  });

  it("formats mixed document footer", () => {
    const entries: CitationEntry[] = [
      { label: "[doc-alpha p.5]", document_id: "doc-alpha", page: 5, source_file: "Alpha_Document.pdf" },
      { label: "[doc-beta p.3]", document_id: "doc-beta", page: 3, source_file: "Beta_Document.pdf" },
    ];
    const footer = buildCitationFooter(entries);
    expect(footer).toContain("Sources:");
    expect(footer).toContain("[1] [doc-alpha p.5] — Alpha_Document.pdf, p.5");
    expect(footer).toContain("[2] [doc-beta p.3] — Beta_Document.pdf, p.3");
  });
});

/* ================================================================== */
/*  Tests: ensureInlineCitations                                       */
/* ================================================================== */

describe("ensureInlineCitations", () => {
  it("returns answer as-is when it already has primary-document citations", () => {
    const answer = "The rule states X [doc-alpha p.5].";
    const chunks = [alphaChunk(5, 0.9)];
    expect(ensureInlineCitations(answer, chunks)).toBe(answer);
  });

  it("returns answer as-is when it already has secondary-document citations", () => {
    const answer = "The paper describes [doc-beta p.3] approach.";
    const chunks = [betaChunk(3, 0.9)];
    expect(ensureInlineCitations(answer, chunks)).toBe(answer);
  });

  it("appends citations when answer has none but chunks exist", () => {
    const answer = "The regulation requires documentation.";
    const chunks = [alphaChunk(5, 0.9), alphaChunk(12, 0.8)];
    const result = ensureInlineCitations(answer, chunks);
    expect(result).toContain("(Sources: [doc-alpha p.5], [doc-alpha p.12])");
    expect(result).toContain(answer);
  });

  it("appends mixed citations when answer has none", () => {
    const answer = "Both documents discuss memory approaches.";
    const chunks = [alphaChunk(5, 0.9), betaChunk(3, 0.8)];
    const result = ensureInlineCitations(answer, chunks);
    expect(result).toContain("[doc-alpha p.5]");
    expect(result).toContain("[doc-beta p.3]");
  });

  it("does not inject citations for refusal answers", () => {
    const answer = "I cannot answer from the indexed evidence.";
    const chunks = [alphaChunk(5, 0.9)];
    expect(ensureInlineCitations(answer, chunks)).toBe(answer);
  });

  it("returns answer as-is when no chunks available", () => {
    const answer = "Some text without evidence.";
    expect(ensureInlineCitations(answer, [])).toBe(answer);
  });
});

/* ================================================================== */
/*  Tests: formatCitationResponse — primary document                  */
/* ================================================================== */

describe("formatCitationResponse — primary document", () => {
  it("formats response with inline primary-document citations", () => {
    const answer = "Section 91.103 requires [doc-alpha p.5] preflight briefings [doc-alpha p.12].";
    const chunks = [alphaChunk(5, 0.9), alphaChunk(12, 0.8)];
    const result = formatCitationResponse(answer, chunks);

    expect(result.has_citations).toBe(true);
    expect(result.cited_answer).toBe(answer);
    expect(result.inline_citations).toEqual(["[doc-alpha p.5]", "[doc-alpha p.12]"]);
    expect(result.citation_entries).toHaveLength(2);
    expect(result.citation_entries[0]).toEqual(
      expect.objectContaining({
        label: "[doc-alpha p.5]",
        document_id: "doc-alpha",
        page: 5,
      }),
    );
    expect(result.citation_entries[1]).toEqual(
      expect.objectContaining({
        label: "[doc-alpha p.12]",
        page: 12,
      }),
    );
    expect(result.citation_footer).toContain("Alpha_Document.pdf");
    expect(result.citation_footer).toContain("[doc-alpha p.5]");
  });

  it("formats response for a single primary-document citation", () => {
    const answer = "The rule [doc-alpha p.1] defines requirements.";
    const chunks = [alphaChunk(1, 0.95)];
    const result = formatCitationResponse(answer, chunks);

    expect(result.has_citations).toBe(true);
    expect(result.inline_citations).toEqual(["[doc-alpha p.1]"]);
    expect(result.citation_entries).toHaveLength(1);
  });

  it("deduplicates repeated primary-document citations", () => {
    const answer = "Rule [doc-alpha p.5] applies. Also see [doc-alpha p.5] again.";
    const chunks = [alphaChunk(5, 0.9), alphaChunk(5, 0.8)];
    const result = formatCitationResponse(answer, chunks);

    expect(result.inline_citations).toEqual(["[doc-alpha p.5]"]);
    expect(result.citation_entries).toHaveLength(1);
  });
});

/* ================================================================== */
/*  Tests: formatCitationResponse — secondary document                */
/* ================================================================== */

describe("formatCitationResponse — secondary document", () => {
  it("formats response with inline secondary-document citations", () => {
    const answer =
      "The paper proposes [doc-beta p.3] a memory architecture [doc-beta p.8].";
    const chunks = [betaChunk(3, 0.9), betaChunk(8, 0.85)];
    const result = formatCitationResponse(answer, chunks);

    expect(result.has_citations).toBe(true);
    expect(result.cited_answer).toBe(answer);
    expect(result.inline_citations).toEqual([
      "[doc-beta p.3]",
      "[doc-beta p.8]",
    ]);
    expect(result.citation_entries).toHaveLength(2);
    expect(result.citation_entries[0]).toEqual(
      expect.objectContaining({
        label: "[doc-beta p.3]",
        document_id: "doc-beta",
        page: 3,
      }),
    );
    expect(result.citation_entries[1]).toEqual(
      expect.objectContaining({
        label: "[doc-beta p.8]",
        page: 8,
      }),
    );
    expect(result.citation_footer).toContain("Beta_Document.pdf");
  });

  it("formats response for a single secondary-document citation", () => {
    const answer = "According to [doc-beta p.1], the method works.";
    const chunks = [betaChunk(1, 0.9)];
    const result = formatCitationResponse(answer, chunks);

    expect(result.has_citations).toBe(true);
    expect(result.inline_citations).toEqual(["[doc-beta p.1]"]);
    expect(result.citation_entries).toHaveLength(1);
  });
});

/* ================================================================== */
/*  Tests: formatCitationResponse — mixed / cross-document             */
/* ================================================================== */

describe("formatCitationResponse — mixed document types", () => {
  it("formats response with both primary and secondary document citations", () => {
    const answer =
      "The regulation [doc-alpha p.5] and the paper [doc-beta p.3] both address memory.";
    const chunks = [alphaChunk(5, 0.9), betaChunk(3, 0.85)];
    const result = formatCitationResponse(answer, chunks);

    expect(result.has_citations).toBe(true);
    expect(result.inline_citations).toEqual(["[doc-alpha p.5]", "[doc-beta p.3]"]);
    expect(result.citation_entries).toHaveLength(2);
    expect(result.citation_entries[0].document_id).toBe("doc-alpha");
    expect(result.citation_entries[1].document_id).toBe("doc-beta");
    expect(result.citation_footer).toContain("Alpha_Document.pdf");
    expect(result.citation_footer).toContain("Beta_Document.pdf");
  });

  it("only includes cited entries (not all retrieved)", () => {
    const answer = "Only the paper [doc-beta p.3] is relevant here.";
    const chunks = [alphaChunk(5, 0.9), betaChunk(3, 0.85), alphaChunk(12, 0.7)];
    const result = formatCitationResponse(answer, chunks);

    expect(result.inline_citations).toEqual(["[doc-beta p.3]"]);
    expect(result.citation_entries).toHaveLength(1);
    expect(result.citation_entries[0].label).toBe("[doc-beta p.3]");
  });
});

/* ================================================================== */
/*  Tests: formatCitationResponse — edge cases                         */
/* ================================================================== */

describe("formatCitationResponse — edge cases", () => {
  it("handles empty answer with no chunks", () => {
    const result = formatCitationResponse("", []);
    expect(result.has_citations).toBe(false);
    expect(result.cited_answer).toBe("");
    expect(result.inline_citations).toEqual([]);
    expect(result.citation_entries).toEqual([]);
    expect(result.citation_footer).toBe("");
  });

  it("handles refusal answer with chunks", () => {
    const answer = "I cannot answer from the indexed evidence.";
    const chunks = [alphaChunk(5, 0.9)];
    const result = formatCitationResponse(answer, chunks);

    expect(result.has_citations).toBe(false);
    expect(result.cited_answer).toBe(answer);
    expect(result.inline_citations).toEqual([]);
    expect(result.citation_entries).toEqual([]);
  });

  it("injects citations when answer lacks them but chunks exist", () => {
    const answer = "The regulation defines pilot requirements.";
    const chunks = [alphaChunk(5, 0.9), betaChunk(3, 0.8)];
    const result = formatCitationResponse(answer, chunks);

    expect(result.has_citations).toBe(true);
    expect(result.cited_answer).toContain("(Sources:");
    expect(result.cited_answer).toContain("[doc-alpha p.5]");
    expect(result.cited_answer).toContain("[doc-beta p.3]");
    expect(result.inline_citations).toContain("[doc-alpha p.5]");
    expect(result.inline_citations).toContain("[doc-beta p.3]");
  });

  it("preserves answer when it has partial citations", () => {
    const answer = "Requirement [doc-alpha p.5] applies broadly.";
    const chunks = [alphaChunk(5, 0.9), alphaChunk(12, 0.8)];
    const result = formatCitationResponse(answer, chunks);

    // Answer already has citations so should not be modified
    expect(result.cited_answer).toBe(answer);
    // Only the cited one should be in entries (not the uncited chunk)
    expect(result.citation_entries).toHaveLength(1);
    expect(result.citation_entries[0].label).toBe("[doc-alpha p.5]");
  });
});

/* ================================================================== */
/*  Tests: Query API response shape simulation                         */
/* ================================================================== */

describe("Query API citation integration", () => {
  it("produces correct API response shape for a primary-document query", () => {
    const answer = "Per [doc-alpha p.5], pilots must complete preflight actions [doc-alpha p.12].";
    const chunks = [alphaChunk(5, 0.95), alphaChunk(12, 0.88)];
    const citationResponse = formatCitationResponse(answer, chunks);

    // Simulate the API response shape
    const apiResponse = {
      question: "What are preflight requirements?",
      answer: citationResponse.cited_answer,
      citations: citationResponse.inline_citations,
      citation_entries: citationResponse.citation_entries,
      citation_footer: citationResponse.citation_footer,
      has_citations: citationResponse.has_citations,
    };

    expect(apiResponse.answer).toContain("[doc-alpha p.5]");
    expect(apiResponse.answer).toContain("[doc-alpha p.12]");
    expect(apiResponse.citations).toEqual(["[doc-alpha p.5]", "[doc-alpha p.12]"]);
    expect(apiResponse.has_citations).toBe(true);
    expect(apiResponse.citation_entries).toHaveLength(2);
    expect(apiResponse.citation_entries.every((e) => e.document_id === "doc-alpha")).toBe(true);
    expect(apiResponse.citation_footer).toContain("Alpha_Document.pdf");
  });

  it("produces correct API response shape for a secondary-document query", () => {
    const answer =
      "The paper introduces [doc-beta p.1] a memory architecture " +
      "with retrieval mechanisms [doc-beta p.7].";
    const chunks = [betaChunk(1, 0.92), betaChunk(7, 0.86)];
    const citationResponse = formatCitationResponse(answer, chunks);

    const apiResponse = {
      question: "What is the main contribution of the secondary document?",
      answer: citationResponse.cited_answer,
      citations: citationResponse.inline_citations,
      citation_entries: citationResponse.citation_entries,
      citation_footer: citationResponse.citation_footer,
      has_citations: citationResponse.has_citations,
    };

    expect(apiResponse.answer).toContain("[doc-beta p.1]");
    expect(apiResponse.answer).toContain("[doc-beta p.7]");
    expect(apiResponse.citations).toEqual(["[doc-beta p.1]", "[doc-beta p.7]"]);
    expect(apiResponse.has_citations).toBe(true);
    expect(apiResponse.citation_entries).toHaveLength(2);
    expect(apiResponse.citation_entries.every((e) => e.document_id === "doc-beta")).toBe(true);
    expect(apiResponse.citation_footer).toContain("Beta_Document.pdf");
  });

  it("produces correct API response shape for cross-document query", () => {
    const answer =
      "The primary document defines [doc-alpha p.5] regulatory requirements while " +
      "the paper proposes [doc-beta p.3] memory systems.";
    const chunks = [alphaChunk(5, 0.9), betaChunk(3, 0.85)];
    const citationResponse = formatCitationResponse(answer, chunks);

    const apiResponse = {
      question: "Compare the two documents",
      answer: citationResponse.cited_answer,
      citations: citationResponse.inline_citations,
      citation_entries: citationResponse.citation_entries,
      citation_footer: citationResponse.citation_footer,
      has_citations: citationResponse.has_citations,
    };

    expect(apiResponse.citations).toEqual(["[doc-alpha p.5]", "[doc-beta p.3]"]);
    expect(apiResponse.citation_entries).toHaveLength(2);
    expect(apiResponse.citation_entries[0].document_id).toBe("doc-alpha");
    expect(apiResponse.citation_entries[1].document_id).toBe("doc-beta");
    expect(apiResponse.citation_footer).toContain("Alpha_Document.pdf");
    expect(apiResponse.citation_footer).toContain("Beta_Document.pdf");
  });

  it("produces correct API response shape for refusal", () => {
    const answer = "I cannot answer from the indexed evidence.";
    const citationResponse = formatCitationResponse(answer, []);

    const apiResponse = {
      question: "Something unanswerable",
      answer: citationResponse.cited_answer,
      citations: citationResponse.inline_citations,
      has_citations: citationResponse.has_citations,
      citation_entries: citationResponse.citation_entries,
    };

    expect(apiResponse.answer).toBe("I cannot answer from the indexed evidence.");
    expect(apiResponse.citations).toEqual([]);
    expect(apiResponse.has_citations).toBe(false);
    expect(apiResponse.citation_entries).toEqual([]);
  });
});
