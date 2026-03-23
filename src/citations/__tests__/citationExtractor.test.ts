/**
 * Tests for the citation extraction module.
 *
 * Validates:
 * - Citation formatting from SourceMetadata (document ids → [doc-id p.X])
 * - Extraction of unique citations from scored chunks
 * - Parsing citations from generated answer text
 * - Validation of citations against allowed evidence sets
 * - Edge cases (empty inputs, duplicates, mixed document ids)
 */

import {
  citationForMetadata,
  citationLabel,
  extractCitationsFromChunks,
  extractCitationsFromAnswer,
  parseCitationsFromAnswer,
  allCitationsSupported,
  findUnsupportedCitations,
  hasCitations,
  buildCitedContext,
  summarizeCitations,
  CITATION_REGEX,
} from "../citationExtractor";
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

/* ================================================================== */
/*  Tests: citationForMetadata / citationLabel                         */
/* ================================================================== */

describe("citationForMetadata", () => {
  it("formats primary document using document id as citation prefix", () => {
    const meta = makeMetadata({ page: 5 });
    expect(citationForMetadata(meta)).toBe("doc-alpha p.5");
  });

  it("formats secondary document using document id as citation prefix", () => {
    const meta = makeMetadata({
      source_file: "Beta_Document.pdf",
      document_id: "doc-beta",
      page: 12,
    });
    expect(citationForMetadata(meta)).toBe("doc-beta p.12");
  });

  it("handles page 1", () => {
    expect(citationForMetadata(makeMetadata({ page: 1 }))).toBe("doc-alpha p.1");
  });

  it("handles large page numbers", () => {
    expect(citationForMetadata(makeMetadata({ page: 999 }))).toBe("doc-alpha p.999");
  });
});

describe("citationLabel", () => {
  it("wraps primary citation in brackets", () => {
    const meta = makeMetadata({ page: 3 });
    expect(citationLabel(meta)).toBe("[doc-alpha p.3]");
  });

  it("wraps secondary citation in brackets", () => {
    const meta = makeMetadata({
      source_file: "Beta_Document.pdf",
      document_id: "doc-beta",
      page: 7,
    });
    expect(citationLabel(meta)).toBe("[doc-beta p.7]");
  });
});

/* ================================================================== */
/*  Tests: extractCitationsFromChunks                                  */
/* ================================================================== */

describe("extractCitationsFromChunks", () => {
  it("returns empty array for no chunks", () => {
    expect(extractCitationsFromChunks([])).toEqual([]);
  });

  it("extracts unique citations preserving order", () => {
    const chunks: ScoredChunk[] = [
      makeScoredChunk("c1", "text1", 0.9, { page: 1 }),
      makeScoredChunk("c2", "text2", 0.8, { page: 1 }),
      makeScoredChunk("c3", "text3", 0.7, { source_file: "Beta_Document.pdf", document_id: "doc-beta", page: 5 }),
    ];
    const result = extractCitationsFromChunks(chunks);
    expect(result).toEqual(["[doc-alpha p.1]", "[doc-beta p.5]"]);
  });

  it("handles mixed document ids", () => {
    const chunks: ScoredChunk[] = [
      makeScoredChunk("c1", "t1", 0.9, { page: 2 }),
      makeScoredChunk("c2", "t2", 0.8, { source_file: "Beta_Document.pdf", document_id: "doc-beta", page: 3 }),
      makeScoredChunk("c3", "t3", 0.7, { page: 4 }),
      makeScoredChunk("c4", "t4", 0.6, { source_file: "Beta_Document.pdf", document_id: "doc-beta", page: 3 }),
    ];
    const result = extractCitationsFromChunks(chunks);
    expect(result).toEqual(["[doc-alpha p.2]", "[doc-beta p.3]", "[doc-alpha p.4]"]);
  });
});

/* ================================================================== */
/*  Tests: parseCitationsFromAnswer / extractCitationsFromAnswer        */
/* ================================================================== */

describe("parseCitationsFromAnswer", () => {
  it("parses regulation-style citations", () => {
    const answer = "The policy states X [doc-alpha p.5] and Y [doc-alpha p.12].";
    const parsed = parseCitationsFromAnswer(answer);
    expect(parsed).toEqual([
      { full: "[doc-alpha p.5]", prefix: "doc-alpha", page: 5 },
      { full: "[doc-alpha p.12]", prefix: "doc-alpha", page: 12 },
    ]);
  });

  it("parses secondary-document citations", () => {
    const answer = "According to [doc-beta p.3], the method works.";
    const parsed = parseCitationsFromAnswer(answer);
    expect(parsed).toEqual([
      { full: "[doc-beta p.3]", prefix: "doc-beta", page: 3 },
    ]);
  });

  it("parses mixed citations", () => {
    const answer =
      "Document alpha says [doc-alpha p.1] while document beta notes [doc-beta p.7] and also [doc-alpha p.1].";
    const parsed = parseCitationsFromAnswer(answer);
    expect(parsed).toHaveLength(3);
    expect(parsed[0].prefix).toBe("doc-alpha");
    expect(parsed[1].prefix).toBe("doc-beta");
    expect(parsed[2].prefix).toBe("doc-alpha");
  });

  it("returns empty array for no citations", () => {
    expect(parseCitationsFromAnswer("No citations here.")).toEqual([]);
  });
});

describe("extractCitationsFromAnswer", () => {
  it("deduplicates citations", () => {
    const answer = "[doc-alpha p.1] some text [doc-alpha p.1] more text [doc-beta p.2]";
    const result = extractCitationsFromAnswer(answer);
    expect(result).toEqual(["[doc-alpha p.1]", "[doc-beta p.2]"]);
  });

  it("returns empty for no citations", () => {
    expect(extractCitationsFromAnswer("plain text")).toEqual([]);
  });
});

/* ================================================================== */
/*  Tests: validation functions                                        */
/* ================================================================== */

describe("allCitationsSupported", () => {
  const allowed = new Set(["[doc-alpha p.1]", "[doc-alpha p.2]", "[doc-beta p.5]"]);

  it("returns true when all citations are in allowed set", () => {
    expect(allCitationsSupported("Answer [doc-alpha p.1] and [doc-beta p.5].", allowed)).toBe(true);
  });

  it("returns false when answer has no citations", () => {
    expect(allCitationsSupported("No citations.", allowed)).toBe(false);
  });

  it("returns false when a citation is not in allowed set", () => {
    expect(allCitationsSupported("Answer [doc-alpha p.99].", allowed)).toBe(false);
  });

  it("returns false when mixed valid and invalid citations", () => {
    expect(allCitationsSupported("[doc-alpha p.1] and [doc-alpha p.99].", allowed)).toBe(false);
  });
});

describe("findUnsupportedCitations", () => {
  const allowed = new Set(["[doc-alpha p.1]", "[doc-beta p.3]"]);

  it("returns empty when all valid", () => {
    expect(findUnsupportedCitations("[doc-alpha p.1] text.", allowed)).toEqual([]);
  });

  it("returns unsupported citations", () => {
    const result = findUnsupportedCitations("[doc-alpha p.1] [doc-alpha p.99] [doc-beta p.50]", allowed);
    expect(result).toEqual(["[doc-alpha p.99]", "[doc-beta p.50]"]);
  });
});

describe("hasCitations", () => {
  it("returns true for primary-document citation", () => {
    expect(hasCitations("text [doc-alpha p.1] more")).toBe(true);
  });

  it("returns true for secondary-document citation", () => {
    expect(hasCitations("text [doc-beta p.10]")).toBe(true);
  });

  it("returns false for no citations", () => {
    expect(hasCitations("just plain text")).toBe(false);
  });

  it("returns false for malformed citations", () => {
    expect(hasCitations("[doc-alpha page 1]")).toBe(false);
    expect(hasCitations("[doc-alpha p.one]")).toBe(false);
  });
});

/* ================================================================== */
/*  Tests: buildCitedContext / summarizeCitations                       */
/* ================================================================== */

describe("buildCitedContext", () => {
  it("prefixes each chunk with its citation label", () => {
    const chunks: ScoredChunk[] = [
      makeScoredChunk("c1", "Rule text", 0.9, { page: 2 }),
      makeScoredChunk("c2", "Paper text", 0.8, { source_file: "Beta_Document.pdf", document_id: "doc-beta", page: 4 }),
    ];
    const result = buildCitedContext(chunks);
    expect(result).toBe("[doc-alpha p.2] Rule text\n\n[doc-beta p.4] Paper text");
  });

  it("returns empty string for no chunks", () => {
    expect(buildCitedContext([])).toBe("");
  });
});

describe("summarizeCitations", () => {
  it("returns summary with correct fields", () => {
    const chunks: ScoredChunk[] = [
      makeScoredChunk("c1", "text1", 0.95, { page: 1 }),
      makeScoredChunk("c2", "text2", 0.85, { source_file: "Beta_Document.pdf", document_id: "doc-beta", page: 3 }),
      makeScoredChunk("c3", "text3", 0.75, { page: 2 }),
      makeScoredChunk("c4", "text4", 0.65, { source_file: "Beta_Document.pdf", document_id: "doc-beta", page: 5 }),
    ];
    const summary = summarizeCitations(chunks, 3);
    expect(summary.sourceCount).toBe(4);
    expect(summary.uniqueSources).toEqual(["[doc-alpha p.1]", "[doc-beta p.3]", "[doc-alpha p.2]"]);
    expect(summary.top).toContain("[doc-alpha p.1]");
    expect(summary.top).toContain("0.950");
    expect(summary.context).toContain("[doc-alpha p.1] text1");
  });

  it("limits to specified count", () => {
    const chunks: ScoredChunk[] = [
      makeScoredChunk("c1", "t1", 0.9, { page: 1 }),
      makeScoredChunk("c2", "t2", 0.8, { page: 2 }),
      makeScoredChunk("c3", "t3", 0.7, { page: 3 }),
    ];
    const summary = summarizeCitations(chunks, 1);
    expect(summary.uniqueSources).toHaveLength(1);
  });
});

/* ================================================================== */
/*  Tests: CITATION_REGEX                                              */
/* ================================================================== */

describe("CITATION_REGEX", () => {
  it("matches primary-document format", () => {
    expect("[doc-alpha p.1]").toMatch(new RegExp(CITATION_REGEX.source));
  });

  it("matches secondary-document format", () => {
    expect("[doc-beta p.42]").toMatch(new RegExp(CITATION_REGEX.source));
  });

  it("does not match invalid formats", () => {
    const re = new RegExp(CITATION_REGEX.source);
    expect("[doc-alpha page 1]").not.toMatch(re);
    expect("[doc-alpha p.one]").not.toMatch(re);
    expect("doc-alpha p.1").not.toMatch(re); // missing brackets
  });
});
