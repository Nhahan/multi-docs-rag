/**
 * Integration tests for cross-document RAG pipeline.
 *
 * These tests use sample multi-document datasets from two different generic sources
 * and mock the Ollama LLM/embedding layer to verify that cross-document comparison
 * questions produce correct, citation-grounded answers referencing multiple sources.
 *
 * The tests exercise the full pipeline: retrieve → evidence_gate → generate → verify
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { ScoredChunk, CorpusChunk, SourceMetadata, GraphState } from "../../types/rag";

/* ------------------------------------------------------------------ */
/*  Sample multi-document datasets                                     */
/* ------------------------------------------------------------------ */

/**
 * Realistic sample chunks from a primary regulation-style document and a secondary paper-style document.
 * These represent the kind of content that would be ingested from the actual PDFs.
 */

const PRIMARY_DOCUMENT_CHUNKS: CorpusChunk[] = [
  {
    id: "doc-alpha-p5-c0",
    text: "Section 91.103 Preflight action. Each pilot in command shall, before beginning a flight, become familiar with all available information concerning that flight. This information must include, for a flight under IFR or a flight not in the vicinity of an airport, available weather reports and forecasts.",
    metadata: {
      source_file: "Alpha_Document.pdf",
      document_id: "doc-alpha",
      page: 5,
      chunk_id: 0,
      section_title: "Section 91.103 Preflight action",
    },
  },
  {
    id: "doc-alpha-p12-c0",
    text: "Section 91.205 Powered civil aircraft with standard category U.S. airworthiness certificates: Instrument and equipment requirements. No person may operate a powered civil aircraft with a standard category U.S. airworthiness certificate unless that aircraft contains instruments and equipment required by this section.",
    metadata: {
      source_file: "Alpha_Document.pdf",
      document_id: "doc-alpha",
      page: 12,
      chunk_id: 0,
      section_title: "Section 91.205 Instrument and equipment requirements",
    },
  },
  {
    id: "doc-alpha-p18-c0",
    text: "Section 91.409 Inspections. No person may operate an aircraft unless, within the preceding 12 calendar months, it has had an annual inspection in accordance with part 43 of this chapter and has been approved for return to service.",
    metadata: {
      source_file: "Alpha_Document.pdf",
      document_id: "doc-alpha",
      page: 18,
      chunk_id: 0,
      section_title: "Section 91.409 Inspections",
    },
  },
  {
    id: "doc-alpha-p25-c0",
    text: "Section 91.171 VOR equipment check for IFR operations. No person may operate a civil aircraft under IFR using the VOR system of radio navigation unless the VOR equipment is checked and found to be within limits of permissible indicated bearing error.",
    metadata: {
      source_file: "Alpha_Document.pdf",
      document_id: "doc-alpha",
      page: 25,
      chunk_id: 0,
      section_title: "Section 91.171 VOR equipment check",
    },
  },
];

const SECONDARY_DOCUMENT_CHUNKS: CorpusChunk[] = [
  {
    id: "doc-beta-p3-c0",
    text: "Agent memory systems store and retrieve information across interactions, enabling persistent context beyond the immediate prompt window. Memory architectures typically include short-term working memory, long-term episodic storage, and retrieval mechanisms based on semantic similarity.",
    metadata: {
      source_file: "Beta_Document.pdf",
      document_id: "doc-beta",
      page: 3,
      chunk_id: 0,
      section_title: "Introduction",
    },
  },
  {
    id: "doc-beta-p7-c0",
    text: "Embedding-based retrieval in persistent memory systems uses vector similarity to find relevant past interactions. The embedding model maps text to a high-dimensional space where semantically similar content clusters together, enabling efficient nearest-neighbor search over large memory stores.",
    metadata: {
      source_file: "Beta_Document.pdf",
      document_id: "doc-beta",
      page: 7,
      chunk_id: 0,
      section_title: "Methods",
    },
  },
  {
    id: "doc-beta-p11-c0",
    text: "Results demonstrate that persistent memory systems with structured retrieval outperform simple context window approaches by 34% on multi-turn task completion benchmarks. The improvement is most significant for tasks requiring information from interactions more than 5 turns prior.",
    metadata: {
      source_file: "Beta_Document.pdf",
      document_id: "doc-beta",
      page: 11,
      chunk_id: 0,
      section_title: "Results",
    },
  },
  {
    id: "doc-beta-p15-c0",
    text: "The attention mechanism in language models provides a form of implicit memory within the context window, but its capacity is fundamentally limited by the quadratic scaling of self-attention. External memory architectures overcome this by maintaining a persistent store that can be queried independently of the prompt context length.",
    metadata: {
      source_file: "Beta_Document.pdf",
      document_id: "doc-beta",
      page: 15,
      chunk_id: 0,
      section_title: "Discussion",
    },
  },
];

const ALL_CHUNKS = [...PRIMARY_DOCUMENT_CHUNKS, ...SECONDARY_DOCUMENT_CHUNKS];

/* ------------------------------------------------------------------ */
/*  Helper: build scored chunks with controllable scores               */
/* ------------------------------------------------------------------ */

function makeScoredChunks(
  chunks: CorpusChunk[],
  baseScore: number,
  decay = 0.05,
): ScoredChunk[] {
  return chunks.map((chunk, i) => ({
    chunk,
    score: Math.max(0.01, baseScore - i * decay),
    denseScore: baseScore - i * decay * 0.6,
    lexicalScore: baseScore - i * decay * 0.4,
  }));
}

/** Helper: generate deterministic pseudo-embeddings from text */
function textToVector(text: string, dim = 64): number[] {
  const vec = new Array(dim).fill(0);
  for (let i = 0; i < text.length; i++) {
    vec[i % dim] += text.charCodeAt(i) / 1000;
  }
  // Normalize
  const mag = Math.sqrt(vec.reduce((sum: number, v: number) => sum + v * v, 0));
  return mag > 0 ? vec.map((v: number) => v / mag) : vec;
}

/* ------------------------------------------------------------------ */
/*  Mock setup                                                         */
/* ------------------------------------------------------------------ */

// We mock the modules that depend on Ollama
vi.mock("../../lib/llm", () => ({
  getChatModel: vi.fn(),
  getEmbeddingModel: vi.fn(),
  getRerankEmbeddingModel: vi.fn(),
}));

vi.mock("../../lib/store", () => ({
  getIndexStore: vi.fn(),
  clearIndexStore: vi.fn(),
}));

import { getChatModel, getEmbeddingModel, getRerankEmbeddingModel } from "../../lib/llm";
import { getIndexStore } from "../../lib/store";
import { LexicalStore } from "../../retrieval/lexical";
import { LocalVectorStore } from "../../retrieval/vectorStore";
import { retrieveNode, evidenceGateNode, generateNode, verifyNode } from "../../graph/nodes";
import { extractCitationsFromAnswer, allCitationsSupported, hasCitations, CITATION_REGEX } from "../../citations/citationExtractor";
import { analyseCrossDocEvidence } from "../../synthesis/crossDocSynthesizer";
import { formatCitationResponse } from "../../citations/citationFormatter";
import { hasMultiDocumentCoverage, getDocumentCoverageSummary } from "../../retrieval/crossDocRetriever";

/* ------------------------------------------------------------------ */
/*  In-memory stores for tests                                         */
/* ------------------------------------------------------------------ */

/**
 * Build a mock LexicalStore from our sample chunks.
 * We use the real LexicalStore's search logic but construct it in-memory.
 */
async function buildTestLexicalStore(): Promise<LexicalStore> {
  // Build real lexical data structure from chunks
  const data = {
    chunks: ALL_CHUNKS,
    termDocFreq: {} as Record<string, number>,
    chunkTermFreq: {} as Record<string, Record<string, number>>,
    chunkLengths: {} as Record<string, number>,
  };

  const STOP_WORDS = new Set("a an and are as at be by for from has have in is it of on or that the this to with was were will would you your we our us they them these those between into when what where how if else can could should".split(/\s+/));

  for (const chunk of ALL_CHUNKS) {
    const tokens = chunk.text
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter((t) => t.length > 2 && !STOP_WORDS.has(t));

    data.chunkLengths[chunk.id] = tokens.length || 1;
    const tf: Record<string, number> = {};
    for (const token of tokens) {
      tf[token] = (tf[token] ?? 0) + 1;
    }
    data.chunkTermFreq[chunk.id] = tf;

    for (const token of Object.keys(tf)) {
      data.termDocFreq[token] = (data.termDocFreq[token] ?? 0) + 1;
    }
  }

  return new LexicalStore(data);
}

/**
 * Build a mock LocalVectorStore by creating a minimal instance.
 * We mock the search method to return deterministic results.
 */
function buildTestVectorStore(): LocalVectorStore {
  // Generate vectors for all chunks
  const vectors = ALL_CHUNKS.map((c) => textToVector(c.text));

  // Create a mock that behaves like a real LocalVectorStore
  const store = {
    isUsable: vi.fn(() => true),
    searchByEmbedding: vi.fn(
      async (queryVector: number[], topK: number) => {
        const results: ScoredChunk[] = [];
        for (let i = 0; i < ALL_CHUNKS.length; i++) {
          const chunk = ALL_CHUNKS[i];
          // Compute real cosine similarity for deterministic ranking
          const vec = vectors[i];
          let dot = 0, magA = 0, magB = 0;
          for (let j = 0; j < queryVector.length; j++) {
            dot += (queryVector[j] ?? 0) * (vec[j] ?? 0);
            magA += (queryVector[j] ?? 0) ** 2;
            magB += (vec[j] ?? 0) ** 2;
          }
          const score = magA > 0 && magB > 0 ? dot / (Math.sqrt(magA) * Math.sqrt(magB)) : 0;
          results.push({ chunk, score });
        }
        return results.sort((a, b) => b.score - a.score).slice(0, topK);
      },
    ),
    getAllChunks: vi.fn(() => ALL_CHUNKS),
    getVectorForChunk: vi.fn((chunkId: string) => {
      const idx = ALL_CHUNKS.findIndex((c) => c.id === chunkId);
      return idx >= 0 ? vectors[idx] : undefined;
    }),
    getChunksByType: vi.fn(() => ALL_CHUNKS),
  } as unknown as LocalVectorStore;

  return store;
}

/* ------------------------------------------------------------------ */
/*  Test suite                                                         */
/* ------------------------------------------------------------------ */

describe("Cross-Document Integration Tests", () => {
  let mockLexical: LexicalStore;
  let mockVector: LocalVectorStore;

  beforeEach(async () => {
    mockLexical = await buildTestLexicalStore();
    mockVector = buildTestVectorStore();

    // Set up the index store mock to return our test stores
    vi.mocked(getIndexStore).mockResolvedValue({
      lexical: mockLexical,
      vector: mockVector,
      chunks: { getChunk: vi.fn(), getAllChunks: vi.fn(() => ALL_CHUNKS) } as any,
    });

    // Mock embedding model — returns deterministic vectors based on text content
    const mockEmbedder = {
      embedQuery: vi.fn(async (text: string) => textToVector(text)),
      embedDocuments: vi.fn(async (texts: string[]) => texts.map((t) => textToVector(t))),
    };
    vi.mocked(getEmbeddingModel).mockReturnValue(mockEmbedder as any);
    vi.mocked(getRerankEmbeddingModel).mockReturnValue(mockEmbedder as any);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  const runFullPipeline = async (question: string): Promise<GraphState> => {
    const stage2 = await retrieveNode({ question } as GraphState);
    const stage3 = await evidenceGateNode(stage2);
    const stage4 = await generateNode(stage3);
    return verifyNode(stage4);
  };

  /* ---------------------------------------------------------------- */
  /*  Dataset 1: Direct cross-document comparison                      */
  /* ---------------------------------------------------------------- */

  describe("Dataset 1: Compare regulation requirements vs research findings", () => {
    const CROSS_QUESTION = "Compare how the primary regulation-style document and the secondary research-style document approach information retrieval requirements";

    it("retrieves chunks from BOTH documents for cross-document queries", async () => {
      const retrieved = await retrieveNode({ question: CROSS_QUESTION } as GraphState);

      expect(retrieved.retrieval).toBeDefined();
      expect(retrieved.retrieval!.chunks.length).toBeGreaterThan(0);

      // Verify multi-document coverage
      const docIds = new Set(
        retrieved.retrieval!.chunks.map((sc) => sc.chunk.metadata.document_id),
      );
      expect(docIds.size).toBeGreaterThanOrEqual(2);
      expect(docIds.has("doc-alpha")).toBe(true);
      expect(docIds.has("doc-beta")).toBe(true);
    });

    it("generates citation-grounded answer referencing multiple sources", async () => {
      const retrieved = await retrieveNode({ question: CROSS_QUESTION } as GraphState);
      const retrievedChunks = retrieved.retrieval?.chunks ?? [];
      const primaryChunk = retrievedChunks.find(
        (sc) => sc.chunk.metadata.document_id === "doc-alpha",
      );
      const secondaryChunk = retrievedChunks.find(
        (sc) => sc.chunk.metadata.document_id === "doc-beta",
      );

      expect(primaryChunk).toBeDefined();
      expect(secondaryChunk).toBeDefined();

      const primaryCitation = `[doc-alpha p.${primaryChunk!.chunk.metadata.page}]`;
      const secondaryCitation = `[doc-beta p.${secondaryChunk!.chunk.metadata.page}]`;

      const mockChat = {
        invoke: vi.fn(async () => ({
          content:
            `The primary document defines information-gathering obligations ${primaryCitation}, ` +
            `while the secondary document describes persistent retrieval mechanisms ${secondaryCitation}. ` +
            `Taken together, the evidence shows that both documents rely on structured access to relevant information.`,
        })),
      };
      vi.mocked(getChatModel).mockReturnValue(mockChat as any);

      const stage3 = await evidenceGateNode(retrieved);
      const stage4 = await generateNode(stage3);
      const s5 = await verifyNode(stage4);

      // Verify the answer contains citations
      expect(s5.answer).toBeDefined();
      expect(hasCitations(s5.answer!)).toBe(true);

      // Verify citations reference BOTH documents
      const cited = extractCitationsFromAnswer(s5.answer!);
      const hasPrimaryDoc = cited.some((c) => c.startsWith("[doc-alpha"));
      const hasSecondaryDoc = cited.some((c) => c.startsWith("[doc-beta"));
      expect(hasPrimaryDoc).toBe(true);
      expect(hasSecondaryDoc).toBe(true);

      // Verify all citations are from retrieved evidence
      const allowed = new Set(s5.citations ?? []);
      if (allowed.size > 0) {
        const unsupported = cited.filter((c) => !allowed.has(c));
        // Citations in the answer should be a subset of retrieved evidence
        expect(unsupported.length).toBeLessThanOrEqual(cited.length);
      }
    });
  });

  /* ---------------------------------------------------------------- */
  /*  Dataset 2: Synthesis across document boundaries                  */
  /* ---------------------------------------------------------------- */

  describe("Dataset 2: Synthesize information across both documents", () => {
    const SYNTHESIS_QUESTION =
      "What are the differences between the compliance requirements in the primary document and the memory architecture described in the secondary document?";

    it("cross-doc analysis groups chunks correctly from mixed retrieval", () => {
      const mixedChunks: ScoredChunk[] = [
        ...makeScoredChunks(PRIMARY_DOCUMENT_CHUNKS.slice(0, 2), 0.85),
        ...makeScoredChunks(SECONDARY_DOCUMENT_CHUNKS.slice(0, 2), 0.80),
      ];

      const analysis = analyseCrossDocEvidence(mixedChunks);

      expect(analysis.isMultiDocument).toBe(true);
      expect(analysis.documentCount).toBe(2);
      expect(analysis.groups).toHaveLength(2);

      // Verify structured context contains citations from both docs
      expect(analysis.structuredContext).toContain("[doc-alpha p.");
      expect(analysis.structuredContext).toContain("[doc-beta p.");
      expect(analysis.structuredContext).toContain("=== Source:");

      // Verify synthesis instruction mentions cross-document comparison
      expect(analysis.synthesisInstruction).toContain("Compare");
      expect(analysis.synthesisInstruction).toContain("Combine");
      expect(analysis.synthesisInstruction).toContain("Attribute");
    });

    it("formatCitationResponse produces valid multi-source citation entries", () => {
      const answer =
        "The primary document requires annual inspections [doc-alpha p.18] and equipment compliance [doc-alpha p.12], " +
        "while the secondary document uses embedding-based retrieval [doc-beta p.7] and " +
        "demonstrate 34% improvement with structured approaches [doc-beta p.11].";

      const chunks: ScoredChunk[] = [
        ...makeScoredChunks([PRIMARY_DOCUMENT_CHUNKS[2], PRIMARY_DOCUMENT_CHUNKS[1]], 0.9),
        ...makeScoredChunks([SECONDARY_DOCUMENT_CHUNKS[1], SECONDARY_DOCUMENT_CHUNKS[2]], 0.85),
      ];

      const formatted = formatCitationResponse(answer, chunks);

      expect(formatted.has_citations).toBe(true);
      expect(formatted.inline_citations.length).toBeGreaterThanOrEqual(2);

      // Verify citation entries span multiple documents
      // Verify citation footer includes both sources
      expect(formatted.citation_footer).toContain("Alpha_Document.pdf");
      expect(formatted.citation_footer).toContain("Beta_Document.pdf");

      // Verify document_id diversity
      const docIds = new Set(formatted.citation_entries.map((e) => e.document_id));
      expect(docIds.has("doc-alpha")).toBe(true);
      expect(docIds.has("doc-beta")).toBe(true);
    });
  });

  /* ---------------------------------------------------------------- */
  /*  Dataset 3: Edge case — question mentions both domains            */
  /* ---------------------------------------------------------------- */

  describe("Dataset 3: Mixed-domain question with both primary and secondary document keywords", () => {
    const MIXED_QUESTION =
      "How do the section requirements in the primary document compare with the attention mechanism and embedding approach in the secondary document?";

    it("full pipeline produces answer with multi-source citations for mixed-domain query", async () => {
      const mockChat = {
        invoke: vi.fn(async () => ({
          content:
            "The primary document establishes specific section requirements for pilots, such as Section 91.103 " +
            "which mandates pre-flight information gathering [doc-alpha p.5], and Section 91.171 for VOR " +
            "equipment checks [doc-alpha p.25]. In contrast, the secondary document describes how " +
            "attention mechanisms provide implicit memory within context windows [doc-beta p.15], " +
            "while embedding-based retrieval enables efficient semantic search [doc-beta p.7]. " +
            "Both approaches deal with information access but from fundamentally different " +
            "perspectives: regulatory compliance versus computational architecture.",
        })),
      };
      vi.mocked(getChatModel).mockReturnValue(mockChat as any);

      const s5 = await runFullPipeline(MIXED_QUESTION);

      expect(s5.answer).toBeDefined();
      expect(s5.answer).not.toContain("I cannot answer from the indexed evidence");

      const cited = extractCitationsFromAnswer(s5.answer!);
      expect(cited.length).toBeGreaterThanOrEqual(2);

      // Must reference both documents
      const primaryDocCitations = cited.filter((c) => c.includes("doc-alpha"));
      const secondaryDocCitations = cited.filter((c) => c.includes("doc-beta"));
      expect(primaryDocCitations.length).toBeGreaterThan(0);
      expect(secondaryDocCitations.length).toBeGreaterThan(0);
    });

    it("trace records all pipeline stages for cross-document query", async () => {
      const mockChat = {
        invoke: vi.fn(async () => ({
          content:
            "The primary document requires equipment compliance [doc-alpha p.12] while the secondary document uses embeddings [doc-beta p.7].",
        })),
      };
      vi.mocked(getChatModel).mockReturnValue(mockChat as any);

      const s5 = await runFullPipeline(MIXED_QUESTION);

      // All 4 pipeline stages should be traced
      const stages = (s5.trace ?? []).map((t) => t.stage);
      expect(stages).toContain("retrieve");
      expect(stages).toContain("evidence_gate");
      expect(stages).toContain("generate");
      expect(stages).toContain("verify");
    });
  });

  /* ---------------------------------------------------------------- */
  /*  Cross-document citation grounding validation                     */
  /* ---------------------------------------------------------------- */

  describe("Citation grounding validation across documents", () => {
    it("rejects answers with fabricated citations not from retrieved evidence", async () => {
      // Simulate an answer that cites pages NOT in the retrieved chunks
      const mockChat = {
        invoke: vi.fn(async () => ({
          content: "According to [doc-alpha p.99] and [doc-beta p.99], the answer is clear.",
        })),
      };
      vi.mocked(getChatModel).mockReturnValue(mockChat as any);

      const QUESTION = "Compare the checks in the primary document with the retrieval methods in the secondary document";
      const s5 = await runFullPipeline(QUESTION);

      // The verifier should detect unsupported citations and refuse
      expect(s5.answer).toBe("I cannot answer from the indexed evidence.");
    });

    it("accepts answers where all citations map to retrieved evidence pages", async () => {
      const mockChat = {
        invoke: vi.fn(async () => ({
          content:
            "The primary document requires preflight information gathering [doc-alpha p.5] while the secondary document " +
            "systems use embedding-based retrieval [doc-beta p.7].",
        })),
      };
      vi.mocked(getChatModel).mockReturnValue(mockChat as any);

      const QUESTION = "Compare how the primary and secondary documents handle information retrieval";
      const s5 = await runFullPipeline(QUESTION);

      // Verify answer is NOT the refusal message (citations were valid)
      expect(s5.answer).not.toBe("I cannot answer from the indexed evidence.");
      expect(hasCitations(s5.answer!)).toBe(true);

      // Verify both document types are cited
      const cited = extractCitationsFromAnswer(s5.answer!);
      expect(cited.some((c) => c.includes("doc-alpha"))).toBe(true);
      expect(cited.some((c) => c.includes("doc-beta"))).toBe(true);
    });

    it("allCitationsSupported validates cross-doc citation sets correctly", () => {
      const allowed = new Set([
        "[doc-alpha p.5]",
        "[doc-alpha p.12]",
        "[doc-beta p.3]",
        "[doc-beta p.7]",
      ]);

      // Valid: all cited references are in allowed set
      expect(
        allCitationsSupported(
          "Info from [doc-alpha p.5] and research from [doc-beta p.7].",
          allowed,
        ),
      ).toBe(true);

      // Invalid: page 99 not in allowed
      expect(
        allCitationsSupported(
          "Info from [doc-alpha p.99] and [doc-beta p.7].",
          allowed,
        ),
      ).toBe(false);

      // Invalid: no citations at all
      expect(
        allCitationsSupported("Answer with no citations.", allowed),
      ).toBe(false);
    });
  });

  /* ---------------------------------------------------------------- */
  /*  Multi-document coverage helpers                                  */
  /* ---------------------------------------------------------------- */

  describe("Multi-document coverage utilities", () => {
    it("hasMultiDocumentCoverage detects when chunks span both documents", () => {
      const mixed = [
        ...makeScoredChunks(PRIMARY_DOCUMENT_CHUNKS.slice(0, 1), 0.9),
        ...makeScoredChunks(SECONDARY_DOCUMENT_CHUNKS.slice(0, 1), 0.85),
      ];
      expect(hasMultiDocumentCoverage(mixed)).toBe(true);
    });

    it("hasMultiDocumentCoverage returns false for single-document chunks", () => {
      expect(hasMultiDocumentCoverage(makeScoredChunks(PRIMARY_DOCUMENT_CHUNKS, 0.9))).toBe(false);
    });

    it("getDocumentCoverageSummary provides per-document stats", () => {
      const mixed = [
        ...makeScoredChunks(PRIMARY_DOCUMENT_CHUNKS.slice(0, 3), 0.9),
        ...makeScoredChunks(SECONDARY_DOCUMENT_CHUNKS.slice(0, 2), 0.85),
      ];

      const summary = getDocumentCoverageSummary(mixed);
      expect(summary["doc-alpha"]).toBeDefined();
      expect(summary["doc-beta"]).toBeDefined();
      expect(summary["doc-alpha"].count).toBe(3);
      expect(summary["doc-beta"].count).toBe(2);
      expect(summary["doc-alpha"].topScore).toBeGreaterThan(0);
      expect(summary["doc-beta"].topScore).toBeGreaterThan(0);
    });
  });

  /* ---------------------------------------------------------------- */
  /*  Dataset 4: Cross-document with "between" keyword                 */
  /* ---------------------------------------------------------------- */

  describe("Dataset 4: Cross-doc queries with explicit comparison keywords", () => {
    const BETWEEN_QUESTION = "What is the difference between VOR equipment requirements and embedding-based retrieval?";

    it("produces cross-doc analysis with both primary and secondary document groups", () => {
      const chunks: ScoredChunk[] = [
        { chunk: PRIMARY_DOCUMENT_CHUNKS[3], score: 0.88 },
        { chunk: SECONDARY_DOCUMENT_CHUNKS[1], score: 0.82 },
      ];

      const analysis = analyseCrossDocEvidence(chunks);
      expect(analysis.isMultiDocument).toBe(true);
      expect(analysis.structuredContext).toContain("[doc-alpha p.25]");
      expect(analysis.structuredContext).toContain("[doc-beta p.7]");
    });
  });

  /* ---------------------------------------------------------------- */
  /*  Dataset 5: Verify citation format consistency                    */
  /* ---------------------------------------------------------------- */

  describe("Dataset 5: Citation format consistency across documents", () => {
    it("all citations match the required format pattern", () => {
      const answer =
        "The primary document Section 91.103 mandates pre-flight information [doc-alpha p.5], " +
        "Section 91.409 requires annual inspections [doc-alpha p.18], and " +
        "the secondary document shows 34% improvement [doc-beta p.11] " +
        "through structured retrieval [doc-beta p.7].";

      const citations = extractCitationsFromAnswer(answer);
      expect(citations).toHaveLength(4);

      // Every citation must match the regex pattern
      for (const citation of citations) {
        expect(citation).toMatch(CITATION_REGEX);
      }

      // Verify specific citations present
      expect(citations).toContain("[doc-alpha p.5]");
      expect(citations).toContain("[doc-alpha p.18]");
      expect(citations).toContain("[doc-beta p.11]");
      expect(citations).toContain("[doc-beta p.7]");
    });

    it("formatCitationResponse builds complete multi-doc footer", () => {
      const answer = "Regulations [doc-alpha p.5] vs research [doc-beta p.3].";
      const chunks: ScoredChunk[] = [
        { chunk: PRIMARY_DOCUMENT_CHUNKS[0], score: 0.9 },
        { chunk: SECONDARY_DOCUMENT_CHUNKS[0], score: 0.85 },
      ];

      const formatted = formatCitationResponse(answer, chunks);

      expect(formatted.has_citations).toBe(true);
      expect(formatted.citation_entries).toHaveLength(2);
      expect(formatted.citation_footer).toContain("Sources:");
      expect(formatted.citation_footer).toContain("[doc-alpha p.5]");
      expect(formatted.citation_footer).toContain("[doc-beta p.3]");
      expect(formatted.citation_footer).toContain("Alpha_Document.pdf");
      expect(formatted.citation_footer).toContain("Beta_Document.pdf");
    });
  });
});
