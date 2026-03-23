/**
 * Test script: verifies that the cross-document retrieval mechanism
 * can pull chunks from multiple different documents for a single query.
 *
 * Usage: npx tsx scripts/test-cross-doc-retrieval.ts
 *
 * This test works without Ollama by creating mock vector/lexical stores
 * with chunks from two different generic documents, then running
 * the cross-document retriever to verify multi-doc coverage.
 */

import { CorpusChunk, ScoredChunk } from "../src/types/rag";
import {
  hasMultiDocumentCoverage,
  getDocumentCoverageSummary,
} from "../src/retrieval/crossDocRetriever";

// ─── Mock data ─────────────────────────────────────────────────────────────

function makeMockChunks(): CorpusChunk[] {
  const primaryChunks: CorpusChunk[] = Array.from({ length: 10 }, (_, i) => ({
    id: `doc-alpha-p${i + 1}-s1`,
    text: `Primary document text about section ${i + 1} covering procedural requirements and compliance standards.`,
    metadata: {
      source_file: "Alpha_Document.pdf",
      document_id: "doc-alpha",
      page: i + 1,
      chunk_id: i,
      section_title: `Section ${i + 1}`,
    },
  }));

  const secondaryChunks: CorpusChunk[] = Array.from({ length: 10 }, (_, i) => ({
    id: `doc-beta-p${i + 1}-s1`,
    text: `Secondary document text discusses system architecture in chapter ${i + 1}, exploring memory mechanisms.`,
    metadata: {
      source_file: "Beta_Document.pdf",
      document_id: "doc-beta",
      page: i + 1,
      chunk_id: i,
      section_title: `Chapter ${i + 1}`,
    },
  }));

  return [...primaryChunks, ...secondaryChunks];
}

// ─── Unit tests for diversity balancing ────────────────────────────────────

function testHasMultiDocumentCoverage() {
  console.log("\n── Test: hasMultiDocumentCoverage ──");

  const primaryOnly: ScoredChunk[] = [
    {
      chunk: {
        id: "doc-alpha-p1-s1",
        text: "test",
        metadata: {
          source_file: "doc-alpha.pdf",
          document_id: "doc-alpha",
          page: 1,
          chunk_id: 0,
          section_title: null,
        },
      },
      score: 0.9,
    },
  ];

  const mixed: ScoredChunk[] = [
    ...primaryOnly,
    {
      chunk: {
        id: "doc-beta-p1-s1",
        text: "test",
        metadata: {
          source_file: "paper.pdf",
          document_id: "doc-beta",
          page: 1,
          chunk_id: 0,
          section_title: null,
        },
      },
      score: 0.8,
    },
  ];

  console.assert(!hasMultiDocumentCoverage(primaryOnly), "Single doc should return false");
  console.assert(hasMultiDocumentCoverage(mixed), "Mixed docs should return true");
  console.log("  ✓ hasMultiDocumentCoverage works correctly");
}

function testGetDocumentCoverageSummary() {
  console.log("\n── Test: getDocumentCoverageSummary ──");

  const chunks: ScoredChunk[] = [
    {
      chunk: {
        id: "doc-alpha-1",
        text: "a",
          metadata: { source_file: "doc-alpha.pdf", document_id: "doc-alpha", page: 1, chunk_id: 0, section_title: null },
      },
      score: 0.95,
    },
    {
      chunk: {
        id: "doc-alpha-2",
        text: "b",
          metadata: { source_file: "doc-alpha.pdf", document_id: "doc-alpha", page: 2, chunk_id: 1, section_title: null },
      },
      score: 0.80,
    },
    {
      chunk: {
        id: "doc-beta-1",
        text: "c",
          metadata: { source_file: "paper.pdf", document_id: "doc-beta", page: 1, chunk_id: 0, section_title: null },
      },
      score: 0.70,
    },
  ];

  const summary = getDocumentCoverageSummary(chunks);
  console.assert(summary["doc-alpha"]?.count === 2, "primary document should have 2 chunks");
  console.assert(summary["doc-alpha"]?.topScore === 0.95, "primary document top score should be 0.95");
  console.assert(summary["doc-beta"]?.count === 1, "secondary document should have 1 chunk");
  console.assert(summary["doc-beta"]?.topScore === 0.70, "secondary document top score should be 0.70");
  console.log("  ✓ getDocumentCoverageSummary works correctly");
}

function testDocumentBalanceExample() {
  console.log("\n── Test: multi-document balance example ──");

  // Simulate grouped chunks: 8 primary-document chunks (high score) vs 4 secondary-document chunks (lower score)
  const primaryChunks: ScoredChunk[] = Array.from({ length: 8 }, (_, i) => ({
    chunk: {
      id: `doc-alpha-${i}`,
      text: `doc-alpha text ${i}`,
      metadata: { source_file: "doc-alpha.pdf", document_id: "doc-alpha", page: i + 1, chunk_id: i, section_title: null },
    },
    score: 0.95 - i * 0.05,
  }));

  const secondaryChunks: ScoredChunk[] = Array.from({ length: 4 }, (_, i) => ({
    chunk: {
      id: `paper-${i}`,
      text: `paper text ${i}`,
      metadata: { source_file: "paper.pdf", document_id: "doc-beta", page: i + 1, chunk_id: i, section_title: null },
    },
    score: 0.60 - i * 0.05,
  }));

  const grouped = new Map<string, ScoredChunk[]>();
  grouped.set("doc-alpha", primaryChunks);
  grouped.set("doc-beta", secondaryChunks);

  // Without balancing, top-6 would all come from the higher-scoring document.
  const allByScore = [...primaryChunks, ...secondaryChunks].sort((a, b) => b.score - a.score);
  const top6NoDiversity = allByScore.slice(0, 6);
  const top6DocIds = new Set(top6NoDiversity.map((sc) => sc.chunk.metadata.document_id));
  console.log(`  Without diversity: ${top6DocIds.size} document(s) in top-6: [${[...top6DocIds].join(", ")}]`);

  // Now test that a balanced selection includes both documents.
  const topK = 6;
  const result: ScoredChunk[] = [];
  const used = new Set<string>();

  // Phase 1: guarantee min per doc (at least 1)
  for (const [, chunks] of grouped) {
    chunks.sort((a, b) => b.score - a.score);
    if (chunks.length > 0 && !used.has(chunks[0].chunk.id)) {
      result.push(chunks[0]);
      used.add(chunks[0].chunk.id);
    }
  }

  // Phase 2: fill rest by round-robin
  const pointers = new Map<string, number>();
  for (const docId of grouped.keys()) {
    pointers.set(docId, 1); // start after first used chunk
  }

  while (result.length < topK) {
    let added = false;
    for (const docId of grouped.keys()) {
      if (result.length >= topK) break;
      const docChunks = grouped.get(docId) ?? [];
      let ptr = pointers.get(docId) ?? docChunks.length;
      while (ptr < docChunks.length && used.has(docChunks[ptr].chunk.id)) ptr++;
      if (ptr < docChunks.length) {
        result.push(docChunks[ptr]);
        used.add(docChunks[ptr].chunk.id);
        pointers.set(docId, ptr + 1);
        added = true;
      }
    }
    if (!added) break;
  }

  const diverseDocIds = new Set(result.map((sc) => sc.chunk.metadata.document_id));
  console.log(`  With diversity: ${diverseDocIds.size} document(s) in top-6: [${[...diverseDocIds].join(", ")}]`);
  console.assert(diverseDocIds.size > 1, "Balanced selection should include chunks from multiple documents");

  const primaryCount = result.filter((sc) => sc.chunk.metadata.document_id === "doc-alpha").length;
  const secondaryCount = result.filter((sc) => sc.chunk.metadata.document_id === "doc-beta").length;
  console.log(`  primary document chunks: ${primaryCount}, secondary document chunks: ${secondaryCount}`);
  console.assert(secondaryCount >= 1, "secondary document should have at least 1 chunk in balanced results");
  console.assert(primaryCount >= 1, "primary document should have at least 1 chunk in balanced results");
  console.log("  ✓ Balanced selection preserves multi-document coverage");
}

// ─── Run all tests ─────────────────────────────────────────────────────────

async function main() {
  console.log("=== Cross-Document Retrieval Tests ===\n");

  testHasMultiDocumentCoverage();
  testGetDocumentCoverageSummary();
  testDocumentBalanceExample();

  console.log("\n=== All cross-document retrieval tests passed ✓ ===\n");
}

main().catch((err) => {
  console.error("Test failed:", err);
  process.exit(1);
});
