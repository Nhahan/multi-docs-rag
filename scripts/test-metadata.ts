/**
 * Verification script for metadata extraction and attachment.
 *
 * Tests that:
 * 1. buildSourceMetadata creates valid metadata with all required fields
 * 2. validateMetadata rejects invalid metadata
 * 3. chunkPageText attaches complete metadata to every chunk
 * 4. enrichMetadata properly overrides fields
 */

import {
  buildSourceMetadata,
  validateMetadata,
  enrichMetadata,
} from "../src/chunking/metadataExtractor";
import { chunkPageText } from "../src/chunking/sectionChunker";
import type { SourceMetadata } from "../src/types/rag";

let passed = 0;
let failed = 0;

const assert = (condition: boolean, label: string) => {
  if (condition) {
    passed++;
    console.log(`  ✓ ${label}`);
  } else {
    failed++;
    console.error(`  ✗ ${label}`);
  }
};

const assertThrows = (fn: () => void, label: string) => {
  try {
    fn();
    failed++;
    console.error(`  ✗ ${label} (expected error, but none thrown)`);
  } catch {
    passed++;
    console.log(`  ✓ ${label}`);
  }
};

async function main() {
  console.log("\n=== Metadata Extraction Tests ===\n");

  // --- 2. buildSourceMetadata ---
  console.log("\n1. buildSourceMetadata – valid metadata:");
  const meta = buildSourceMetadata({
    source_file: "test.pdf",
    document_id: "test-doc",
    page: 1,
    chunk_id: 0,
    section_title: "Introduction",
  });
  assert(meta.source_file === "test.pdf", "source_file set");
  assert(meta.document_id === "test-doc", "document_id set");
  assert(meta.page === 1, "page set");
  assert(meta.chunk_id === 0, "chunk_id set");
  assert(meta.section_title === "Introduction", "section_title set");

  // With null section_title
  const meta2 = buildSourceMetadata({
    source_file: "alpha-document.pdf",
    document_id: "doc-alpha",
    page: 5,
    chunk_id: 10,
    section_title: null,
  });
  assert(meta2.section_title === null, "section_title can be null");

  // --- 3. validateMetadata – rejects invalid ---
  console.log("\n2. validateMetadata – rejects invalid metadata:");
  assertThrows(() => validateMetadata({ source_file: "" } as any), "Rejects empty source_file");
  assertThrows(() => validateMetadata({ ...meta, page: -1 }), "Rejects negative page");
  assertThrows(() => validateMetadata({ ...meta, chunk_id: -1 }), "Rejects negative chunk_id");
  validateMetadata(meta);
  assert(true, "Accepts valid metadata shape");

  // --- 4. chunkPageText – metadata attachment ---
  console.log("\n3. chunkPageText – metadata on every chunk:");
  const sampleText = [
    "Abstract",
    "This paper presents a novel approach to memory management in LLM agents.",
    "We demonstrate improved performance across multiple benchmarks.",
    "",
    "Introduction",
    "Large language models have shown remarkable capabilities in various tasks.",
    "However, managing long-term memory remains a challenge.",
    "In this work, we propose a hierarchical memory architecture.",
  ].join("\n");

  const chunks = await chunkPageText({
    pageText: sampleText,
    pageNumber: 3,
    sourceFile: "Beta_Document.pdf",
    documentId: "doc-beta",
    chunkSize: 200,
    chunkOverlap: 20,
  });

  assert(chunks.length > 0, `Produced ${chunks.length} chunks`);

  for (const chunk of chunks) {
    assert(chunk.metadata.source_file === "Beta_Document.pdf", `Chunk ${chunk.metadata.chunk_id}: source_file present`);
    assert(chunk.metadata.document_id === "doc-beta", `Chunk ${chunk.metadata.chunk_id}: document_id present`);
    assert(chunk.metadata.page === 3, `Chunk ${chunk.metadata.chunk_id}: page present`);
    assert(typeof chunk.metadata.chunk_id === "number", `Chunk ${chunk.metadata.chunk_id}: chunk_id is number`);
    assert(chunk.metadata.section_title === null, `Chunk ${chunk.metadata.chunk_id}: section_title is null`);
  }

  // Verify generic page splitting does not synthesize section labels.
  const uniqueSections = new Set(chunks.map((c) => c.metadata.section_title));
  assert(uniqueSections.size === 1 && uniqueSections.has(null), "No synthetic section labels were assigned");

  // --- 5. enrichMetadata ---
  console.log("\n4. enrichMetadata – overrides fields:");
  const enriched = enrichMetadata(meta, { chunk_id: 42 });
  assert(enriched.chunk_id === 42, "chunk_id overridden to 42");
  assert(enriched.source_file === "test.pdf", "source_file preserved");

  // --- Summary ---
  console.log(`\n=== Results: ${passed} passed, ${failed} failed ===\n`);
  if (failed > 0) process.exit(1);
}

main().catch((err) => {
  console.error("Test runner error:", err);
  process.exit(1);
});
