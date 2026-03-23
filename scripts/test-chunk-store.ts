/**
 * Unit tests for ChunkStore – run with: npx tsx scripts/test-chunk-store.ts
 */
import { ChunkStore } from "../src/retrieval/chunkStore";
import { CorpusChunk } from "../src/types/rag";
import { writeFile, unlink, mkdir } from "node:fs/promises";
import { join } from "node:path";

const tmpPath = join(process.cwd(), "data/index/.test-chunk-store.json");

const sampleChunks: CorpusChunk[] = [
  {
    id: "doc-alpha-p1-s0",
    text: "Title 21, Part 820 — Quality System Regulation",
    metadata: {
      source_file: "Alpha_Document.pdf",
      document_id: "doc-alpha",
      page: 1,
      chunk_id: 0,
      section_title: "Part 820",
    },
  },
  {
    id: "doc-alpha-p1-s1",
    text: "Subpart A — General Provisions",
    metadata: {
      source_file: "Alpha_Document.pdf",
      document_id: "doc-alpha",
      page: 1,
      chunk_id: 1,
      section_title: "Subpart A",
    },
  },
  {
    id: "doc-beta-p3-s0",
    text: "We propose a tiered memory architecture for LLM agents.",
    metadata: {
      source_file: "Beta_Document.pdf",
      document_id: "doc-beta",
      page: 3,
      chunk_id: 0,
      section_title: "Introduction",
    },
  },
  {
    id: "doc-beta-p5-s2",
    text: "Results show 40% improvement in recall.",
    metadata: {
      source_file: "Beta_Document.pdf",
      document_id: "doc-beta",
      page: 5,
      chunk_id: 2,
      section_title: "Results",
    },
  },
];

let passed = 0;
let failed = 0;

function assert(condition: boolean, label: string) {
  if (condition) {
    passed++;
    console.log(`  ✓ ${label}`);
  } else {
    failed++;
    console.error(`  ✗ ${label}`);
  }
}

async function runTests() {
  console.log("ChunkStore tests\n");

  // --- fromChunks ---
  console.log("fromChunks:");
  const store = ChunkStore.fromChunks(sampleChunks);
  assert(store.size === 4, "size is 4");
  assert(store.getAll().length === 4, "getAll returns 4 chunks");

  // --- getById ---
  console.log("\ngetById:");
  const c1 = store.getById("doc-alpha-p1-s0");
  assert(c1 !== undefined, "finds doc-alpha-p1-s0");
  assert(c1?.metadata.document_id === "doc-alpha", "correct document_id");
  assert(c1?.metadata.page === 1, "correct page");
  assert(store.getById("nonexistent") === undefined, "undefined for missing id");

  // --- getByDocumentId ---
  console.log("\ngetByDocumentId:");
  const primaryChunks = store.getByDocumentId("doc-alpha");
  assert(primaryChunks.length === 2, "doc-alpha has 2 chunks");
  const paperChunks = store.getByDocumentId("doc-beta");
  assert(paperChunks.length === 2, "doc-beta has 2 chunks");
  assert(store.getByDocumentId("unknown").length === 0, "unknown doc returns empty array");

  // --- getDocumentIds ---
  console.log("\ngetDocumentIds:");
  const docIds = store.getDocumentIds();
  assert(docIds.includes("doc-alpha"), "includes doc-alpha");
  assert(docIds.includes("doc-beta"), "includes doc-beta");
  assert(docIds.length === 2, "exactly 2 document ids");

  // --- getSummary ---
  console.log("\ngetSummary:");
  const summary = store.getSummary();
  assert(summary["doc-alpha"] === 2, "summary doc-alpha=2");
  assert(summary["doc-beta"] === 2, "summary doc-beta=2");

  // --- duplicate prevention ---
  console.log("\nduplicates:");
  store.addChunks([sampleChunks[0]]);
  assert(store.size === 4, "duplicate not added");

  // --- save & load round-trip ---
  console.log("\nsave & load:");
  await store.save(tmpPath);
  const loaded = await ChunkStore.load(tmpPath);
  assert(loaded.size === 4, "loaded size is 4");
  assert(loaded.getById("doc-beta-p5-s2")?.text === "Results show 40% improvement in recall.", "loaded chunk text matches");
  assert(loaded.getByDocumentId("doc-alpha").length === 2, "loaded doc-alpha count matches");

  // --- clear ---
  console.log("\nclear:");
  store.clear();
  assert(store.size === 0, "size is 0 after clear");
  assert(store.getByDocumentId("doc-alpha").length === 0, "no doc-alpha after clear");

  // cleanup
  try { await unlink(tmpPath); } catch {}

  console.log(`\n${passed} passed, ${failed} failed`);
  process.exit(failed > 0 ? 1 : 0);
}

runTests().catch((err) => {
  console.error(err);
  process.exit(1);
});
