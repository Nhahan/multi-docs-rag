#!/usr/bin/env tsx
/**
 * Smoke test: a fixture-backed hard multi-document scenario
 *
 * Uses a fixture-specific cross-document query and validates retrieval/citation
 * behavior using dynamically discovered corpus document ids instead of fixed names.
 */

if (typeof process.loadEnvFile === "function") {
  process.loadEnvFile();
}

import { runRagQuery } from "../src/graph/run";
import type { GraphState, ScoredChunk } from "../src/types/rag";
import { appConfig } from "../src/lib/config";
import { findCorpusDocBySourceFile } from "./smoke/utils";

interface AssertionResult {
  label: string;
  passed: boolean;
  detail?: string;
}

type CitationRegex = RegExp;

const REFUSAL_TEXT = appConfig.messages.insufficientEvidence;

const escapeRegExp = (value: string) =>
  value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const citationRegex = (prefix: string): CitationRegex =>
  new RegExp(`\\[${escapeRegExp(prefix)} p\\.\\d+\\]`, "g");

const inferDocumentIds = (): string[] => {
  const primary = findCorpusDocBySourceFile("CFR - Code of Federal Regulations.pdf", 0);
  const secondary = findCorpusDocBySourceFile("Agent_Memory_Below_the_Prompt.pdf", 1);
  return [primary?.document_id, secondary?.document_id].filter(
    (value): value is string => Boolean(value),
  );
};

const TARGET_DOC_IDS = inferDocumentIds();
const PRIMARY_DOC_ID = TARGET_DOC_IDS[0] ?? "document-1";
const SECONDARY_DOC_ID = TARGET_DOC_IDS[1] ?? "document-2";
const QUERY =
  `문서 "${SECONDARY_DOC_ID}" 와 문서 "${PRIMARY_DOC_ID}" 만 근거로 짧은 구조화된 메모를 작성해줘. 1) "${SECONDARY_DOC_ID}" 에서 Apple M4 Pro 기준 8K context의 FP16/Q4 agent capacity와 4K cold/warm/hot latency 수치를 정확히 적고, 2) "${PRIMARY_DOC_ID}" 에서 이 시스템과 가장 직접 관련 있는 control 세 가지를 section number와 핵심 요구사항과 함께 적고, 3) "${SECONDARY_DOC_ID}" 가 명시적으로 뒷받침하지 않아서 준수했다고 말하면 안 되는 control 한 가지를 분명히 구분해줘. 법적 결론은 내리지 말고, 문서에 있는 근거와 근거 부족만 구분해줘.`;

function assert(label: string, ok: boolean, detail?: string): AssertionResult {
  return { label, passed: ok, detail };
}

function uniqueRegexMatches(text: string, pattern: RegExp): string[] {
  const flags = pattern.flags.includes("g") ? pattern.flags : `${pattern.flags}g`;
  const globalRe = new RegExp(pattern.source, flags);
  const set = new Set<string>();
  const matches = text.match(globalRe) ?? [];
  for (const match of matches) {
    set.add(match);
  }
  return [...set];
}

function runAssertions(result: GraphState): AssertionResult[] {
  const assertions: AssertionResult[] = [];
  const answer = result.answer ?? "";
  const chunks = result.retrieval?.chunks ?? [];
  const allDocIds = [...new Set(chunks.map((chunk) => chunk.chunk.metadata.document_id))];

  const targetDocIds =
    TARGET_DOC_IDS.length >= 2 ? TARGET_DOC_IDS : [...allDocIds].slice(0, 2);
  const docA = targetDocIds[0];
  const docB = targetDocIds[1];

  const grouped = new Map<string, ScoredChunk[]>();
  for (const chunk of chunks) {
    const docId = chunk.chunk.metadata.document_id;
    const list = grouped.get(docId);
    if (list) {
      list.push(chunk);
    } else {
      grouped.set(docId, [chunk]);
    }
  }

  const docAChunks = docA ? grouped.get(docA) ?? [] : [];
  const docBChunks = docB ? grouped.get(docB) ?? [] : [];
  const targetCandidates = [docA, docB].filter((id): id is string => Boolean(id));
  const targetRegexes = targetCandidates.map((id) => citationRegex(id));
  const citationMatches = targetRegexes.map((re) => uniqueRegexMatches(answer, re));
  const [docACitationMatches, docBCitationMatches] = citationMatches;
  const distinctPart11Sections = [...new Set(answer.match(/(?:§\s*)?11\.\d+/gu) ?? [])];

  assertions.push(
    assert(
      "Cross-route evidence includes both documents",
      chunks.length > 0 && docAChunks.length > 0 && docBChunks.length > 0,
      `retrieved chunks: ${chunks.length}, ${docA ?? "doc-a"}=${docAChunks.length}, ${docB ?? "doc-b"}=${docBChunks.length}`,
    ),
  );

  assertions.push(
    assert(
      `Answer includes [${docA ?? "doc-a"}] citations`,
      Boolean(docACitationMatches.length),
      docACitationMatches.join(", "),
    ),
  );

  assertions.push(
    assert(
      "Answer includes at least one citation",
      Boolean((docACitationMatches.length || docBCitationMatches.length)),
      [docACitationMatches.join(", "), docBCitationMatches.join(", ")].filter(Boolean).join(" / "),
    ),
  );

  assertions.push(
    assert(
      "Answer is substantive",
      answer.length > 0 && !answer.includes(REFUSAL_TEXT),
      `${answer.length} chars`,
    ),
  );

  assertions.push(
    assert(
      "Retrieval includes paper chunk evidence with Apple M4/8K signals",
      docAChunks.some((entry) =>
        /apple\s*m4\s*pro|8k|q4|fp16|latenc/i.test(entry.chunk.text),
      ) ||
        docBChunks.some((entry) =>
          /apple\s*m4\s*pro|8k|q4|fp16|latenc/i.test(entry.chunk.text),
        ),
      `Document chunks searched: ${docAChunks.length + docBChunks.length}`,
    ),
  );

  assertions.push(
    assert(
      "Unsupported Part 11 controls are explicitly marked as unsupported or insufficiently supported",
      /not supported|insufficient|근거가 부족|지원되지|포함되어 있지 않|명시적으로 뒷받침하지 않|없음|없다|제공된 문서에는|명시되지 않/u.test(answer),
      answer.slice(0, 320),
    ),
  );

  assertions.push(
    assert(
      "Answer does not invent multiple unsupported Part 11 section numbers",
      distinctPart11Sections.length <= 1,
      distinctPart11Sections.join(", "),
    ),
  );

  if (result.trace?.length) {
    const failed = result.trace.filter((t) => t.status === "failed");
    assertions.push(
      assert(
        "No failed pipeline stages",
        failed.length === 0,
        failed.length > 0
          ? `failed: ${failed.map((f) => f.stage).join(", ")}`
          : `${result.trace.length} stages`,
      ),
    );
  } else {
    assertions.push(assert("No failed pipeline stages", false, "trace missing"));
  }

  return assertions;
}

function printSummary(result: GraphState) {
  const chunks = result.retrieval?.chunks ?? [];
  const allDocIds = [...new Set(chunks.map((chunk) => chunk.chunk.metadata.document_id))];
  const targetDocIds = TARGET_DOC_IDS.length >= 2 ? TARGET_DOC_IDS : allDocIds.slice(0, 2);
  const docs = targetDocIds.map((docId) => `${docId}:${chunks.filter((chunk) => chunk.chunk.metadata.document_id === docId).length}`);

  console.log(
    "Quality:",
    result.quality
      ? `passed=${result.quality.passed}, top=${result.quality.top_score.toFixed(3)}, avg=${result.quality.avg_score.toFixed(3)}`
      : "N/A",
  );
  console.log("Retrieved chunks:", chunks.length, `(${docs.join(", ")})`);
  console.log("Answer preview:", result.answer?.slice(0, 280));
  console.log("Citations:", (result.citations ?? []).slice(0, 16).join(", "));
  console.log(
    "Trace:",
    (result.trace ?? [])
      .map((t) => `${t.stage}=${t.status}`)
      .join(" -> "),
  );
}

function reportResults(assertions: AssertionResult[]): boolean {
  let allPassed = true;
  console.log("\n── Assertions ──────────────────────────────");
  for (const assertion of assertions) {
    const icon = assertion.passed ? "✅" : "❌";
    console.log(`  ${icon} ${assertion.label}${assertion.detail ? ` — ${assertion.detail}` : ""}`);
    if (!assertion.passed) allPassed = false;
  }
  console.log("────────────────────────────────────────────");
  console.log(allPassed ? "All assertions passed." : "Some assertions FAILED.");
  return allPassed;
}

async function main() {
  console.log("Running hard multi-document fixture smoke test…");
  console.log("Q:", QUERY);

  const result: GraphState = await runRagQuery(QUERY, true);
  printSummary(result);
  const assertions = runAssertions(result);
  const passed = reportResults(assertions);

  if (!passed) {
    process.exit(1);
  }
}

main().catch((err) => {
  console.error("Smoke test crashed:", err instanceof Error ? err.message : String(err));
  process.exit(1);
});
