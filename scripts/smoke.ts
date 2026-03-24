#!/usr/bin/env tsx
/**
 * Smoke test: Combined fixture-backed RAG pipeline validation
 *
 * Runs fixture-specific questions against the generic RAG pipeline and
 * validates citations and evidence availability for each.
 *
 * Usage:
 *   npx tsx scripts/smoke.ts
 *   npm run smoke
 *
 * Exit code 0 = all pass, 1 = at least one failure or crash.
 */

if (typeof process.loadEnvFile === "function") {
  process.loadEnvFile();
}

import { runRagQuery } from "../src/graph/run";
import type { GraphState } from "../src/types/rag";
import { appConfig } from "../src/lib/config";
import { findCorpusDocBySourceFile } from "./smoke/utils";

// ── Types ────────────────────────────────────────────────────────────

interface AssertionResult {
  label: string;
  passed: boolean;
  detail?: string;
}

interface QuestionSpec {
  tag: string;
  question: string;
  expectedCitationRules: Array<{ label: string; re: RegExp; required: boolean }>;
  expectedDocIds?: string[];
  allowInsufficientEvidence?: boolean;
}

const escapeRegExp = (value: string) =>
  value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const citationRegex = (prefix: string) =>
  new RegExp(`\\[${escapeRegExp(prefix)} p\\.\\d+\\]`, "g");

const primaryFixtureDoc =
  findCorpusDocBySourceFile("CFR - Code of Federal Regulations.pdf", 0);
const secondaryFixtureDoc =
  findCorpusDocBySourceFile("Agent_Memory_Below_the_Prompt.pdf", 1);
const primaryDocId = primaryFixtureDoc?.document_id ?? "document-1";
const secondaryDocId = secondaryFixtureDoc?.document_id ?? primaryDocId;

function countMatches(re: RegExp, text: string) {
  const pattern = new RegExp(re.source, re.flags.includes("g") ? re.flags : `${re.flags}g`);
  return text.match(pattern) ?? [];
}

// ── Helpers ──────────────────────────────────────────────────────────

function assert(label: string, ok: boolean, detail?: string): AssertionResult {
  return { label, passed: ok, detail };
}

function printBanner(text: string) {
  const line = "═".repeat(60);
  console.log(`\n${line}`);
  console.log(`  ${text}`);
  console.log(line);
}

function printSummary(result: GraphState) {
  console.log(
    "  Evidence:",
    result.evidence
      ? `passed=${result.evidence.passed}, top=${result.evidence.top_score.toFixed(3)}, avg=${result.evidence.avg_score.toFixed(3)}`
      : "N/A",
  );
  console.log("  Answer (first 200 chars):");
  console.log("   ", (result.answer ?? "").slice(0, 200));
  if (result.citations?.length) {
    console.log("  Citations:", result.citations.join(", "));
  }
  if (result.trace?.length) {
    console.log(
      "  Trace:",
      result.trace
        .map(
          (t) =>
            `${t.stage}=${t.status}${typeof t.score === "number" ? `(${t.score.toFixed(3)})` : ""}`,
        )
        .join(" → "),
    );
  }
}

function runAssertions(result: GraphState, spec: QuestionSpec): AssertionResult[] {
  const assertions: AssertionResult[] = [];
  const answer = result.answer ?? "";
  const chunks = result.retrieval?.chunks ?? [];
  const trace = result.trace ?? [];
  const retrievedDocIds = new Set(chunks.map((c) => c.chunk.metadata.document_id));
  const refusal = appConfig.messages.insufficientEvidence;
  const isInsufficientEvidence = answer.includes(refusal);

  // 1. Retrieved chunks cover expected document ids
  const expectedDocIds = spec.expectedDocIds ?? [];
  if (expectedDocIds.length > 0) {
    const coveredExpectedCount = expectedDocIds.filter((docId) => retrievedDocIds.has(docId)).length;
    assertions.push(
      assert(
        `Retrieved chunks include expected document ids [${expectedDocIds.join(", ")}]`,
        coveredExpectedCount === expectedDocIds.length,
        `retrieved: [${[...retrievedDocIds].join(", ")}], chunks=${chunks.length}`,
      ),
    );
  } else {
    assertions.push(
      assert(
        "Retrieved chunks include evidence",
        chunks.length > 0,
        `${chunks.length} chunks`,
      ),
    );
  }

  // 2. Answer contains expected citation formats
  for (const rule of spec.expectedCitationRules) {
    const matches = countMatches(rule.re, answer);
    assertions.push(
      assert(
        `Answer contains ${rule.label} citation`,
        spec.allowInsufficientEvidence && isInsufficientEvidence
          ? true
          : rule.required
            ? matches.length > 0
            : true,
        `found ${matches.length}: ${matches.slice(0, 5).join(", ")}`,
      ),
    );
  }

  // 3. Pipeline trace has no failed stages
  const failedStages = trace.filter((t) => t.status === "failed");
  assertions.push(
    assert(
      "No pipeline stages failed",
      spec.allowInsufficientEvidence && isInsufficientEvidence
        ? failedStages.every((stage) => stage.stage === "verify")
        : failedStages.length === 0,
      failedStages.length > 0
        ? `failed: ${failedStages.map((t) => t.stage).join(", ")}`
        : `${trace.length} stages completed`,
    ),
  );

  // 4. Answer is non-empty and substantive
  assertions.push(
    assert(
      "Answer is substantive (not empty or refusal)",
      answer.length > 0 && (spec.allowInsufficientEvidence ? true : !isInsufficientEvidence),
      `${answer.length} chars`,
    ),
  );

  // 5. Citations array in state has expected entries
  const stateCitations = result.citations ?? [];
  for (const rule of spec.expectedCitationRules) {
    const matchingCitations = stateCitations.filter((c) => countMatches(rule.re, c).length > 0);
    assertions.push(
      assert(
        `GraphState citations array has ${rule.label} entries`,
        spec.allowInsufficientEvidence && isInsufficientEvidence
          ? true
          : rule.required
            ? matchingCitations.length > 0
            : true,
        `${matchingCitations.length} matching citation(s)`,
      ),
    );
  }

  return assertions;
}

function reportSection(tag: string, assertions: AssertionResult[]): boolean {
  let allPassed = true;
  console.log(`\n── [${tag}] Assertions ──────────────────────`);
  for (const r of assertions) {
    const icon = r.passed ? "✅" : "❌";
    console.log(`  ${icon} ${r.label}${r.detail ? ` — ${r.detail}` : ""}`);
    if (!r.passed) allPassed = false;
  }
  console.log(
    allPassed
      ? `  ✅ [${tag}] ALL PASSED`
      : `  ❌ [${tag}] SOME FAILED`,
  );
  return allPassed;
}

// ── Question Specs ───────────────────────────────────────────────────

const QUESTIONS: QuestionSpec[] = [
  {
    tag: "Primary fixture",
    question:
      `In document "${primaryDocId}", what section covers information collection requirements for public comments?`,
    expectedDocIds: [primaryDocId],
    allowInsufficientEvidence: true,
    expectedCitationRules: [
      {
        label: `[${primaryDocId} p.<page>]`,
        re: citationRegex(primaryDocId),
        required: true,
      },
    ],
  },
  {
    tag: "Secondary fixture",
    question:
    `In document "${secondaryDocId}", what is the main point of the abstract and what limitation does it highlight?`,
    expectedDocIds: [secondaryDocId],
    expectedCitationRules: [
      {
        label: `[${secondaryDocId} p.<page>]`,
        re: citationRegex(secondaryDocId),
        required: true,
      },
    ],
  },
  {
    tag: "Cross-doc fixture",
    question:
      `Compare the treatment of memory and compliance requirements across documents "${primaryDocId}" and "${secondaryDocId}".`,
    expectedDocIds: [primaryDocId, secondaryDocId],
    expectedCitationRules: [
      {
        label: `[${primaryDocId} p.<page>]`,
        re: citationRegex(primaryDocId),
        required: true,
      },
      {
        label: `[${secondaryDocId} p.<page>]`,
        re: citationRegex(secondaryDocId),
        required: true,
      },
    ],
  },
];

// ── Main Entry Point ─────────────────────────────────────────────────

async function main() {
  console.log("╔══════════════════════════════════════════════════════════╗");
  console.log("║         Multi-Document RAG Smoke Test                   ║");
  console.log("║   Testing fixture-specific single-doc and multi-doc QA   ║");
  console.log("╚══════════════════════════════════════════════════════════╝");

  const allResults: { tag: string; passed: boolean }[] = [];

  for (const spec of QUESTIONS) {
    printBanner(`[${spec.tag}] ${spec.question}`);

    console.log("\n  Running pipeline…");
    const result: GraphState = await runRagQuery(spec.question, true);

    printSummary(result);

    const assertions = runAssertions(result, spec);
    const passed = reportSection(spec.tag, assertions);
    allResults.push({ tag: spec.tag, passed });
  }

  // ── Final Summary ────────────────────────────────────────────────

  const totalPassed = allResults.filter((r) => r.passed).length;
  const totalQuestions = allResults.length;

  console.log("\n╔══════════════════════════════════════════════════════════╗");
  console.log("║                    FINAL RESULTS                        ║");
  console.log("╠══════════════════════════════════════════════════════════╣");
  for (const r of allResults) {
    const icon = r.passed ? "✅ PASS" : "❌ FAIL";
    console.log(`║  ${icon}  ${r.tag.padEnd(46)}║`);
  }
  console.log("╠══════════════════════════════════════════════════════════╣");
  const summaryLine =
    totalPassed === totalQuestions
      ? `✅ ${totalPassed}/${totalQuestions} question suites passed`
      : `❌ ${totalPassed}/${totalQuestions} question suites passed`;
  console.log(`║  ${summaryLine.padEnd(56)}║`);
  console.log("╚══════════════════════════════════════════════════════════╝");

  if (totalPassed < totalQuestions) {
    process.exit(1);
  }

  console.log("\nSmoke test completed successfully.");
}

main().catch((err) => {
  console.error("\n❌ Smoke test crashed:", err instanceof Error ? err.message : err);
  if (err instanceof Error && err.stack) {
    console.error(err.stack);
  }
  process.exit(1);
});
