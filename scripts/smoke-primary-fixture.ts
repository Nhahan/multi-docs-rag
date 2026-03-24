/**
 * Smoke test: fixture-focused single-document RAG pipeline
 *
 * Imports the RAG pipeline, runs a fixture-specific single-document question, and asserts:
 *   1. Retrieved evidence includes the configured primary fixture document
 *   2. The generated answer contains at least one citation for that document
 *   3. The pipeline trace completes without failures
 */

if (typeof process.loadEnvFile === "function") {
  process.loadEnvFile();
}

import { runRagQuery } from "../src/graph/run";
import type { GraphState } from "../src/types/rag";
import { appConfig } from "../src/lib/config";
import { findCorpusDocBySourceFile } from "./smoke/utils";

// ── Helpers ──────────────────────────────────────────────────────────

const escapeRegExp = (value: string) =>
  value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const citationRegex = (prefix: string) =>
  new RegExp(`\\[${escapeRegExp(prefix)} p\\.\\d+\\]`, "g");

const primaryFixtureDoc = findCorpusDocBySourceFile("CFR - Code of Federal Regulations.pdf", 0);
const PRIMARY_DOC_PREFIX = primaryFixtureDoc?.document_id ?? "document-1";
const PRIMARY_DOC_CITATION_RE = citationRegex(PRIMARY_DOC_PREFIX);

interface AssertionResult {
  label: string;
  passed: boolean;
  detail?: string;
}

function assert(label: string, ok: boolean, detail?: string): AssertionResult {
  return { label, passed: ok, detail };
}

function reportResults(results: AssertionResult[]): boolean {
  let allPassed = true;
  console.log("\n── Assertions ──────────────────────────────");
  for (const r of results) {
    const icon = r.passed ? "✅" : "❌";
    console.log(`  ${icon} ${r.label}${r.detail ? ` — ${r.detail}` : ""}`);
    if (!r.passed) allPassed = false;
  }
  console.log("────────────────────────────────────────────");
  console.log(allPassed ? "All assertions passed." : "Some assertions FAILED.");
  return allPassed;
}

// ── Main ─────────────────────────────────────────────────────────────

async function main() {
  const question =
    `In document "${PRIMARY_DOC_PREFIX}", what section covers information collection requirements for public comments?`;

  console.log("Running primary-fixture single-document smoke test…");
  console.log("Q:", question);

  const result: GraphState = await runRagQuery(question, true);

  // Print summary
  console.log(
    "Evidence:",
    result.evidence
      ? `passed=${result.evidence.passed}, top=${result.evidence.top_score.toFixed(3)}, avg=${result.evidence.avg_score.toFixed(3)}`
      : "N/A",
  );
  console.log("Answer (first 300 chars):\n ", result.answer?.slice(0, 300));
  if (result.citations?.length) {
    console.log("Citations:", result.citations.join(", "));
  }
  if (result.trace?.length) {
    console.log(
      "Trace:",
      result.trace
        .map(
          (t) =>
            `${t.stage}=${t.status}${typeof t.score === "number" ? ` (${t.score.toFixed(3)})` : ""}`,
        )
        .join(" -> "),
    );
  }

  // ── Assertions ───────────────────────────────────────────────────

  const assertions: AssertionResult[] = [];

  // 1. Retrieved chunks include the primary fixture document
  const chunks = result.retrieval?.chunks ?? [];
  const hasTargetCitationDoc = chunks.some((c) => c.chunk.metadata.document_id === PRIMARY_DOC_PREFIX);
  const docIdsSeen = Array.from(new Set(chunks.map((c) => c.chunk.metadata.document_id)));
  assertions.push(
    assert(
      `Retrieved evidence includes ${PRIMARY_DOC_PREFIX}`,
      chunks.length > 0 && hasTargetCitationDoc,
      `${chunks.length} chunks, sources: [${docIdsSeen.join(", ")}]`,
    ),
  );

  // 2. Answer contains at least one primary-document citation
  const answer = result.answer ?? "";
  const hasPrimaryCitation = PRIMARY_DOC_CITATION_RE.test(answer);
  const primaryMatches = answer.match(PRIMARY_DOC_CITATION_RE) ?? [];
  assertions.push(
    assert(
      `Answer contains [${PRIMARY_DOC_PREFIX} p.<page>] citation`,
      hasPrimaryCitation,
      `found ${primaryMatches.length}: ${primaryMatches.slice(0, 5).join(", ")}`,
    ),
  );

  // 3. Pipeline trace has no "failed" stages
  const trace = result.trace ?? [];
  const failedStages = trace.filter((t) => t.status === "failed");
  assertions.push(
    assert(
      "No pipeline stages failed",
      failedStages.length === 0,
      failedStages.length > 0
        ? `failed: ${failedStages.map((t) => t.stage).join(", ")}`
        : `${trace.length} stages completed`,
    ),
  );

  // 4. Answer is non-empty and not the refusal phrase
  const refusal = appConfig.messages.insufficientEvidence;
  assertions.push(
    assert(
      "Answer is substantive (not empty or refusal)",
      answer.length > 0 && !answer.includes(refusal),
      `${answer.length} chars`,
    ),
  );

  // 5. Citations array in GraphState matches inline citations
  const stateCitations = result.citations ?? [];
  const statePrimaryCitations = stateCitations.filter((c) =>
    c.match(citationRegex(PRIMARY_DOC_PREFIX)),
  );
  assertions.push(
    assert(
      `GraphState citations array has ${PRIMARY_DOC_PREFIX} entries`,
      statePrimaryCitations.length > 0,
      `${statePrimaryCitations.length} ${PRIMARY_DOC_PREFIX} citation(s) in state`,
    ),
  );

  // ── Report ─────────────────────────────────────────────────────

  const allPassed = reportResults(assertions);
  if (!allPassed) {
    process.exit(1);
  }
}

main().catch((err) => {
  console.error(
    "Smoke test crashed:",
    err instanceof Error ? err.message : err,
  );
  process.exit(1);
});
