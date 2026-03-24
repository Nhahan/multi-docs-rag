#!/usr/bin/env tsx

if (typeof process.loadEnvFile === "function") {
  process.loadEnvFile();
}

import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { runRagQuery } from "../src/graph/run";
import { appConfig } from "../src/lib/config";

type ManualReviewCase = {
  id: string;
  category: string;
  question: string;
  reviewer_focus?: string[];
  notes?: string;
};

type ManualReviewSpec = {
  title?: string;
  description?: string;
  cases: ManualReviewCase[];
};

const DEFAULT_CASES_PATH = "scripts/manual-review/cases.local.json";

const getArgValue = (flag: string): string | undefined => {
  const index = process.argv.indexOf(flag);
  if (index === -1) return undefined;
  return process.argv[index + 1];
};

const excerpt = (text: string, limit = 220) =>
  text.replace(/\s+/gu, " ").trim().slice(0, limit);

const printDivider = (label: string) => {
  const line = "═".repeat(72);
  console.log(`\n${line}`);
  console.log(label);
  console.log(line);
};

const loadCases = async (filePath: string): Promise<ManualReviewSpec> => {
  const raw = await readFile(resolve(process.cwd(), filePath), "utf8");
  return JSON.parse(raw) as ManualReviewSpec;
};

const main = async () => {
  const filePath = getArgValue("--file") ?? DEFAULT_CASES_PATH;
  const onlyCase = getArgValue("--case");
  const adHocQuestion = getArgValue("--question");
  const spec = adHocQuestion
    ? {
        title: "Ad-hoc manual review",
        description: "Single ad-hoc manual review question provided on the command line.",
        cases: [
          {
            id: getArgValue("--id") ?? "adhoc",
            category: getArgValue("--category") ?? "ad-hoc",
            question: adHocQuestion,
            reviewer_focus: ["manual semantic review"],
          },
        ],
      }
    : await loadCases(filePath);
  const selectedCases = onlyCase
    ? spec.cases.filter((item) => item.id === onlyCase)
    : spec.cases;

  if (selectedCases.length === 0) {
    throw new Error(`No manual review cases matched. file=${filePath} case=${onlyCase ?? "ALL"}`);
  }

  console.log(`Manual review set: ${spec.title ?? "Unnamed set"}`);
  if (spec.description) {
    console.log(spec.description);
  }
  console.log(`Cases: ${selectedCases.length}`);
  console.log(`Refusal text: ${appConfig.messages.insufficientEvidence}`);

  for (const item of selectedCases) {
    printDivider(`${item.id} [${item.category}]`);
    console.log(`Question: ${item.question}`);
    if (item.notes) {
      console.log(`Notes: ${item.notes}`);
    }
    if (item.reviewer_focus?.length) {
      console.log(`Reviewer focus: ${item.reviewer_focus.join(", ")}`);
    }

    const result = await runRagQuery(item.question, true);
    const chunks = result.retrieval?.chunks ?? [];
    const documentMix = Array.from(
      chunks
        .reduce((acc, chunk) => {
        const key = chunk.chunk.metadata.document_id;
        const existing = acc.get(key);
        if (existing) {
          existing.count += 1;
          existing.topScore = Math.max(existing.topScore, chunk.score);
        } else {
          acc.set(key, {
            documentId: key,
            sourceFile: chunk.chunk.metadata.source_file,
            count: 1,
            topScore: chunk.score,
          });
        }
        return acc;
      }, new Map<string, { documentId: string; sourceFile: string; count: number; topScore: number }>())
        .values(),
    );

    console.log("\nAnswer:");
    console.log(result.answer ?? "");

    console.log("\nCitations:");
    console.log((result.citations ?? []).length ? result.citations!.join(", ") : "(none)");

    console.log("\nEvidence availability:");
    if (result.evidence) {
      console.log(
        `passed=${result.evidence.passed} candidate_count=${result.evidence.candidate_count} top=${result.evidence.top_score.toFixed(3)} avg=${result.evidence.avg_score.toFixed(3)}`,
      );
      if (result.evidence.reasons.length > 0) {
        console.log(result.evidence.reasons.join("\n"));
      }
    } else {
      console.log("(none)");
    }

    console.log("\nDocument mix:");
    if (documentMix.length === 0) {
      console.log("(none)");
    } else {
      for (const doc of documentMix.sort((a, b) => b.count - a.count || b.topScore - a.topScore)) {
        console.log(
          `- ${doc.documentId} (${doc.sourceFile}): chunks=${doc.count}, top_score=${doc.topScore.toFixed(3)}`,
        );
      }
    }

    console.log("\nTop chunks:");
    for (const chunk of chunks.slice(0, 6)) {
      console.log(
        `- [${chunk.chunk.metadata.document_id} p.${chunk.chunk.metadata.page}] score=${chunk.score.toFixed(3)}${
          typeof chunk.rerankScore === "number" ? ` rerank=${chunk.rerankScore.toFixed(3)}` : ""
        } ${excerpt(chunk.chunk.text)}`,
      );
    }

    console.log("\nTrace:");
    for (const trace of result.trace ?? []) {
      console.log(
        `- ${trace.stage}: ${trace.status}${typeof trace.score === "number" ? ` (${trace.score.toFixed(3)})` : ""} ${trace.message}`,
      );
      if (trace.details && Object.keys(trace.details).length > 0) {
        console.log(`  details: ${JSON.stringify(trace.details, null, 2)}`);
      }
    }
  }
};

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
