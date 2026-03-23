import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { appConfig } from "../lib/config";
import {
  allCitationsSupported,
  citationLabel,
  extractCitationsFromAnswer,
  extractCitationsFromChunks,
  findInvalidCitationLikeTags,
  findUnsupportedCitations,
  hasCitations,
  summarizeCitations,
} from "../citations/citationExtractor";
import { ensureInlineCitations } from "../citations/citationFormatter";
import { getChatModel } from "../lib/llm";
import { getIndexStore } from "../lib/store";
import { buildCrossDocSystemPrompt, buildRagSystemPrompt } from "../prompts/rag";
import { hasMultiDocumentCoverage, getDocumentCoverageSummary, retrieveCrossDocument } from "../retrieval/crossDocRetriever";
import { EvidenceAvailability, GraphState, PipelineStage, PipelineStageTrace, PipelineStatus, ScoredChunk } from "../types/rag";
import { analyseCrossDocEvidence, crossDocCoverageSummary } from "../synthesis/crossDocSynthesizer";

const EVIDENCE_WARNING = appConfig.messages.insufficientEvidence;

const appendTrace = (
  traces: PipelineStageTrace[] = [],
  stage: PipelineStage,
  status: PipelineStatus,
  message: string,
  score?: number,
  details?: Record<string, unknown>,
) => [...traces, { stage, status, message, score, details }];

const evaluateEvidenceAvailability = (chunks: ScoredChunk[]): EvidenceAvailability => {
  const candidate_count = chunks.length;
  const top_score = chunks[0]?.score ?? 0;
  const avg_score = chunks.length === 0 ? 0 : chunks.reduce((sum, chunk) => sum + chunk.score, 0) / chunks.length;
  const reasons: string[] = [];

  if (candidate_count === 0) {
    reasons.push("No evidence chunks were retrieved.");
  }

  return {
    passed: candidate_count > 0,
    candidate_count,
    top_score,
    avg_score,
    reasons,
  };
};

const parseChatResponse = (content: unknown): string => {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((value) => {
        if (typeof value === "string") return value;
        if (value && typeof value === "object" && "text" in value) return String((value as { text?: unknown }).text ?? "");
        return "";
      })
      .filter(Boolean)
      .join("\n")
      .trim();
  }
  if (content && typeof content === "object" && "toString" in content) return String(content);
  return "";
};

type CitationValidationSnapshot = {
  uniqueCited: string[];
  hasAnyCitation: boolean;
  unsupportedCitations: string[];
  invalidCitationLikeTags: string[];
  passed: boolean;
};

const validateAnswerCitations = (
  answer: string,
  available: Set<string>,
): CitationValidationSnapshot => {
  const uniqueCited = extractCitationsFromAnswer(answer);
  const hasAnyCitation = hasCitations(answer);
  const unsupportedCitations = findUnsupportedCitations(answer, available);
  const invalidCitationLikeTags = findInvalidCitationLikeTags(answer, available);
  const passed =
    hasAnyCitation &&
    unsupportedCitations.length === 0 &&
    invalidCitationLikeTags.length === 0 &&
    allCitationsSupported(answer, available);

  return {
    uniqueCited,
    hasAnyCitation,
    unsupportedCitations,
    invalidCitationLikeTags,
    passed,
  };
};

const normalizeNonCitationBracketTags = (
  answer: string,
  available: Set<string>,
): string => {
  const invalidTags = findInvalidCitationLikeTags(answer, available);
  let normalized = answer;

  for (const tag of invalidTags) {
    const inner = tag.slice(1, -1).trim();
    if (/\bp\.\d+\b/i.test(inner) || inner.startsWith("p.")) {
      continue;
    }
    normalized = normalized.split(tag).join(inner);
  }

  return normalized;
};

type GroundingVerdict = {
  supported: boolean;
  reason: string;
};

const parseGroundingVerdict = (raw: string): GroundingVerdict | null => {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/u);
  const candidate = fenced?.[1]?.trim() ?? trimmed;

  try {
    const parsed = JSON.parse(candidate) as unknown;
    if (
      parsed &&
      typeof parsed === "object" &&
      typeof (parsed as { supported?: unknown }).supported === "boolean"
    ) {
      return {
        supported: Boolean((parsed as { supported: boolean }).supported),
        reason:
          typeof (parsed as { reason?: unknown }).reason === "string"
            ? String((parsed as { reason?: unknown }).reason)
            : "",
      };
    }
  } catch {
    return null;
  }

  return null;
};

const verifyGroundingWithModel = async (
  question: string,
  answer: string,
  chunks: ScoredChunk[],
  citedLabels: Set<string>,
): Promise<GroundingVerdict | null> => {
  const citedChunks = chunks.filter((chunk) =>
    citedLabels.has(citationLabel(chunk.chunk.metadata)),
  );

  if (citedChunks.length === 0) {
    return null;
  }

  const context = citedChunks
    .map((chunk) => {
      const label = citationLabel(chunk.chunk.metadata);
      return `${label}\n${chunk.chunk.text}`;
    })
    .join("\n\n");

  const chat = getChatModel();
  const response = await chat.invoke([
    new SystemMessage(`
You are verifying whether a RAG answer is fully supported by the cited evidence for the user's question.

Return JSON only: {"supported":true|false,"reason":"..."}

Mark supported=false if any of these are true:
- the answer includes a factual claim not directly supported by the cited context
- the answer gives a numeric value not explicitly stated in the cited context
- the answer treats an index row, catalog line, taxonomy label, or bare heading as if it were a descriptive requirement or control
- the answer cites text that is not actually relevant to the user's question
- the answer overgeneralizes from cited evidence

Be conservative. If support is ambiguous, return supported=false.
`),
    new HumanMessage(
      `Question:\n${question}\n\nAnswer:\n${answer}\n\nCited context:\n${context}`,
    ),
  ]);

  return parseGroundingVerdict(parseChatResponse(response.content));
};

export const retrieveNode = async (state: GraphState): Promise<GraphState> => {
  const { lexical, vector } = await getIndexStore({ allowDenseMissing: true });
  const retrieval = await retrieveCrossDocument(lexical, vector, {
    query: state.question,
    topK: appConfig.retrieval.topK,
  });

  const chunks = retrieval.chunks;
  const summary = summarizeCitations(chunks);
  const documentCoverage = getDocumentCoverageSummary(chunks);
  const multiDoc = hasMultiDocumentCoverage(chunks);
  const fallbackInfo =
    retrieval.dense_enabled === false
      ? "lexical-only"
      : retrieval.lexical_enabled === false
        ? "dense-only"
        : "dense+lexical";

  return {
    ...state,
    retrieval,
    trace: appendTrace(
      state.trace,
      "retrieve",
      chunks.length === 0 ? "failed" : "passed",
      `Unified retrieval returned ${summary.sourceCount} chunks.`,
      chunks[0]?.score,
      {
        topChunkSummary: summary.top,
        multiDocCoverage: multiDoc,
        documentCoverage,
        denseMode: fallbackInfo,
        degradationReasons: retrieval.degradation_reasons ?? [],
        denseEnabled: retrieval.dense_enabled,
        lexicalEnabled: retrieval.lexical_enabled,
      },
    ),
  };
};

export const evidenceGateNode = async (state: GraphState): Promise<GraphState> => {
  const primaryRetrieval = state.retrieval;
  if (!primaryRetrieval) {
    const quality: EvidenceAvailability = {
      passed: false,
      candidate_count: 0,
      top_score: 0,
      avg_score: 0,
      reasons: ["No retrieval result returned from retrieval node."],
    };
    return {
      ...state,
      quality,
      trace: appendTrace(state.trace, "evidence_gate", "failed", "Evidence gate cannot run without retrieval result.", 0, {
        failure: true,
        quality,
      }),
    };
  }

  const quality = evaluateEvidenceAvailability(primaryRetrieval.chunks);
  const coverageSummary = getDocumentCoverageSummary(primaryRetrieval.chunks);

  const qualityPassed = quality.passed;
  return {
    ...state,
    quality,
    trace: appendTrace(
      state.trace,
      "evidence_gate",
      qualityPassed ? "passed" : "degraded",
      qualityPassed ? "Evidence is available for generation." : "No evidence is available for generation.",
      quality.avg_score,
      {
        evidenceQuality: quality,
        coverageSummary,
        retrieverReranked: primaryRetrieval.reranked,
      },
    ),
  };
};

export const generateNode = async (state: GraphState): Promise<GraphState> => {
  const chunks = state.retrieval?.chunks ?? [];
  const citations = extractCitationsFromChunks(chunks);
  const hasCrossCoverage = hasMultiDocumentCoverage(chunks);
  const qualityPassed = state.quality?.passed ?? false;

  if (chunks.length === 0) {
    return {
      ...state,
      answer: EVIDENCE_WARNING,
      citations: [],
      trace: appendTrace(state.trace, "generate", "failed", "No chunks available for generation.", 0, { evidenceCount: 0 }),
    };
  }

  const analysis = analyseCrossDocEvidence(chunks);
  const systemPrompt = hasCrossCoverage
    ? buildCrossDocSystemPrompt(analysis.structuredContext, analysis.synthesisInstruction, !qualityPassed)
    : buildRagSystemPrompt(summarizeCitations(chunks).context, !qualityPassed);

  const qualityPrompt = qualityPassed
    ? "Proceed only with evidence-backed claims and include citations."
    : "Available evidence is limited. Provide a conservative, citation-grounded response and avoid unsupported claims.";

  const chat = getChatModel();
  const response = await chat.invoke([
    new SystemMessage(systemPrompt),
    new HumanMessage(
      `Question: ${state.question}\n\n` +
        `${hasCrossCoverage ? `Using ${analysis.documentCount} source document group(s).` : ""}\n` +
        `Retrieved ${chunks.length} candidate chunks.\n${qualityPrompt}\n\n` +
        `Use citation labels only from available chunks: ${citations.join(", ") || "none"}.`,
    ),
  ]);

  const answerRaw = parseChatResponse(response.content).trim();
  const draftedAnswer = answerRaw.length > 0 ? answerRaw : EVIDENCE_WARNING;
  const answer =
    draftedAnswer === EVIDENCE_WARNING
      ? draftedAnswer
      : ensureInlineCitations(draftedAnswer, chunks);

  return {
    ...state,
    answer,
    citations: extractCitationsFromChunks(chunks),
    trace: appendTrace(
      state.trace,
      "generate",
      "passed",
      hasCrossCoverage
        ? `Generated response from ${analysis.documentCount} document(s) using ${chunks.length} chunks.`
        : `Generated response from ${chunks.length} chunk(s).`,
      undefined,
      {
        qualityPassed,
        sourceCount: chunks.length,
        isMultiDocument: analysis.isMultiDocument,
        documentCount: analysis.documentCount,
        crossDocCoverage: crossDocCoverageSummary(analysis),
        chunkSample: analysis.structuredContext.slice(0, 300),
      },
    ),
  };
};

export const verifyNode = async (state: GraphState): Promise<GraphState> => {
  const available = new Set(state.citations?.map((citation) => citation) ?? []);
  const answer = state.answer ?? "";
  const hasEvidence = (state.retrieval?.chunks.length ?? 0) > 0;

  if (!hasEvidence) {
    return {
      ...state,
      answer: EVIDENCE_WARNING,
      trace: appendTrace(state.trace, "verify", "failed", "No evidence retrieved for verification.", 0, {
        finalAnswer: EVIDENCE_WARNING,
      }),
    };
  }

  if (answer.includes(EVIDENCE_WARNING)) {
    return {
      ...state,
      trace: appendTrace(
        state.trace,
        "verify",
        "passed",
        "Model reported insufficient evidence and returned the standard insufficiency message.",
        0.95,
        {
          availableCitations: [...available],
        },
      ),
    };
  }

  let finalAnswer = normalizeNonCitationBracketTags(answer, available);
  const validation = validateAnswerCitations(finalAnswer, available);

  const usedCitationSet = new Set(validation.uniqueCited);

  if (!validation.passed) {
    return {
      ...state,
      answer: EVIDENCE_WARNING,
      citations: [],
      trace: appendTrace(
        state.trace,
        "verify",
        validation.hasAnyCitation ? "failed" : "failed",
        validation.hasAnyCitation
          ? validation.invalidCitationLikeTags.length > 0
            ? "Generated answer contained malformed citation-like tags."
            : "Claims were not fully supported by retrieved evidence."
          : "Generated answer had no valid citation tags.",
        0.31,
        {
          citedCitations: validation.uniqueCited,
          availableCitations: [...available],
          invalidCitation: validation.unsupportedCitations.length > 0,
          unsupportedCitations: validation.unsupportedCitations,
          invalidCitationLikeTags: validation.invalidCitationLikeTags,
        },
      ),
    };
  }

  const groundingVerdict = await verifyGroundingWithModel(
    state.question,
    finalAnswer,
    state.retrieval?.chunks ?? [],
    usedCitationSet,
  );

  if (groundingVerdict && !groundingVerdict.supported) {
    return {
      ...state,
      answer: EVIDENCE_WARNING,
      citations: [],
      trace: appendTrace(
        state.trace,
        "verify",
        "failed",
        "Grounding verifier rejected the generated answer.",
        0.2,
        {
          citedCitations: validation.uniqueCited,
          availableCitations: [...available],
          verifierReason: groundingVerdict.reason,
        },
      ),
    };
  }

  return {
    ...state,
    answer: finalAnswer,
    citations: usedCitationSet.size > 0 ? [...usedCitationSet] : state.citations,
    trace: appendTrace(
      state.trace,
      "verify",
      "passed",
      `Verified ${usedCitationSet.size} citation(s) against retrieved evidence.`,
      undefined,
      {
        citedCitations: validation.uniqueCited,
        availableCitations: [...available],
      },
    ),
  };
};
