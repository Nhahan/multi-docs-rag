import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { appConfig } from "../lib/config";
import { buildEvidenceContext, summarizeCitations } from "../citations/citationExtractor";
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

type GroundingVerdict = {
  supported: boolean;
  reason: string;
};

type ItemGroundingVerdict = {
  supported: boolean;
  reason: string;
};

type RequestedItemPlan = {
  items: string[];
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

const parseItemGroundingVerdict = (raw: string): ItemGroundingVerdict | null => {
  const parsed = parseGroundingVerdict(raw);
  return parsed ? { supported: parsed.supported, reason: parsed.reason } : null;
};

const parseRequestedItemPlan = (raw: string): RequestedItemPlan | null => {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/u);
  const candidate = fenced?.[1]?.trim() ?? trimmed;

  try {
    const parsed = JSON.parse(candidate) as unknown;
    if (
      parsed &&
      typeof parsed === "object" &&
      Array.isArray((parsed as { items?: unknown }).items)
    ) {
      const items = (parsed as { items: unknown[] }).items
        .filter((value): value is string => typeof value === "string")
        .map((value) => value.trim())
        .filter(Boolean)
        .slice(0, 8);
      return { items };
    }
  } catch {
    return null;
  }

  return null;
};

const planRequestedItems = async (question: string): Promise<string[]> => {
  const chat = getChatModel();
  try {
    const response = await chat.invoke([
      new SystemMessage(`
Break a user question into the explicit answer items it requests.

Rules:
- Return JSON only: {"items":["..."]}.
- Preserve exact identifiers, document ids, filenames, quoted strings, and numbers when relevant.
- Each item should be an answerable sub-request, not a retrieval keyword list.
- Keep items concise.
- Do not invent extra items beyond what the user explicitly asked for.
- If the question is already a single request, return one item.
`),
      new HumanMessage(`Question:\n${question}`),
    ]);
    const plan = parseRequestedItemPlan(parseChatResponse(response.content));
    if (plan && plan.items.length > 0) {
      return plan.items;
    }
  } catch {
    // fall through
  }
  return [question];
};

const normalizeAnswerPresentation = async (
  question: string,
  answer: string,
): Promise<string> => {
  const chat = getChatModel();
  const response = await chat.invoke([
    new SystemMessage(`
You are rewriting a grounded RAG answer for presentation only.

Preserve all supported claims and all explicit unsupported/insufficient-evidence statements.
Do not add new facts.
Do not remove supported facts.
Do not include citation markup, source names, file names, page numbers, document locations, or separate Source/Sources/provenance sections unless the user explicitly asked for them.
Return only the rewritten user-facing answer.
`),
    new HumanMessage(`Question:\n${question}\n\nDraft answer:\n${answer}`),
  ]);

  const normalized = parseChatResponse(response.content).trim();
  return normalized.length > 0 ? normalized : answer;
};

const verifyGroundingWithModel = async (
  question: string,
  requestedItems: string[],
  answer: string,
  chunks: ScoredChunk[],
): Promise<GroundingVerdict | null> => {
  if (chunks.length === 0) {
    return null;
  }

  const context = chunks
    .map((chunk) => {
      const { document_id, source_file, page } = chunk.chunk.metadata;
      return `${document_id} (${source_file}) p.${page}\n${chunk.chunk.text}`;
    })
    .join("\n\n");

  const chat = getChatModel();
  const response = await chat.invoke([
    new SystemMessage(`
You are verifying whether a RAG answer is fully supported by the retrieved evidence for the user's question.

Return JSON only: {"supported":true|false,"reason":"..."}

Mark supported=false if any of these are true:
- the answer includes a factual claim not directly supported by the retrieved context
- the answer gives a numeric value not explicitly stated in the retrieved context
- the answer treats an index row, catalog line, taxonomy label, or bare heading as if it were a descriptive requirement or control
- the answer cites text that is not actually relevant to the user's question
- the answer overgeneralizes from retrieved evidence
- the answer includes source-location citation markup, source attributions, or document-location formatting that was not requested by the user
- the answer adds a separate "Source", "Sources", provenance, or document-reference section that the user did not ask for
- the answer claims to satisfy a requested item even though that item is not actually supported by the retrieved evidence

Do not mark the answer unsupported merely because it explicitly says some requested items are not supported by the retrieved evidence.
If the answer gives grounded claims for supported parts and clearly labels the remaining parts as unsupported, insufficient, or not available from the retrieved evidence, that is acceptable and should be marked supported=true.

Be conservative. If support is ambiguous, return supported=false.
`),
    new HumanMessage(
      `Question:\n${question}\n\nRequested items:\n- ${requestedItems.join("\n- ")}\n\nAnswer:\n${answer}\n\nRetrieved context:\n${context}`,
    ),
  ]);

  return parseGroundingVerdict(parseChatResponse(response.content));
};

const verifyRequestedItemsWithModel = async (
  question: string,
  requestedItems: string[],
  answer: string,
  chunks: ScoredChunk[],
): Promise<ItemGroundingVerdict | null> => {
  if (chunks.length === 0 || requestedItems.length === 0) {
    return null;
  }

  const context = chunks
    .map((chunk) => {
      const { document_id, source_file, page } = chunk.chunk.metadata;
      return `${document_id} (${source_file}) p.${page}\n${chunk.chunk.text}`;
    })
    .join("\n\n");

  const chat = getChatModel();
  const response = await chat.invoke([
    new SystemMessage(`
You are verifying whether an answer handles each requested item correctly against retrieved evidence.

Return JSON only: {"supported":true|false,"reason":"..."}.

Return supported=true if:
- supported requested items are answered with evidence-backed claims, and
- unsupported requested items are explicitly marked as not supported by the retrieved evidence, insufficient, or unavailable.

Return supported=false if:
- the answer claims to satisfy an item that is not actually supported by the retrieved evidence
- the answer omits a clearly supported requested item without explicitly marking it unsupported
- the answer invents extra requested items or extra unsupported items
- the answer uses source attributions or citation formatting that the user did not ask for
`),
    new HumanMessage(
      `Question:\n${question}\n\nRequested items:\n- ${requestedItems.join("\n- ")}\n\nAnswer:\n${answer}\n\nRetrieved context:\n${context}`,
    ),
  ]);

  return parseItemGroundingVerdict(parseChatResponse(response.content));
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
    const evidence: EvidenceAvailability = {
      passed: false,
      candidate_count: 0,
      top_score: 0,
      avg_score: 0,
      reasons: ["No retrieval result returned from retrieval node."],
    };
    return {
      ...state,
      evidence,
      trace: appendTrace(state.trace, "evidence_gate", "failed", "Evidence gate cannot run without retrieval result.", 0, {
        failure: true,
        evidence,
      }),
    };
  }

  const evidence = evaluateEvidenceAvailability(primaryRetrieval.chunks);
  const coverageSummary = getDocumentCoverageSummary(primaryRetrieval.chunks);

  const evidencePassed = evidence.passed;
  return {
    ...state,
    evidence,
    trace: appendTrace(
      state.trace,
      "evidence_gate",
      evidencePassed ? "passed" : "degraded",
      evidencePassed ? "Evidence is available for generation." : "No evidence is available for generation.",
      evidence.avg_score,
      {
        evidenceAvailability: evidence,
        coverageSummary,
        retrieverReranked: primaryRetrieval.reranked,
      },
    ),
  };
};

export const generateNode = async (state: GraphState): Promise<GraphState> => {
  const chunks = state.retrieval?.chunks ?? [];
  const hasCrossCoverage = hasMultiDocumentCoverage(chunks);
  const evidencePassed = state.evidence?.passed ?? false;
  const requestedItems =
    state.requested_items && state.requested_items.length > 0
      ? state.requested_items
      : await planRequestedItems(state.question);

  if (chunks.length === 0) {
    return {
      ...state,
      requested_items: requestedItems,
      answer: EVIDENCE_WARNING,
      citations: [],
      trace: appendTrace(state.trace, "generate", "failed", "No chunks available for generation.", 0, { evidenceCount: 0 }),
    };
  }

  const analysis = analyseCrossDocEvidence(chunks);
  const systemPrompt = hasCrossCoverage
    ? buildCrossDocSystemPrompt(analysis.structuredContext, analysis.synthesisInstruction, !evidencePassed)
    : buildRagSystemPrompt(buildEvidenceContext(chunks), !evidencePassed);

  const qualityPrompt = evidencePassed
    ? "Proceed only with evidence-backed claims."
    : "Available evidence is limited. Provide a conservative response and avoid unsupported claims.";

  const chat = getChatModel();
  const response = await chat.invoke([
    new SystemMessage(systemPrompt),
    new HumanMessage(
      `Question: ${state.question}\n\n` +
        `Requested answer items:\n- ${requestedItems.join("\n- ")}\n\n` +
        `${hasCrossCoverage ? `Using ${analysis.documentCount} source document group(s).` : ""}\n` +
        `Retrieved ${chunks.length} candidate chunks.\n${qualityPrompt}`,
    ),
  ]);

  const answerRaw = parseChatResponse(response.content).trim();
  const answer = answerRaw.length > 0 ? answerRaw : EVIDENCE_WARNING;

  return {
    ...state,
    requested_items: requestedItems,
    answer,
    citations: [],
    trace: appendTrace(
      state.trace,
      "generate",
      "passed",
      hasCrossCoverage
        ? `Generated response from ${analysis.documentCount} document(s) using ${chunks.length} chunks.`
        : `Generated response from ${chunks.length} chunk(s).`,
      undefined,
      {
        evidencePassed,
        requestedItems,
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
  const answer = (state.answer ?? "").trim();
  const hasEvidence = (state.retrieval?.chunks.length ?? 0) > 0;
  const requestedItems =
    state.requested_items && state.requested_items.length > 0
      ? state.requested_items
      : await planRequestedItems(state.question);

  if (!hasEvidence) {
    return {
      ...state,
      requested_items: requestedItems,
      answer: EVIDENCE_WARNING,
      citations: [],
      trace: appendTrace(state.trace, "verify", "failed", "No evidence retrieved for verification.", 0, {
        finalAnswer: EVIDENCE_WARNING,
      }),
    };
  }

  if (!answer || answer.includes(EVIDENCE_WARNING)) {
    return {
      ...state,
      requested_items: requestedItems,
      answer: EVIDENCE_WARNING,
      citations: [],
      trace: appendTrace(
        state.trace,
        "verify",
        "passed",
        "Model reported insufficient evidence and returned the standard insufficiency message.",
        0.95,
        {},
      ),
    };
  }

  const normalizedAnswer = await normalizeAnswerPresentation(
    state.question,
    answer,
  );

  const itemVerdict = await verifyRequestedItemsWithModel(
    state.question,
    requestedItems,
    normalizedAnswer,
    state.retrieval?.chunks ?? [],
  );

  if (itemVerdict && !itemVerdict.supported) {
    return {
      ...state,
      requested_items: requestedItems,
      answer: EVIDENCE_WARNING,
      citations: [],
      trace: appendTrace(
        state.trace,
        "verify",
        "failed",
        "Requested-item grounding verifier rejected the generated answer.",
        0.2,
        {
          verifierReason: itemVerdict.reason,
        },
      ),
    };
  }

  const groundingVerdict = await verifyGroundingWithModel(
    state.question,
    requestedItems,
    normalizedAnswer,
    state.retrieval?.chunks ?? [],
  );

  if (groundingVerdict && !groundingVerdict.supported) {
    return {
      ...state,
      requested_items: requestedItems,
      answer: EVIDENCE_WARNING,
      citations: [],
      trace: appendTrace(
        state.trace,
        "verify",
        "failed",
        "Grounding verifier rejected the generated answer.",
        0.2,
        {
          verifierReason: groundingVerdict.reason,
        },
      ),
    };
  }

  return {
    ...state,
    requested_items: requestedItems,
    answer: normalizedAnswer,
    citations: [],
    trace: appendTrace(
      state.trace,
      "verify",
      "passed",
      "Verified generated answer against retrieved evidence.",
      undefined,
      {},
    ),
  };
};
