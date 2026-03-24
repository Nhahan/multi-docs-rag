import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { appConfig } from "../lib/config";
import { buildEvidenceContext, summarizeCitations } from "../citations/citationExtractor";
import { getChatModel } from "../lib/llm";
import { getIndexStore } from "../lib/store";
import { buildCrossDocSystemPrompt, buildRagSystemPrompt } from "../prompts/rag";
import { hasMultiDocumentCoverage, getDocumentCoverageSummary, retrieveCrossDocument } from "../retrieval/crossDocRetriever";
import { retrieveHybrid } from "../retrieval/hybrid";
import { expandRetrievalQueries } from "../retrieval/queryExpander";
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

const dedupeNormalizedStrings = (values: string[]): string[] => {
  const seen = new Set<string>();
  const deduped: string[] = [];
  for (const value of values) {
    const normalized = value.replace(/\s+/gu, " ").trim();
    if (!normalized) continue;
    const key = normalized.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(normalized);
  }
  return deduped;
};

const extractReferencedDocumentIds = (text: string): string[] => {
  const normalized = text.toLowerCase();
  const matches: string[] = [];

  for (const entry of appConfig.corpus) {
    if (
      normalized.includes(entry.document_id.toLowerCase()) ||
      normalized.includes(entry.source_file.toLowerCase())
    ) {
      matches.push(entry.document_id);
    }
  }

  return dedupeNormalizedStrings(matches);
};

const getRetrievedDocumentIds = (chunks: ScoredChunk[]): Set<string> =>
  new Set(
    chunks
      .map((chunk) => chunk.chunk.metadata.document_id)
      .filter((value): value is string => typeof value === "string" && value.length > 0),
  );

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

type ItemAnswerPlan = {
  item: string;
  supported: boolean;
  answer: string;
};

type SingleItemResolution = {
  supported: boolean;
  answer: string;
};

type RequiredPhrasePlan = {
  phrases: string[];
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

const parseItemAnswerPlan = (raw: string): ItemAnswerPlan[] => {
  const trimmed = raw.trim();
  if (!trimmed) return [];
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/u);
  const candidate = fenced?.[1]?.trim() ?? trimmed;

  try {
    const parsed = JSON.parse(candidate) as unknown;
    const items =
      parsed &&
      typeof parsed === "object" &&
      Array.isArray((parsed as { items?: unknown }).items)
        ? (parsed as { items: unknown[] }).items
        : [];

    return items
      .filter(
        (value): value is { item: string; supported: boolean; answer: string } =>
          Boolean(value) &&
          typeof value === "object" &&
          typeof (value as { item?: unknown }).item === "string" &&
          typeof (value as { supported?: unknown }).supported === "boolean" &&
          typeof (value as { answer?: unknown }).answer === "string",
      )
      .map((value) => {
        const item = value.item.trim();
        const answer = value.answer.trim();
        if (!value.supported) {
          if (
            answer.length > 0 &&
            /not supported by the retrieved evidence/i.test(answer)
          ) {
            return {
              item,
              supported: false,
              answer,
            };
          }
          return {
            item,
            supported: false,
            answer: `The requested information about ${item} is not supported by the retrieved evidence.`,
          };
        }
        return {
          item,
          supported: true,
          answer,
        };
      })
      .filter((value) => value.item.length > 0 && value.answer.length > 0);
  } catch {
    return [];
  }
};

const parseSingleItemResolution = (raw: string): SingleItemResolution | null => {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/u);
  const candidate = fenced?.[1]?.trim() ?? trimmed;

  try {
    const parsed = JSON.parse(candidate) as unknown;
    if (
      parsed &&
      typeof parsed === "object" &&
      typeof (parsed as { supported?: unknown }).supported === "boolean" &&
      typeof (parsed as { answer?: unknown }).answer === "string"
    ) {
      return {
        supported: Boolean((parsed as { supported: boolean }).supported),
        answer: String((parsed as { answer: string }).answer).trim(),
      };
    }
  } catch {
    return null;
  }

  return null;
};

const parseRequiredPhrasePlan = (raw: string): RequiredPhrasePlan | null => {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/u);
  const candidate = fenced?.[1]?.trim() ?? trimmed;

  try {
    const parsed = JSON.parse(candidate) as unknown;
    if (
      parsed &&
      typeof parsed === "object" &&
      Array.isArray((parsed as { phrases?: unknown }).phrases)
    ) {
      return {
        phrases: (parsed as { phrases: unknown[] }).phrases
          .filter((value): value is string => typeof value === "string")
          .map((value) => value.trim())
          .filter(Boolean),
      };
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
Break the question into explicit answer items.

Rules:
- Return JSON only: {"items":["..."]}.
- Each item must be an answerable sub-request, not a keyword list.
- Keep exact identifiers, numbers, filenames, and quoted strings when relevant.
- Keep each item concise and specific.
- For multi-part questions, split into one item per requested fact.
- Keep only the subject needed for that item instead of carrying every named document into every item.
- For metrics or measurements tied to a named source, phrase them as what is reported in or by that source.
- Do not invent extra items.
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

const formatRequestedItemResponses = (items: ItemAnswerPlan[]): string => {
  if (items.length === 0) {
    return EVIDENCE_WARNING;
  }

  if (items.length === 1) {
    return items[0].answer;
  }

  return items
    .map((entry) => `- ${entry.answer}`)
    .join("\n\n");
};

const generateItemResponsesWithModel = async (
  question: string,
  requestedItems: string[],
  chunks: ScoredChunk[],
  isMultiDocument: boolean,
): Promise<ItemAnswerPlan[]> => {
  if (requestedItems.length === 0 || chunks.length === 0) {
    return [];
  }

  const chat = getChatModel();
  const context = isMultiDocument
    ? analyseCrossDocEvidence(chunks).structuredContext
    : buildEvidenceContext(chunks);
  const response = await chat.invoke([
    new SystemMessage(`
You are answering a multi-part question from retrieved evidence.

Return JSON only in this shape:
{"items":[{"item":"...","supported":true|false,"answer":"..."}]}

Rules:
- Preserve each requested item exactly.
- For each requested item, decide whether the retrieved evidence directly supports it.
- If supported, write a concise self-contained user-facing answer for that item using only retrieved evidence.
- Supported answers must be complete sentences, not bare values, fragments, labels, or isolated numbers.
- If a supported value is stated only under an explicit condition, environment, hardware profile, memory budget, context length, or other qualifier, include that qualifier in the answer instead of presenting the value as unconditional.
- Preserve the requested qualifier exactly when it matters semantically. Cold, warm, and hot are distinct requested conditions and must not be substituted for one another.
- If a requested item is answered by an explicit table cell, comparison row, metric row, caption, or short factual sentence in the retrieved evidence, treat it as supported even if the evidence is concise or tabular.
- If the question asks for a single value or metric, answer with the value for the requested subject only. Do not add baseline, comparison, or alternative-system values unless the requested item explicitly asks for comparison.
- If unsupported, set supported=false and answer with a concise self-contained statement that names the requested subject and includes the phrase "not supported by the retrieved evidence".
- If the retrieved evidence explicitly states that a configuration fits, supports, allows, or reaches a quantity under the same requested condition, treat that quantity as the supported answer.
- Do not include citation markup, source names, file names, page numbers, document locations, or provenance sections.
- Do not invent facts, identifiers, controls, section numbers, or numeric values.
- If a requested item names a specific governing document, standard, policy set, or source family, do not substitute a different governing document, standard, policy set, or source family.
- If the retrieved evidence does not support the requested item from the named governing source, mark that item unsupported instead of answering from a different source or from general background knowledge.
- Do not use evidence for one requested item to answer a different requested item.
- Do not answer a requested item with a value belonging to a different system, baseline, or comparison target unless the item explicitly asks for that comparison target.
- Do not answer a requested item with a value belonging to a different qualifier, state, mode, or condition than the one named in the item.
- Do not mark an item unsupported if the retrieved evidence explicitly contains the answer in tabular or comparison form.
`),
    new HumanMessage(
      `Question:\n${question}\n\nRequested items:\n- ${requestedItems.join("\n- ")}\n\nRetrieved context:\n${context}`,
    ),
  ]);

  const parsed = parseItemAnswerPlan(parseChatResponse(response.content));
  const retrievedDocumentIds = getRetrievedDocumentIds(chunks);
  if (parsed.length > 0) {
    const byItem = new Map(parsed.map((entry) => [entry.item, entry]));
    return requestedItems.map((item) => {
      const referencedDocumentIds = extractReferencedDocumentIds(item);
      const entry = byItem.get(item);
      if (
        referencedDocumentIds.length > 0 &&
        referencedDocumentIds.some((documentId) => !retrievedDocumentIds.has(documentId))
      ) {
        return {
          item,
          supported: false,
          answer: `The requested information about ${item} is not supported by the retrieved evidence.`,
        };
      }
      return entry ?? {
        item,
        supported: false,
        answer: "Not supported by the retrieved evidence.",
      };
    });
  }

  return requestedItems.map((item) => ({
    item,
    supported: false,
    answer: "Not supported by the retrieved evidence.",
  }));
};

const repairUnsupportedItemsWithModel = async (
  question: string,
  itemResponses: ItemAnswerPlan[],
  chunks: ScoredChunk[],
): Promise<ItemAnswerPlan[]> => {
  if (itemResponses.length === 0 || chunks.length === 0) {
    return itemResponses;
  }

  const retrievedDocumentIds = getRetrievedDocumentIds(chunks);
  const { lexical, vector } = await getIndexStore({ allowDenseMissing: true });

  const repaired: ItemAnswerPlan[] = [];

  for (const entry of itemResponses) {
      const referencedDocumentIds = extractReferencedDocumentIds(entry.item);
      if (
        referencedDocumentIds.length > 0 &&
        referencedDocumentIds.some((documentId) => !retrievedDocumentIds.has(documentId))
      ) {
        repaired.push(entry);
        continue;
      }

      try {
        const focusedQueries = dedupeNormalizedStrings([
          entry.item,
          ...(await expandRetrievalQueries(entry.item).catch(() => [entry.item])),
        ]);
        const focusedRetrieval = await retrieveHybrid(lexical, vector, {
          query: entry.item,
          queries: focusedQueries,
          topK: Math.min(4, appConfig.retrieval.topK),
          candidateK: Math.min(appConfig.retrieval.candidateK, 8),
          expandQueries: false,
          filterDocumentIds: referencedDocumentIds,
        });
        const candidateChunks = focusedRetrieval.chunks.length > 0 ? focusedRetrieval.chunks : chunks;
        const chat = getChatModel();
        const focusedContext = buildEvidenceContext(candidateChunks);
        const response = await chat.invoke(
          `Question:\n${question}\n\nRequested item:\n${entry.item}\n\nRetrieved context:\n${focusedContext}\n\nReturn JSON only: {"supported":true|false,"answer":"..."}.

Rules:
- Resolve only this item from the retrieved context.
- Preserve the requested qualifier, subject, and governing source.
- If the item is supported only under a stated condition or environment, include that condition.
- If unsupported, return supported=false and include the phrase "not supported by the retrieved evidence".
- Do not invent values or substitute a different qualifier, subject, or governing source.`,
        );

        const resolved = parseSingleItemResolution(parseChatResponse(response.content));
        if (!resolved || !resolved.answer) {
          repaired.push(entry);
          continue;
        }

        repaired.push({
          item: entry.item,
          supported: resolved.supported,
          answer: resolved.answer,
        });
      } catch {
        repaired.push(entry);
      }
  }

  return repaired;
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
Verify whether the answer is supported by the retrieved evidence.

Return JSON only: {"supported":true|false,"reason":"..."}.

Return supported=false if the answer:
- adds unsupported facts or numbers
- uses irrelevant evidence
- claims to satisfy a requested item without support
- adds citation or provenance formatting the user did not ask for

If supported parts are grounded and unsupported parts are clearly marked unsupported, that is acceptable.
Be conservative.
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
Verify whether the answer handles each requested item correctly.

Return JSON only: {"supported":true|false,"reason":"..."}.

Return supported=true when:
- supported items are answered from the retrieved evidence
- unsupported items are clearly marked unsupported

Return supported=false when the answer:
- claims support that is not in the evidence
- omits a clearly supported item without marking it unsupported
- invents extra items
- adds citation or provenance formatting the user did not ask for

If a value is explicitly present in the retrieved context, treat it as supported. Preserve stated qualifiers and conditions.
`),
    new HumanMessage(
      `Question:\n${question}\n\nRequested items:\n- ${requestedItems.join("\n- ")}\n\nAnswer:\n${answer}\n\nRetrieved context:\n${context}`,
    ),
  ]);

  return parseItemGroundingVerdict(parseChatResponse(response.content));
};

export const retrieveNode = async (state: GraphState): Promise<GraphState> => {
  const requestedItems = await planRequestedItems(state.question);
  const expandedItemQueries = dedupeNormalizedStrings(
    (
      await Promise.all(
        requestedItems.map((item) =>
          expandRetrievalQueries(item).catch(() => [item]),
        ),
      )
    ).flat(),
  );
  const { lexical, vector } = await getIndexStore({ allowDenseMissing: true });
  const retrieval = await retrieveCrossDocument(lexical, vector, {
    query: state.question,
    additionalQueries: expandedItemQueries,
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
    requested_items: requestedItems,
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
        requestedItems,
        expandedItemQueries,
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
  const itemResponses = await repairUnsupportedItemsWithModel(
    state.question,
    requestedItems.map((item) => ({
      item,
      supported: false,
      answer: `The requested information about ${item} is not supported by the retrieved evidence.`,
    })),
    chunks,
  );
  const supportedCount = itemResponses.filter((item) => item.supported).length;
  const answer =
    itemResponses.length > 0
      ? formatRequestedItemResponses(itemResponses)
      : EVIDENCE_WARNING;

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
        supportedItemCount: supportedCount,
        itemResponses,
        draftAnswer: answer,
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

  const itemVerdict = await verifyRequestedItemsWithModel(
    state.question,
    requestedItems,
    answer,
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

  return {
    ...state,
    requested_items: requestedItems,
    answer,
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
