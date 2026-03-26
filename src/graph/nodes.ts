import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { appConfig } from "../lib/config";
import { buildEvidenceContext, summarizeCitations } from "../citations/citationExtractor";
import { getChatModel } from "../lib/llm";
import { getIndexStore } from "../lib/store";
import { buildCrossDocSystemPrompt, buildRagSystemPrompt } from "../prompts/rag";
import { hasMultiDocumentCoverage, getDocumentCoverageSummary, retrieveCrossDocument } from "../retrieval/crossDocRetriever";
import { EvidenceAvailability, GraphState, ItemContextBundle, PipelineStage, PipelineStageTrace, PipelineStatus, RequestedItemDescriptor, ScoredChunk } from "../types/rag";
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

const tokenizeReferenceText = (value: string): string[] =>
  value
    .toLowerCase()
    .split(/[^a-z0-9]+/u)
    .map((token) => token.trim())
    .filter((token) => token.length >= 3);

const extractReferencedDocumentIds = (text: string): string[] => {
  const normalized = text.toLowerCase();
  const queryTokens = new Set(tokenizeReferenceText(text));
  const tokenFrequency = new Map<string, number>();

  const corpusTokens = appConfig.corpus.map((entry) => {
    const tokens = dedupeNormalizedStrings([
      ...tokenizeReferenceText(entry.document_id),
      ...tokenizeReferenceText(entry.source_file),
    ]);
    for (const token of tokens) {
      tokenFrequency.set(token, (tokenFrequency.get(token) ?? 0) + 1);
    }
    return { entry, tokens };
  });

  const matches: { documentId: string; position: number }[] = [];

  for (const { entry, tokens } of corpusTokens) {
    let position = Number.POSITIVE_INFINITY;
    if (
      normalized.includes(entry.document_id.toLowerCase()) ||
      normalized.includes(entry.source_file.toLowerCase())
    ) {
      const directPositions = [
        normalized.indexOf(entry.document_id.toLowerCase()),
        normalized.indexOf(entry.source_file.toLowerCase()),
      ].filter((value) => value >= 0);
      position = Math.min(...directPositions);
      matches.push({ documentId: entry.document_id, position });
      continue;
    }

    const distinctiveTokenPositions = tokens
      .filter((token) => queryTokens.has(token) && tokenFrequency.get(token) === 1)
      .map((token) => normalized.indexOf(token))
      .filter((value) => value >= 0);
    if (distinctiveTokenPositions.length > 0) {
      position = Math.min(...distinctiveTokenPositions);
      matches.push({ documentId: entry.document_id, position });
    }
  }

  return dedupeNormalizedStrings(
    matches
      .sort((left, right) => left.position - right.position)
      .map((entry) => entry.documentId),
  );
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
  items?: ItemAnswerPlan[];
  raw_response?: string;
};

type RequestedItemPlan = {
  items: RequestedItemDescriptor[];
};

type RetrievalTermPlan = {
  terms: string[];
};

type ItemAnswerPlan = {
  item: string;
  supported: boolean;
  answer: string;
  support_text?: string;
  reasoning?: string;
  primary_controls?: string[];
  supporting_safeguards?: string[];
  matched_pairs?: string[];
};



type ItemScopedContext = {
  descriptor: RequestedItemDescriptor;
  primaryChunks: ScoredChunk[];
  supportingChunks: ScoredChunk[];
};

const toItemContextBundle = (scopedContext: ItemScopedContext): ItemContextBundle => ({
  item: scopedContext.descriptor.item,
  primary_source_ids: scopedContext.descriptor.primary_source_ids,
  supporting_source_ids: scopedContext.descriptor.supporting_source_ids,
  primary_chunks: scopedContext.primaryChunks,
  supporting_chunks: scopedContext.supportingChunks,
});


const parseGroundingVerdict = (raw: string): GroundingVerdict | null => {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/u);
  const candidate = fenced?.[1]?.trim() ?? trimmed;
  const normalizedCandidate = candidate.replace(/,\s*([}\]])/gu, "$1");

  try {
    const parsed = JSON.parse(normalizedCandidate) as unknown;
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
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/u);
  const candidate = fenced?.[1]?.trim() ?? trimmed;
  const normalizedCandidate = candidate.replace(/,\s*([}\]])/gu, "$1");

  try {
    const parsed = JSON.parse(normalizedCandidate) as unknown;
    if (
      !parsed ||
      typeof parsed !== "object" ||
      typeof (parsed as { supported?: unknown }).supported !== "boolean"
    ) {
      return null;
    }

    const items =
      Array.isArray((parsed as { items?: unknown }).items)
        ? parseItemAnswerPlan(JSON.stringify({ items: (parsed as { items: unknown[] }).items }))
        : [];

    return {
      supported: Boolean((parsed as { supported: boolean }).supported),
      reason:
        typeof (parsed as { reason?: unknown }).reason === "string"
          ? String((parsed as { reason?: unknown }).reason)
          : "",
      items,
      raw_response: raw,
    };
  } catch {
    return null;
  }
};

const parseRequestedItemPlan = (raw: string): RequestedItemPlan | null => {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/u);
  const candidate = fenced?.[1]?.trim() ?? trimmed;
  const normalizedCandidate = candidate.replace(/,\s*([}\]])/gu, "$1");

  try {
    const parsed = JSON.parse(normalizedCandidate) as unknown;
    if (
      parsed &&
      typeof parsed === "object" &&
      Array.isArray((parsed as { items?: unknown }).items)
    ) {
      const items = (parsed as { items: unknown[] }).items
        .map((value) => {
          if (typeof value === "string") {
            const item = value.trim();
            return item
              ? {
                  item,
                  retrieval_query: item,
                  supporting_retrieval_query: item,
                  primary_source_ids: [],
                  supporting_source_ids: [],
                }
              : null;
          }
          if (
            value &&
            typeof value === "object" &&
            typeof (value as { item?: unknown }).item === "string"
          ) {
            const item = String((value as { item: string }).item).trim();
            const retrieval_query =
              typeof (value as { retrieval_query?: unknown }).retrieval_query === "string"
                ? String((value as { retrieval_query?: unknown }).retrieval_query).trim()
                : item;
            const supporting_retrieval_query =
              typeof (value as { supporting_retrieval_query?: unknown }).supporting_retrieval_query === "string"
                ? String((value as { supporting_retrieval_query?: unknown }).supporting_retrieval_query).trim()
                : item;
            const primary_source_ids = Array.isArray((value as { primary_source_ids?: unknown }).primary_source_ids)
              ? (value as { primary_source_ids: unknown[] }).primary_source_ids
                  .filter((entry): entry is string => typeof entry === "string")
                  .map((entry) => entry.trim())
                  .filter(Boolean)
              : [];
            const supporting_source_ids = Array.isArray((value as { supporting_source_ids?: unknown }).supporting_source_ids)
              ? (value as { supporting_source_ids: unknown[] }).supporting_source_ids
                  .filter((entry): entry is string => typeof entry === "string")
                  .map((entry) => entry.trim())
                  .filter(Boolean)
              : [];
            return item
              ? {
                  item,
                  retrieval_query: retrieval_query || item,
                  supporting_retrieval_query: supporting_retrieval_query || item,
                  primary_source_ids,
                  supporting_source_ids,
                }
              : null;
          }
          return null;
        })
        .filter((value): value is RequestedItemDescriptor => Boolean(value))
        .slice(0, 8);
      return { items };
    }
  } catch {
    return null;
  }

  return null;
};

const parseRetrievalTermPlan = (raw: string): RetrievalTermPlan | null => {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/u);
  const candidate = fenced?.[1]?.trim() ?? trimmed;
  const normalizedCandidate = candidate.replace(/,\s*([}\]])/gu, "$1");

  try {
    const parsed = JSON.parse(normalizedCandidate) as unknown;
    if (!parsed || typeof parsed !== "object" || !Array.isArray((parsed as { terms?: unknown }).terms)) {
      return null;
    }
    return {
      terms: dedupeNormalizedStrings(
        ((parsed as { terms: unknown[] }).terms ?? [])
          .filter((value): value is string => typeof value === "string")
          .map((value) => value.trim())
          .filter(Boolean),
      ).slice(0, 8),
    };
  } catch {
    return null;
  }
};

const corpusDocumentIds = new Set(appConfig.corpus.map((entry) => entry.document_id));

const resolveExplicitDocumentIds = (values: string[]): string[] =>
  dedupeNormalizedStrings(values.flatMap((value) => extractReferencedDocumentIds(value))).filter((value) =>
    corpusDocumentIds.has(value),
  );

const resolveDocumentIds = (values: string[], fallbackText: string): string[] => {
  const resolved = dedupeNormalizedStrings(
    values.flatMap((value) => extractReferencedDocumentIds(value)),
  ).filter((value) => corpusDocumentIds.has(value));

  if (resolved.length > 0) {
    return resolved;
  }

  return extractReferencedDocumentIds(fallbackText).filter((value) => corpusDocumentIds.has(value));
};

const stripDocumentReferences = (text: string): string => {
  let stripped = text;
  for (const entry of appConfig.corpus) {
    const patterns = [entry.document_id, entry.source_file]
      .map((value) => value.trim())
      .filter((value) => value.length > 0)
      .map((value) => value.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&"));
    for (const pattern of patterns) {
      stripped = stripped.replace(new RegExp(pattern, "igu"), " ");
    }
  }
  return stripped.replace(/\s+/gu, " ").trim();
};

const stripReferencedSourceTokens = (text: string, sourceIds: string[]): string => {
  let stripped = text;
  const referencedEntries = appConfig.corpus.filter((entry) => sourceIds.includes(entry.document_id));
  for (const entry of referencedEntries) {
    const tokens = dedupeNormalizedStrings([
      ...tokenizeReferenceText(entry.document_id),
      ...tokenizeReferenceText(entry.source_file),
    ]).filter((token) => token.length >= 3);
    for (const token of tokens) {
      stripped = stripped.replace(new RegExp(`\\b${token.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&")}\\b`, "igu"), " ");
    }
  }
  return stripped.replace(/\s+/gu, " ").trim();
};

const buildSourceScopedQuery = (
  sourceIds: string[],
  preferredText: string,
  fallbackText: string,
  excludedSourceIds: string[] = [],
): string => {
  const sourcePrefix = sourceIds.join(" ").trim();
  const topic = stripReferencedSourceTokens(
    stripDocumentReferences(preferredText || fallbackText),
    excludedSourceIds,
  );
  if (sourcePrefix && topic) {
    return `${sourcePrefix} ${topic}`.trim();
  }
  return sourcePrefix || topic || fallbackText.trim();
};

const normalizeRequestedItemDescriptor = (descriptor: RequestedItemDescriptor): RequestedItemDescriptor => {
  const primaryFallbackText = `${descriptor.retrieval_query}\n${descriptor.item}`;
  const supportingFallbackText = `${descriptor.supporting_retrieval_query}\n${descriptor.item}`;
  const declaredPrimarySourceIds = resolveExplicitDocumentIds(descriptor.primary_source_ids);
  const declaredSupportingSourceIds = resolveExplicitDocumentIds(descriptor.supporting_source_ids);
  const queryPrimarySourceIds = extractReferencedDocumentIds(descriptor.retrieval_query).filter((value) =>
    corpusDocumentIds.has(value),
  );
  const querySupportingSourceIds = extractReferencedDocumentIds(
    descriptor.supporting_retrieval_query,
  ).filter((value) => corpusDocumentIds.has(value));

  const shouldInferSupportingFromQuery =
    declaredSupportingSourceIds.length > 0 ||
    (declaredPrimarySourceIds.length === 0 &&
      queryPrimarySourceIds.length > 1 &&
      querySupportingSourceIds.length > 0);

  let supportingSourceIds = dedupeNormalizedStrings(
    (shouldInferSupportingFromQuery ? querySupportingSourceIds : declaredSupportingSourceIds).filter(Boolean),
  );
  let primarySourceIds = dedupeNormalizedStrings(
    (
      declaredPrimarySourceIds.length > 0
        ? declaredPrimarySourceIds
        : resolveDocumentIds(descriptor.primary_source_ids, primaryFallbackText)
    ).filter((documentId) => !supportingSourceIds.includes(documentId)),
  );

  if (primarySourceIds.length === 0 && supportingSourceIds.length > 1) {
    const orderedPrimaryCandidates = resolveDocumentIds([], primaryFallbackText).filter((documentId) =>
      supportingSourceIds.includes(documentId),
    );
    const primarySourceId = orderedPrimaryCandidates[0];
    if (primarySourceId) {
      primarySourceIds = [primarySourceId];
      supportingSourceIds = supportingSourceIds.filter((documentId) => documentId !== primarySourceId);
    }
  }

  const retrievalQuery = buildSourceScopedQuery(
    primarySourceIds,
    descriptor.retrieval_query,
    descriptor.item,
    supportingSourceIds,
  );
  const supportingRetrievalQuery =
    supportingSourceIds.length > 0
      ? buildSourceScopedQuery(
          supportingSourceIds,
          descriptor.supporting_retrieval_query,
          descriptor.item,
          primarySourceIds,
        )
      : "";

  return {
    item: descriptor.item,
    retrieval_query: retrievalQuery,
    supporting_retrieval_query: supportingRetrievalQuery,
    primary_source_ids: primarySourceIds,
    supporting_source_ids: supportingSourceIds,
  };
};

const getRequestedItemLabels = (requestedItems: RequestedItemDescriptor[]): string[] =>
  requestedItems.map((descriptor) => descriptor.item);

const buildItemScopedContexts = (
  requestedItems: RequestedItemDescriptor[],
  chunks: ScoredChunk[],
): ItemScopedContext[] =>
  requestedItems.map((descriptor) => {
    const primaryChunks =
      descriptor.primary_source_ids.length > 0
        ? chunks.filter((chunk) => descriptor.primary_source_ids.includes(chunk.chunk.metadata.document_id))
        : chunks;
    const supportingChunks =
      descriptor.supporting_source_ids.length > 0
        ? chunks.filter((chunk) => descriptor.supporting_source_ids.includes(chunk.chunk.metadata.document_id))
        : [];

    return {
      descriptor,
      primaryChunks,
      supportingChunks,
    };
  });

const ITEM_CONTEXT_PRIMARY_LIMIT = appConfig.retrieval.topK;
const ITEM_CONTEXT_SUPPORTING_LIMIT = appConfig.retrieval.topK * 2;

const extractBridgeTermsFromPrimaryContext = async (
  item: string,
  primaryChunks: ScoredChunk[],
): Promise<string[]> => {
  if (primaryChunks.length === 0) {
    return [];
  }

  const chat = getChatModel();
  const primaryContext = buildEvidenceContext(primaryChunks.slice(0, ITEM_CONTEXT_PRIMARY_LIMIT));
  const response = await chat.invoke([
    new SystemMessage(`
Return JSON only:
{"terms":["..."]}

Extract short generic control, safeguard, record-handling, or compliance-category phrases from the governing-source context that should be used to retrieve matching behavior descriptions from another source.

Rules:
- Use only category phrases visible in or directly implied by the governing-source context.
- Prefer short phrases such as access control, audit trails, authority checks, record integrity, validation, electronic records, electronic signatures, confidentiality, or record linking when those categories are present.
- Do not return source names, law names, section numbers, product names, page numbers, or implementation-specific nouns.
- Do not explain. Return only the JSON object.
`),
    new HumanMessage(`Requested item:\n${item}\n\nGoverning-source context:\n${primaryContext}`),
  ]);

  return parseRetrievalTermPlan(parseChatResponse(response.content))?.terms ?? [];
};

const hydrateItemContextBundles = async (
  question: string,
  requestedItems: RequestedItemDescriptor[],
  chunks: ScoredChunk[],
): Promise<ItemContextBundle[]> => {
  const globalContexts = buildItemScopedContexts(requestedItems, chunks);
  const { lexical, vector } = await getIndexStore({ allowDenseMissing: true });
  const questionScopedChunks = await retrieveCrossDocument(lexical, vector, {
    query: question,
    additionalQueries: [],
    topK: appConfig.retrieval.topK,
  }).then((retrieval) => retrieval.chunks);

  return Promise.all(
    globalContexts.map(async (scopedContext) => {
      const { descriptor } = scopedContext;
      const primaryChunks =
        descriptor.primary_source_ids.length > 0
          ? await Promise.all(
              descriptor.primary_source_ids.map(async (documentId) => {
                const retrieval = await retrieveCrossDocument(lexical, vector, {
                  query: descriptor.retrieval_query,
                  additionalQueries: [],
                  topK: ITEM_CONTEXT_PRIMARY_LIMIT,
                  filterDocumentIds: [documentId],
                });
                return retrieval.chunks;
              }),
            ).then((bundles) => mergeBundleChunks(bundles).slice(0, ITEM_CONTEXT_PRIMARY_LIMIT))
          : scopedContext.primaryChunks.slice(0, ITEM_CONTEXT_PRIMARY_LIMIT);

      const bridgeTerms =
        descriptor.supporting_source_ids.length > 0
          ? await extractBridgeTermsFromPrimaryContext(descriptor.item, primaryChunks)
          : [];

      const supportingChunks =
        descriptor.supporting_source_ids.length > 0
          ? await Promise.all(
              descriptor.supporting_source_ids.map(async (documentId) => {
                const retrieval = await retrieveCrossDocument(lexical, vector, {
                  query: buildSourceScopedQuery(
                    [documentId],
                    bridgeTerms.join(" "),
                    question,
                    descriptor.primary_source_ids,
                  ),
                  additionalQueries: [],
                  topK: appConfig.retrieval.topK,
                  filterDocumentIds: [documentId],
                });
                return retrieval.chunks;
              }),
            ).then((bundles) =>
              mergeBundleChunks([
                ...bundles,
                questionScopedChunks.filter((chunk) =>
                  descriptor.supporting_source_ids.includes(chunk.chunk.metadata.document_id),
                ),
              ]).slice(0, ITEM_CONTEXT_SUPPORTING_LIMIT),
            )
          : scopedContext.supportingChunks.slice(0, ITEM_CONTEXT_SUPPORTING_LIMIT);

      return {
        item: descriptor.item,
        primary_source_ids: descriptor.primary_source_ids,
        supporting_source_ids: descriptor.supporting_source_ids,
        primary_chunks: primaryChunks,
        supporting_chunks: supportingChunks,
      };
    }),
  );
};

const parseItemAnswerPlan = (raw: string): ItemAnswerPlan[] => {
  const trimmed = raw.trim();
  if (!trimmed) return [];
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/u);
  const candidate = fenced?.[1]?.trim() ?? trimmed;
  const normalizedCandidate = candidate.replace(/,\s*([}\]])/gu, "$1");

  try {
    const parsed = JSON.parse(normalizedCandidate) as unknown;
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
        const support_text =
          typeof (value as { support_text?: unknown }).support_text === "string"
            ? String((value as { support_text?: unknown }).support_text).trim()
            : "";
        const reasoning =
          typeof (value as { reasoning?: unknown }).reasoning === "string"
            ? String((value as { reasoning?: unknown }).reasoning).trim()
            : "";
        const primary_controls =
          Array.isArray((value as { primary_controls?: unknown }).primary_controls)
            ? ((value as { primary_controls?: unknown[] }).primary_controls ?? [])
                .filter((entry): entry is string => typeof entry === "string")
                .map((entry) => entry.trim())
                .filter((entry) => entry.length > 0)
            : [];
        const supporting_safeguards =
          Array.isArray((value as { supporting_safeguards?: unknown }).supporting_safeguards)
            ? ((value as { supporting_safeguards?: unknown[] }).supporting_safeguards ?? [])
                .filter((entry): entry is string => typeof entry === "string")
                .map((entry) => entry.trim())
                .filter((entry) => entry.length > 0)
            : [];
        const matched_pairs =
          Array.isArray((value as { matched_pairs?: unknown }).matched_pairs)
            ? ((value as { matched_pairs?: unknown[] }).matched_pairs ?? [])
                .filter((entry): entry is string => typeof entry === "string")
                .map((entry) => entry.trim())
                .filter((entry) => entry.length > 0)
            : [];
        const normalizedSupported = Boolean(value.supported) || matched_pairs.length > 0;
        if (!value.supported) {
          if (
            answer.length > 0 &&
            /not supported by the retrieved evidence/i.test(answer)
          ) {
            return {
              item,
              supported: normalizedSupported,
              answer,
              support_text,
              reasoning,
              primary_controls,
              supporting_safeguards,
              matched_pairs,
            };
          }
          return {
            item,
            supported: normalizedSupported,
            answer: normalizedSupported
              ? answer.length > 0
                ? answer
                : support_text
              : `The requested information about ${item} is not supported by the retrieved evidence.`,
            support_text,
            reasoning,
            primary_controls,
            supporting_safeguards,
            matched_pairs,
          };
        }
        return {
          item,
          supported: normalizedSupported,
          answer,
          support_text,
          reasoning,
          primary_controls,
          supporting_safeguards,
          matched_pairs,
        };
      })
      .filter((value) => value.item.length > 0 && value.answer.length > 0);
  } catch {
    return [];
  }
};



const planRequestedItems = async (
  question: string,
): Promise<{ requestedItems: RequestedItemDescriptor[]; retrievalQueries: string[] }> => {
  const chat = getChatModel();
  try {
    const response = await chat.invoke([
      new SystemMessage(`
Break the question into explicit answer items.

Rules:
- Return JSON only in this shape: {"items":[{"item":"...","retrieval_query":"...","supporting_retrieval_query":"...","primary_source_ids":["..."],"supporting_source_ids":["..."]}]}.
- Each item must be an answerable factual sub-request, not a keyword list or a procedural instruction.
- Keep exact identifiers, numbers, filenames, and quoted strings when relevant.
- If the question names a source by exact document id or file name, preserve that exact source string in the corresponding item and retrieval_query. Do not replace it with an abbreviation or paraphrase.
- If an item is source-specific, keep the exact corpus document id verbatim in the item label, retrieval_query, and supporting_retrieval_query where that source is referenced. Do not replace an exact document id with a generic label such as "paper", "CFR", or "document".
- Keep each item concise and specific.
- For multi-part questions, split into one item per requested fact.
- Keep only the subject needed for that item instead of carrying every named document into every item.
- For metrics or measurements tied to a named source, phrase them as what is reported in or by that source.
- When a question asks for a value from a named document, treat the document as the reporting source, not as the system identity. Prefer phrasing like "8K FP16 agent count reported in agent-memory-below-the-prompt" over phrasing like "agent count for agent-memory-below-the-prompt".
- Use short factual item labels, factual questions, or noun phrases. Do not use imperative instructions such as "report", "determine", "state", or conditional phrasing such as "if not".
- For regulation, policy, control, or section items that relate a governing source to another named system, method, or topic, keep the governing source but express the relevant governed behavior or topic in natural language instead of vague phrases like "this system".
- For items asking whether a governing source supports matching controls, requirements, or sections for a target system or method, prefer an answerable noun phrase such as "Matching CFR controls for ..." or "Matching policy requirements for ..." instead of a yes/no item phrased with "whether".
- For items asking whether a governing source supports controls, requirements, or sections for another source, prefer a category-level item about matching control categories, safeguard categories, or requirement categories rather than a broad whole-system compliance judgment.
- When a supporting source explicitly describes safeguards, record-handling properties, access restrictions, auditability, validation needs, confidentiality, integrity, or erasure requirements, express the item around those matching categories instead of around the entire target system as a whole.
- For that kind of cross-source item, do not phrase the item around low-level implementation mechanisms when the supporting source also provides higher-level safeguard or compliance categories. Prefer the higher-level categories.
- Do not create a separate item that only restates what to say when evidence is insufficient. Keep unsupported handling inside the same factual item.
- retrieval_query must be a short evidence-seeking query for the primary source of that item.
- supporting_retrieval_query must be a short evidence-seeking query for the supporting source context of that same item.
- retrieval_query should preserve the governing or reporting source and the concrete topic terms needed for the primary source, but remove answer instructions and extra wording.
- supporting_retrieval_query should preserve the supporting source and the concrete target-system behavior, capability, safeguard, or topic needed from that supporting source.
- For governing-source support items, supporting_retrieval_query should prefer explicit safeguard or compliance phrases from the supporting source over low-level implementation nouns when those phrases are present in the supporting source.
- retrieval_query should name or anchor only the primary source, not the supporting source.
- supporting_retrieval_query should name or anchor only the supporting source, not the primary source.
- If an item has no supporting source, supporting_source_ids must be [] and supporting_retrieval_query must be an empty string.
- When an item asks whether a named governing source supports controls, requirements, or sections for a target system or method, make retrieval_query center the governing source and the concrete governed behavior or topic. Make supporting_retrieval_query center the supporting source and the target-system behaviors or safeguards that must be matched.
- For that kind of item, make retrieval_query center the governing-source control categories or safeguard categories that could match, not a broad compliance verdict for the whole target system.
- Do not let a reporting source or target system name dominate retrieval_query when the requested governing source is different.
- primary_source_ids must list the exact corpus document ids that are the main evidence source for the item.
- supporting_source_ids may list exact corpus document ids that provide target-system or comparison context for the item, but are not the main governing or reporting source.
- For a metric reported in a named document, that reporting document belongs in primary_source_ids.
- For an item asking whether a governing source supports matching controls or requirements for another named system or method, the governing source belongs in primary_source_ids and the target system source belongs in supporting_source_ids.
- For any cross-source item, primary_source_ids must contain exactly one governing or reporting source id and must never be empty. supporting_source_ids must not repeat the primary source id.
- Do not invent extra items.
`),
      new HumanMessage(`Question:\n${question}`),
    ]);
    const plan = parseRequestedItemPlan(parseChatResponse(response.content));
    if (plan && plan.items.length > 0) {
      const requestedItems = plan.items.map(normalizeRequestedItemDescriptor);
      const retrievalQueries = dedupeNormalizedStrings(
        requestedItems.map((entry) => entry.retrieval_query || entry.item),
      );
      return { requestedItems, retrievalQueries };
    }
  } catch {
    // fall through
  }
  return {
    requestedItems: [
      normalizeRequestedItemDescriptor({
        item: question,
        retrieval_query: question,
        supporting_retrieval_query: question,
        primary_source_ids: [],
        supporting_source_ids: [],
      }),
    ],
    retrievalQueries: [question],
  };
};

const mergeBundleChunks = (bundles: ScoredChunk[][]): ScoredChunk[] => {
  const merged = new Map<string, ScoredChunk>();

  for (const bundle of bundles) {
    for (const chunk of bundle) {
      if (!merged.has(chunk.chunk.id)) {
        merged.set(chunk.chunk.id, chunk);
      }
    }
  }

  return [...merged.values()];
};

const formatFinalItemAnswer = (
  item: ItemAnswerPlan,
  descriptor?: RequestedItemDescriptor,
): string => {
  if (item.supported) {
    const answer = item.answer.trim();
    const supportText = (item.support_text ?? "").trim();
    const reasoning = (item.reasoning ?? "").trim();
    const matchedPairs = item.matched_pairs ?? [];
    const primaryControls = item.primary_controls ?? [];
    const looksBareScalar =
      answer.length > 0 &&
      answer.length <= 32 &&
      !/[.!?]/u.test(answer) &&
      !/\s{2,}/u.test(answer) &&
      !/[a-z]{3,}/iu.test(answer);
    const looksLikeUnsupportedTemplate = /not supported by the retrieved evidence/i.test(answer);
    const structuredSupportSummary =
      supportText ||
      (matchedPairs.length > 0 ? matchedPairs.join("; ") : "") ||
      (primaryControls.length > 0 ? primaryControls.join("; ") : "");
    const normalizedSupportedAnswer =
      looksLikeUnsupportedTemplate && structuredSupportSummary
        ? /[.!?]$/u.test(structuredSupportSummary)
          ? structuredSupportSummary
          : `${structuredSupportSummary}.`
        : looksBareScalar && supportText
        ? /[.!?]$/u.test(supportText)
          ? supportText
          : `${supportText}.`
        : answer;
    return normalizedSupportedAnswer;
  }

  const reasoning = (item.reasoning ?? "").trim();
  if (!reasoning) {
    return item.answer;
  }

  if (item.answer.toLowerCase().includes(reasoning.toLowerCase())) {
    return item.answer;
  }

  return `${item.answer} ${reasoning}`;
};

const formatRequestedItemResponses = (
  items: ItemAnswerPlan[],
  requestedItems?: RequestedItemDescriptor[],
): string => {
  if (items.length === 0) {
    return EVIDENCE_WARNING;
  }

  const descriptorByItem = new Map(
    (requestedItems ?? []).map((descriptor) => [descriptor.item, descriptor]),
  );

  if (items.length === 1) {
    return formatFinalItemAnswer(items[0], descriptorByItem.get(items[0].item));
  }

  return items
    .map(
      (entry) =>
        `- ${entry.item}: ${formatFinalItemAnswer(entry, descriptorByItem.get(entry.item))}`,
    )
    .join("\n\n");
};

const formatItemScopedContexts = (scopedContexts: ItemScopedContext[]): string =>
  scopedContexts
    .map(({ descriptor, primaryChunks, supportingChunks }) => {
      const primaryContext =
        primaryChunks.length > 0
          ? buildEvidenceContext(primaryChunks.slice(0, Math.min(ITEM_CONTEXT_PRIMARY_LIMIT, primaryChunks.length)))
          : "(none)";
      const supportingContext =
        supportingChunks.length > 0
          ? buildEvidenceContext(supportingChunks.slice(0, Math.min(ITEM_CONTEXT_SUPPORTING_LIMIT, supportingChunks.length)))
          : "(none)";

      return [
        `Item: ${descriptor.item}`,
        `Primary source ids: ${descriptor.primary_source_ids.join(", ") || "(none)"}`,
        `Supporting source ids: ${descriptor.supporting_source_ids.join(", ") || "(none)"}`,
        `Primary context:\n${primaryContext}`,
        `Supporting context:\n${supportingContext}`,
      ].join("\n\n");
    })
    .join("\n\n---\n\n");

const itemContextsFromBundles = (
  requestedItems: RequestedItemDescriptor[],
  bundles: ItemContextBundle[] | undefined,
  fallbackChunks: ScoredChunk[],
): ItemScopedContext[] => {
  if (!bundles || bundles.length === 0) {
    return buildItemScopedContexts(requestedItems, fallbackChunks);
  }

  const byItem = new Map(bundles.map((bundle) => [bundle.item, bundle]));
  return requestedItems.map((descriptor) => {
    const bundle = byItem.get(descriptor.item);
    if (!bundle) {
      return {
        descriptor,
        primaryChunks:
          descriptor.primary_source_ids.length > 0
            ? fallbackChunks.filter((chunk) => descriptor.primary_source_ids.includes(chunk.chunk.metadata.document_id))
            : fallbackChunks,
        supportingChunks:
          descriptor.supporting_source_ids.length > 0
            ? fallbackChunks.filter((chunk) => descriptor.supporting_source_ids.includes(chunk.chunk.metadata.document_id))
            : [],
      };
    }

    return {
      descriptor,
      primaryChunks: bundle.primary_chunks,
      supportingChunks: bundle.supporting_chunks,
    };
  });
};

const generateItemResponsesWithModel = async (
  question: string,
  requestedItems: RequestedItemDescriptor[],
  chunks: ScoredChunk[],
  itemContextBundles?: ItemContextBundle[],
): Promise<ItemAnswerPlan[]> => {
  if (requestedItems.length === 0 || chunks.length === 0) {
    return [];
  }

  const chat = getChatModel();
  const itemLabels = getRequestedItemLabels(requestedItems);
  const context = formatItemScopedContexts(itemContextsFromBundles(requestedItems, itemContextBundles, chunks));
  const response = await chat.invoke([
    new SystemMessage(`
You are answering a multi-part question from retrieved evidence.

Return JSON only in this shape:
{"items":[{"item":"...","supported":true|false,"answer":"...","support_text":"...","reasoning":"...","primary_controls":["..."],"supporting_safeguards":["..."],"matched_pairs":["..."]}]}

Rules:
- Preserve each requested item exactly.
- For each requested item, decide whether the retrieved evidence directly supports it.
- If supported, write a concise self-contained user-facing answer for that item using only retrieved evidence.
- Keep every field brief. Prefer short phrases over full quotations.
- support_text must be a single short phrase or sentence fragment of at most 25 words.
- reasoning must be a single short sentence of at most 20 words.
- primary_controls, supporting_safeguards, and matched_pairs must contain at most 4 entries each, and each entry must stay under 12 words.
- For every item, populate support_text with the brief evidence phrase, row, control name, requirement text, or metric wording that most directly supports the decision.
- For every item, populate reasoning with one short sentence explaining why that support_text does or does not answer the requested item.
- Supported answers must be complete sentences, not bare values, fragments, labels, or isolated numbers.
- If a supported value is stated only under an explicit condition, environment, hardware profile, memory budget, context length, or other qualifier, include that qualifier in the answer instead of presenting the value as unconditional.
- Preserve the requested qualifier exactly when it matters semantically. Cold, warm, and hot are distinct requested conditions and must not be substituted for one another.
- Each item has a primary context and an optional supporting context.
- Answer an item from its primary context. Use supporting context only to interpret the target system, target behavior, or comparison target named in the item.
- Do not use supporting context as the governing or reporting source for the item.
- If a requested item is answered by an explicit table cell, comparison row, metric row, caption, or short factual sentence in the retrieved evidence, treat it as supported even if the evidence is concise or tabular.
- If the question asks for a single value or metric, answer with the value for the requested subject only. Do not add baseline, comparison, or alternative-system values unless the requested item explicitly asks for comparison.
- If unsupported, set supported=false and answer with a concise self-contained statement that names the requested subject and includes the phrase "not supported by the retrieved evidence".
- If unsupported, support_text must name what the retrieved evidence actually contains and what is still missing.
- If the retrieved evidence explicitly states that a configuration fits, supports, allows, or reaches a quantity under the same requested condition, treat that quantity as the supported answer.
- If a requested item asks whether a named governing source supports matching controls, requirements, or sections for a target system or method, mark it supported when the governing-source text explicitly lists controls or requirements that match the target system's described behaviors, safeguards, or record-handling needs, even if the governing source does not name the target system verbatim.
- For that kind of supported item, answer by summarizing the matching controls or requirement categories from the governing source instead of repeating the unsupported template, and use support_text to name the matching control clauses or requirement categories.
- For that kind of supported item, name only controls, requirement categories, or legal identifiers that are explicitly visible in the primary context. Do not invent subsection letters, clause numbers, or citations that are not directly shown.
- support_text for that kind of item must stay close to the governing-source wording. Prefer short control phrases copied or lightly paraphrased from the primary context over reconstructed legal references.
- Do not mark that kind of item unsupported only because the governing source and the target system come from different application domains. Judge the item by whether the controls, safeguards, or record-handling requirements match the target behaviors described in the supporting context.
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
      `Question:\n${question}\n\nRequested items:\n- ${itemLabels.join("\n- ")}\n\nItem-scoped retrieved context:\n${context}`,
    ),
  ]);

  const parsed = parseItemAnswerPlan(parseChatResponse(response.content));
  const retrievedDocumentIds = getRetrievedDocumentIds(chunks);
  if (parsed.length > 0) {
    const byItem = new Map(parsed.map((entry) => [entry.item, entry]));
    return requestedItems.map((descriptor) => {
      const entry = byItem.get(descriptor.item);
      if (
        descriptor.primary_source_ids.length > 0 &&
        descriptor.primary_source_ids.some((documentId) => !retrievedDocumentIds.has(documentId))
      ) {
        return {
          item: descriptor.item,
          supported: false,
          answer: `The requested information about ${descriptor.item} is not supported by the retrieved evidence.`,
          support_text: "",
          reasoning: "",
        };
      }
      return entry ?? {
        item: descriptor.item,
        supported: false,
        answer: "Not supported by the retrieved evidence.",
        support_text: "",
        reasoning: "",
      };
    });
  }

  return requestedItems.map((descriptor) => ({
    item: descriptor.item,
    supported: false,
    answer: "Not supported by the retrieved evidence.",
    support_text: "",
    reasoning: "",
  }));
};

const generateCrossSourceMatchesWithModel = async (
  question: string,
  requestedItems: RequestedItemDescriptor[],
  baseItemResponses: ItemAnswerPlan[],
  chunks: ScoredChunk[],
  itemContextBundles?: ItemContextBundle[],
): Promise<ItemAnswerPlan[]> => {
  if (requestedItems.length === 0 || chunks.length === 0) {
    return [];
  }

  const chat = getChatModel();
  const scopedContexts = itemContextsFromBundles(requestedItems, itemContextBundles, chunks);

  return Promise.all(
    scopedContexts.map(async (scopedContext) => {
      const currentResponse =
        baseItemResponses.find((entry) => entry.item === scopedContext.descriptor.item) ?? null;
      const response = await chat.invoke([
        new SystemMessage(`
You are revising a candidate answer for an item that compares a primary governing source against a supporting source that describes a target system or method.

Return JSON only in this shape:
{"items":[{"item":"...","supported":true|false,"answer":"...","support_text":"...","reasoning":"..."}]}

Rules:
- Preserve each requested item exactly.
- First judge whether the current candidate answer is correct given the item-scoped retrieved context.
- Keep every field brief. Prefer short phrases over full quotations.
- support_text must be a single short phrase or sentence fragment of at most 25 words.
- reasoning must be a single short sentence of at most 20 words.
- primary_controls, supporting_safeguards, and matched_pairs must contain at most 4 entries each, and each entry must stay under 12 words.
- Compare the primary-source controls, requirements, sections, or safeguards against the supporting-source behaviors, safeguards, records, access patterns, or data-handling needs.
- Treat the item as supported when the primary-source text explicitly lists controls or requirements that match the target system's described behaviors or safeguards, even if the primary source does not name the target system verbatim.
- Do not reject the item only because the governing source and the target system come from different domains.
- Match by control or safeguard category, not by implementation-specific nouns. A primary-source control can support a supporting-source behavior when both concern the same category such as access restriction, record integrity, record authenticity, auditability, validation, confidentiality, or record attribution, even if the implementation details differ.
- If the supporting context explicitly names safeguard categories, compliance categories, or record-handling categories, use those explicit categories as the matching target instead of lower-level implementation details.
- If the supporting context explicitly states that a safeguard, record type, or operational boundary maps onto compliance requirements, and the primary context lists controls in the same category, treat that as direct support.
- If one primary-context chunk explicitly lists control or requirement categories that match the supporting safeguards, treat the item as supported even when other retrieved primary-context chunks from the same source are unrelated.
- If the supporting context mentions a different law, policy family, or regulatory source than the primary governing source, do not require the primary source to reproduce that other law or source. Use only the safeguard, control, audit, access, integrity, confidentiality, record, or erasure categories described around that mention as the matching target.
- Do not treat the name of another law, regulation, or policy family as a safeguard category by itself. If the supporting context mentions another law or regulation, extract the concrete safeguard or control category described around that mention instead of the law name.
- If the current candidate answer is unsupported but the context shows matching controls or requirements, rewrite it as supported.
- If the current candidate answer is supported but the context does not show matching controls or requirements, rewrite it as unsupported.
- Before deciding supported, list the explicit governing-source control phrases in primary_controls and the explicit target-system safeguard or compliance phrases in supporting_safeguards.
- Then list only the actual category-level matches between those two lists in matched_pairs.
- If matched_pairs is non-empty, treat the item as supported. If matched_pairs is empty, treat the item as unsupported.
- If supported, answer by summarizing the matching control or requirement categories from the primary source.
- support_text must name the specific matching control phrases, requirement categories, or safeguard text from the primary source.
- Name only controls, requirement categories, or legal identifiers that are explicitly visible in the primary context. Do not invent subsection letters, clause numbers, or citations that are not directly shown.
- Keep support_text close to the primary-source wording. Prefer short copied or lightly paraphrased control phrases over reconstructed legal references.
- reasoning must explain in one short sentence which supporting-source behaviors or safeguards those primary-source controls match.
- Mark supported=false only when the primary context does not provide matching controls or requirements for the behaviors described in the supporting context.
- If unsupported, answer must include the phrase "not supported by the retrieved evidence", support_text must say what the primary source contains, and reasoning must explain what matching behavior or safeguard is still missing.
- Do not invent controls, sections, or behaviors that are not in the retrieved context.
- Do not include citations, provenance, or page references.
`),
        new HumanMessage(
          `Question:\n${question}\n\nRequested item:\n- ${scopedContext.descriptor.item}\n\nCurrent candidate answer:\n${JSON.stringify(currentResponse ?? { item: scopedContext.descriptor.item, supported: false, answer: "" })}\n\nItem-scoped retrieved context:\n${formatItemScopedContexts([scopedContext])}`,
        ),
      ]);

      const parsed = parseItemAnswerPlan(parseChatResponse(response.content));
      return (
        parsed.find((entry) => entry.item === scopedContext.descriptor.item) ?? {
          item: scopedContext.descriptor.item,
          supported: false,
          answer: "Not supported by the retrieved evidence.",
          support_text: "",
          reasoning: "",
        }
      );
    }),
  );
};

const verifyRequestedItemsWithModel = async (
  question: string,
  requestedItems: RequestedItemDescriptor[],
  itemResponses: ItemAnswerPlan[],
  answer: string,
  chunks: ScoredChunk[],
  itemContextBundles?: ItemContextBundle[],
): Promise<ItemGroundingVerdict | null> => {
  if (chunks.length === 0 || requestedItems.length === 0) {
    return null;
  }

  const scopedContexts = itemContextsFromBundles(requestedItems, itemContextBundles, chunks);
  const context = formatItemScopedContexts(scopedContexts);

  const chat = getChatModel();
  const itemResponseSummary = itemResponses
    .map((entry) =>
      JSON.stringify({
        item: entry.item,
        supported: entry.supported,
        answer: entry.answer,
        support_text: entry.support_text ?? "",
        reasoning: entry.reasoning ?? "",
      }),
    )
    .join("\n");
  const response = await chat.invoke([
    new SystemMessage(`
Verify whether the answer handles each requested item correctly, and correct it when the retrieved evidence supports a better grounded resolution.

Return JSON only:
{"supported":true|false,"reason":"...","items":[{"item":"...","supported":true|false,"answer":"...","support_text":"...","reasoning":"...","primary_controls":["..."],"supporting_safeguards":["..."],"matched_pairs":["..."]}]}.

Use the items array to provide the grounded final resolution for every requested item.
If the current answer or resolved items are wrong but the retrieved evidence supports a corrected answer, correct them in the items array.
Set supported=true when the items array provides a fully grounded final resolution for all requested items.
Set supported=false only when the retrieved evidence is insufficient to produce a grounded final resolution for one or more requested items.
- Keep every field brief. Prefer short phrases over full quotations.
- support_text must be a single short phrase or sentence fragment of at most 25 words.
- reasoning must be a single short sentence of at most 20 words.
- primary_controls, supporting_safeguards, and matched_pairs must contain at most 4 entries each, and each entry must stay under 12 words.

For supported items, require that the item's answer matches its support_text and that the support_text directly supports the exact requested item rather than a nearby metric, qualifier, row, or condition.
If a value is explicitly present in the retrieved context, treat it as supported only when it matches the requested subject and qualifier. Preserve stated qualifiers and conditions.
For items asking whether a governing source supports matching controls or requirements for a target system or method, treat the item as supported when the governing-source text explicitly lists controls or requirements that match the target system's described behaviors or safeguards, even if the governing source does not name the target system verbatim.
Do not reject that kind of item only because the governing source and target system are from different domains. Reject it only when the retrieved governing-source text lacks matching controls or requirements for the behaviors described in the supporting context.
If the supporting context explicitly names safeguard categories, compliance categories, or record-handling categories, use those explicit categories as the matching target instead of lower-level implementation details.
If the supporting context explicitly states that a safeguard, record type, or operational boundary maps onto compliance requirements, and the governing-source text lists controls in the same category, treat that as direct support.
If one or more governing-source chunks explicitly list matching control or requirement categories, treat the item as supported even when other retrieved chunks from the same governing source are unrelated.
If the supporting context mentions a different law, policy family, or regulatory source than the governing source, do not require the governing source to reproduce that other law or source. Use only the safeguard, control, audit, access, integrity, confidentiality, record, or erasure categories described around that mention as the matching target.
Do not treat the name of another law, regulation, or policy family as a safeguard category by itself. If the supporting context mentions another law or regulation, extract the concrete safeguard or control category described around that mention instead of the law name.
For that kind of supported item, the answer must summarize the matching control or requirement categories from the governing source in plain language and must not invent clause letters, subsection ids, or citations that are not explicitly shown in the retrieved primary context.
For that kind of item, populate primary_controls with explicit governing-source control phrases, supporting_safeguards with explicit target-system safeguard or compliance phrases, and matched_pairs with the actual category-level matches between them. If matched_pairs is non-empty, treat the item as supported. If matched_pairs is empty, treat the item as unsupported.
`),
    new HumanMessage(
      `Question:\n${question}\n\nRequested items:\n- ${getRequestedItemLabels(requestedItems).join("\n- ")}\n\nResolved items:\n${itemResponseSummary}\n\nAnswer:\n${answer}\n\nItem-scoped retrieved context:\n${context}`,
    ),
  ]);

  const rawResponse = parseChatResponse(response.content);
  const parsed = parseItemGroundingVerdict(rawResponse);
  return (
    parsed ?? {
      supported: false,
      reason: "Verifier returned an unparseable response.",
      items: [],
      raw_response: rawResponse,
    }
  );
};

const resolveCrossSourceItemWithModel = async (
  question: string,
  descriptor: RequestedItemDescriptor,
  bundle: ItemContextBundle,
  current: ItemAnswerPlan | null,
): Promise<ItemAnswerPlan | null> => {
  if (descriptor.supporting_source_ids.length === 0) {
    return null;
  }

  const chat = getChatModel();
  const scopedContext = formatItemScopedContexts([
    {
      descriptor,
      primaryChunks: bundle.primary_chunks,
      supportingChunks: bundle.supporting_chunks,
    },
  ]);

  const response = await chat.invoke([
    new SystemMessage(`
Return JSON only:
{"items":[{"item":"...","supported":true|false,"answer":"...","support_text":"...","reasoning":"...","primary_controls":["..."],"supporting_safeguards":["..."],"matched_pairs":["..."]}]}

Resolve one cross-source item from a primary governing source and a supporting source.

Rules:
- Extract explicit control or requirement phrases from the primary context into primary_controls.
- Extract explicit safeguard, record-handling, compliance-category, or operational-boundary phrases from the supporting context into supporting_safeguards.
- Populate matched_pairs only when a primary control phrase and a supporting safeguard phrase share the same control category.
- Match by control category, not by domain. Examples of valid category matches when they are explicitly present: authority or access checks, audit trails, independent/addressable records, record integrity, validation, confidentiality, record linking, or attribution.
- If matched_pairs is non-empty, set supported=true and answer by summarizing the matching primary controls in plain language.
- If matched_pairs is empty, set supported=false and answer with the unsupported template.
- Do not invent clause ids, subsection letters, or controls not visible in the primary context.
- Do not use law names by themselves as supporting safeguards.
`),
    new HumanMessage(
      `Question:\n${question}\n\nRequested item:\n${descriptor.item}\n\nCurrent resolution:\n${JSON.stringify(current ?? { item: descriptor.item, supported: false, answer: "" })}\n\nItem-scoped context:\n${scopedContext}`,
    ),
  ]);

  const parsed = parseItemAnswerPlan(parseChatResponse(response.content));
  return parsed.find((entry) => entry.item === descriptor.item) ?? null;
};

export const retrieveNode = async (state: GraphState): Promise<GraphState> => {
  const { lexical, vector } = await getIndexStore({ allowDenseMissing: true });
  const explicitSourceIds = extractReferencedDocumentIds(state.question);
  const retrievalQueries =
    explicitSourceIds.length > 0
      ? explicitSourceIds.map((documentId) =>
          buildSourceScopedQuery(
            [documentId],
            state.question,
            state.question,
            explicitSourceIds.filter((value) => value !== documentId),
          ),
        )
      : [state.question];
  const retrieval =
    explicitSourceIds.length > 0
      ? await Promise.all(
          explicitSourceIds.map(async (documentId, index) =>
            retrieveCrossDocument(lexical, vector, {
              query: retrievalQueries[index] ?? state.question,
              additionalQueries: [],
              topK: appConfig.retrieval.topK,
              filterDocumentIds: [documentId],
            }),
          ),
        ).then((bundleResults) => ({
          question: state.question,
          chunks: mergeBundleChunks(bundleResults.map((result) => result.chunks)),
          reranked: bundleResults.some((result) => result.reranked),
          degradation_reasons: Array.from(
            new Set(bundleResults.flatMap((result) => result.degradation_reasons ?? [])),
          ),
          dense_enabled: bundleResults.every((result) => result.dense_enabled !== false),
          lexical_enabled: bundleResults.every((result) => result.lexical_enabled !== false),
        }))
      : await retrieveCrossDocument(lexical, vector, {
          query: state.question,
          additionalQueries: [],
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
        retrievalQueries,
        explicitSourceIds,
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
      : (await planRequestedItems(state.question)).requestedItems;
  const itemContexts =
    state.item_contexts && state.item_contexts.length > 0
      ? state.item_contexts
      : await hydrateItemContextBundles(state.question, requestedItems, chunks);

  if (chunks.length === 0) {
    return {
      ...state,
      requested_items: requestedItems,
      answer: EVIDENCE_WARNING,
      citations: [],
      trace: appendTrace(state.trace, "generate", "failed", "No chunks available for generation.", 0, { evidenceCount: 0 }),
    };
  }

  const itemResponses = await generateItemResponsesWithModel(
    state.question,
    requestedItems,
    chunks,
    itemContexts,
  );
  const finalChunks = chunks;
  const analysis = analyseCrossDocEvidence(finalChunks);
  const supportedCount = itemResponses.filter((item) => item.supported).length;
  const answer =
    itemResponses.length > 0
      ? formatRequestedItemResponses(itemResponses, requestedItems)
      : EVIDENCE_WARNING;

  return {
    ...state,
    retrieval: state.retrieval
      ? {
          ...state.retrieval,
          chunks: finalChunks,
        }
      : state.retrieval,
    requested_items: requestedItems,
    item_contexts: itemContexts,
    item_responses: itemResponses,
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
        requestedItems: getRequestedItemLabels(requestedItems),
        itemContextCoverage: itemContexts.map((context) => ({
          item: context.item,
          primaryDocuments: Array.from(new Set(context.primary_chunks.map((chunk) => chunk.chunk.metadata.document_id))),
          supportingDocuments: Array.from(new Set(context.supporting_chunks.map((chunk) => chunk.chunk.metadata.document_id))),
          primaryTopPages: context.primary_chunks.slice(0, 6).map((chunk) => chunk.chunk.metadata.page),
          supportingTopPages: context.supporting_chunks.slice(0, 6).map((chunk) => chunk.chunk.metadata.page),
        })),
        supportedItemCount: supportedCount,
        itemResponses,
        draftAnswer: answer,
        sourceCount: finalChunks.length,
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
  const itemResponses = state.item_responses ?? [];
  const requestedItems =
    state.requested_items && state.requested_items.length > 0
      ? state.requested_items
      : (await planRequestedItems(state.question)).requestedItems;
  const itemContexts =
    state.item_contexts && state.item_contexts.length > 0
      ? state.item_contexts
      : await hydrateItemContextBundles(state.question, requestedItems, state.retrieval?.chunks ?? []);

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
    itemResponses,
    answer,
    state.retrieval?.chunks ?? [],
    itemContexts,
  );

  const correctedItemResponses =
    itemVerdict?.items && itemVerdict.items.length > 0
      ? requestedItems.map((descriptor) => {
          const corrected = itemVerdict.items?.find((entry) => entry.item === descriptor.item);
          const existing = itemResponses.find((entry) => entry.item === descriptor.item);
          return (
            corrected ??
            existing ?? {
              item: descriptor.item,
              supported: false,
              answer: `The requested information about ${descriptor.item} is not supported by the retrieved evidence.`,
              support_text: "",
              reasoning: "",
            }
          );
        })
      : itemResponses;
  const finalItemResponses = correctedItemResponses;
  const hasGroundedItemResolution =
    finalItemResponses.length === requestedItems.length &&
    finalItemResponses.every(
      (entry) => entry.item.trim().length > 0 && entry.answer.trim().length > 0,
    );
  const correctedAnswer =
    finalItemResponses.length > 0
      ? formatRequestedItemResponses(finalItemResponses, requestedItems)
      : answer;

  if (itemVerdict && !itemVerdict.supported && !hasGroundedItemResolution) {
    return {
      ...state,
      requested_items: requestedItems,
      item_contexts: itemContexts,
      item_responses: finalItemResponses,
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
          correctedItemResponses: finalItemResponses,
          verifierRawResponse: itemVerdict.raw_response ?? "",
        },
      ),
    };
  }

  return {
    ...state,
    requested_items: requestedItems,
    item_contexts: itemContexts,
    item_responses: finalItemResponses,
    answer: correctedAnswer,
    citations: [],
    trace: appendTrace(
      state.trace,
      "verify",
      itemVerdict && !itemVerdict.supported ? "passed" : "passed",
      itemVerdict && !itemVerdict.supported
        ? "Verifier returned a grounded corrected item resolution."
        : "Verified generated answer against retrieved evidence.",
      undefined,
      itemVerdict
        ? {
            verifierReason: itemVerdict.reason,
            correctedItemResponses: finalItemResponses,
            correctedAnswer,
            verifierRawResponse: itemVerdict.raw_response ?? "",
            hasGroundedItemResolution,
          }
        : {},
    ),
  };
};
