import { appConfig } from "../lib/config";
import { getChatModel } from "../lib/llm";

const QUERY_LIMIT = 4;
const expansionCache = new Map<string, string[]>();

const normalizeQuery = (value: string): string =>
  value.replace(/\s+/gu, " ").trim();

const dedupeQueries = (queries: string[]): string[] => {
  const seen = new Set<string>();
  const unique: string[] = [];

  for (const raw of queries) {
    const query = normalizeQuery(raw);
    if (!query) continue;
    const key = query.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    unique.push(query);
    if (unique.length >= QUERY_LIMIT) break;
  }

  return unique;
};

const extractTextContent = (content: unknown): string => {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") return part;
        if (part && typeof part === "object" && "text" in part) {
          const text = (part as { text?: unknown }).text;
          return typeof text === "string" ? text : "";
        }
        return "";
      })
      .join("\n");
  }
  return "";
};

const parseExpandedQueries = (raw: string): string[] => {
  const trimmed = raw.trim();
  if (!trimmed) return [];

  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/u);
  const candidate = fenced?.[1]?.trim() ?? trimmed;

  try {
    const parsed = JSON.parse(candidate) as unknown;
    if (Array.isArray(parsed)) {
      return parsed.filter((entry): entry is string => typeof entry === "string");
    }
    if (
      parsed &&
      typeof parsed === "object" &&
      "queries" in parsed &&
      Array.isArray((parsed as { queries?: unknown }).queries)
    ) {
      return (parsed as { queries: unknown[] }).queries.filter(
        (entry): entry is string => typeof entry === "string",
      );
    }
  } catch {
    // fall through
  }

  return trimmed
    .split(/\n+/u)
    .map((line) => line.replace(/^[-*\d.)\s]+/u, "").trim())
    .filter(Boolean);
};

const buildExpansionPrompt = (question: string): string => `
Rewrite the question into up to ${QUERY_LIMIT - 1} additional retrieval queries.

Rules:
- Keep exact identifiers, filenames, numbers, quoted strings, and qualifiers unchanged.
- Preserve scope. Do not broaden to comparisons, baselines, or related systems unless asked.
- If helpful, split a multi-part question into shorter evidence-seeking queries.
- If helpful, include one English query for cross-lingual retrieval.
- For policy, control, or section-identification items, keep the governing source and the concrete behavior or topic terms together in at least one query.
- Keep queries short and concrete.
- Return JSON only: {"queries":["..."]}.

Question:
${question}
`;

export const expandRetrievalQueries = async (
  question: string,
  _context?: unknown,
): Promise<string[]> => {
  const questionText = normalizeQuery(question);
  if (!questionText) return [];

  const cacheKey = `${appConfig.models.baseUrl}|${appConfig.models.chatModel}|${questionText}`;
  const cached = expansionCache.get(cacheKey);
  if (cached) {
    return cached;
  }

  const baseQueries = [questionText];

  try {
    const model = getChatModel();
    const response = await model.invoke(buildExpansionPrompt(questionText));
    const responseText = extractTextContent(response.content);
    const expanded = parseExpandedQueries(responseText);
    const queries = dedupeQueries([questionText, ...expanded]);
    expansionCache.set(cacheKey, queries);
    return queries;
  } catch {
    expansionCache.set(cacheKey, baseQueries);
    return baseQueries;
  }
};
