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
Rewrite the following user question into up to ${QUERY_LIMIT - 1} additional retrieval queries for arbitrary document search.

Requirements:
- Keep exact identifiers, document ids, filenames, numbers, and quoted strings unchanged.
- If the user question is not in the same language as the likely source text, include a translated English retrieval query.
- If the question contains multiple subparts, include shorter evidence-seeking queries that preserve the original meaning.
- Prefer concrete noun phrases and keywords that are likely to appear verbatim in source documents.
- Avoid vague rewrites that mainly repeat abstract words like "document", "system", "control", or "requirements" without concrete concepts.
- When the question asks for compliance, controls, limits, metrics, or requirements, generate retrieval queries that name the specific concepts implied by the task instead of only repeating the generic category words.
- Do not invent facts or narrow the task to a specific domain beyond what is implied by the user question.
- Return JSON only in this shape: {"queries":["..."]}.

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
