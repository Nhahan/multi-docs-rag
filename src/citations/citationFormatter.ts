/**
 * Citation formatting for Query API responses.
 *
 * Transforms raw pipeline output (answer + retrieved chunks) into a
 * structured, citation-safe response with:
 * - inline citations preserved/injected
 * - stable citation metadata entries
 * - a readable source footer
 */

import { ScoredChunk, SourceMetadata } from "../types/rag";
import {
  citationLabel,
  extractCitationsFromAnswer,
  extractCitationsFromChunks,
  hasCitations,
} from "./citationExtractor";
import { appConfig } from "../lib/config";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

export interface CitationEntry {
  label: string;
  document_id: string;
  page: number;
  source_file: string;
}

export interface FormattedCitationResponse {
  cited_answer: string;
  inline_citations: string[];
  citation_entries: CitationEntry[];
  citation_footer: string;
  has_citations: boolean;
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

export const buildCitationEntry = (metadata: SourceMetadata): CitationEntry => ({
  label: citationLabel(metadata),
  document_id: metadata.document_id,
  page: metadata.page,
  source_file: metadata.source_file,
});

export const buildCitationEntries = (chunks: ScoredChunk[]): CitationEntry[] => {
  const seen = new Set<string>();
  const entries: CitationEntry[] = [];

  for (const sc of chunks) {
    const metadata = sc?.chunk?.metadata;
    if (!metadata) continue;
    const label = citationLabel(metadata);
    if (seen.has(label)) continue;
    seen.add(label);
    entries.push(buildCitationEntry(metadata));
  }

  return entries;
};

export const filterCitedEntries = (
  answer: string,
  allEntries: CitationEntry[],
): CitationEntry[] => {
  const cited = new Set(extractCitationsFromAnswer(answer));
  return allEntries.filter((entry) => cited.has(entry.label));
};

/* ------------------------------------------------------------------ */
/*  Footer                                                             */
/* ------------------------------------------------------------------ */

export const buildCitationFooter = (entries: CitationEntry[]): string => {
  if (entries.length === 0) return "";

  const lines = entries.map(
    (entry, i) =>
      `[${i + 1}] ${entry.label} — ${entry.source_file}, p.${entry.page}`,
  );

  return `Sources:\n${lines.join("\n")}`;
};

/* ------------------------------------------------------------------ */
/*  Inline citations                                                   */
/* ------------------------------------------------------------------ */

export const ensureInlineCitations = (
  answer: string,
  chunks: ScoredChunk[],
): string => {
  const insufficiencyMessage = appConfig.messages.insufficientEvidence;

  if (answer.includes(insufficiencyMessage)) {
    return answer;
  }

  if (hasCitations(answer)) {
    return answer;
  }

  const available = extractCitationsFromChunks(chunks);
  if (available.length === 0) {
    return answer;
  }

  return `${answer}\n\n(Sources: ${available.join(", ")})`;
};

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

export const formatCitationResponse = (
  answer: string,
  chunks: ScoredChunk[],
): FormattedCitationResponse => {
  const citedAnswer = ensureInlineCitations(answer, chunks);
  const inlineCitations = extractCitationsFromAnswer(citedAnswer);
  const allEntries = buildCitationEntries(chunks);

  const citedEntries =
    inlineCitations.length > 0
      ? filterCitedEntries(citedAnswer, allEntries)
      : [];

  return {
    cited_answer: citedAnswer,
    inline_citations: inlineCitations,
    citation_entries: citedEntries,
    citation_footer: buildCitationFooter(citedEntries),
    has_citations: inlineCitations.length > 0,
  };
};
