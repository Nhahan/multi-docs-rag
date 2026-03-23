import { CorpusChunk } from "../types/rag";
import { buildSourceMetadata } from "./metadataExtractor";

/** Options for controlling chunk size and overlap. */
export interface ChunkingOptions {
  /** Maximum characters per chunk */
  chunkSize?: number;
  /** Character overlap between consecutive chunks */
  chunkOverlap?: number;
}

/**
 * Split a single page's extracted text into chunks.
 */
const splitParagraphs = (text: string): string[] =>
  text
    .split(/\n\s*\n/gu)
    .map((block) => block.replace(/[ \t]+\n/gu, "\n").replace(/[ \t]{2,}/gu, " ").trim())
    .filter((block) => block.length > 0);

const splitLongText = (text: string, maxLength: number): string[] => {
  if (text.length <= maxLength) {
    return [text];
  }

  if (typeof Intl !== "undefined" && typeof Intl.Segmenter === "function") {
    const segmenter = new Intl.Segmenter("en", { granularity: "sentence" });
    const sentences = [...segmenter.segment(text)]
      .map((entry) => entry.segment.trim())
      .filter((entry) => entry.length > 0);

    if (sentences.length > 1) {
      const chunks: string[] = [];
      let current = "";

      for (const sentence of sentences) {
        const candidate = current ? `${current} ${sentence}` : sentence;
        if (candidate.length <= maxLength) {
          current = candidate;
          continue;
        }
        if (current) {
          chunks.push(current);
        }
        current = sentence;
      }

      if (current) {
        chunks.push(current);
      }

      if (chunks.every((chunk) => chunk.length <= maxLength)) {
        return chunks;
      }
    }
  }

  const chunks: string[] = [];
  for (let start = 0; start < text.length; start += maxLength) {
    chunks.push(text.slice(start, start + maxLength).trim());
  }
  return chunks.filter((chunk) => chunk.length > 0);
};

const packParagraphs = (
  paragraphs: string[],
  maxLength?: number,
): string[] => {
  if (!maxLength || maxLength <= 0) {
    return [paragraphs.join("\n\n").trim()].filter((entry) => entry.length > 0);
  }

  const chunks: string[] = [];
  let current = "";

  for (const paragraph of paragraphs) {
    const parts = splitLongText(paragraph, maxLength);
    for (const part of parts) {
      const candidate = current ? `${current}\n\n${part}` : part;
      if (candidate.length <= maxLength) {
        current = candidate;
        continue;
      }
      if (current) {
        chunks.push(current);
      }
      current = part;
    }
  }

  if (current) {
    chunks.push(current);
  }

  return chunks;
};

export const chunkPageText = async ({
  pageText,
  pageNumber,
  sourceFile,
  documentId,
  chunkSize,
  chunkOverlap,
}: {
  pageText: string;
  pageNumber: number;
  sourceFile: string;
  documentId: string;
  chunkSize?: number;
  chunkOverlap?: number;
}): Promise<CorpusChunk[]> => {
  const normalizedPageText = pageText.trim();
  if (!normalizedPageText) {
    return [];
  }

  const paragraphs = splitParagraphs(normalizedPageText);
  const chunks = packParagraphs(
    paragraphs.length > 0 ? paragraphs : [normalizedPageText],
    chunkSize,
  );
  const overlap = chunkOverlap && chunkOverlap > 0 ? chunkOverlap : 0;
  const out: CorpusChunk[] = [];
  let localIndex = 0;

  let previousTail = "";
  for (const text of chunks) {
    if (!text.trim()) {
      continue;
    }

    const chunkText =
      overlap > 0 && previousTail
        ? `${previousTail}\n\n${text.trim()}`
        : text.trim();

    const metadata = buildSourceMetadata({
      source_file: sourceFile,
      document_id: documentId,
      page: pageNumber,
      chunk_id: localIndex,
      section_title: null,
    });

    out.push({
      id: `${documentId}-p${pageNumber}-s${localIndex}`,
      text: chunkText,
      metadata,
    });
    previousTail =
      overlap > 0
        ? chunkText.slice(Math.max(0, chunkText.length - overlap)).trim()
        : "";
    localIndex += 1;
  }

  return out;
};
