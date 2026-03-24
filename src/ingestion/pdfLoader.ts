import { readFile, access } from "node:fs/promises";
import { constants } from "node:fs";
import { basename } from "node:path";
import { CorpusChunk } from "../types/rag";
import { chunkPageText } from "../chunking/sectionChunker";
import { extractPdfDocumentInfo, PdfDocumentInfo, enrichMetadata } from "../chunking/metadataExtractor";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

/** A single page extracted from a PDF with its 1-based page number. */
export interface ExtractedPage {
  /** 1-based page number */
  page: number;
  /** Raw text content extracted from the page */
  text: string;
}

/** Summary information about a parsed PDF document. */
export interface PdfParseResult {
  /** Original file path that was parsed */
  filePath: string;
  /** Basename of the PDF file (e.g. "document.pdf") */
  fileName: string;
  /** Total number of pages in the PDF */
  totalPages: number;
  /** Pages that contained at least some text */
  nonEmptyPageCount: number;
  /** All extracted pages (including empty ones) */
  pages: ExtractedPage[];
  /** Document-level metadata extracted from the PDF info dictionary */
  documentInfo: PdfDocumentInfo;
}

/* ------------------------------------------------------------------ */
/*  Internals                                                          */
/* ------------------------------------------------------------------ */

/**
 * Extract text content from a single pdf.js page object.
 * Joins text items with spaces (within a line) and newlines (between lines),
 * preserving basic reading order.
 */
const extractTextFromPage = async (page: any): Promise<string> => {
  const textContent = await page.getTextContent();
  const items = textContent.items as Array<{
    str?: string;
    hasEOL?: boolean;
    transform?: number[];
  }>;

  if (!items || items.length === 0) return "";

  const rows = items
    .map((item, index) => ({
      text: item.str ?? "",
      x: Array.isArray(item.transform) ? item.transform[4] ?? 0 : 0,
      y: Array.isArray(item.transform) ? item.transform[5] ?? 0 : 0,
      hasEOL: Boolean(item.hasEOL),
      index,
    }))
    .filter((item) => item.text.trim().length > 0);

  if (rows.length === 0) return "";

  const lineBuckets = new Map<number, typeof rows>();
  const lineTolerance = 2;
  for (const row of rows) {
    const key = Math.round(row.y / lineTolerance) * lineTolerance;
    const bucket = lineBuckets.get(key) ?? [];
    bucket.push(row);
    lineBuckets.set(key, bucket);
  }

  const lines = [...lineBuckets.entries()]
    .sort((a, b) => b[0] - a[0])
    .map(([, bucket]) =>
      bucket
        .sort((a, b) => a.x - b.x || a.index - b.index)
        .map((entry) => entry.text)
        .join(" ")
        .replace(/\s+/gu, " ")
        .trim(),
    )
    .filter((line) => line.length > 0);

  return lines
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
};

const stripRepeatedBoilerplate = (pages: ExtractedPage[]): ExtractedPage[] => {
  const nonEmptyPages = pages.filter((page) => page.text.trim().length > 0);
  const majorityThreshold = Math.ceil(nonEmptyPages.length / 2);
  if (majorityThreshold <= 1) {
    return pages;
  }

  const normalizeLineSignature = (line: string): string =>
    line
      .toLowerCase()
      .replace(/\d+/gu, "#")
      .replace(/[a-z]:\\[^\s]+/gu, "<path>")
      .replace(/\s+/gu, " ")
      .trim();

  const lineDocumentFrequency = new Map<string, number>();
  for (const page of nonEmptyPages) {
    const uniqueLines = new Set(
      page.text
        .split("\n")
        .map((line) => normalizeLineSignature(line))
        .filter((line) => line.length > 0),
    );
    for (const line of uniqueLines) {
      lineDocumentFrequency.set(line, (lineDocumentFrequency.get(line) ?? 0) + 1);
    }
  }

  return pages.map((page) => {
    if (!page.text.trim()) {
      return page;
    }

    const cleaned = page.text
      .split("\n")
      .map((line) => line.replace(/\s+/gu, " ").trim())
      .filter((line) => {
        if (!line) return false;
        const frequency = lineDocumentFrequency.get(normalizeLineSignature(line)) ?? 0;
        return frequency < majorityThreshold;
      })
      .join("\n")
      .replace(/\n{3,}/gu, "\n\n")
      .trim();

    return { ...page, text: cleaned };
  });
};

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

/**
 * Parse a local PDF file and extract text content page-by-page.
 *
 * Returns a `PdfParseResult` containing every page's raw text along with
 * its 1-based page number.  Empty pages are included (with `text: ""`).
 *
 * @throws {Error} if the file does not exist or cannot be read
 * @throws {Error} if the file is not a valid PDF
 */
export const parsePdf = async (filePath: string): Promise<PdfParseResult> => {
  // Validate file exists before attempting to read
  try {
    await access(filePath, constants.R_OK);
  } catch {
    throw new Error(`PDF file not found or not readable: ${filePath}`);
  }

  const buffer = await readFile(filePath);
  if (buffer.length === 0) {
    throw new Error(`PDF file is empty: ${filePath}`);
  }

  // pdfjs-dist requires Uint8Array, not Node Buffer
  const bytes = new Uint8Array(buffer);

  // Dynamic import – pdfjs-dist legacy build works in Node without a canvas
  const pdfjs = await import("pdfjs-dist/legacy/build/pdf.mjs");

  let pdf: any;
  try {
    pdf = await pdfjs.getDocument({ data: bytes }).promise;
  } catch (err: any) {
    throw new Error(
      `Failed to parse PDF "${filePath}": ${err?.message ?? String(err)}`
    );
  }

  const pages: ExtractedPage[] = [];

  for (let index = 1; index <= pdf.numPages; index += 1) {
    const page = await pdf.getPage(index);
    const text = await extractTextFromPage(page);
    pages.push({ page: index, text });
  }

  const nonEmptyPageCount = pages.filter((p) => p.text.length > 0).length;

  // Extract document-level metadata from PDF info dictionary
  const documentInfo = await extractPdfDocumentInfo(pdf);

  return {
    filePath,
    fileName: basename(filePath),
    totalPages: pdf.numPages,
    nonEmptyPageCount,
    pages,
    documentInfo,
  };
};

/**
 * Convenience wrapper that calls `parsePdf` and returns just the pages array.
 * Backward-compatible with the original `loadPdfPages` signature.
 */
export const loadPdfPages = async (
  filePath: string
): Promise<ExtractedPage[]> => {
  const result = await parsePdf(filePath);
  return result.pages;
};

/**
 * Full pipeline: parse PDF → section-aware chunking → CorpusChunk[].
 *
 * Each chunk is enriched with validated SourceMetadata containing:
 * - source_file: original PDF filename
 * - document_id: corpus-level identifier
 * - page: 1-based page number from the PDF
 * - chunk_id: globally unique (within document) chunk index
 * - section_title: optional section label
 */
export const buildChunksFromPdf = async (params: {
  filePath: string;
  sourceFile: string;
  documentId: string;
  sectionAwareOptions?: {
    chunkSize?: number;
    chunkOverlap?: number;
  };
}): Promise<CorpusChunk[]> => {
  const parseResult = await parsePdf(params.filePath);
  const pages = stripRepeatedBoilerplate(parseResult.pages);
  const chunks: CorpusChunk[] = [];
  let chunkId = 0;

  if (parseResult.documentInfo.title || parseResult.documentInfo.author) {
    console.log(
      `[metadata] PDF info for "${params.sourceFile}": title="${parseResult.documentInfo.title ?? "N/A"}", author="${parseResult.documentInfo.author ?? "N/A"}"`
    );
  }

  for (const page of pages) {
    // Skip pages with no text content
    if (!page.text.trim()) continue;

    const pieces = await chunkPageText({
      pageText: page.text,
      pageNumber: page.page,
      sourceFile: params.sourceFile,
      documentId: params.documentId,
      chunkSize: params.sectionAwareOptions?.chunkSize,
      chunkOverlap: params.sectionAwareOptions?.chunkOverlap,
    });

    for (const piece of pieces) {
      // Reassign chunk_id to be globally unique within this document
      chunks.push({
        ...piece,
        metadata: enrichMetadata(piece.metadata, { chunk_id: chunkId }),
      });
      chunkId += 1;
    }
  }

  return chunks;
};
