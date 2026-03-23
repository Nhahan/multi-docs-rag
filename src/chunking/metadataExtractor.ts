/**
 * Metadata extraction and enrichment for chunk metadata.
 *
 * This module is intentionally document-agnostic:
 * - No document-type-specific heading rules
 * - No domain-specific assumptions
 * - No heading-guess heuristics; section labels are assigned by chunking strategy
 */

import { SourceMetadata } from "../types/rag";

/* ------------------------------------------------------------------ */
/*  Document-level metadata (from PDF info dict)                       */
/* ------------------------------------------------------------------ */

/** Optional metadata parsed from the PDF's info dictionary. */
export interface PdfDocumentInfo {
  title?: string;
  author?: string;
  subject?: string;
  keywords?: string;
  creationDate?: string;
}

/**
 * Extract document-level metadata from a pdfjs document object.
 * Gracefully returns empty object if metadata is unavailable.
 */
export const extractPdfDocumentInfo = async (pdfDocument: any): Promise<PdfDocumentInfo> => {
  try {
    const metadata = await pdfDocument.getMetadata();
    const info = metadata?.info ?? {};
    return {
      title: info.Title || undefined,
      author: info.Author || undefined,
      subject: info.Subject || undefined,
      keywords: info.Keywords || undefined,
      creationDate: info.CreationDate || undefined,
    };
  } catch {
    return {};
  }
};

/* ------------------------------------------------------------------ */
/*  Metadata construction & validation                                 */
/* ------------------------------------------------------------------ */

/**
 * Build a SourceMetadata object with all required fields.
 */
export const buildSourceMetadata = (params: {
  source_file: string;
  document_id: string;
  page: number;
  chunk_id: number;
  section_title: string | null;
}): SourceMetadata => {
  validateMetadata(params);
  return {
    source_file: params.source_file,
    document_id: params.document_id,
    page: params.page,
    chunk_id: params.chunk_id,
    section_title: params.section_title,
  };
};

/**
 * Validate that a SourceMetadata object has all required fields
 * with proper types and values.
 *
 * @throws {Error} if validation fails
 */
export const validateMetadata = (meta: Partial<SourceMetadata>): void => {
  const errors: string[] = [];

  if (!meta.source_file || typeof meta.source_file !== "string") {
    errors.push("source_file must be a non-empty string");
  }
  if (!meta.document_id || typeof meta.document_id !== "string") {
    errors.push("document_id must be a non-empty string");
  }
  if (typeof meta.page !== "number" || meta.page < 1) {
    errors.push("page must be a positive number (1-based)");
  }
  if (typeof meta.chunk_id !== "number" || meta.chunk_id < 0) {
    errors.push("chunk_id must be a non-negative number");
  }
  if (meta.section_title !== null && typeof meta.section_title !== "string") {
    errors.push("section_title must be a string or null");
  }
  if (errors.length > 0) {
    throw new Error(`Invalid chunk metadata: ${errors.join("; ")}`);
  }
};

/**
 * Enrich an existing metadata object with additional inferred fields.
 * Used to add or correct metadata after initial extraction.
 */
export const enrichMetadata = (
  base: SourceMetadata,
  overrides: Partial<SourceMetadata>,
): SourceMetadata => {
  const enriched = { ...base, ...overrides };
  validateMetadata(enriched);
  return enriched;
};
