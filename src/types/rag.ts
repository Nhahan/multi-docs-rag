export type DocumentType = string;

export type PipelineStage =
  | "retrieve"
  | "evidence_gate"
  | "generate"
  | "verify";

export type PipelineStatus = "passed" | "degraded" | "failed" | "skipped";

export interface SourceMetadata {
  source_file: string;
  document_id: string;
  page: number;
  chunk_id: number;
  section_title: string | null;
}

export interface CorpusChunk {
  id: string;
  text: string;
  metadata: SourceMetadata;
}

export interface ScoredChunk {
  chunk: CorpusChunk;
  score: number;
  denseScore?: number;
  lexicalScore?: number;
  rerankScore?: number;
}

export interface EvidenceAvailability {
  passed: boolean;
  candidate_count: number;
  top_score: number;
  avg_score: number;
  reasons: string[];
}

export interface PipelineStageTrace {
  stage: PipelineStage;
  status: PipelineStatus;
  message: string;
  score?: number;
  details?: Record<string, unknown>;
}

export interface PipelineResult {
  evidence: EvidenceAvailability | null;
  trace: PipelineStageTrace[];
}

export interface RetrievalResult {
  question: string;
  chunks: ScoredChunk[];
  reranked: boolean;
  degradation_reasons?: string[];
  dense_enabled?: boolean;
  lexical_enabled?: boolean;
}

export interface GraphState {
  question: string;
  retrieval?: RetrievalResult;
  requested_items?: string[];
  answer?: string;
  citations?: string[];
  debug?: boolean;
  evidence?: EvidenceAvailability;
  trace?: PipelineStageTrace[];
}

export interface CorpusDocConfig {
  source_file: string;
  document_id: string;
}
