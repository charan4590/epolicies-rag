// Mirrors backend/app/schemas/api_models.py. Keep both in sync.

export type CitationVerification = "supported" | "weak" | "unverified";
export type VerificationStatus = "verified" | "revised" | "partially_verified" | "not_applicable";
export type DocumentStatus = "pending" | "indexed" | "failed" | "deleted";
export type SearchMode = "hybrid" | "vector" | "keyword";

export interface ApiErrorBody {
  error: { code: string; message: string; request_id: string | null };
}

export interface HistoryMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  question: string;
  conversation_id?: string | null;
  history?: HistoryMessage[];
  include_diagnostics?: boolean;
  persist?: boolean;
}

export interface Citation {
  label: string;
  chunk_id: string;
  document_id: string;
  document_title: string;
  relative_path: string;
  heading_path: string[];
  section_paths: string[][];
  passage: string;
  source_urls: string[];
  document_source_url: string | null;
  verification: CitationVerification;
}

export interface ConflictInfo {
  description: string;
  labels: string[];
}

export interface ClaimInfo {
  statement: string;
  labels: string[];
  support_type: string;
  status: string;
  missing_qualifications: string[];
}

export interface VerificationInfo {
  status: VerificationStatus;
  revised: boolean;
  llm_checked: boolean;
  removed_citations: string[];
  claims: ClaimInfo[];
  notes: string[];
}

export interface RankedItem {
  rank: number;
  chunk_id: string;
  title: string;
  heading_path: string[];
  similarity?: number;
  score?: number;
  hits?: number;
}

export interface ChunkBrief {
  chunk_id: string;
  document_id: string;
  title: string;
  relative_path: string;
  heading_path: string[];
  vector_rank: number | null;
  vector_similarity: number | null;
  keyword_rank: number | null;
  keyword_score: number | null;
  exact_rank: number | null;
  rrf_score: number;
}

export interface SearchDiagnostics {
  lists: Record<string, RankedItem[]>;
  fused: ChunkBrief[];
  query_embeddings: number;
  query_embedding_cache_hits: number;
  duplicates_collapsed: number;
  timings_ms: Record<string, number>;
}

export interface RetrievalRoundInfo {
  round: number;
  queries: string[];
  retrieved: number;
  new_chunks: number;
  search: SearchDiagnostics;
  referenced_policies?: string[];
  assessment: {
    sufficient: boolean;
    confidence: string;
    missing_information: string[];
    follow_up_queries: string[];
    method: string;
  } | null;
}

export interface ChatDiagnostics {
  plan: {
    standalone_question: string;
    planner: string;
    identifiers: string[];
    exact_phrases: string[];
    search_queries: string[];
    retrieval_emphasis: string;
    needs_multiple_documents: boolean;
    ambiguity: string;
    notes: string[];
  };
  rounds: RetrievalRoundInfo[];
  retrieval_rounds: number;
  stop_reason: string;
  retrieved_chunks: number;
  context_chunks: ChunkBrief[];
  relevant_chunk_ids?: string[];
  models: { generation: string; embedding: string; embedding_dimensions: number };
  llm_usage: { calls: number; prompt_tokens: number; output_tokens: number; total_tokens: number; by_stage: Record<string, number> };
  timings_ms: Record<string, number>;
}

export interface ChatResponse {
  answer: string;
  citations: Citation[];
  insufficient_evidence: boolean;
  missing_information: string[];
  conflicts: ConflictInfo[];
  verification: VerificationInfo;
  standalone_question: string;
  conversation_id: string | null;
  message_id: string | null;
  diagnostics: ChatDiagnostics | null;
}

export type StreamEvent =
  | { type: "stage"; stage: string; message: string; round?: number }
  | { type: "final"; response: ChatResponse }
  | { type: "error"; status: number; error: ApiErrorBody["error"] };

export interface ConversationSummary {
  conversation_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface ConversationMessage {
  message_id: string;
  role: "user" | "assistant";
  content: string;
  payload: Partial<ChatResponse>;
  created_at: string;
}

export interface ConversationDetail {
  conversation_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: ConversationMessage[];
}

export interface DocumentSummary {
  document_id: string;
  relative_path: string;
  filename: string;
  title: string;
  status: DocumentStatus;
  chunk_count: number;
  source_url: string | null;
  size_bytes: number;
  is_empty: boolean | null;
  last_error: string | null;
  last_indexed_at: string | null;
  file_modified_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface DocumentList {
  items: DocumentSummary[];
  total: number;
  page: number;
  page_size: number;
}

export interface ChunkSummary {
  chunk_id: string;
  chunk_index: number;
  heading_path: string[];
  token_count: number;
  char_count: number;
  source_urls: string[];
  preview: string;
}

export interface DocumentDetail extends DocumentSummary {
  encoding: string | null;
  content_hash: string | null;
  source_hash: string | null;
  embedding_config: string | null;
  failure_count: number;
  metadata: {
    headings?: { level: number; text: string }[];
    urls?: string[];
    warnings?: string[];
    empty?: boolean;
    [key: string]: unknown;
  };
  deleted_at: string | null;
  chunks: ChunkSummary[];
}

export interface SourceStatus {
  source_dir_configured: boolean;
  source_dir_exists: boolean;
  discovered: number;
  new: string[];
  modified: string[];
  unchanged: number;
  deleted: string[];
  unreadable: { path: string; error: string }[];
  error: string | null;
}

export interface HealthResponse {
  status: "ok" | "degraded" | "error";
  version: string;
  database: {
    ok: boolean;
    error?: string;
    latency_ms?: number;
    pgvector_enabled?: boolean;
    pgvector_version?: string | null;
    schema_initialized?: boolean;
    vector_dimensions?: number | null;
    dimensions_match?: boolean;
  };
  index: {
    indexed_documents: number;
    failed_documents: number;
    pending_documents: number;
    deleted_documents: number;
    chunks: number;
    cached_embeddings: number;
    embedding_configs: string[];
    last_indexed_at: string | null;
  } | null;
  gemini: { configured: boolean; backend?: "api_key" | "vertex"; generation_model: string | null; embedding_model: string; embedding_dimensions: number };
  config: Record<string, unknown>;
}

export interface SearchHit {
  rank: number;
  chunk_id: string;
  document_id: string;
  document_title: string;
  relative_path: string;
  heading_path: string[];
  text: string;
  source_urls: string[];
  rrf_score: number;
  vector_rank: number | null;
  vector_similarity: number | null;
  keyword_rank: number | null;
  keyword_score: number | null;
  exact_rank: number | null;
}

export interface SearchResponse {
  query: string;
  mode: SearchMode;
  identifiers: string[];
  hits: SearchHit[];
  diagnostics: SearchDiagnostics;
}
