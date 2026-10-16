// Centralised HTTP client for the FastAPI backend.
// Only NEXT_PUBLIC_API_BASE_URL is read here; no secrets ever reach the browser.

import type {
  ApiErrorBody,
  ChatRequest,
  ChatResponse,
  ConversationDetail,
  ConversationSummary,
  DocumentDetail,
  DocumentList,
  DocumentStatus,
  HealthResponse,
  SearchMode,
  SearchResponse,
  SourceStatus,
  StreamEvent,
} from "@/lib/types";

function resolveBaseUrl(): string {
  // Must be referenced literally so Next.js can inline it at build time.
  const raw = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
  try {
    const url = new URL(raw);
    if (url.protocol !== "http:" && url.protocol !== "https:") throw new Error("bad protocol");
    return url.toString().replace(/\/$/, "");
  } catch {
    return "http://localhost:8000";
  }
}

export const API_BASE_URL = resolveBaseUrl();
const DEFAULT_TIMEOUT_MS = 30_000;
const CHAT_TIMEOUT_MS = 200_000;

export class ApiError extends Error {
  constructor(
    message: string,
    readonly code: string,
    readonly status: number,
    readonly requestId: string | null = null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function friendlyNetworkError(error: unknown): ApiError {
  if (error instanceof DOMException && error.name === "AbortError") {
    return new ApiError("The request was cancelled.", "aborted", 0);
  }
  if (error instanceof DOMException && error.name === "TimeoutError") {
    return new ApiError("The server took too long to respond.", "timeout", 0);
  }
  return new ApiError(
    `Cannot reach the ePolicies API at ${API_BASE_URL}. Check that the backend is running.`,
    "network_error",
    0,
  );
}

async function toApiError(response: Response): Promise<ApiError> {
  let body: Partial<ApiErrorBody> | null = null;
  try {
    body = (await response.json()) as Partial<ApiErrorBody>;
  } catch {
    body = null;
  }
  const err = body?.error;
  return new ApiError(
    err?.message ?? `Request failed with status ${response.status}.`,
    err?.code ?? "http_error",
    response.status,
    err?.request_id ?? response.headers.get("X-Request-ID"),
  );
}

interface RequestOptions {
  method?: "GET" | "POST" | "DELETE";
  body?: unknown;
  signal?: AbortSignal;
  timeoutMs?: number;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const timeout = AbortSignal.timeout(options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
  const signal = options.signal ? AbortSignal.any([options.signal, timeout]) : timeout;
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method: options.method ?? "GET",
      headers: options.body !== undefined ? { "Content-Type": "application/json" } : undefined,
      body: options.body !== undefined ? JSON.stringify(options.body) : undefined,
      signal,
      cache: "no-store",
    });
  } catch (error) {
    throw friendlyNetworkError(error);
  }
  if (!response.ok) throw await toApiError(response);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

function query(params: Record<string, string | number | undefined | null>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") search.set(key, String(value));
  }
  const text = search.toString();
  return text ? `?${text}` : "";
}

export const api = {
  health: (signal?: AbortSignal) => request<HealthResponse>("/health", { signal, timeoutMs: 10_000 }),

  chat: (body: ChatRequest, signal?: AbortSignal) =>
    request<ChatResponse>("/chat", { method: "POST", body, signal, timeoutMs: CHAT_TIMEOUT_MS }),

  listConversations: (limit = 30) => request<ConversationSummary[]>(`/conversations${query({ limit })}`),
  getConversation: (id: string) => request<ConversationDetail>(`/conversations/${encodeURIComponent(id)}`),
  deleteConversation: (id: string) =>
    request<void>(`/conversations/${encodeURIComponent(id)}`, { method: "DELETE" }),

  listDocuments: (params: { page?: number; pageSize?: number; q?: string; status?: DocumentStatus | "" }) =>
    request<DocumentList>(
      `/documents${query({ page: params.page, page_size: params.pageSize, q: params.q, status: params.status })}`,
    ),
  getDocument: (id: string) => request<DocumentDetail>(`/documents/${encodeURIComponent(id)}`),
  sourceStatus: () => request<SourceStatus>("/ingest/source-status", { timeoutMs: 60_000 }),

  search: (q: string, mode: SearchMode = "hybrid", topK = 10) =>
    request<SearchResponse>(`/search${query({ q, mode, top_k: topK })}`, { timeoutMs: 60_000 }),
};

/**
 * POST /chat/stream and dispatch Server-Sent Events. Resolves with the final
 * ChatResponse; rejects with ApiError on HTTP, network or workflow errors.
 * Aborting `signal` closes the connection, which cancels the backend workflow.
 */
export async function chatStream(
  body: ChatRequest,
  onEvent: (event: StreamEvent) => void,
  signal: AbortSignal,
): Promise<ChatResponse> {
  const timeout = AbortSignal.timeout(CHAT_TIMEOUT_MS);
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
      signal: AbortSignal.any([signal, timeout]),
      cache: "no-store",
    });
  } catch (error) {
    throw friendlyNetworkError(error);
  }
  if (!response.ok || !response.body) throw await toApiError(response);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let final: ChatResponse | null = null;

  const handleBlock = (block: string) => {
    const data = block
      .split("\n")
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart())
      .join("\n");
    if (!data) return; // comments / keep-alives
    const event = JSON.parse(data) as StreamEvent;
    onEvent(event);
    if (event.type === "final") final = event.response;
    if (event.type === "error") {
      throw new ApiError(event.error.message, event.error.code, event.status, event.error.request_id);
    }
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        handleBlock(buffer.slice(0, boundary));
        buffer = buffer.slice(boundary + 2);
        boundary = buffer.indexOf("\n\n");
      }
    }
    if (buffer.trim()) handleBlock(buffer);
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw friendlyNetworkError(error);
  } finally {
    reader.releaseLock();
  }
  if (!final) throw new ApiError("The response ended before an answer was received.", "incomplete_stream", 0);
  return final;
}
