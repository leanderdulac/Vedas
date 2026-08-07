const API_BASE = import.meta.env.VITE_API_BASE ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      /* ignore */
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json() as Promise<T>;
}

export type Tradition = {
  id: string;
  name_sa: string;
  name_en: string;
  name_pt: string;
  description_pt: string;
  description_en: string;
  icon: string;
  color: string;
  order: number;
  document_count?: number;
};

export type DocumentSummary = {
  id: string;
  title: string;
  source_url?: string;
  tradition: string;
  language: string;
  license?: string;
  char_count: number;
  retrieved_at?: string;
  preview: string;
  text?: string;
};

export type SearchHit = {
  chunk_id: string;
  doc_id?: string;
  text: string;
  title?: string;
  tradition?: string;
  language?: string;
  license?: string;
  source_url?: string;
  score?: number;
  chunk_index?: number;
};

export type AskResponse = {
  query: string;
  answer: string;
  provider: string;
  model?: string | null;
  retrieval_backend: string;
  n_hits: number;
  hits?: SearchHit[];
};

export type Stats = {
  documents: number;
  total_chars: number;
  by_tradition: Record<string, number>;
  by_language: Record<string, number>;
  corpus_exists: boolean;
  embeddings?: Record<string, unknown>;
};

export type Health = {
  status: string;
  version: string;
  corpus: Stats;
  embedding_index_numpy: boolean;
  database: { reachable?: boolean; counts?: Record<string, number> };
  llm_providers: Record<string, { available?: boolean; model?: string }>;
};

export const api = {
  health: () => request<Health>("/api/v1/health"),
  stats: () => request<Stats>("/api/v1/stats"),
  traditions: () => request<{ items: Tradition[] }>("/api/v1/traditions"),
  tradition: (id: string) =>
    request<{ tradition: Tradition; documents: { items: DocumentSummary[]; total: number } }>(
      `/api/v1/traditions/${encodeURIComponent(id)}`
    ),
  documents: (params: {
    tradition?: string;
    language?: string;
    q?: string;
    limit?: number;
    offset?: number;
  }) => {
    const sp = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== "") sp.set(k, String(v));
    });
    return request<{
      total: number;
      items: DocumentSummary[];
      facets: { traditions: string[]; languages: string[] };
    }>(`/api/v1/documents?${sp.toString()}`);
  },
  document: (id: string) =>
    request<DocumentSummary>(`/api/v1/documents/${encodeURIComponent(id)}`),
  search: (body: {
    query: string;
    top_k?: number;
    tradition?: string;
    language?: string;
    backend?: string;
  }) =>
    request<{ query: string; retrieval_backend: string; hits: SearchHit[] }>(
      "/api/v1/search",
      { method: "POST", body: JSON.stringify(body) }
    ),
  ask: (body: {
    query: string;
    top_k?: number;
    tradition?: string;
    language?: string;
    backend?: string;
    provider?: string;
  }) =>
    request<AskResponse>("/api/v1/ask", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  /**
   * SSE stream for /api/v1/ask/stream.
   * Calls onMeta (hits), onToken (text deltas), onDone (full payload).
   */
  askStream: async (
    body: {
      query: string;
      top_k?: number;
      tradition?: string;
      language?: string;
      backend?: string;
      provider?: string;
    },
    handlers: {
      onMeta?: (data: { hits?: SearchHit[]; retrieval_backend?: string; n_hits?: number }) => void;
      onToken?: (text: string) => void;
      onProvider?: (data: { provider?: string; model?: string | null }) => void;
      onDone?: (data: AskResponse) => void;
      onError?: (detail: string) => void;
    }
  ): Promise<void> => {
    const res = await fetch(`${API_BASE}/api/v1/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
    });
    if (!res.ok || !res.body) {
      let detail = res.statusText;
      try {
        const j = await res.json();
        detail = j.detail || JSON.stringify(j);
      } catch {
        /* ignore */
      }
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let eventName = "message";

    const dispatch = (name: string, dataRaw: string) => {
      let data: Record<string, unknown> = {};
      try {
        data = JSON.parse(dataRaw) as Record<string, unknown>;
      } catch {
        data = { text: dataRaw };
      }
      if (name === "meta") {
        handlers.onMeta?.(data as { hits?: SearchHit[]; retrieval_backend?: string; n_hits?: number });
      } else if (name === "token") {
        handlers.onToken?.(String((data as { text?: string }).text || ""));
      } else if (name === "provider") {
        handlers.onProvider?.(data as { provider?: string; model?: string | null });
      } else if (name === "done") {
        handlers.onDone?.(data as unknown as AskResponse);
      } else if (name === "error") {
        handlers.onError?.(String((data as { detail?: string }).detail || dataRaw));
      }
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n");
      buffer = parts.pop() || "";
      let dataLines: string[] = [];
      for (const line of parts) {
        if (line.startsWith("event:")) {
          eventName = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          dataLines.push(line.slice(5).trim());
        } else if (line === "") {
          if (dataLines.length) {
            dispatch(eventName, dataLines.join("\n"));
            dataLines = [];
            eventName = "message";
          }
        }
      }
    }
    if (buffer.trim()) {
      // flush trailing
      const lines = buffer.split("\n");
      let dataLines: string[] = [];
      for (const line of lines) {
        if (line.startsWith("event:")) eventName = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      if (dataLines.length) dispatch(eventName, dataLines.join("\n"));
    }
  },
};
