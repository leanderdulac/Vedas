import { createSseParser } from './sse';

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

export type VerseUnit = {
  work: string;
  book?: number | null;
  hymn?: number | null;
  verse?: number | null;
  verse_id: string;
  locator: string;
  heading?: string | null;
  text: string;
};

export type DocumentSummary = {
  id: string;
  title: string;
  source_url?: string;
  tradition: string;
  language: string;
  license?: string;
  work?: string | null;
  char_count: number;
  retrieved_at?: string;
  preview: string;
  text?: string;
  units?: VerseUnit[];
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
  work?: string | null;
  verse_id?: string | null;
  locator?: string | null;
  book?: number | null;
  hymn?: number | null;
  verse?: number | null;
  verse_end?: number | null;
  heading?: string | null;
};

export type VerseWitness = {
  role: "sa" | "iast" | "en" | "pt" | string;
  language?: string;
  text: string;
  title?: string;
  doc_id?: string;
  license?: string;
  source_url?: string;
  locator?: string;
};

export type VerseBundle = {
  verse_id: string;
  locator: string;
  work?: string;
  witnesses: VerseWitness[];
  has_sanskrit?: boolean;
};

export type VerseExplanation = {
  verse_id: string;
  locator?: string;
  lang: "pt" | "en";
  provider?: string;
  explanation: string;
  witnesses?: VerseWitness[];
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
  verse: (verseId: string) =>
    request<VerseBundle>(`/api/v1/verses/${encodeURIComponent(verseId)}`),
  explainVerse: (verseId: string, lang: "pt" | "en", provider = "auto") =>
    request<VerseExplanation>(`/api/v1/verses/${encodeURIComponent(verseId)}/explain`, {
      method: "POST",
      body: JSON.stringify({ lang, provider }),
    }),
  verseAudioUrl: (verseId: string) =>
    `${API_BASE}/api/v1/verses/${encodeURIComponent(verseId)}/audio`,
  mediaCached: () =>
    request<{ images: string[]; videos: string[] }>("/api/v1/media/cached"),
  verseImageUrl: (verseId: string) =>
    `${API_BASE}/api/v1/verses/${encodeURIComponent(verseId)}/image`,
  startVerseVideo: (verseId: string) =>
    request<{ verse_id: string; status: string; ready?: boolean }>(
      `/api/v1/verses/${encodeURIComponent(verseId)}/video`,
      { method: "POST", body: JSON.stringify({}) }
    ),
  verseVideoStatus: (verseId: string) =>
    request<{ verse_id: string; status: string; ready?: boolean }>(
      `/api/v1/verses/${encodeURIComponent(verseId)}/video`
    ),
  verseVideoFileUrl: (verseId: string) =>
    `${API_BASE}/api/v1/verses/${encodeURIComponent(verseId)}/video/file`,
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
    history?: { role: "user" | "assistant"; content: string }[];
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
      history?: { role: "user" | "assistant"; content: string }[];
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

    const parse = createSseParser(dispatch);
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        parse(decoder.decode(value, { stream: true }));
      }
      parse(decoder.decode());
    } finally {
      reader.releaseLock();
    }
  },
};
