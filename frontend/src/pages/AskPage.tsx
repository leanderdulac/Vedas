import { FormEvent, useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api, AskResponse, SearchHit } from "../api/client";
import HitCard from "../components/HitCard";

type Turn = {
  role: "user" | "assistant";
  content: string;
  meta?: AskResponse;
  hits?: SearchHit[];
  streaming?: boolean;
};

export default function AskPage() {
  const [params] = useSearchParams();
  const initial = params.get("q") || "";

  const [query, setQuery] = useState(initial);
  const [tradition, setTradition] = useState("");
  const [provider, setProvider] = useState("auto");
  const [topK, setTopK] = useState(10);
  const [useStream, setUseStream] = useState(true);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (initial.trim()) {
      void runAsk(initial.trim());
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function runAskStream(q: string) {
    setTurns((prev) => [
      ...prev,
      { role: "user", content: q },
      { role: "assistant", content: "", streaming: true, hits: [] },
    ]);

    let answer = "";
    let hits: SearchHit[] = [];
    let providerUsed = provider;
    let modelUsed: string | null | undefined;
    let retrieval = "";

    await api.askStream(
      {
        query: q,
        top_k: topK,
        tradition: tradition || undefined,
        provider,
        backend: "auto",
      },
      {
        onMeta: (data) => {
          hits = data.hits || [];
          retrieval = data.retrieval_backend || "";
          setTurns((prev) => {
            const copy = [...prev];
            const last = copy[copy.length - 1];
            if (last?.role === "assistant") {
              copy[copy.length - 1] = { ...last, hits };
            }
            return copy;
          });
        },
        onProvider: (data) => {
          providerUsed = data.provider || providerUsed;
          modelUsed = data.model;
        },
        onToken: (text) => {
          answer += text;
          setTurns((prev) => {
            const copy = [...prev];
            const last = copy[copy.length - 1];
            if (last?.role === "assistant") {
              copy[copy.length - 1] = {
                ...last,
                content: answer,
                streaming: true,
                hits,
              };
            }
            return copy;
          });
        },
        onDone: (data) => {
          answer = data.answer || answer;
          hits = data.hits || hits;
          setTurns((prev) => {
            const copy = [...prev];
            const last = copy[copy.length - 1];
            if (last?.role === "assistant") {
              copy[copy.length - 1] = {
                role: "assistant",
                content: answer,
                streaming: false,
                hits,
                meta: {
                  query: q,
                  answer,
                  provider: data.provider || providerUsed,
                  model: data.model ?? modelUsed,
                  retrieval_backend: data.retrieval_backend || retrieval,
                  n_hits: data.n_hits ?? hits.length,
                  hits,
                },
              };
            }
            return copy;
          });
        },
        onError: (detail) => {
          throw new Error(detail);
        },
      }
    );
  }

  async function runAsk(q: string) {
    setLoading(true);
    setError(null);
    try {
      if (useStream) {
        await runAskStream(q);
      } else {
        setTurns((prev) => [...prev, { role: "user", content: q }]);
        const res = await api.ask({
          query: q,
          top_k: topK,
          tradition: tradition || undefined,
          provider,
          backend: "auto",
        });
        setTurns((prev) => [
          ...prev,
          {
            role: "assistant",
            content: res.answer,
            meta: res,
            hits: res.hits,
          },
        ]);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      setTurns((prev) => {
        const copy = [...prev];
        const last = copy[copy.length - 1];
        if (last?.role === "assistant" && last.streaming) {
          copy[copy.length - 1] = {
            role: "assistant",
            content: `Não foi possível responder: ${msg}`,
            streaming: false,
          };
          return copy;
        }
        return [
          ...copy,
          { role: "assistant", content: `Não foi possível responder: ${msg}` },
        ];
      });
    } finally {
      setLoading(false);
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    const q = query.trim();
    if (!q || loading) return;
    setQuery("");
    void runAsk(q);
  }

  const lastHits = [...turns].reverse().find((t) => t.hits?.length)?.hits || [];

  return (
    <div className="container">
      <div className="section-head">
        <div>
          <h1>Perguntar às fontes</h1>
          <p className="muted">
            Q&A com recuperação de trechos e resposta fundamentada. Streaming SSE quando
            disponível; sem chave xAI, usa modo extrativo com citações.
          </p>
        </div>
      </div>

      <div
        className="grid"
        style={{ gridTemplateColumns: "minmax(0, 1.4fr) minmax(280px, 0.9fr)", gap: "1rem" }}
      >
        <div className="stack">
          <div className="card chat-history" style={{ minHeight: 320 }}>
            {turns.length === 0 && (
              <div className="empty">
                Faça uma pergunta sobre o corpus. Ex.: “What is the Self according to the Isha
                Upanishad?”
              </div>
            )}
            {turns.map((t, i) => (
              <div
                key={i}
                className={`bubble ${t.role === "user" ? "bubble-user" : "bubble-assistant"}`}
              >
                <div className="dim" style={{ fontSize: "0.78rem", marginBottom: 6 }}>
                  {t.role === "user" ? "Você" : "Veda Knowledge"}
                  {t.streaming && " · streaming…"}
                  {t.meta && (
                    <>
                      {" "}
                      · {t.meta.provider}
                      {t.meta.model ? `/${t.meta.model}` : ""} · {t.meta.retrieval_backend}
                    </>
                  )}
                </div>
                <div style={{ whiteSpace: "pre-wrap" }}>
                  {t.content || (t.streaming ? "…" : "")}
                </div>
              </div>
            ))}
            {loading && !turns.some((t) => t.streaming) && (
              <div className="loading">Consultando corpus e gerando resposta…</div>
            )}
          </div>

          <form className="card stack" onSubmit={onSubmit}>
            <div className="field">
              <label htmlFor="ask-q">Pergunta</label>
              <textarea
                id="ask-q"
                className="textarea"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Pergunte com base nas fontes autorizadas…"
                rows={3}
              />
            </div>
            <div className="form-row">
              <div className="field">
                <label htmlFor="ask-provider">Provider</label>
                <select
                  id="ask-provider"
                  className="select"
                  value={provider}
                  onChange={(e) => setProvider(e.target.value)}
                >
                  <option value="auto">auto (xAI se disponível)</option>
                  <option value="extractive">extractive (citações)</option>
                  <option value="xai">xai (SpaceXAI)</option>
                  <option value="local">local (gpt2)</option>
                </select>
              </div>
              <div className="field">
                <label htmlFor="ask-trad">Tradição</label>
                <select
                  id="ask-trad"
                  className="select"
                  value={tradition}
                  onChange={(e) => setTradition(e.target.value)}
                >
                  <option value="">Qualquer</option>
                  <option value="vedic">vedic</option>
                  <option value="upanishad">upanishad</option>
                  <option value="itihasa">itihasa</option>
                  <option value="vaishnava">vaishnava</option>
                  <option value="yoga">yoga</option>
                  <option value="purana">purana</option>
                  <option value="grammar">grammar</option>
                </select>
              </div>
              <div className="field">
                <label htmlFor="ask-k">Top-K</label>
                <input
                  id="ask-k"
                  className="input"
                  type="number"
                  min={1}
                  max={15}
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value) || 5)}
                />
              </div>
            </div>
            <label
              style={{ display: "flex", alignItems: "center", gap: 8, fontSize: "0.9rem" }}
            >
              <input
                type="checkbox"
                checked={useStream}
                onChange={(e) => setUseStream(e.target.checked)}
              />
              Streaming (SSE)
            </label>
            <div style={{ display: "flex", gap: "0.6rem", flexWrap: "wrap" }}>
              <button className="btn btn-primary" type="submit" disabled={loading}>
                {loading ? "Pensando…" : "Enviar"}
              </button>
              <button
                className="btn btn-ghost"
                type="button"
                onClick={() => {
                  setTurns([]);
                  setError(null);
                }}
              >
                Limpar conversa
              </button>
            </div>
            {error && <div className="error">{error}</div>}
          </form>
        </div>

        <aside className="stack">
          <div className="card">
            <h3 className="panel-title">Fontes recuperadas</h3>
            <p className="muted" style={{ marginTop: 0, fontSize: "0.92rem" }}>
              Trechos usados na última resposta. Sempre confira a licença e o contexto original.
            </p>
          </div>
          {lastHits.length === 0 && <div className="empty">As citações aparecerão aqui.</div>}
          {lastHits.map((h, i) => (
            <HitCard key={h.chunk_id || i} hit={h} index={i} />
          ))}
        </aside>
      </div>

      <style>{`
        @media (max-width: 960px) {
          .container > .grid {
            grid-template-columns: 1fr !important;
          }
        }
      `}</style>
    </div>
  );
}
