import { FormEvent, useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api, DocumentSummary } from "../api/client";
import DocumentCard from "../components/DocumentCard";

export default function LibraryPage() {
  const [params, setParams] = useSearchParams();
  const [items, setItems] = useState<DocumentSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [facets, setFacets] = useState<{ traditions: string[]; languages: string[] }>({
    traditions: [],
    languages: [],
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const tradition = params.get("tradition") || "";
  const language = params.get("language") || "";
  const q = params.get("q") || "";

  const [draftQ, setDraftQ] = useState(q);

  useEffect(() => {
    setDraftQ(q);
  }, [q]);

  useEffect(() => {
    setLoading(true);
    setError(null);
    api
      .documents({ tradition: tradition || undefined, language: language || undefined, q: q || undefined, limit: 48 })
      .then((res) => {
        setItems(res.items);
        setTotal(res.total);
        setFacets(res.facets);
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, [tradition, language, q]);

  function applyFilters(e?: FormEvent) {
    e?.preventDefault();
    const next = new URLSearchParams();
    if (draftQ.trim()) next.set("q", draftQ.trim());
    if (tradition) next.set("tradition", tradition);
    if (language) next.set("language", language);
    setParams(next);
  }

  function setFilter(key: "tradition" | "language", value: string) {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value);
    else next.delete(key);
    setParams(next);
  }

  return (
    <div className="container">
      <div className="section-head" style={{ marginBottom: "1.25rem" }}>
        <div>
          <h1>Biblioteca</h1>
          <p className="muted">Documentos ingeridos com metadados de tradição, idioma e licença.</p>
        </div>
        <div className="chip">{total} resultados</div>
      </div>

      <form className="filters card" onSubmit={applyFilters}>
        <div className="field">
          <label htmlFor="lib-q">Buscar no catálogo</label>
          <input
            id="lib-q"
            className="input"
            value={draftQ}
            onChange={(e) => setDraftQ(e.target.value)}
            placeholder="título ou trecho…"
          />
        </div>
        <div className="field">
          <label htmlFor="lib-trad">Tradição</label>
          <select
            id="lib-trad"
            className="select"
            value={tradition}
            onChange={(e) => setFilter("tradition", e.target.value)}
          >
            <option value="">Todas</option>
            {facets.traditions.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor="lib-lang">Idioma</label>
          <select
            id="lib-lang"
            className="select"
            value={language}
            onChange={(e) => setFilter("language", e.target.value)}
          >
            <option value="">Todos</option>
            {facets.languages.map((l) => (
              <option key={l} value={l}>
                {l}
              </option>
            ))}
          </select>
        </div>
        <button className="btn btn-primary" type="submit">
          Filtrar
        </button>
      </form>

      {loading && <div className="loading">Carregando biblioteca…</div>}
      {error && <div className="error">{error}</div>}
      {!loading && !error && items.length === 0 && (
        <div className="empty">
          Nenhum documento. Rode:{" "}
          <code>python -m vedic_pipeline ingest --manifest fixtures/sources_local.json</code>
        </div>
      )}

      <div className="grid grid-auto">
        {items.map((d) => (
          <DocumentCard key={d.id} doc={d} />
        ))}
      </div>
    </div>
  );
}
