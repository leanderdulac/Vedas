import { FormEvent, useEffect, useState } from "react";
import { api, SearchHit } from "../api/client";
import HitCard from "../components/HitCard";

const DEFAULT_TRADITIONS = [
  { id: "vedic", name: "Védico" },
  { id: "upanishad", name: "Upaniṣads" },
  { id: "itihasa", name: "Itihāsa" },
  { id: "vaishnava", name: "Vaishnava" },
  { id: "yoga", name: "Yoga" },
  { id: "grammar", name: "Vyākaraṇa / Gramática" },
];

export default function SearchPage() {
  const [query, setQuery] = useState("Self and oneness");
  const [tradition, setTradition] = useState("");
  const [availableTraditions, setAvailableTraditions] = useState(DEFAULT_TRADITIONS);
  const [language, setLanguage] = useState("");
  const [topK, setTopK] = useState(5);
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [backend, setBackend] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.traditions()
      .then((res) => {
        if (res?.items && res.items.length > 0) {
          setAvailableTraditions(
            res.items.map((t) => ({ id: t.id, name: t.name_pt || t.name_sa || t.name_en || t.id }))
          );
        }
      })
      .catch(() => {
        // mantém fallback local se offline
      });
  }, []);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.search({
        query: query.trim(),
        top_k: topK,
        tradition: tradition || undefined,
        language: language || undefined,
        backend: "auto",
      });
      setHits(res.hits || []);
      setBackend(res.retrieval_backend);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setHits([]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="container">
      <div className="section-head">
        <div>
          <h1>Busca semântica</h1>
          <p className="muted">
            Recuperação por embeddings — encontra trechos por significado, não só por palavra exata.
          </p>
        </div>
      </div>

      <form className="card stack" onSubmit={onSubmit}>
        <div className="field">
          <label htmlFor="search-q">Consulta</label>
          <input
            id="search-q"
            className="input"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ex.: immortality through knowledge"
          />
        </div>
        <div className="form-row">
          <div className="field">
            <label htmlFor="s-trad">Tradição</label>
            <select id="s-trad" className="select" value={tradition} onChange={(e) => setTradition(e.target.value)}>
              <option value="">Qualquer</option>
              {availableTraditions.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.name}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="s-lang">Idioma</label>
            <select id="s-lang" className="select" value={language} onChange={(e) => setLanguage(e.target.value)}>
              <option value="">Qualquer</option>
              <option value="sa">sa</option>
              <option value="en">en</option>
            </select>
          </div>
          <div className="field">
            <label htmlFor="s-k">Top-K</label>
            <input
              id="s-k"
              className="input"
              type="number"
              min={1}
              max={20}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value) || 5)}
            />
          </div>
        </div>
        <div>
          <button className="btn btn-primary" type="submit" disabled={loading}>
            {loading ? "Buscando…" : "Buscar"}
          </button>
        </div>
      </form>

      {error && <div className="error" style={{ marginTop: "1rem" }}>{error}</div>}

      {backend && (
        <div className="dim" style={{ margin: "1rem 0 0.5rem" }}>
          Backend de recuperação: <strong>{backend}</strong> · {hits.length} trechos
        </div>
      )}

      <div className="stack" style={{ marginTop: "1rem" }}>
        {hits.map((h, i) => (
          <HitCard key={h.chunk_id || i} hit={h} index={i} />
        ))}
      </div>
    </div>
  );
}
