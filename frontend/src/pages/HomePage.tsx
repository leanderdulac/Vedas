import { FormEvent, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, Stats, Tradition } from "../api/client";
import TraditionCard from "../components/TraditionCard";

const ILLUSTRATED = [
  { verse_id: "BG.2.47", locator: "BG 2.47", caption: "Gītā — o direito à ação", doc: "011a40c5947dba3222591227" },
  { verse_id: "BG.1.1", locator: "BG 1.1", caption: "Gītā — o campo do dharma", doc: "c934846207ae4c61a26fa405" },
  { verse_id: "RV.1.1.1", locator: "RV 1.1.1", caption: "Ṛgveda — Agni, o purohita", doc: "e18671807016d8fe318ec4dc" },
  { verse_id: "RV.10.129.1", locator: "RV 10.129.1", caption: "Nāsadīya — antes do ser", doc: "e31ef3d419d690118c5a4d08" },
  { verse_id: "VS.1.1", locator: "VS 1.1", caption: "Yajurveda — o rito da aurora", doc: "772d8a9d4938a33ad8ebe5df" },
  { verse_id: "AV.1.1.1", locator: "AV 1.1.1", caption: "Atharvaveda — o senhor da fala", doc: "5086dfc25601fc1601434bf0" },
];

export default function HomePage() {
  const nav = useNavigate();
  const [traditions, setTraditions] = useState<Tradition[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [q, setQ] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [illustrated, setIllustrated] = useState<typeof ILLUSTRATED>([]);

  useEffect(() => {
    Promise.all([
      api.traditions(),
      api.stats(),
      api.mediaCached().catch(() => ({ images: [] as string[], videos: [] as string[] })),
    ])
      .then(([t, s, media]) => {
        setTraditions(t.items);
        setStats(s);
        const have = new Set(media.images);
        setIllustrated(ILLUSTRATED.filter((item) => have.has(item.verse_id)));
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  function onAsk(e: FormEvent) {
    e.preventDefault();
    if (!q.trim()) return;
    nav(`/perguntar?q=${encodeURIComponent(q.trim())}`);
  }

  return (
    <div className="container">
      <section className="hero">
        <div className="hero-panel">
          <div className="chip">Biblioteca · Busca · RAG</div>
          <h1 style={{ marginTop: "0.8rem" }}>O verso sânscrito, narrado e explicado</h1>
          <p className="shloka">ॐ पूर्णमदः पूर्णमिदं पूर्णात्पूर्णमुदच्यते ॥</p>
          <p className="muted" style={{ maxWidth: "40rem", marginTop: "0.6rem" }}>
            O original em sânscrito é a fonte. Cada mantra pode ser ouvido, ilustrado
            e explicado em português e inglês, sempre com citação da edição licenciada.
          </p>

          <form onSubmit={onAsk} style={{ marginTop: "1.4rem", maxWidth: 640 }}>
            <div className="field">
              <label htmlFor="home-q">Pergunte às fontes</label>
              <div style={{ display: "flex", gap: "0.6rem", flexWrap: "wrap" }}>
                <input
                  id="home-q"
                  className="input"
                  style={{ flex: 1, minWidth: 220 }}
                  placeholder="Ex.: O que é o Ātman na Īśā Upaniṣad?"
                  value={q}
                  onChange={(e) => setQ(e.target.value)}
                />
                <button className="btn btn-primary" type="submit">
                  Perguntar
                </button>
              </div>
            </div>
          </form>

          <div className="hero-actions">
            <Link className="btn btn-secondary" to="/biblioteca">
              Abrir biblioteca
            </Link>
            <Link className="btn btn-ghost" to="/busca">
              Busca semântica
            </Link>
          </div>

          {stats && (
            <div className="stats-row">
              <div className="stat">
                <strong>{stats.documents}</strong>
                <span>documentos</span>
              </div>
              <div className="stat">
                <strong>{(stats.total_chars / 1000).toFixed(1)}k</strong>
                <span>caracteres</span>
              </div>
              <div className="stat">
                <strong>{Object.keys(stats.by_tradition || {}).length}</strong>
                <span>tradições no corpus</span>
              </div>
              <div className="stat">
                <strong>{Object.keys(stats.by_language || {}).length}</strong>
                <span>idiomas</span>
              </div>
            </div>
          )}
        </div>
      </section>

      {error && <div className="error" style={{ marginTop: "1rem" }}>{error}</div>}

      {illustrated.length > 0 && (
        <section className="section">
          <div className="section-head">
            <div>
              <h2>O que o verso narra</h2>
              <p className="muted">
                Imagens e vídeos gerados a partir do sentido do mantra. Abra o verso para ouvir,
                explicar e ver o filme curto.
              </p>
            </div>
          </div>
          <div className="illustrations-grid">
            {illustrated.map((item) => (
              <Link
                key={item.verse_id}
                className="illustration-card"
                to={`/documento/${item.doc}#${encodeURIComponent(item.verse_id)}`}
              >
                <img src={api.verseImageUrl(item.verse_id)} alt={item.caption} />
                <div className="illustration-meta">
                  <span className="illustration-locator">{item.locator}</span>
                  <span>{item.caption}</span>
                </div>
              </Link>
            ))}
          </div>
        </section>
      )}

      <section className="section">
        <div className="section-head">
          <div>
            <h2>Tradições</h2>
            <p className="muted">
              Navegue pelos eixos clássicos do conhecimento. Cada cartão filtra a biblioteca.
            </p>
          </div>
        </div>
        <div className="grid grid-auto">
          {traditions.map((t) => (
            <TraditionCard key={t.id} t={t} />
          ))}
        </div>
      </section>

      <section className="section">
        <div className="grid grid-2">
          <div className="card">
            <h3 className="panel-title">Como funciona</h3>
            <ol className="muted" style={{ margin: 0, paddingLeft: "1.2rem" }}>
              <li>Ingestão apenas de fontes com licença aceita</li>
              <li>Corpus JSONL com metadados (tradição, idioma, URL)</li>
              <li>Embeddings para recuperação semântica</li>
              <li>Respostas RAG com trechos citáveis</li>
            </ol>
          </div>
          <div className="card">
            <h3 className="panel-title">Sugestões de estudo</h3>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
              {[
                "What is the Self in the Isha Upanishad?",
                "Explain equanimity in the Gita sample",
                "वृद्धिरादैच् meaning",
                "पूर्णमदः पूर्णमिदं",
              ].map((s) => (
                <Link
                  key={s}
                  className="btn btn-secondary"
                  style={{ fontSize: "0.88rem", padding: "0.45rem 0.85rem" }}
                  to={`/perguntar?q=${encodeURIComponent(s)}`}
                >
                  {s}
                </Link>
              ))}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
