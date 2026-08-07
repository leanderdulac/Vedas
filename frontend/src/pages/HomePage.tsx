import { FormEvent, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, Stats, Tradition } from "../api/client";
import TraditionCard from "../components/TraditionCard";

export default function HomePage() {
  const nav = useNavigate();
  const [traditions, setTraditions] = useState<Tradition[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [q, setQ] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.traditions(), api.stats()])
      .then(([t, s]) => {
        setTraditions(t.items);
        setStats(s);
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
          <h1 style={{ marginTop: "0.8rem" }}>Conhecimento védico com fontes citáveis</h1>
          <p className="shloka">ॐ पूर्णमदः पूर्णमिदं पूर्णात्पूर्णमुदच्यते ॥</p>
          <p className="muted" style={{ maxWidth: "40rem", marginTop: "0.6rem" }}>
            Explore textos autorizados — Veda, Upaniṣad, Itihāsa, Vaiṣṇava, Jyotiṣa e
            vyākaraṇa — com busca semântica e perguntas fundamentadas no corpus.
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
