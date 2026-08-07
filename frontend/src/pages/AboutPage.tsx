import { useEffect, useState } from "react";
import { api, Health } from "../api/client";

export default function AboutPage() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch((e: Error) => setError(e.message));
  }, []);

  return (
    <div className="container" style={{ maxWidth: 820 }}>
      <h1>Sobre o Veda Knowledge</h1>
      <p className="muted">
        Aplicação full-stack para estudar literatura védica e vaishnava a partir de fontes com
        licença explícita. O sistema combina catálogo textual, embeddings e geração opcional via
        SpaceXAI (xAI).
      </p>

      <section className="section card">
        <h2>Princípios</h2>
        <ul className="muted">
          <li>
            <strong>Licenciamento:</strong> bloqueio de fontes sem licença aceita
          </li>
          <li>
            <strong>Metadados:</strong> tradição, idioma, URL e data de coleta
          </li>
          <li>
            <strong>Citabilidade:</strong> respostas com trechos recuperados
          </li>
          <li>
            <strong>Unicode/Sânscrito:</strong> Devanāgarī e IAST no pipeline
          </li>
        </ul>
      </section>

      <section className="section card">
        <h2>Licenças aceitas</h2>
        <div className="doc-meta">
          {["public-domain", "cc0", "cc-by", "cc-by-sa", "authorized", "official-api"].map((l) => (
            <span key={l} className="chip chip-green">
              {l}
            </span>
          ))}
        </div>
        <p className="muted" style={{ marginBottom: 0 }}>
          Material da BBT/Vedabase somente com autorização ou API oficial — não use cópias não
          autorizadas no treino.
        </p>
      </section>

      <section className="section card">
        <h2>Status do sistema</h2>
        {error && <div className="error">{error}</div>}
        {!health && !error && <div className="loading">Consultando /api/v1/health…</div>}
        {health && (
          <div className="stack">
            <div className="stats-row" style={{ marginTop: 0 }}>
              <div className="stat">
                <strong>{health.version}</strong>
                <span>versão API</span>
              </div>
              <div className="stat">
                <strong>{health.corpus?.documents ?? 0}</strong>
                <span>documentos</span>
              </div>
              <div className="stat">
                <strong>{health.embedding_index_numpy ? "sim" : "não"}</strong>
                <span>índice numpy</span>
              </div>
              <div className="stat">
                <strong>{health.database?.reachable ? "ok" : "—"}</strong>
                <span>PostgreSQL</span>
              </div>
            </div>
            <div className="muted" style={{ fontSize: "0.92rem" }}>
              Providers LLM:{" "}
              {Object.entries(health.llm_providers || {})
                .map(([k, v]) => `${k}${v.available ? "✓" : ""}`)
                .join(" · ")}
            </div>
          </div>
        )}
      </section>

      <section className="section card">
        <h2 className="deva">शान्तिः</h2>
        <p className="shloka" style={{ fontSize: "1.15rem" }}>
          ॐ शान्तिः शान्तिः शान्तिः ॥
        </p>
        <p className="muted" style={{ marginBottom: 0 }}>
          Esta interface é uma ferramenta de estudo. Para decisões rituais, teológicas ou legais,
          consulte ācāryas e fontes primárias autorizadas.
        </p>
      </section>
    </div>
  );
}
