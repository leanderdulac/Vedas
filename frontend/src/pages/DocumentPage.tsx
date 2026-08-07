import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, DocumentSummary } from "../api/client";

export default function DocumentPage() {
  const { id } = useParams();
  const [doc, setDoc] = useState<DocumentSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!id) return;
    setLoading(true);
    api
      .document(id)
      .then(setDoc)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) return <div className="container loading">Carregando documento…</div>;
  if (error) return <div className="container error">{error}</div>;
  if (!doc) return <div className="container empty">Documento não encontrado.</div>;

  return (
    <div className="container reader">
      <Link to="/biblioteca" className="btn btn-ghost" style={{ marginBottom: "1rem", paddingLeft: 0 }}>
        ← Biblioteca
      </Link>

      <article className="card">
        <h1 style={{ fontSize: "clamp(1.7rem, 3vw, 2.4rem)" }}>{doc.title}</h1>
        <div className="doc-meta">
          <span className="chip chip-saffron">{doc.tradition}</span>
          <span className="chip">{doc.language}</span>
          {doc.license && <span className="chip chip-green">{doc.license}</span>}
        </div>
        <div className="dim" style={{ fontSize: "0.9rem", marginBottom: "1.25rem" }}>
          {doc.char_count.toLocaleString()} caracteres
          {doc.source_url ? ` · ${doc.source_url}` : ""}
          {doc.retrieved_at ? ` · ${new Date(doc.retrieved_at).toLocaleString()}` : ""}
        </div>
        <div className="reader-body">{doc.text || doc.preview}</div>
      </article>

      <div style={{ marginTop: "1rem", display: "flex", gap: "0.6rem", flexWrap: "wrap" }}>
        <Link
          className="btn btn-primary"
          to={`/perguntar?q=${encodeURIComponent(`Resuma e explique: ${doc.title}`)}`}
        >
          Perguntar sobre este texto
        </Link>
        <Link className="btn btn-secondary" to={`/busca`}>
          Busca semântica
        </Link>
      </div>
    </div>
  );
}
