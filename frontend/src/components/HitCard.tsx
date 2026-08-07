import { Link } from "react-router-dom";
import type { SearchHit } from "../api/client";

export default function HitCard({ hit, index }: { hit: SearchHit; index: number }) {
  return (
    <article className="card hit">
      <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem" }}>
        <div>
          <strong>
            [{index + 1}] {hit.title || "Trecho"}
          </strong>
          <div className="doc-meta" style={{ marginTop: 6 }}>
            {hit.tradition && <span className="chip chip-saffron">{hit.tradition}</span>}
            {hit.language && <span className="chip">{hit.language}</span>}
            {hit.license && <span className="chip chip-green">{hit.license}</span>}
          </div>
        </div>
        {typeof hit.score === "number" && (
          <div className="hit-score">{(hit.score * 100).toFixed(1)}%</div>
        )}
      </div>
      <p className="hit-text">{hit.text}</p>
      <div style={{ marginTop: "0.75rem", display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
        {hit.doc_id && (
          <Link to={`/documento/${hit.doc_id}`} className="btn btn-ghost" style={{ padding: "0.35rem 0.7rem" }}>
            Ver documento
          </Link>
        )}
        {hit.source_url && (
          <span className="dim" style={{ fontSize: "0.82rem", alignSelf: "center" }}>
            {hit.source_url}
          </span>
        )}
      </div>
    </article>
  );
}
