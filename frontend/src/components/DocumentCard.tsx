import { Link } from "react-router-dom";
import type { DocumentSummary } from "../api/client";

export default function DocumentCard({ doc }: { doc: DocumentSummary }) {
  return (
    <Link to={`/documento/${doc.id}`} className="card card-link">
      <h3 style={{ marginBottom: 4 }}>{doc.title}</h3>
      <div className="doc-meta">
        <span className="chip chip-saffron">{doc.tradition}</span>
        <span className="chip">{doc.language}</span>
        {doc.license && <span className="chip chip-green">{doc.license}</span>}
      </div>
      <p className="doc-preview">{doc.preview}</p>
      <div className="dim" style={{ marginTop: "0.75rem", fontSize: "0.82rem" }}>
        {doc.char_count.toLocaleString()} caracteres
      </div>
    </Link>
  );
}
