import { Link } from "react-router-dom";
import type { Tradition } from "../api/client";

const SAFE_COLOR = /^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$/;

function safeColor(input: string | undefined, fallback = "#c2410c"): string {
  if (input && SAFE_COLOR.test(input.trim())) return input.trim();
  return fallback;
}

export default function TraditionCard({ t }: { t: Tradition }) {
  const icon = t.icon === "lotus" ? "❀" : t.icon;
  const color = safeColor(t.color);
  return (
    <Link
      to={`/biblioteca?tradition=${encodeURIComponent(t.id)}`}
      className="card card-link"
      style={{ borderTop: `3px solid ${color}` }}
    >
      <div className="tradition-icon" style={{ color }}>
        {icon}
      </div>
      <h3 className="deva" style={{ marginBottom: 2 }}>
        {t.name_sa}
      </h3>
      <div className="muted" style={{ marginBottom: "0.6rem" }}>
        {t.name_pt} · {t.name_en}
      </div>
      <p className="doc-preview" style={{ margin: 0 }}>
        {t.description_pt}
      </p>
      <div style={{ marginTop: "0.9rem" }}>
        <span className="chip">{t.document_count ?? 0} docs</span>
      </div>
    </Link>
  );
}
