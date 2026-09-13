import { useState } from "react";
import { Link } from "react-router-dom";
import type { SearchHit } from "../api/client";
import { copyCitationText, displayHitText, documentHref, hitHeading } from "../api/citation";

export default function HitCard({ hit, index }: { hit: SearchHit; index: number }) {
  const [copied, setCopied] = useState(false);
  const href = documentHref(hit);
  const title = hitHeading(hit);

  const handleCopyCitation = () => {
    navigator.clipboard.writeText(copyCitationText(hit)).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <article className="card hit">
      <div className="hit-head">
        <div className="hit-head-main">
          <strong className="hit-title">
            <span className="hit-index">[{index + 1}]</span> {title}
          </strong>
          <div className="doc-meta">
            {hit.locator && <span className="chip chip-gold">{hit.locator}</span>}
            {hit.tradition && <span className="chip chip-saffron">{hit.tradition}</span>}
            {hit.language && <span className="chip">{hit.language}</span>}
            {hit.license && <span className="chip chip-green">{hit.license}</span>}
          </div>
        </div>
        {typeof hit.score === "number" && (
          <div className="hit-score" title="Pontuação de ordenação; não representa probabilidade de acerto">
            Score {hit.score.toFixed(3)}
          </div>
        )}
      </div>
      <p className="hit-text">{displayHitText(hit)}</p>
      <div className="hit-actions">
        {href && (
          <Link to={href} className="btn btn-ghost hit-action">
            Ver {hit.locator ? hit.locator : "documento"}
          </Link>
        )}
        <button
          type="button"
          onClick={handleCopyCitation}
          className="btn btn-secondary hit-action"
          title="Copiar citação acadêmica para a área de transferência"
        >
          {copied ? "Copiado! ✓" : "Copiar citação"}
        </button>
        {hit.source_url && <span className="dim hit-source">{hit.source_url}</span>}
      </div>
    </article>
  );
}
