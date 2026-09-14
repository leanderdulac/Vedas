import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { Link, useLocation, useParams, useSearchParams } from "react-router-dom";
import { api, DocumentSummary, VerseUnit } from "../api/client";
import VerseCard from "../components/VerseCard";

export default function DocumentPage() {
  const { id } = useParams();
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const highlightQuery = searchParams.get("highlight")?.trim() || "";
  const hashId = (() => {
    try {
      return decodeURIComponent((location.hash || "").replace(/^#/, ""));
    } catch {
      return "";
    }
  })();

  const [doc, setDoc] = useState<DocumentSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [cachedMedia, setCachedMedia] = useState<{ images: Set<string>; videos: Set<string> }>({
    images: new Set(),
    videos: new Set(),
  });
  const highlightRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!id) return;
    setLoading(true);
    api
      .document(id)
      .then(setDoc)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, [id]);

  useEffect(() => {
    api
      .mediaCached()
      .then((media) =>
        setCachedMedia({
          images: new Set(media.images),
          videos: new Set(media.videos),
        })
      )
      .catch(() => undefined);
  }, []);

  const units = doc?.units || [];
  const targetId = useMemo(() => {
    if (hashId) return hashId;
    if (!highlightQuery || units.length === 0) return "";
    const needle = highlightQuery.slice(0, 60).toLowerCase();
    const match = units.find((unit) => unit.text.toLowerCase().includes(needle));
    return match?.verse_id || "";
  }, [hashId, highlightQuery, units]);

  useEffect(() => {
    if (loading || !doc || !highlightRef.current) return;
    highlightRef.current.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [loading, doc, targetId]);

  if (loading) return <div className="container loading">Carregando documento…</div>;
  if (error) return <div className="container error">{error}</div>;
  if (!doc) return <div className="container empty">Documento não encontrado.</div>;

  const fullText = doc.text || doc.preview || "";

  const renderPlain = () => {
    if (!highlightQuery || !fullText) {
      return fullText;
    }
    const searchSnippet = highlightQuery.slice(0, 60).toLowerCase();
    const idx = fullText.toLowerCase().indexOf(searchSnippet);
    if (idx === -1) {
      return fullText;
    }
    const before = fullText.slice(0, idx);
    const matched = fullText.slice(idx, idx + searchSnippet.length);
    const after = fullText.slice(idx + searchSnippet.length);
    return (
      <>
        {before}
        <mark ref={highlightRef} className="reader-highlight">
          {matched}
        </mark>
        {after}
      </>
    );
  };

  const renderUnits = (items: VerseUnit[]) => {
    let lastHeading = "";
    return items.map((unit) => {
      const showHeading = Boolean(unit.heading && unit.heading !== lastHeading);
      if (unit.heading) lastHeading = unit.heading;
      const isTarget = targetId === unit.verse_id;
      return (
        <Fragment key={unit.verse_id}>
          {showHeading && <h2 className="verse-heading">{unit.heading}</h2>}
          <VerseCard
            unit={unit}
            isTarget={isTarget}
            highlightRef={isTarget ? (el) => { highlightRef.current = el; } : undefined}
            hasImage={cachedMedia.images.has(unit.verse_id)}
            hasVideo={cachedMedia.videos.has(unit.verse_id)}
          />
        </Fragment>
      );
    });
  };

  return (
    <div className="container reader">
      <Link to="/biblioteca" className="btn btn-ghost reader-back">
        ← Biblioteca
      </Link>

      <header className="card reader-header">
        <h1 className="reader-title">{doc.title}</h1>
        <div className="doc-meta">
          <span className="chip chip-saffron">{doc.tradition}</span>
          <span className="chip">{doc.language}</span>
          {doc.work && <span className="chip chip-gold">{doc.work}</span>}
          {doc.license && <span className="chip chip-green">{doc.license}</span>}
        </div>
        <div className="dim reader-byline">
          <span>{doc.char_count.toLocaleString()} caracteres</span>
          {units.length ? <span>{units.length} unidades</span> : null}
          {doc.source_url ? <span className="reader-source">{doc.source_url}</span> : null}
          {doc.retrieved_at ? <span>{new Date(doc.retrieved_at).toLocaleString()}</span> : null}
        </div>
      </header>

      <p className="muted" style={{ margin: "0.4rem 0 0.8rem" }}>
        No verso: ouvir, explicar (PT/EN), ilustrar e conversar. Ilustrações em cache aparecem
        sozinhas. Gerar uma nova imagem ou vídeo exige permissão <strong>image</strong> e{" "}
        <strong>video</strong> na chave em console.x.ai.
      </p>
      <article className="card reader-content">
        {units.length > 0 ? (
          <div className="reader-verses">{renderUnits(units)}</div>
        ) : (
          <div className="reader-body">{renderPlain()}</div>
        )}
      </article>

      <div className="reader-actions">
        <Link
          className="btn btn-primary"
          to={`/perguntar?q=${encodeURIComponent(`Resuma e explique: ${doc.title}`)}`}
        >
          Perguntar sobre este texto
        </Link>
        <Link className="btn btn-secondary" to="/busca">
          Busca semântica
        </Link>
      </div>
    </div>
  );
}
