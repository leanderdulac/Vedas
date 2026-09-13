import type { SearchHit } from "./client";

export function citationLabel(hit: SearchHit): string {
  const locator = (hit.locator || "").trim();
  const title = (hit.title || "").trim();
  if (locator && title) return `${locator} — ${title}`;
  return locator || title || "Trecho";
}

export function hitHeading(hit: SearchHit): string {
  return (hit.title || "").trim() || "Trecho";
}

export function displayHitText(hit: SearchHit): string {
  return (hit.text || "").replace(/^\[(?:RV|BG|YS)[^\]]+\]\s*/gm, "").trim();
}

export function documentHref(hit: SearchHit): string | null {
  if (!hit.doc_id) return null;
  const params = new URLSearchParams();
  if (hit.text) params.set("highlight", hit.text.slice(0, 100));
  const query = params.toString();
  const hash = hit.verse_id ? `#${encodeURIComponent(hit.verse_id)}` : "";
  return `/documento/${encodeURIComponent(hit.doc_id)}${query ? `?${query}` : ""}${hash}`;
}

export function copyCitationText(hit: SearchHit): string {
  const locator = (hit.locator || "").trim();
  const work = locator ? locator : hit.title || "Obra Védica";
  const title = hit.title || "Obra Védica";
  const source = hit.source_url || "Corpus Veda Knowledge";
  const license = hit.license || "Domínio Público";
  const quote = (hit.text || "").trim();
  const head = locator && title !== locator ? `${work}. ${title}` : work;
  return `"${quote}"\n— ${head}. Fonte: ${source}. Licença: ${license}.\nRecuperado via Veda Knowledge.`;
}
