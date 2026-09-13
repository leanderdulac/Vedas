#!/usr/bin/env python3
"""Gera o manifest de saṃhitās do Vedic Heritage Portal a partir do sitemap
oficial (WordPress) e grava em fixtures/sources_vedicheritage.json.

Cada entrada é uma página de texto do portal (RV por sūkta, YV por capítulo,
AV por kanda+sūkta), com titulo legivel, tradition "vedic", language "sa" e
license "gov-ind" (conteudo governamental com atribuicao — ver
docs/SOURCES_REVIEW.md). O script e idempotente e reflete o estado atual do
portal; os numeros podem mudar se o portal for reestruturado.

Uso:
  python scripts/build_vedicheritage_manifest.py [--out PATH]
"""

from __future__ import annotations

import argparse
import re
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

UA = "VedaKnowledgePipeline/2.0 (+research; authorized sources only)"
SITEMAP_INDEX = "https://vedicheritage.gov.in/sitemap.xml"

# Páginas que NÃO são texto de saṃhitā (menus, introduções, utilitários).
_DROP_SUBSTR = (
    "-introduction",
    "-audio",
    "-video",
    "-learning-purpose",
    "-learning",
    "kathopanisad",
)
_DROP_EXACT = {
    "https://vedicheritage.gov.in/samhitas",
    "https://vedicheritage.gov.in/samhitas/",
    "https://vedicheritage.gov.in/samhitas/rigveda",
    "https://vedicheritage.gov.in/samhitas/rigveda/",
    "https://vedicheritage.gov.in/samhitas/yajurveda",
    "https://vedicheritage.gov.in/samhitas/yajurveda/",
    "https://vedicheritage.gov.in/samhitas/samaveda-samhitas",
    "https://vedicheritage.gov.in/samhitas/samaveda-samhitas/",
    "https://vedicheritage.gov.in/samhitas/atharvaveda-samhitas",
    "https://vedicheritage.gov.in/samhitas/atharvaveda-samhitas/",
    # landings de recensão (sem texto integral)
    "https://vedicheritage.gov.in/samhitas/rigveda/shakala-samhita",
    "https://vedicheritage.gov.in/samhitas/rigveda/ashvalayana-samhita",
    "https://vedicheritage.gov.in/samhitas/rigveda/shakala-samhita/mandal-01",
    "https://vedicheritage.gov.in/samhitas/rigveda/shakala-samhita-2/mandal-07",
    "https://vedicheritage.gov.in/samhitas/yajurveda/vajasneyi-madhyandina-samhita",
    "https://vedicheritage.gov.in/samhitas/yajurveda/vajasaneyi-kanva-samhita",
    "https://vedicheritage.gov.in/samhitas/yajurveda/taittiriya-samhita",
    "https://vedicheritage.gov.in/samhitas/samaveda-samhitas/kauthuma-samhita",
    "https://vedicheritage.gov.in/samhitas/samaveda-samhitas/jaiminiya-samhita",
    "https://vedicheritage.gov.in/samhitas/samaveda-samhitas/ranayaniya-samhita",
    "https://vedicheritage.gov.in/samhitas/atharvaveda-samhitas/shaunaka-samhita",
    "https://vedicheritage.gov.in/samhitas/atharvaveda-samhitas/paippalada-samhita",
}
# Reprints/páginas numéricas ambíguas do RV que não correspondem a um sūkta limpo.
_DROP_RV_REPRINT = re.compile(r"samhitas/rigveda/shakala-samhita/(\d+-2|mandal-\d+)$")


def _get(url: str, timeout: int = 40) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return ""


def collect_samhita_urls() -> list[str]:
    index = _get(SITEMAP_INDEX)
    subs = re.findall(r"<loc>(.*?)</loc>", index)
    urls: set[str] = set()
    for sub in subs:
        if "wp-sitemap-posts-page-" not in sub:
            continue
        xml = _get(sub)
        for loc in re.findall(r"<loc>(.*?)</loc>", xml):
            urls.add(loc.rstrip("/"))
    return sorted(u for u in urls if "/samhitas/" in u)


# --- identidade / título por Veda ----------------------------------------

_RV_LONG = re.compile(r"rigveda-shakala-samhita-mandal(?:e|a)?-(\d+)-sukta-(\d+)", re.I)
_RV_SHORT = re.compile(r"rigveda-shakala-shakha-mandala-(\d+)-sukta-(\d+)", re.I)
_RV_MNN = re.compile(r"/m(\d{2})-(\d{3})/?$")
_RV_PAIR = re.compile(r"mandal(?:e|a)?-(\d+)-sukta-(\d+)", re.I)
_AV = re.compile(r"shaunaka-samhita/kanda-(\d+)-sukta-(\d+)$", re.I)
_YV_CH = re.compile(r"chapter-(\d+)$", re.I)
_YV_KANVA = re.compile(r"vajasaneyi-kanva-samhita-chapter-(\d+)$", re.I)


def _rv_identity(url: str) -> tuple[int, int] | None:
    for rx in (_RV_LONG, _RV_SHORT, _RV_MNN, _RV_PAIR):
        m = rx.search(url)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def _title(url: str) -> str:
    if _AV.search(url):
        k, s = _AV.search(url).groups()
        return f"Atharvaveda Shaunaka Samhita — Kanda {int(k)}, Sukta {int(s)}"
    m = _YV_KANVA.search(url)
    if m:
        return f"Krishna Yajurveda (Kanva) Samhita — Chapter {int(m.group(1))}"
    m = _YV_CH.search(url)
    if m:
        return f"Yajurveda Samhita — Chapter {int(m.group(1))}"
    mid = _rv_identity(url)
    if mid:
        return f"Rigveda Shakala Samhita — Mandala {mid[0]}, Sukta {mid[1]}"
    # página de texto não reconhecida: título descritivo a partir do slug
    slug = url.rstrip("/").split("/")[-1]
    return " ".join(w.capitalize() for w in re.split(r"[-_]", slug) if w)


def build_sources(urls: list[str]) -> list[dict]:
    rv_seen: dict[tuple[int, int], str] = {}
    sources: list[dict] = []
    for url in urls:
        if any(s in url for s in _DROP_SUBSTR) or url in _DROP_EXACT:
            continue
        if "/samhitas/rigveda/" in url:
            if _DROP_RV_REPRINT.search(url):
                continue
            mid = _rv_identity(url)
            if mid is None:
                continue  # RV menu/ambíguo não tratado
            # dedup: mantém a URL canônica "long" por (mandala,sukta)
            prev = rv_seen.get(mid)
            if prev is not None:
                if _RV_LONG.search(url) and not _RV_LONG.search(prev):
                    rv_seen[mid] = url
                continue
            rv_seen[mid] = url
        elif "/samhitas/atharvaveda" in url and not _AV.search(url) or "/samhitas/yajurveda/" in url and not (_YV_CH.search(url) or _YV_KANVA.search(url)):
            continue
        if "/samhitas/samaveda" in url:
            continue  # portal não expõe texto das saṃhitās do Sāma no sitemap

        # RV canonical (long) entra; outros esquemas já resolvidos acima
        if "/samhitas/rigveda/" in url:
            mid = _rv_identity(url)
            if rv_seen.get(mid) != url:
                continue

        sources.append(
            {
                "url": url,
                "title": _title(url),
                "tradition": "vedic",
                "language": "sa",
                "license": "gov-ind",
                "source_class": "vedicheritage",
            }
        )
    sources.sort(key=lambda s: (s["title"], s["url"]))
    return sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="fixtures/sources_vedicheritage.json",
        help="Caminho de saída do manifesto (default: fixtures/sources_vedicheritage.json)",
    )
    args = parser.parse_args()

    import json

    urls = collect_samhita_urls()
    sources = build_sources(urls)
    payload = {
        "_meta": {
            "generated_at": datetime.now(UTC).isoformat(),
            "portal": "https://vedicheritage.gov.in/sitemap.xml",
            "samhita_urls_collected": len(urls),
            "text_sources": len(sources),
            "note": (
                "Manifest gerado a partir do sitemap oficial do Vedic Heritage Portal. "
                "Texto Devanagari reutilizável com atribuição (licença gov-ind). "
                "Regenere com: python scripts/build_vedicheritage_manifest.py"
            ),
        },
        "sources": sources,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"coletadas {len(urls)} URLs de samhita; gravadas {len(sources)} fontes de texto em {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
