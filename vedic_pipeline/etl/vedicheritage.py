"""Adaptador do Vedic Heritage Portal (vedicheritage.gov.in).

Portal do Ministério da Cultura da Índia com as saṃhitās dos quatro Vedas,
Brāhmaṇas, Āraṇyakas e Upaniṣads em Devanāgarī. Conteúdo governamental
reutilizável com atribuição -> token de licenca ``gov-ind``
(ver ``docs/SOURCES_REVIEW.md``).

Estrutura observada nas páginas (WordPress):
  - o texto canônico dos mantras fica num bloco ``<div id="videotext">``
    dentro de ``#videoid``, com cada linha num ``<br>`` e versos numerados
    em algarismos devanagáricos (ex.: ``॥१७॥``);
  - o mesmo texto reaparece em modais (``.modal-fade``) para exibição de
    vídeo/áudio — eles são excluídos para não duplicar o documento;
  - o texto usa marcas de svara combinantes (``॒``/``॑``), preservadas no
    armazenamento (a busca as ignora via ``fold_for_search``).

O adaptador extrai o Devanāgarī canônico e produz um registro de corpus por
página (linguagem ``sa``, licença ``gov-ind``), reutilizando
``detect_work``/``parse_document_units`` para alinhamento de versos quando a
página tem estrutura reconhecível; caso contrário, o fallback por caracteres
do ``chunk_records`` entra em ação.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from vedic_pipeline.common.corpus import content_fingerprint, stable_id, utc_now_iso
from vedic_pipeline.common.sanskrit import has_devanagari
from vedic_pipeline.etl.structure import detect_work

logger = logging.getLogger("vedic_pipeline.etl.vedicheritage")

VEDICHERITAGE_HOSTS = frozenset(
    {"vedicheritage.gov.in", "www.vedicheritage.gov.in", "vedicheritage.gov.in/"}
)
VEDICHERITAGE_ATTRIBUTION = (
    "Vedic Heritage Portal (Ministry of Culture, Government of India) — "
    "https://vedicheritage.gov.in. Devanagari text reproduced with attribution "
    "under the portal's content-reuse policy; ancient Vedic works (public domain)."
)

# Segmentos de path -> tradição do catálogo.
_TRADITION_BY_PATH: tuple[tuple[str, str], ...] = (
    ("/samhitas/yajurveda", "vedic"),
    ("/samhitas/rigveda", "vedic"),
    ("/samhitas/samaveda", "vedic"),
    ("/samhitas/atharvaveda", "vedic"),
    ("/brahmanas", "vedic"),
    ("/aranyakas", "vedic"),
    ("/upanishads", "upanishad"),
)


def is_vedicheritage_url(url: str) -> bool:
    """True se a URL aponta para o portal vedicheritage.gov.in."""
    from urllib.parse import urlparse

    host = (urlparse(url or "").hostname or "").strip().lower()
    return host in VEDICHERITAGE_HOSTS


def _page_title(soup: Any) -> str | None:
    """Título legível a partir do <title> ou de um heading da página."""
    title_tag = soup.find("title")
    if title_tag and title_tag.get_text(strip=True):
        t = re.sub(r"\s*\|\s*Vedic Heritage Portal\s*$", "", title_tag.get_text(strip=True)).strip()
        if t:
            return t
    for sel in ("h1", ".entry-title", "h2.hdngtxtnew"):
        node = soup.select_one(sel)
        if node and node.get_text(strip=True):
            return node.get_text(strip=True)
    return None


def _tradition_from_path(url: str) -> str:
    from urllib.parse import urlparse

    path = (urlparse(url or "").path or "").lower()
    for seg, trad in _TRADITION_BY_PATH:
        if path.startswith(seg):
            return trad
    return "vedic"


def _devanagari_len(text: str) -> int:
    from vedic_pipeline.common.sanskrit import DEVANAGARI_RANGE

    return len(DEVANAGARI_RANGE.findall(text or ""))


def _scripture_node(soup: Any):
    """Localiza o bloco do texto canônico.

    O texto dos mantras vive em ``#videotext`` (classe ``fnt-shobhika-reg``);
    páginas reais podem conter vários ``#videoid`` (alguns vazios/placeholders).
    Seleciona, entre os blocos ``#videotext`` não-modais com Devanāgarī, o de
    maior conteúdo. Fallback: ``article``, ``#content``, ``main``, ``#primary``.
    """
    candidates = [
        el
        for el in soup.select("#videotext, .fnt-shobhika-reg")
        if has_devanagari(el.get_text())
    ]
    if candidates:
        return max(candidates, key=lambda el: _devanagari_len(el.get_text()))
    for sel in ("article", "#content", "#primary", "main", "#main"):
        node = soup.select_one(sel)
        if node is not None and has_devanagari(node.get_text()):
            return node
    return None


def main_devanagari(html: str) -> str:
    """Extrai o Devanāgarī canônico da página como texto com quebras por linha.

    Remove nav/header/footer/scripts e modais (que duplicam o texto), então
    junta os nós-folha dos segmentos com Devanāgarī do bloco principal.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for node in soup.select(
        "script, style, noscript, nav, header, footer, .modal, figure, ins"
    ):
        node.decompose()

    node = _scripture_node(soup)
    if node is None:
        # Fallback genérico: qualquer nó-folha com Devanāgarī, em ordem.
        node = soup

    lines: list[str] = []
    for el in node.find_all(string=True):
        t = el.strip()
        if t and has_devanagari(t):
            lines.append(t)
    return "\n".join(lines).strip()


def _collapse_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text)


def build_record(
    url: str,
    html: str,
    *,
    title: str | None = None,
    tradition: str | None = None,
) -> dict[str, Any] | None:
    """Monta um registro de corpus a partir do HTML de uma página do portal."""
    text = _collapse_blank_lines(main_devanagari(html))
    if not text or not has_devanagari(text):
        return None
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    parsed_title = title or _page_title(soup) or ""
    if not parsed_title:
        stem = re.split(r"[_\-.]", Path(url).stem)[0]
        parsed_title = stem.title() or "Vedic Heritage text"
    work = detect_work(parsed_title, url)
    fp = content_fingerprint(text)
    return {
        "id": stable_id(url, parsed_title, fp[:16]),
        "text": text,
        "source_url": url,
        "title": parsed_title,
        "work": work,
        "tradition": (tradition or _tradition_from_path(url)).lower(),
        "language": "sa",
        "license": "gov-ind",
        "retrieved_at": utc_now_iso(),
        "fingerprint": fp,
        "char_count": len(text),
        "attribution": VEDICHERITAGE_ATTRIBUTION,
    }


def parse_vedicheritage_file(path: Any, source: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Lê um arquivo HTML já baixado e devolve os registros de corpus (0..1)."""
    src = source or {}
    html = Path(path).read_text(encoding="utf-8", errors="replace")
    rec = build_record(src.get("url") or str(path), html, title=src.get("title"))
    if rec is None:
        return []
    return [rec]


def looks_vedicheritage(html: str) -> bool:
    """Heurística: a página parece ser do portal vedicheritage.gov.in."""
    if "vedicheritage.gov.in" in html[:4000]:
        return True
    return "class=\"fnt-shobhika-reg\"" in html or "class=\"hdngtxtnew\"" in html or (
        "videotext" in html and has_devanagari(html)
    )
