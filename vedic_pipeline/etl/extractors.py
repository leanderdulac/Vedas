"""Extração de texto: HTML, PDF, EPUB, TXT."""

from __future__ import annotations

import logging
from pathlib import Path

from vedic_pipeline.common.text import normalize_whitespace

logger = logging.getLogger("vedic_pipeline.etl")


def extract_txt(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def extract_html(path: Path) -> str:
    from bs4 import BeautifulSoup

    html = extract_txt(path)
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def extract_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falha ao extrair página de %s: %s", path, exc)
            t = ""
        if t.strip():
            parts.append(t)
    return "\n\n".join(parts)


def extract_epub(path: Path) -> str:
    import ebooklib
    from bs4 import BeautifulSoup
    from ebooklib import epub

    book = epub.read_epub(str(path))
    parts: list[str] = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        try:
            content = item.get_content()
            soup = BeautifulSoup(content, "lxml")
            for tag in soup(["script", "style"]):
                tag.decompose()
            t = soup.get_text(separator="\n")
            if t.strip():
                parts.append(t)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falha em item EPUB de %s: %s", path, exc)
    return "\n\n".join(parts)


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        text = extract_txt(path)
    elif suffix in {".html", ".htm", ".xhtml"}:
        text = extract_html(path)
    elif suffix == ".pdf":
        text = extract_pdf(path)
    elif suffix == ".epub":
        text = extract_epub(path)
    elif suffix in {".json", ".xml", ".bin"}:
        text = extract_txt(path)
    else:
        logger.warning("Extensão desconhecida %s — tentando como texto", suffix)
        text = extract_txt(path)
    return normalize_whitespace(text)
