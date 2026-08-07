"""Download de fontes autorizadas (HTTP ou file://)."""

from __future__ import annotations

import logging
import mimetypes
import re
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from vedic_pipeline.common.constants import DEFAULT_RAW_DIR
from vedic_pipeline.common.corpus import stable_id

logger = logging.getLogger("vedic_pipeline.crawler")


def guess_extension(url: str, content_type: str | None) -> str:
    path = urlparse(url).path.lower()
    for ext in (".pdf", ".epub", ".html", ".htm", ".txt", ".json", ".xml"):
        if path.endswith(ext):
            return ".html" if ext == ".htm" else ext
    if content_type:
        ct = content_type.split(";")[0].strip().lower()
        mapping = {
            "application/pdf": ".pdf",
            "application/epub+zip": ".epub",
            "text/html": ".html",
            "text/plain": ".txt",
            "application/json": ".json",
            "text/xml": ".xml",
            "application/xml": ".xml",
        }
        if ct in mapping:
            return mapping[ct]
        guessed = mimetypes.guess_extension(ct)
        if guessed:
            return guessed
    return ".bin"


def _download_file_uri(url: str, dest: Path) -> Path:
    parsed = urlparse(url)
    src = Path(unquote(parsed.path))
    if not src.exists():
        raise FileNotFoundError(f"Arquivo local não encontrado: {src}")
    shutil.copy2(src, dest)
    logger.info("Copiado file:// %s -> %s", src, dest)
    return dest


def download_source(
    src: dict[str, Any],
    raw_dir: Path = DEFAULT_RAW_DIR,
    timeout: float = 60.0,
) -> Path:
    """Baixa o recurso (http/https ou file://) e grava em data/raw/."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = src["url"]
    title_slug = re.sub(r"[^\w\-]+", "_", (src.get("title") or "doc"))[:60]

    if url.startswith("file://") or (not urlparse(url).scheme and Path(url).exists()):
        local_path = url if not url.startswith("file://") else url
        if not url.startswith("file://"):
            # caminho relativo/absoluto no manifesto
            src_path = Path(url)
            ext = src_path.suffix or ".txt"
            dest = raw_dir / f"{title_slug}_{stable_id(str(src_path))}{ext}"
            shutil.copy2(src_path, dest)
            logger.info("Copiado local %s -> %s", src_path, dest)
            return dest
        ext = Path(unquote(urlparse(url).path)).suffix or ".txt"
        dest = raw_dir / f"{title_slug}_{stable_id(url)}{ext}"
        return _download_file_uri(url, dest)

    import httpx

    headers = {
        "User-Agent": "VedicKnowledgePipeline/1.1 (+research; authorized sources only)"
    }
    logger.info("Baixando: %s", url)
    with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers) as client:
        resp = client.get(url)
        resp.raise_for_status()
        ext = guess_extension(url, resp.headers.get("content-type"))
        dest = raw_dir / f"{title_slug}_{stable_id(url)}{ext}"
        dest.write_bytes(resp.content)
        logger.info("Salvo em %s (%d bytes)", dest, len(resp.content))
        return dest
