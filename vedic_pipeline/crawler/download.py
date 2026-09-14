"""Download de fontes autorizadas (HTTP ou file://)."""

from __future__ import annotations

import ipaddress
import logging
import mimetypes
import os
import re
import shutil
import socket
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from vedic_pipeline.common.constants import DEFAULT_RAW_DIR, PROJECT_ROOT
from vedic_pipeline.common.corpus import stable_id

logger = logging.getLogger("vedic_pipeline.crawler")


def is_safe_url(url: str) -> tuple[bool, str]:
    """Valida se a URL é segura contra SSRF e esquemas não autorizados."""
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        return False, f"Esquema de URL não suportado: '{scheme}' (esperado http ou https)"

    hostname = parsed.hostname
    if not hostname:
        return False, "URL sem hostname válido"

    lower_host = hostname.lower()
    if lower_host in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return False, f"Acesso a loopback bloqueado por segurança: {hostname}"

    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        return False, f"Falha na resolução de DNS para {hostname}: {exc}"

    for *_, sockaddr in addr_info:
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False, f"Endereço IP restrito/privado bloqueado: {ip_str}"
        except ValueError:
            return False, f"Endereço IP inválido retornado: {ip_str}"

    return True, "ok"


def max_download_bytes() -> int:
    """Limite de download (bytes). Default 25MB; configurável via VEDIC_MAX_DOWNLOAD_BYTES."""
    raw = os.environ.get("VEDIC_MAX_DOWNLOAD_BYTES", "26214400").strip()
    try:
        return max(1024, int(raw))
    except ValueError:
        return 26214400


def is_safe_local_path(path: str | Path, base_dir: Path | None = None) -> tuple[bool, Path | None, str]:
    """Valida que o arquivo local existe e está confinado dentro do diretório permitido (LFI prevention)."""
    base = (base_dir or PROJECT_ROOT).resolve()
    target = Path(path).resolve()
    if not target.is_relative_to(base):
        return False, None, f"Caminho local fora do diretório permitido ({base}): {target}"
    if not target.exists():
        return False, None, f"Arquivo local não encontrado: {target}"
    return True, target, "ok"


def canonicalize_source_url(url: str) -> str:
    """O site ao vivo do Sacred Texts virou um invólucro JS; o HTML estático está no archive."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host in {"sacred-texts.com", "www.sacred-texts.com"}:
        return parsed._replace(scheme="https", netloc="archive.sacred-texts.com").geturl()
    return url


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


def download_source(
    src: dict[str, Any],
    raw_dir: Path = DEFAULT_RAW_DIR,
    timeout: float = 60.0,
) -> Path:
    """Baixa o recurso (http/https ou file://) e grava em data/raw/ com validações de segurança."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = canonicalize_source_url((src.get("url") or "").strip())
    if not url:
        raise ValueError("URL da fonte vazia")

    title_slug = re.sub(r"[^\w\-]+", "_", (src.get("title") or "doc"))[:60]

    # Arquivos locais (file:// ou caminho relativo no repositório)
    if url.startswith("file://") or (not urlparse(url).scheme and Path(url).exists()):
        raw_path = unquote(urlparse(url).path) if url.startswith("file://") else url
        safe, resolved, reason = is_safe_local_path(raw_path)
        if not safe or resolved is None:
            raise ValueError(f"Fonte local rejeitada por segurança: {reason}")

        ext = resolved.suffix or ".txt"
        dest = raw_dir / f"{title_slug}_{stable_id(str(resolved))}{ext}"
        shutil.copy2(resolved, dest)
        logger.info("Copiado local seguro %s -> %s", resolved, dest)
        return dest

    # URLs remotas: validação SSRF
    safe, reason = is_safe_url(url)
    if not safe:
        raise ValueError(f"URL rejeitada por segurança (SSRF): {reason}")

    import httpx

    headers = {
        "User-Agent": "VedicKnowledgePipeline/2.0 (+research; authorized sources only)"
    }
    logger.info("Baixando: %s", url)
    limit = max_download_bytes()
    current = url
    content: bytes | None = None
    content_type: str | None = None
    # Redirects manuais com revalidação SSRF a cada salto (máx 5).
    with httpx.Client(follow_redirects=False, timeout=timeout, headers=headers) as client:
        for _ in range(6):
            safe, reason = is_safe_url(current)
            if not safe:
                raise ValueError(f"URL rejeitada por segurança (SSRF): {reason}")
            resp = client.get(current)
            if resp.status_code in (301, 302, 303, 307, 308):
                location = resp.headers.get("location")
                if not location:
                    raise ValueError(f"Redirect sem Location em {current}")
                from urllib.parse import urljoin

                nxt = urljoin(current, location)
                logger.info("Redirect %s -> %s", current, nxt)
                current = canonicalize_source_url(nxt)
                continue
            resp.raise_for_status()
            declared = resp.headers.get("content-length")
            if declared:
                try:
                    if int(declared) > limit:
                        raise ValueError(f"Conteúdo excede limite de {limit} bytes (Content-Length={declared})")
                except ValueError as exc:
                    if "excede limite" in str(exc):
                        raise
            content_type = resp.headers.get("content-type")
            # Streaming com teto para evitar OOM/disco.
            chunks: list[bytes] = []
            total = 0
            for piece in resp.iter_bytes(chunk_size=65536):
                if not piece:
                    continue
                total += len(piece)
                if total > limit:
                    raise ValueError(f"Conteúdo excede limite de {limit} bytes durante download de {current}")
                chunks.append(piece)
            content = b"".join(chunks)
            break
        else:
            raise ValueError("Muitos redirects (limite 5)")
    if content is None:
        raise ValueError(f"Download falhou para {url}")
    ext = guess_extension(current, content_type)
    dest = raw_dir / f"{title_slug}_{stable_id(url)}{ext}"
    dest.write_bytes(content)
    logger.info("Salvo em %s (%d bytes)", dest, len(content))
    return dest
