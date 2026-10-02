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
from urllib.parse import unquote, urljoin, urlparse, urlunparse

from vedic_pipeline.common.constants import DEFAULT_RAW_DIR, PROJECT_ROOT
from vedic_pipeline.common.corpus import stable_id

logger = logging.getLogger("vedic_pipeline.crawler")

_BLOCKED_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0", "[::1]"})


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    ):
        return True
    if isinstance(ip, ipaddress.IPv6Address):
        mapped = ip.ipv4_mapped
        if mapped is not None:
            return _is_blocked_ip(mapped)
    return False


def resolve_public_targets(url: str) -> tuple[bool, str, list[str]]:
    """Resolve DNS uma vez e devolve só IPs públicos; qualquer IP restrito falha."""
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        return False, f"Esquema de URL não suportado: '{scheme}' (esperado http ou https)", []

    hostname = parsed.hostname
    if not hostname:
        return False, "URL sem hostname válido", []

    lower_host = hostname.lower().strip("[]")
    if lower_host in _BLOCKED_HOSTS:
        return False, f"Acesso a loopback bloqueado por segurança: {hostname}", []

    try:
        literal = ipaddress.ip_address(lower_host)
    except ValueError:
        literal = None
    if literal is not None:
        if _is_blocked_ip(literal):
            return False, f"Endereço IP restrito/privado bloqueado: {lower_host}", []
        return True, "ok", [str(literal)]

    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        return False, f"Falha na resolução de DNS para {hostname}: {exc}", []

    ips: list[str] = []
    seen: set[str] = set()
    for *_, sockaddr in addr_info:
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return False, f"Endereço IP inválido retornado: {ip_str}", []
        if _is_blocked_ip(ip):
            return False, f"Endereço IP restrito/privado bloqueado: {ip_str}", []
        if ip_str not in seen:
            seen.add(ip_str)
            ips.append(ip_str)
    if not ips:
        return False, f"Nenhum endereço público para {hostname}", []
    return True, "ok", ips


def is_safe_url(url: str) -> tuple[bool, str]:
    """Valida se a URL é segura contra SSRF e esquemas não autorizados."""
    ok, reason, _ips = resolve_public_targets(url)
    return ok, reason


def pin_url_to_ip(url: str, ip: str) -> str:
    """Troca o hostname pelo IP já validado; o caller define Host + SNI."""
    parsed = urlparse(url)
    if not parsed.hostname:
        raise ValueError("URL sem hostname válido")
    port = parsed.port
    host_part = f"[{ip}]" if ":" in ip and not ip.startswith("[") else ip
    netloc = f"{host_part}:{port}" if port is not None else host_part
    if parsed.username:
        auth = parsed.username
        if parsed.password:
            auth = f"{auth}:{parsed.password}"
        netloc = f"{auth}@{netloc}"
    return urlunparse(parsed._replace(netloc=netloc))


def prepare_pinned_request(url: str) -> tuple[str, str, dict[str, str], dict[str, Any]]:
    """Resolve + valida e pina a URL no primeiro IP público.

    Returns (pinned_url, original_hostname, extra_headers, extensions).
    """
    ok, reason, ips = resolve_public_targets(url)
    if not ok or not ips:
        raise ValueError(f"URL rejeitada por segurança (SSRF): {reason}")
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    pinned_ip = ips[0]
    try:
        ipaddress.ip_address(hostname.strip("[]"))
        hostname_is_ip = True
    except ValueError:
        hostname_is_ip = False

    pinned_url = url if hostname_is_ip else pin_url_to_ip(url, pinned_ip)
    headers: dict[str, str] = {}
    extensions: dict[str, Any] = {}
    if not hostname_is_ip:
        host_header = f"{hostname}:{parsed.port}" if parsed.port else hostname
        headers["Host"] = host_header
        if (parsed.scheme or "").lower() == "https":
            extensions["sni_hostname"] = hostname
    return pinned_url, hostname, headers, extensions


def max_download_bytes() -> int:
    """Limite de download (bytes). Default 25MB; configurável via VEDIC_MAX_DOWNLOAD_BYTES."""
    raw = os.environ.get("VEDIC_MAX_DOWNLOAD_BYTES", "26214400").strip()
    try:
        return max(1024, int(raw))
    except ValueError:
        return 26214400


def fetch_url_bytes(
    url: str,
    *,
    timeout: float = 60.0,
    limit: int | None = None,
    headers: dict[str, str] | None = None,
    max_redirects: int = 5,
) -> tuple[bytes, str | None, str]:
    """Baixa com TCP pinado no IP validado + stream com teto. Revalida cada redirect.

    Returns (content, content_type, final_url).
    """
    import httpx

    cap = limit if limit is not None else max_download_bytes()
    extra = dict(headers or {})
    extra.setdefault(
        "User-Agent",
        "VedicKnowledgePipeline/2.0 (+research; authorized sources only)",
    )
    current = url
    content: bytes | None = None
    content_type: str | None = None

    with httpx.Client(follow_redirects=False, timeout=timeout, headers=extra) as client:
        for _ in range(max_redirects + 1):
            pinned_url, _host, pin_headers, extensions = prepare_pinned_request(current)
            req_headers = {**extra, **pin_headers}
            with client.stream(
                "GET",
                pinned_url,
                headers=req_headers,
                extensions=extensions or None,
            ) as resp:
                if resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get("location")
                    if not location:
                        raise ValueError(f"Redirect sem Location em {current}")
                    nxt = urljoin(current, location)
                    logger.info("Redirect %s -> %s", current, nxt)
                    current = canonicalize_source_url(nxt)
                    continue
                resp.raise_for_status()
                declared = resp.headers.get("content-length")
                if declared:
                    try:
                        declared_n = int(declared)
                    except ValueError as exc:
                        raise ValueError(f"Content-Length inválido: {declared}") from exc
                    if declared_n > cap:
                        raise ValueError(
                            f"Conteúdo excede limite de {cap} bytes (Content-Length={declared})"
                        )
                content_type = resp.headers.get("content-type")
                chunks: list[bytes] = []
                total = 0
                for piece in resp.iter_bytes(chunk_size=65536):
                    if not piece:
                        continue
                    total += len(piece)
                    if total > cap:
                        raise ValueError(
                            f"Conteúdo excede limite de {cap} bytes durante download de {current}"
                        )
                    chunks.append(piece)
                content = b"".join(chunks)
                break
        else:
            raise ValueError("Muitos redirects (limite 5)")
    if content is None:
        raise ValueError(f"Download falhou para {url}")
    return content, content_type, current


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

    # URLs remotos: validação SSRF + pin de IP + stream com teto
    safe, reason = is_safe_url(url)
    if not safe:
        raise ValueError(f"URL rejeitada por segurança (SSRF): {reason}")

    logger.info("Baixando: %s", url)
    content, content_type, current = fetch_url_bytes(url, timeout=timeout)
    ext = guess_extension(current, content_type)
    dest = raw_dir / f"{title_slug}_{stable_id(url)}{ext}"
    dest.write_bytes(content)
    logger.info("Salvo em %s (%d bytes)", dest, len(content))
    return dest
