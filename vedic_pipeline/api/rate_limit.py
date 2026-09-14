"""Limitação de taxa em memória para endpoints públicos de geração/busca.

Bloqueia abuso/DoS leve nas rotas que disparam custo (LLM gerativo, TTS,
Imagine) ou leitura pesada. Janela fixa por (grupo, IP cliente).

Configuração via env:
  VEDIC_RATE_LIMIT_ENABLED      (default "true")
  VEDIC_RATE_LIMIT_PER_MINUTE   (default 180)
Os valores são lidos a cada requisição, então ajuste não exige restart.

Limitação é aplicada apenas às rotas caras; leituras (documents, traditions,
stats, health, static) e metrics não entram.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict

# Prefixos de rota -> grupo de limite. Dois grupos separados mantêm contadores
# independentes (geração via LLM x mídia via TTS/Imagine).
_RULES: tuple[tuple[str, str], ...] = (
    ("/ask", "generation"),
    ("/search", "generation"),
    ("/api/v1/ask", "generation"),
    ("/api/v1/search", "generation"),
    ("/api/v1/verses", "media"),
    ("/metrics", "metrics"),
)

# Limites por grupo (req/min). Geração/mídia são caras — teto menor.
_GROUP_LIMITS: dict[str, int] = {
    "generation": 60,
    "media": 120,
    "metrics": 30,
}

# (grupo, ip) -> (janela_inicio, contagem)
_HITS: dict[tuple[str, str], tuple[float, int]] = defaultdict(lambda: (0.0, 0))
_LOCK = threading.Lock()
_WINDOW_S = 60.0


def rate_limit_enabled() -> bool:
    val = os.environ.get("VEDIC_RATE_LIMIT_ENABLED", "true").strip().lower()
    return val not in {"0", "false", "off", "no"}


def per_minute_limit() -> int:
    raw = os.environ.get("VEDIC_RATE_LIMIT_PER_MINUTE", "180").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 180


def limit_for_group(group: str) -> int:
    """Teto por grupo: min(global, teto do grupo) para conter custo/DoS."""
    base = per_minute_limit()
    cap = _GROUP_LIMITS.get(group)
    if cap is None:
        return base
    return min(base, cap)


def _client_ip(request_or_ip: str | None, forwarded_for: str | None = None) -> str:
    """Resolve IP do cliente respeitando proxy confiável (Caddy/uvicorn --proxy-headers)."""
    if forwarded_for and os.environ.get("VEDIC_TRUST_PROXY_HEADERS", "true").strip().lower() not in {"0", "false", "off", "no"}:
        # X-Forwarded-For: client, proxy1, proxy2 — primeiro é o cliente.
        first = forwarded_for.split(",")[0].strip()
        if first:
            return first[:64]
    return (request_or_ip or "unknown")[:64]


def _group_for_path(path: str) -> str | None:
    for prefix, group in _RULES:
        if path.startswith(prefix):
            return group
    return None


def allow_request(client_ip: str | None, path: str, forwarded_for: str | None = None) -> bool:
    """Retorna True se a requisição pode prosseguir, False se estourou o limite."""
    if not rate_limit_enabled():
        return True
    group = _group_for_path(path)
    if group is None:
        return True

    ip = _client_ip(client_ip, forwarded_for)
    key = (group, ip)
    limit = limit_for_group(group)
    now = time.monotonic()
    with _LOCK:
        # Evita crescimento ilimitado: purga janelas expiradas ocasionalmente.
        if len(_HITS) > 20000:
            expired = [k for k, (s, _) in _HITS.items() if now - s >= _WINDOW_S]
            for k in expired:
                _HITS.pop(k, None)
        start, count = _HITS[key]
        if now - start >= _WINDOW_S:
            _HITS[key] = (now, 1)
            return True
        if count >= limit:
            return False
        _HITS[key] = (start, count + 1)
    return True


def reset_rate_limits() -> None:
    """Limpa o estado de rate limiting (útil em testes e ao trocar config)."""
    with _LOCK:
        _HITS.clear()
