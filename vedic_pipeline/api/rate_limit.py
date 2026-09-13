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
)

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


def _group_for_path(path: str) -> str | None:
    for prefix, group in _RULES:
        if path.startswith(prefix):
            return group
    return None


def allow_request(client_ip: str | None, path: str) -> bool:
    """Retorna True se a requisição pode prosseguir, False se estourou o limite."""
    if not rate_limit_enabled():
        return True
    group = _group_for_path(path)
    if group is None:
        return True

    key = (group, client_ip or "unknown")
    limit = per_minute_limit()
    now = time.monotonic()
    with _LOCK:
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
