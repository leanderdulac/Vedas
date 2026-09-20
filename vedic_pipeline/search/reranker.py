"""Módulo de Re-ranqueamento Cross-Encoder para busca e RAG védico."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from vedic_pipeline.common.constants import PROJECT_ROOT

logger = logging.getLogger("vedic_pipeline.search.reranker")

_RERANKER_INSTANCE: Any = None
_RERANKER_INITIALIZED: bool = False
_RERANKER_CACHE_KEY: tuple[bool, str] | None = None

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANKER_DIR = PROJECT_ROOT / "artifacts" / "reranker"


def get_reranker_model_name() -> str:
    raw = os.environ.get("VEDIC_RERANKER_MODEL", "").strip()
    return raw or DEFAULT_RERANKER_MODEL


def is_reranker_enabled() -> bool:
    """Opt-in: o CE só carrega com valor explícito true/1/on/yes.

    Default off — o híbrido sozinho já passa o gold set; o MiniLM ms-marco
    genérico regride Nasadiya. Ver docs/reranker_decision.md.
    """
    val = os.environ.get("VEDIC_ENABLE_RERANKER", "false").strip().lower()
    return val in {"1", "true", "on", "yes"}


def looks_like_filesystem_ref(name: str) -> bool:
    """True se o valor parece um caminho local (não um id Hugging Face)."""
    raw = (name or "").strip()
    if not raw:
        return False
    expanded = Path(raw).expanduser()
    if expanded.exists() or (PROJECT_ROOT / raw).exists():
        return True
    return raw.startswith(("./", "../", "/", "~", "artifacts/", "data/"))


def resolve_reranker_model_source(name: str | None = None) -> str:
    """Resolve HF id ou diretório local (cwd ou raiz do projeto) para load.

    `VEDIC_RERANKER_MODEL=artifacts/reranker` vira caminho absoluto se o
    diretório existir, para o CrossEncoder não tratar o path como repo HF.
    """
    raw = (name if name is not None else get_reranker_model_name()).strip()
    raw = raw or DEFAULT_RERANKER_MODEL
    path = Path(raw).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(PROJECT_ROOT / path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return raw


def load_cross_encoder(model_source: str) -> Any:
    """Carrega o CrossEncoder (HF id ou diretório local já resolvido)."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_source)


def _cache_key() -> tuple[bool, str]:
    return (is_reranker_enabled(), resolve_reranker_model_source())


def get_reranker() -> Any | None:
    """Retorna instância singleton do CrossEncoder ou None se desabilitado/falhar.

    Recarrega sozinho se `VEDIC_ENABLE_RERANKER` ou `VEDIC_RERANKER_MODEL`
    mudarem no processo (necessário para o harness de eval A/B).
    """
    global _RERANKER_INSTANCE, _RERANKER_INITIALIZED, _RERANKER_CACHE_KEY
    key = _cache_key()
    if _RERANKER_INITIALIZED and key == _RERANKER_CACHE_KEY:
        return _RERANKER_INSTANCE

    _RERANKER_INSTANCE = None
    _RERANKER_CACHE_KEY = key
    enabled, model_source = key

    if not enabled:
        logger.info("Cross-Encoder Reranker desabilitado por configuração")
        _RERANKER_INITIALIZED = True
        return None

    try:
        logger.info("Carregando Cross-Encoder Reranker: %s", model_source)
        _RERANKER_INSTANCE = load_cross_encoder(model_source)
        logger.info("Cross-Encoder Reranker carregado com sucesso")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Não foi possível carregar CrossEncoder (%s): %s", model_source, exc)
        _RERANKER_INSTANCE = None

    _RERANKER_INITIALIZED = True
    return _RERANKER_INSTANCE


def invalidate_reranker() -> None:
    """Invalida o singleton do reranker para recarga ou testes."""
    global _RERANKER_INSTANCE, _RERANKER_INITIALIZED, _RERANKER_CACHE_KEY
    _RERANKER_INSTANCE = None
    _RERANKER_INITIALIZED = False
    _RERANKER_CACHE_KEY = None


def _restore_env(key: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = previous


@contextmanager
def reranker_runtime(*, enabled: bool, model: str | None = None) -> Iterator[None]:
    """Ajusta env + invalida o singleton; restaura ao sair."""
    previous_enable = os.environ.get("VEDIC_ENABLE_RERANKER")
    previous_model = os.environ.get("VEDIC_RERANKER_MODEL")
    os.environ["VEDIC_ENABLE_RERANKER"] = "true" if enabled else "false"
    if model is not None:
        resolved = resolve_reranker_model_source(model) if model else ""
        if resolved:
            os.environ["VEDIC_RERANKER_MODEL"] = resolved
        else:
            os.environ.pop("VEDIC_RERANKER_MODEL", None)
    invalidate_reranker()
    try:
        yield
    finally:
        _restore_env("VEDIC_ENABLE_RERANKER", previous_enable)
        _restore_env("VEDIC_RERANKER_MODEL", previous_model)
        invalidate_reranker()


def rerank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int = 5,
    rerank_weight: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Aplica Cross-Encoder sobre a lista de chunks candidatos.
    Combina o score do cross-encoder com o score anterior dos chunks.
    """
    if not chunks:
        return []

    model = get_reranker()
    if model is None or not query.strip():
        # Fallback gracioso para a lista original
        return chunks[:top_k]

    pairs = [(query, str(c.get("text") or "")) for c in chunks]

    try:
        import numpy as np

        raw_scores = model.predict(pairs)
        # Normalização sigmóide dos logits
        scores = 1.0 / (1.0 + np.exp(-np.array(raw_scores, dtype=np.float32)))

        for i, c in enumerate(chunks):
            ce_score = round(float(scores[i]), 4)
            c["_cross_score"] = ce_score
            prev_score = float(c.get("score") or 0.0)
            c["score"] = round((1.0 - rerank_weight) * prev_score + rerank_weight * ce_score, 4)

        chunks.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
        return chunks[:top_k]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Falha durante inferência do CrossEncoder: %s", exc)
        return chunks[:top_k]
