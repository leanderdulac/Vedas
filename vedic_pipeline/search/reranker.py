"""Módulo de Re-ranqueamento Cross-Encoder para busca e RAG védico."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("vedic_pipeline.search.reranker")

_RERANKER_INSTANCE: Any = None
_RERANKER_INITIALIZED: bool = False

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_reranker_model_name() -> str:
    return os.environ.get("VEDIC_RERANKER_MODEL", DEFAULT_RERANKER_MODEL).strip()


def is_reranker_enabled() -> bool:
    """Opt-in: o CE genérico só carrega com valor explícito true/1/on/yes.

    Default off — o híbrido sozinho já passa o gold set; o MiniLM ms-marco
    genérico regride Nasadiya. Ver docs/reranker_decision.md.
    """
    val = os.environ.get("VEDIC_ENABLE_RERANKER", "false").strip().lower()
    return val in {"1", "true", "on", "yes"}


def get_reranker() -> Any | None:
    """Retorna instância singleton do CrossEncoder ou None se desabilitado/falhar."""
    global _RERANKER_INSTANCE, _RERANKER_INITIALIZED
    if _RERANKER_INITIALIZED:
        return _RERANKER_INSTANCE

    if not is_reranker_enabled():
        logger.info("Cross-Encoder Reranker desabilitado por configuração")
        _RERANKER_INITIALIZED = True
        return None

    model_name = get_reranker_model_name()
    try:
        from sentence_transformers import CrossEncoder

        logger.info("Carregando Cross-Encoder Reranker: %s", model_name)
        _RERANKER_INSTANCE = CrossEncoder(model_name)
        logger.info("Cross-Encoder Reranker carregado com sucesso")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Não foi possível carregar CrossEncoder (%s): %s", model_name, exc)
        _RERANKER_INSTANCE = None

    _RERANKER_INITIALIZED = True
    return _RERANKER_INSTANCE


def invalidate_reranker() -> None:
    """Invalida o singleton do reranker para recarga ou testes."""
    global _RERANKER_INSTANCE, _RERANKER_INITIALIZED
    _RERANKER_INSTANCE = None
    _RERANKER_INITIALIZED = False


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
