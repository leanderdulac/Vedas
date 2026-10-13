"""Pipeline ask = multi-query retrieve + hybrid rerank + generate."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from vedic_pipeline.common.constants import DEFAULT_EMBED_DIR, DEFAULT_EMBEDDING_MODEL
from vedic_pipeline.common.style import clean_prose
from vedic_pipeline.llm.generate import generate_answer
from vedic_pipeline.search.entity import (
    ENTITY_MAX_TOKENS,
    EntityQuery,
    coverage_note,
    entity_context_chars,
    entity_top_k,
    parse_entity_query,
)
from vedic_pipeline.search.hybrid import expand_query, hybrid_rerank
from vedic_pipeline.search.rag import build_rag_prompt, get_index, retrieve
from vedic_pipeline.storage.db import get_database_url

logger = logging.getLogger("vedic_pipeline.llm.ask")

# defaults mais generosos para inferência sobre corpus amplo
DEFAULT_TOP_K = 10
DEFAULT_FETCH_K = 36  # mais candidatos → diversificação por doc funciona melhor


_EN_HINT = re.compile(r"\b(?:what|who|how|why|which|is|are|does|the|according)\b", re.IGNORECASE)
_PT_HINT = re.compile(r"\b(?:que|qual|quem|como|por|segundo|é|são|o|a|os|as|do|da|no|na)\b", re.IGNORECASE)


def _answer_lang(query: str) -> str:
    """Idioma provável da resposta (o modelo responde no idioma da pergunta)."""
    en = len(_EN_HINT.findall(query or ""))
    pt = len(_PT_HINT.findall(query or ""))
    return "en" if en > pt else "pt"


def _entity_plan(
    query: str, top_k: int, max_tokens: int, hybrid: bool
) -> tuple[EntityQuery | None, int, int]:
    """Consulta de entidade pede mais trechos e uma resposta mais longa."""
    entity = parse_entity_query(query) if hybrid else None
    if entity is None:
        return None, top_k, max_tokens
    return entity, max(top_k, entity_top_k()), max(max_tokens, ENTITY_MAX_TOKENS)


def _entity_note(entity: EntityQuery | None, index_dir: Path) -> str | None:
    if entity is None:
        return None
    try:
        chunks = get_index(Path(index_dir)).get("chunks") or []
        return coverage_note(entity, chunks) if chunks else None
    except Exception:  # noqa: BLE001
        logger.warning("Cobertura da entidade indisponível", exc_info=True)
        return None


def _prompt_for(query: str, hits: list[dict[str, Any]], entity: EntityQuery | None, index_dir: Path):
    if entity is None:
        return build_rag_prompt(query, hits)
    return build_rag_prompt(
        query,
        hits,
        entity_note=_entity_note(entity, index_dir),
        max_context_chars=entity_context_chars(),
    )


def retrieve_hits(
    query: str,
    *,
    backend: str = "auto",
    index_dir: Path = DEFAULT_EMBED_DIR,
    top_k: int = DEFAULT_TOP_K,
    tradition: str | None = None,
    language: str | None = None,
    embed_model: str = DEFAULT_EMBEDDING_MODEL,
    hybrid: bool = True,
) -> tuple[list[dict[str, Any]], str]:
    """
    backend: auto | numpy | pgvector
    Usa expansão de consulta + fusão híbrida para melhor recall/precision.
    """
    backend = (backend or "auto").lower()
    fetch_k = max(top_k * 2, DEFAULT_FETCH_K)

    # O modelo gravado no índice numpy é a fonte da verdade — evita que o
    # pgvector consulte com um modelo diferente do usado para indexar.
    if embed_model == DEFAULT_EMBEDDING_MODEL:
        try:
            meta = (get_index(Path(index_dir)).get("meta") or {}) if Path(index_dir).exists() else {}
            stored = meta.get("model_name")
            if stored and str(stored).strip():
                embed_model = str(stored)
        except Exception:  # noqa: BLE001
            pass

    if backend == "auto":
        if get_database_url():
            try:
                from vedic_pipeline.storage.db import check_db

                st = check_db()
                n = (st.get("counts") or {}).get("embeddings") or 0
                backend = "pgvector" if st.get("reachable") and int(n) > 0 else "numpy"
            except Exception:  # noqa: BLE001
                backend = "numpy"
        else:
            backend = "numpy"

    variants = expand_query(query)
    entity = parse_entity_query(query) if hybrid else None
    if entity and entity.display.casefold() not in {v.casefold() for v in variants}:
        # "Narada Muni" → também "Narada" no denso (o título não identifica).
        variants.insert(1, entity.display)
    fused: list[dict[str, Any]] = []
    seen: set[str] = set()

    for vq in variants[:3]:
        if backend == "pgvector":
            from vedic_pipeline.storage.vectors import search_pgvector

            batch = search_pgvector(
                vq,
                top_k=fetch_k,
                model_name=embed_model,
                tradition=tradition,
                language=language,
            )
        else:
            batch = retrieve(
                vq,
                index_dir=Path(index_dir),
                top_k=fetch_k,
                tradition=tradition,
                language=language,
            )
        for h in batch:
            cid = str(h.get("chunk_id") or "")
            if cid and cid not in seen:
                seen.add(cid)
                fused.append(h)

    all_chunks = None
    if hybrid:
        try:
            if backend == "numpy" or Path(index_dir).exists():
                idx = get_index(Path(index_dir))
                all_chunks = [
                    chunk for chunk in (idx.get("chunks") or [])
                    if (not tradition or (chunk.get("tradition") or "").lower() == tradition.lower())
                    and (not language or (chunk.get("language") or "").lower() == language.lower())
                ]
        except Exception:  # noqa: BLE001
            all_chunks = None

        # all_chunks alimenta léxico + injeção de locator (RV id + obra/coleção).
        hits = hybrid_rerank(
            query,
            fused,
            all_chunks=all_chunks,
            top_k=top_k,
            max_per_doc=2,
        )
        used = f"{backend}+hybrid"
    else:
        from vedic_pipeline.search.hybrid import diversify_by_doc

        hits = diversify_by_doc(fused, top_k=top_k, max_per_doc=2)
        used = backend

    return hits, used


def ask(
    query: str,
    *,
    backend: str = "auto",
    index_dir: Path = DEFAULT_EMBED_DIR,
    top_k: int = DEFAULT_TOP_K,
    tradition: str | None = None,
    language: str | None = None,
    provider: str = "auto",
    model: str | None = None,
    embed_model: str = DEFAULT_EMBEDDING_MODEL,
    include_hits: bool = True,
    include_prompt: bool = False,
    hybrid: bool = True,
    max_tokens: int = 1800,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    entity, top_k, max_tokens = _entity_plan(query, top_k, max_tokens, hybrid)
    hits, used_backend = retrieve_hits(
        query,
        backend=backend,
        index_dir=index_dir,
        top_k=top_k,
        tradition=tradition,
        language=language,
        embed_model=embed_model,
        hybrid=hybrid,
    )
    prompt = _prompt_for(query, hits, entity, index_dir)
    gen = generate_answer(
        prompt["system"],
        prompt["user"],
        provider=provider,
        model=model,
        hits=hits,
        max_tokens=max_tokens,
        history=history,
    )

    out: dict[str, Any] = {
        "query": query,
        "answer": clean_prose(gen["answer"], lang=_answer_lang(query)),
        "provider": gen["provider"],
        "model": gen.get("model"),
        "retrieval_backend": used_backend,
        "n_hits": len(hits),
        "corpus_note": (
            "Respostas usam apenas o corpus licenciado indexado. "
            "Amplie com sources autorizadas para cobertura total da śruti/smṛti."
        ),
    }
    if include_hits:
        out["hits"] = hits
    if include_prompt:
        out["rag_prompt"] = prompt
    return out


def ask_stream_events(
    query: str,
    *,
    backend: str = "auto",
    index_dir: Path = DEFAULT_EMBED_DIR,
    top_k: int = DEFAULT_TOP_K,
    tradition: str | None = None,
    language: str | None = None,
    provider: str = "auto",
    model: str | None = None,
    embed_model: str = DEFAULT_EMBEDDING_MODEL,
    hybrid: bool = True,
    max_tokens: int = 1800,
    history: list[dict[str, str]] | None = None,
):
    """
    Gera eventos SSE-friendly (dicts) para /ask/stream:
      meta → token* → done | error
    """
    from vedic_pipeline.llm.generate import stream_answer

    entity, top_k, max_tokens = _entity_plan(query, top_k, max_tokens, hybrid)
    hits, used_backend = retrieve_hits(
        query,
        backend=backend,
        index_dir=index_dir,
        top_k=top_k,
        tradition=tradition,
        language=language,
        embed_model=embed_model,
        hybrid=hybrid,
    )
    prompt = _prompt_for(query, hits, entity, index_dir)
    yield {
        "event": "meta",
        "data": {
            "query": query,
            "retrieval_backend": used_backend,
            "n_hits": len(hits),
            "hits": hits,
        },
    }
    full_parts: list[str] = []
    provider_used = provider
    model_used = model
    try:
        for chunk in stream_answer(
            prompt["system"],
            prompt["user"],
            provider=provider,
            model=model,
            hits=hits,
            max_tokens=max_tokens,
            history=history,
        ):
            if chunk.get("type") == "token":
                t = chunk.get("text") or ""
                full_parts.append(t)
                yield {"event": "token", "data": {"text": t}}
            elif chunk.get("type") == "meta":
                provider_used = chunk.get("provider") or provider_used
                model_used = chunk.get("model") or model_used
                yield {
                    "event": "provider",
                    "data": {
                        "provider": provider_used,
                        "model": model_used,
                    },
                }
        # o texto transmitido é bruto; o "done" leva a versão limpa, que a UI
        # usa para substituir o que foi exibido durante o streaming
        answer = clean_prose("".join(full_parts), lang=_answer_lang(query))
        yield {
            "event": "done",
            "data": {
                "query": query,
                "answer": answer,
                "provider": provider_used,
                "model": model_used,
                "retrieval_backend": used_backend,
                "n_hits": len(hits),
                "hits": hits,
            },
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("ask_stream falhou")
        yield {"event": "error", "data": {"detail": str(exc)}}
