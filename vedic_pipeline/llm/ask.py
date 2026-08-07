"""Pipeline ask = multi-query retrieve + hybrid rerank + generate."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from vedic_pipeline.common.constants import DEFAULT_EMBED_DIR, DEFAULT_EMBEDDING_MODEL
from vedic_pipeline.llm.generate import generate_answer
from vedic_pipeline.search.hybrid import expand_query, hybrid_rerank
from vedic_pipeline.search.rag import build_rag_prompt, get_index, retrieve
from vedic_pipeline.storage.db import get_database_url

logger = logging.getLogger("vedic_pipeline.llm.ask")

# defaults mais generosos para inferência sobre corpus amplo
DEFAULT_TOP_K = 10
DEFAULT_FETCH_K = 36  # mais candidatos → diversificação por doc funciona melhor


def retrieve_hits(
    query: str,
    *,
    backend: str = "auto",
    index_dir: Path = DEFAULT_EMBED_DIR,
    top_k: int = DEFAULT_TOP_K,
    tradition: Optional[str] = None,
    language: Optional[str] = None,
    embed_model: str = DEFAULT_EMBEDDING_MODEL,
    hybrid: bool = True,
) -> tuple[list[dict[str, Any]], str]:
    """
    backend: auto | numpy | pgvector
    Usa expansão de consulta + fusão híbrida para melhor recall/precision.
    """
    backend = (backend or "auto").lower()
    fetch_k = max(top_k * 2, DEFAULT_FETCH_K)

    if backend == "auto":
        if get_database_url():
            try:
                from vedic_pipeline.storage.db import check_db

                st = check_db()
                n = (st.get("counts") or {}).get("embeddings") or 0
                if st.get("reachable") and int(n) > 0:
                    backend = "pgvector"
                else:
                    backend = "numpy"
            except Exception:  # noqa: BLE001
                backend = "numpy"
        else:
            backend = "numpy"

    variants = expand_query(query)
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
                all_chunks = idx.get("chunks")
        except Exception:  # noqa: BLE001
            all_chunks = None

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
    tradition: Optional[str] = None,
    language: Optional[str] = None,
    provider: str = "auto",
    model: Optional[str] = None,
    embed_model: str = DEFAULT_EMBEDDING_MODEL,
    include_hits: bool = True,
    include_prompt: bool = False,
    hybrid: bool = True,
    max_tokens: int = 1800,
) -> dict[str, Any]:
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
    prompt = build_rag_prompt(query, hits)
    gen = generate_answer(
        prompt["system"],
        prompt["user"],
        provider=provider,
        model=model,
        hits=hits,
        max_tokens=max_tokens,
    )

    out: dict[str, Any] = {
        "query": query,
        "answer": gen["answer"],
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
    tradition: Optional[str] = None,
    language: Optional[str] = None,
    provider: str = "auto",
    model: Optional[str] = None,
    embed_model: str = DEFAULT_EMBEDDING_MODEL,
    hybrid: bool = True,
    max_tokens: int = 1800,
):
    """
    Gera eventos SSE-friendly (dicts) para /ask/stream:
      meta → token* → done | error
    """
    from vedic_pipeline.llm.generate import stream_answer

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
    prompt = build_rag_prompt(query, hits)
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
        answer = "".join(full_parts).strip()
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
