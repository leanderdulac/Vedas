"""API REST da aplicação Veda Knowledge — pipeline + endpoints da UI."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from vedic_pipeline.common.constants import (
    ALLOWED_LICENSES,
    DEFAULT_BASE_MODEL,
    DEFAULT_CORPUS,
    DEFAULT_EMBED_DIR,
)
from vedic_pipeline.common.corpus import utc_now_iso

logger = logging.getLogger("vedic_pipeline.api")

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


def create_app():
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    from vedic_pipeline.api.schemas import (
        AskRequest,
        BuildIndexRequest,
        IngestRequest,
        SearchRequest,
        TokenizeRequest,
        TrainRequest,
    )

    app = FastAPI(
        title="Veda Knowledge",
        description=(
            "Aplicação de conhecimento védico: biblioteca licenciada, "
            "busca semântica e Q&A RAG (SpaceXAI/xAI)."
        ),
        version="2.0.0",
    )

    origins = os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000",
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in origins if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------ health
    @app.get("/health")
    @app.get("/api/v1/health")
    def health() -> dict[str, Any]:
        from vedic_pipeline.api.catalog_service import corpus_stats
        from vedic_pipeline.llm.generate import list_providers
        from vedic_pipeline.storage.db import check_db, get_database_url

        stats = corpus_stats()
        embed_ok = (DEFAULT_EMBED_DIR / "embeddings.npy").exists()
        return {
            "status": "ok",
            "service": "veda-knowledge",
            "version": "2.0.0",
            "time": utc_now_iso(),
            "corpus": stats,
            "embedding_index_numpy": embed_ok,
            "database_url_set": bool(get_database_url()),
            "database": check_db(),
            "llm_providers": list_providers(),
            "allowed_licenses": sorted(ALLOWED_LICENSES),
            "default_base_model": DEFAULT_BASE_MODEL,
        }

    # ------------------------------------------------------------------ app v1
    @app.get("/api/v1/stats")
    def api_stats() -> dict[str, Any]:
        from vedic_pipeline.api.catalog_service import corpus_stats

        stats = corpus_stats()
        embed_meta = {}
        meta_path = DEFAULT_EMBED_DIR / "index_meta.json"
        if meta_path.exists():
            import json

            embed_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {
            **stats,
            "embeddings": embed_meta,
            "time": utc_now_iso(),
        }

    @app.get("/api/v1/traditions")
    def api_traditions() -> dict[str, Any]:
        from vedic_pipeline.api.catalog_service import corpus_stats
        from vedic_pipeline.api.traditions import list_traditions

        stats = corpus_stats()
        counts = stats.get("by_tradition") or {}
        items = []
        for t in list_traditions():
            item = dict(t)
            item["document_count"] = int(counts.get(t["id"], 0))
            items.append(item)
        return {"items": items}

    @app.get("/api/v1/traditions/{tradition_id}")
    def api_tradition_detail(tradition_id: str) -> dict[str, Any]:
        from vedic_pipeline.api.catalog_service import list_documents
        from vedic_pipeline.api.traditions import get_tradition

        t = get_tradition(tradition_id)
        if not t:
            raise HTTPException(status_code=404, detail="Tradição não encontrada")
        docs = list_documents(tradition=tradition_id, limit=100)
        return {"tradition": t, "documents": docs}

    @app.get("/api/v1/documents")
    def api_documents(
        tradition: Optional[str] = None,
        language: Optional[str] = None,
        q: Optional[str] = None,
        limit: int = Query(default=24, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        from vedic_pipeline.api.catalog_service import list_documents

        return list_documents(
            tradition=tradition,
            language=language,
            q=q,
            limit=limit,
            offset=offset,
        )

    @app.get("/api/v1/documents/{doc_id}")
    def api_document(doc_id: str) -> dict[str, Any]:
        from vedic_pipeline.api.catalog_service import get_document

        doc = get_document(doc_id, include_text=True)
        if not doc:
            raise HTTPException(status_code=404, detail="Documento não encontrado")
        return doc

    @app.post("/api/v1/search")
    @app.post("/search")
    def api_search(body: SearchRequest) -> dict[str, Any]:
        from vedic_pipeline.llm.ask import retrieve_hits
        from vedic_pipeline.search.rag import build_rag_prompt

        try:
            hits, used = retrieve_hits(
                body.query,
                backend=body.backend,
                index_dir=Path(body.index_dir),
                top_k=body.top_k,
                tradition=body.tradition,
                language=body.language,
            )
            payload: dict[str, Any] = {
                "query": body.query,
                "top_k": body.top_k,
                "retrieval_backend": used,
                "hits": hits,
            }
            if body.include_prompt:
                payload["rag_prompt"] = build_rag_prompt(body.query, hits)
            return payload
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Erro em /search")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/v1/ask")
    @app.post("/ask")
    def api_ask(body: AskRequest) -> dict[str, Any]:
        from vedic_pipeline.llm.ask import ask

        try:
            return ask(
                body.query,
                backend=body.backend,
                index_dir=Path(body.index_dir),
                top_k=body.top_k,
                tradition=body.tradition,
                language=body.language,
                provider=body.provider,
                model=body.model,
                include_hits=body.include_hits,
                include_prompt=body.include_prompt,
                hybrid=body.hybrid,
            )
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"{exc}. Construa o índice com: "
                    "python -m vedic_pipeline build-index"
                ),
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Erro em /ask")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/v1/ask/stream")
    @app.post("/ask/stream")
    def api_ask_stream(body: AskRequest):
        """Server-Sent Events: meta → token* → done | error."""
        import json as _json

        from fastapi.responses import StreamingResponse

        from vedic_pipeline.llm.ask import ask_stream_events

        def event_gen():
            try:
                for ev in ask_stream_events(
                    body.query,
                    backend=body.backend,
                    index_dir=Path(body.index_dir),
                    top_k=body.top_k,
                    tradition=body.tradition,
                    language=body.language,
                    provider=body.provider,
                    model=body.model,
                    hybrid=body.hybrid,
                ):
                    name = ev.get("event") or "message"
                    payload = _json.dumps(ev.get("data") or {}, ensure_ascii=False)
                    yield f"event: {name}\ndata: {payload}\n\n"
            except FileNotFoundError as exc:
                err = _json.dumps({"detail": str(exc)}, ensure_ascii=False)
                yield f"event: error\ndata: {err}\n\n"
            except Exception as exc:  # noqa: BLE001
                logger.exception("Erro em /ask/stream")
                err = _json.dumps({"detail": str(exc)}, ensure_ascii=False)
                yield f"event: error\ndata: {err}\n\n"

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ----------------------------------------------------------- pipeline ops
    @app.post("/ingest")
    def api_ingest(body: IngestRequest) -> dict[str, Any]:
        from vedic_pipeline.crawler.ingest import ingest_manifest

        try:
            stats = ingest_manifest(
                body.manifest,
                corpus_path=Path(body.corpus),
                min_chars=body.min_chars,
            )
            if body.sync_db:
                from vedic_pipeline.storage.catalog import sync_corpus_to_db

                stats["db_sync"] = sync_corpus_to_db(Path(body.corpus))
            return stats
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Erro em /ingest")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/tokenize")
    def api_tokenize(body: TokenizeRequest) -> dict[str, Any]:
        from vedic_pipeline.train.tokenizer import (
            evaluate_tokenizer_compression,
            train_bpe_tokenizer,
        )

        try:
            out = train_bpe_tokenizer(
                corpus_path=Path(body.corpus),
                out_dir=Path(body.out_dir),
                vocab_size=body.vocab_size,
                min_frequency=body.min_frequency,
            )
            metrics = evaluate_tokenizer_compression(out, Path(body.corpus))
            return {
                "tokenizer_dir": str(out),
                "vocab_size": body.vocab_size,
                "compression": metrics,
            }
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Erro em /tokenize")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/train")
    def api_train(body: TrainRequest) -> dict[str, Any]:
        from vedic_pipeline.train.model import train_causal_model

        try:
            out = train_causal_model(
                corpus_path=Path(body.corpus),
                out_dir=Path(body.out_dir),
                base_model=body.base_model,
                tokenizer_dir=Path(body.tokenizer_dir),
                epochs=body.epochs,
                block_size=body.block_size,
                batch_size=body.batch_size,
                learning_rate=body.learning_rate,
                max_steps=body.max_steps,
                fp16=body.fp16,
            )
            return {"model_dir": str(out), "base_model": body.base_model}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Erro em /train")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/build-index")
    def api_build_index(body: BuildIndexRequest) -> dict[str, Any]:
        from vedic_pipeline.search.embeddings import build_embedding_index
        from vedic_pipeline.search.rag import get_index

        backend = body.backend.lower()
        result: dict[str, Any] = {"backend": backend}
        try:
            if backend in {"numpy", "both"}:
                meta = build_embedding_index(
                    corpus_path=Path(body.corpus),
                    out_dir=Path(body.out_dir),
                    model_name=body.model_name,
                    chunk_size=body.chunk_size,
                    overlap=body.overlap,
                )
                get_index(Path(body.out_dir), reload=True)
                result["numpy"] = meta
            if backend in {"pgvector", "both"}:
                from vedic_pipeline.storage.vectors import build_pgvector_index

                result["pgvector"] = build_pgvector_index(
                    corpus_path=Path(body.corpus),
                    model_name=body.model_name,
                    chunk_size=body.chunk_size,
                    overlap=body.overlap,
                )
            if backend not in {"numpy", "pgvector", "both"}:
                raise HTTPException(
                    status_code=400,
                    detail="backend deve ser numpy | pgvector | both",
                )
            return result
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Erro em /build-index")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/db/init")
    def api_db_init() -> dict[str, Any]:
        from vedic_pipeline.storage.db import init_schema

        try:
            return init_schema()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/db/sync")
    def api_db_sync(corpus: str = str(DEFAULT_CORPUS)) -> dict[str, Any]:
        from vedic_pipeline.storage.catalog import sync_corpus_to_db

        try:
            return sync_corpus_to_db(Path(corpus))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    # -------------------------------------------------------- static frontend
    if FRONTEND_DIST.exists():
        assets = FRONTEND_DIST / "assets"
        if assets.exists():
            app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

        @app.get("/")
        def spa_index():
            return FileResponse(FRONTEND_DIST / "index.html")

        @app.get("/{full_path:path}")
        def spa_fallback(full_path: str):
            # não engolir rotas de API
            if full_path.startswith("api/") or full_path in {
                "health",
                "docs",
                "openapi.json",
                "redoc",
                "ingest",
                "tokenize",
                "train",
                "build-index",
                "search",
                "ask",
            }:
                raise HTTPException(status_code=404, detail="Not found")
            candidate = FRONTEND_DIST / full_path
            if candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(FRONTEND_DIST / "index.html")

    return app
