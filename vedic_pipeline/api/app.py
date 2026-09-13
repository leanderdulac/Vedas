"""API REST da aplicação Veda Knowledge — pipeline + endpoints da UI."""

import logging
import os
import secrets
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from vedic_pipeline import __version__ as _app_version
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
    from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    from vedic_pipeline.api.metrics import METRICS
    from vedic_pipeline.api.request_policy import authorize_generation
    from vedic_pipeline.api.schemas import (
        AskRequest,
        BuildIndexRequest,
        ExplainRequest,
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
        version=_app_version,
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

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        import time

        start = time.time()
        response = await call_next(request)
        duration = time.time() - start
        METRICS.record_http_request(
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_s=duration,
        )
        return response

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        from fastapi.responses import JSONResponse

        from vedic_pipeline.api.rate_limit import allow_request

        if not allow_request(request.client.host if request.client else None, request.url.path):
            # Middleware roda fora do ExceptionMiddleware do router: retorna Response
            # diretamente em vez de lançar HTTPException (que viraria 500).
            return JSONResponse(
                status_code=429,
                content={"detail": "Muitas requisições — aguarde um instante"},
            )
        return await call_next(request)

    def require_pipeline_token(authorization: str | None = Header(default=None)) -> None:
        expected = os.environ.get("VEDIC_PIPELINE_API_TOKEN", "")
        if not expected:
            raise HTTPException(status_code=503, detail="Operações HTTP do pipeline desabilitadas; configure VEDIC_PIPELINE_API_TOKEN ou use a CLI")
        supplied = authorization.removeprefix("Bearer ") if authorization and authorization.startswith("Bearer ") else ""
        if not secrets.compare_digest(supplied.encode(), expected.encode()):
            raise HTTPException(status_code=401, detail="Token do pipeline inválido", headers={"WWW-Authenticate": "Bearer"})

    # ------------------------------------------------------------------ metrics & health
    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        from vedic_pipeline.api.catalog_service import corpus_stats

        stats = corpus_stats()
        content = METRICS.render_prometheus(
            doc_count=int(stats.get("documents") or 0),
            chunk_count=int(stats.get("chunks") or 0),
        )
        return Response(content=content, media_type="text/plain; version=0.0.4; charset=utf-8")

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
            "version": _app_version,
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
        tradition: str | None = None,
        language: str | None = None,
        q: str | None = None,
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

    @app.get("/api/v1/verses/{verse_id}")
    def api_verse(verse_id: str) -> dict[str, Any]:
        from vedic_pipeline.api.verse_service import get_verse

        verse = get_verse(verse_id)
        if not verse:
            raise HTTPException(status_code=404, detail="Verso não encontrado no índice")
        return verse

    @app.post("/api/v1/verses/{verse_id}/explain")
    def api_verse_explain(
        verse_id: str,
        body: ExplainRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        from vedic_pipeline.api.verse_service import explain_verse

        provider, model = authorize_generation(body.provider, body.model, authorization)
        try:
            return explain_verse(verse_id, lang=body.lang, provider=provider, model=model)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/v1/verses/{verse_id}/audio")
    def api_verse_audio(verse_id: str, authorization: str | None = Header(default=None)):
        from fastapi.responses import FileResponse

        from vedic_pipeline.api.request_policy import authorize_media
        from vedic_pipeline.api.verse_service import get_verse, recitation_text
        from vedic_pipeline.llm.tts import cached_verse_audio, tts_available

        if not tts_available():
            raise HTTPException(
                status_code=503,
                detail="TTS indisponível (sem XAI_API_KEY). O leitor usa a voz do navegador.",
            )
        # Gera áudio pago via xAI: respeita a mesma política de token do /ask.
        authorize_media(authorization)
        bundle = get_verse(verse_id)
        if not bundle:
            raise HTTPException(status_code=404, detail="Verso não encontrado no índice")
        text = recitation_text(bundle)
        if not text:
            raise HTTPException(status_code=404, detail="Verso sem texto para narrar")
        lang = "sa" if bundle.get("has_sanskrit") else "en"
        try:
            path = cached_verse_audio(verse_id, text, language=lang)
        except Exception as exc:  # noqa: BLE001
            logger.exception("TTS falhou para %s", verse_id)
            raise HTTPException(status_code=503, detail=f"Falha ao narrar o verso: {exc}") from exc
        return FileResponse(path, media_type="audio/mpeg", filename=f"{verse_id}.mp3")

    @app.get("/api/v1/media/cached")
    def api_cached_media() -> dict[str, list[str]]:
        from vedic_pipeline.llm.imagine import list_cached_media

        return list_cached_media()

    @app.get("/api/v1/verses/{verse_id}/image")
    def api_verse_image(
        verse_id: str,
        refresh: bool = Query(default=False),
        cached: bool = Query(default=False),
        authorization: str | None = Header(default=None),
    ):
        from fastapi.responses import FileResponse

        from vedic_pipeline.api.request_policy import authorize_media
        from vedic_pipeline.llm.imagine import generate_verse_image, image_path

        dest = image_path(verse_id)
        if dest.exists() and dest.stat().st_size > 1000 and not refresh:
            return FileResponse(dest, media_type="image/jpeg")
        if cached:
            raise HTTPException(status_code=404, detail="Ilustração ainda não gerada")
        # Só a geração nova é paga (xAI Imagine); leitura em cache fica aberta.
        authorize_media(authorization)
        try:
            path = generate_verse_image(verse_id, force=refresh)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Imagine imagem falhou para %s", verse_id)
            raise HTTPException(status_code=503, detail=f"Falha ao ilustrar o verso: {exc}") from exc
        return FileResponse(path, media_type="image/jpeg")

    @app.post("/api/v1/verses/{verse_id}/video")
    def api_verse_video_start(verse_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        from vedic_pipeline.api.request_policy import authorize_media
        from vedic_pipeline.llm.imagine import start_verse_video, video_path

        if video_path(verse_id).exists():
            return {"verse_id": verse_id, "status": "done", "ready": True}
        # Gera still + vídeo pagos (xAI Imagine): política de token do /ask.
        authorize_media(authorization)
        try:
            return start_verse_video(verse_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Imagine vídeo falhou para %s", verse_id)
            raise HTTPException(status_code=503, detail=f"Falha ao gerar vídeo: {exc}") from exc

    @app.get("/api/v1/verses/{verse_id}/video")
    def api_verse_video_status(verse_id: str) -> dict[str, Any]:
        from vedic_pipeline.llm.imagine import poll_verse_video, video_path

        if video_path(verse_id).exists():
            return {"verse_id": verse_id, "status": "done", "ready": True}
        try:
            return poll_verse_video(verse_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/v1/verses/{verse_id}/video/file")
    def api_verse_video_file(verse_id: str):
        from fastapi.responses import FileResponse

        from vedic_pipeline.llm.imagine import poll_verse_video, video_path

        dest = video_path(verse_id)
        if not dest.exists():
            job = poll_verse_video(verse_id)
            if not job.get("ready"):
                raise HTTPException(status_code=404, detail="Vídeo ainda não está pronto")
        return FileResponse(video_path(verse_id), media_type="video/mp4", filename=f"{verse_id}.mp4")

    @app.post("/api/v1/search")
    @app.post("/search")
    def api_search(body: SearchRequest) -> dict[str, Any]:
        from vedic_pipeline.llm.ask import retrieve_hits
        from vedic_pipeline.search.rag import build_rag_prompt

        try:
            import time

            t0 = time.time()
            hits, used = retrieve_hits(
                body.query,
                backend=body.backend,
                index_dir=Path(body.index_dir),
                top_k=body.top_k,
                tradition=body.tradition,
                language=body.language,
            )
            METRICS.record_search(used, time.time() - t0)
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
            raise HTTPException(status_code=404, detail="Índice ou corpus não encontrado") from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Erro em /search")
            raise HTTPException(status_code=500, detail="Erro interno ao processar a busca") from exc

    @app.post("/api/v1/ask")
    @app.post("/ask")
    def api_ask(body: AskRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        from vedic_pipeline.llm.ask import ask

        provider, model = authorize_generation(body.provider, body.model, authorization)
        try:
            return ask(
                body.query,
                backend=body.backend,
                index_dir=Path(body.index_dir),
                top_k=body.top_k,
                tradition=body.tradition,
                language=body.language,
                provider=provider,
                model=model,
                include_hits=body.include_hits,
                include_prompt=body.include_prompt,
                hybrid=body.hybrid,
                history=[{"role": t.role, "content": t.content} for t in body.history],
            )
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Índice não encontrado. Construa o índice com: "
                    "python -m vedic_pipeline build-index"
                ),
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Erro em /ask")
            raise HTTPException(status_code=500, detail="Erro interno ao processar a pergunta") from exc

    @app.post("/api/v1/ask/stream")
    @app.post("/ask/stream")
    def api_ask_stream(body: AskRequest, authorization: str | None = Header(default=None)):
        """Server-Sent Events: meta → token* → done | error."""
        import json as _json

        from fastapi.responses import StreamingResponse

        from vedic_pipeline.llm.ask import ask_stream_events

        provider, model = authorize_generation(body.provider, body.model, authorization)
        def event_gen():
            try:
                for ev in ask_stream_events(
                    body.query,
                    backend=body.backend,
                    index_dir=Path(body.index_dir),
                    top_k=body.top_k,
                    tradition=body.tradition,
                    language=body.language,
                    provider=provider,
                    model=model,
                    hybrid=body.hybrid,
                    history=[{"role": t.role, "content": t.content} for t in body.history],
                ):
                    name = ev.get("event") or "message"
                    payload = _json.dumps(ev.get("data") or {}, ensure_ascii=False)
                    yield f"event: {name}\ndata: {payload}\n\n"
            except FileNotFoundError:
                err = _json.dumps({"detail": "Índice não encontrado"}, ensure_ascii=False)
                yield f"event: error\ndata: {err}\n\n"
            except Exception:  # noqa: BLE001
                logger.exception("Erro em /ask/stream")
                err = _json.dumps({"detail": "Erro interno ao processar resposta via streaming"}, ensure_ascii=False)
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
    @app.post("/ingest", dependencies=[Depends(require_pipeline_token)])
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
            raise HTTPException(status_code=404, detail="Manifesto não encontrado") from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Erro em /ingest")
            raise HTTPException(status_code=500, detail="Erro interno durante a ingestão do manifesto") from exc

    @app.post("/tokenize", dependencies=[Depends(require_pipeline_token)])
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
            raise HTTPException(status_code=404, detail="Corpus não encontrado para tokenização") from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Erro em /tokenize")
            raise HTTPException(status_code=500, detail="Erro interno durante o treinamento do tokenizador") from exc

    @app.post("/train", dependencies=[Depends(require_pipeline_token)])
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
            raise HTTPException(status_code=404, detail="Corpus ou diretório não encontrado para treino") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Erro em /train")
            raise HTTPException(status_code=500, detail="Erro interno durante o treinamento do modelo") from exc

    @app.post("/build-index", dependencies=[Depends(require_pipeline_token)])
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
            raise HTTPException(status_code=404, detail="Corpus não encontrado para indexação") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Erro em /build-index")
            raise HTTPException(status_code=500, detail="Erro interno durante a construção do índice") from exc

    @app.post("/db/init", dependencies=[Depends(require_pipeline_token)])
    def api_db_init() -> dict[str, Any]:
        from vedic_pipeline.storage.db import init_schema

        try:
            return init_schema()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Erro em /db/init")
            raise HTTPException(status_code=503, detail="Serviço de banco de dados indisponível") from exc

    @app.post("/db/sync", dependencies=[Depends(require_pipeline_token)])
    def api_db_sync(corpus: str = str(DEFAULT_CORPUS)) -> dict[str, Any]:
        from vedic_pipeline.storage.catalog import sync_corpus_to_db

        try:
            return sync_corpus_to_db(Path(corpus))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Erro em /db/sync")
            raise HTTPException(status_code=503, detail="Serviço de banco de dados indisponível") from exc

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
            # não engolir rotas de API nem subrotas de infraestrutura
            api_prefixes = (
                "api",
                "db",
                "ask",
                "health",
                "docs",
                "openapi.json",
                "redoc",
                "ingest",
                "tokenize",
                "train",
                "build-index",
                "search",
            )
            for p in api_prefixes:
                if full_path == p or full_path.startswith(f"{p}/"):
                    raise HTTPException(status_code=404, detail="Not found")

            root = FRONTEND_DIST.resolve()
            candidate = (root / full_path).resolve()
            if not candidate.is_relative_to(root):
                raise HTTPException(status_code=404, detail="Not found")
            if candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(FRONTEND_DIST / "index.html")

    return app
