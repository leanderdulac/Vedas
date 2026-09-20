"""CLI unificada do pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from vedic_pipeline.common.constants import (
    DEFAULT_BASE_MODEL,
    DEFAULT_CORPUS,
    DEFAULT_EMBED_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL_DIR,
    DEFAULT_RAW_DIR,
    DEFAULT_TOKENIZER_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vedic_pipeline",
        description="Pipeline védico: ingestão, ETL, tokenização, treino e RAG.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ing = sub.add_parser("ingest", help="Baixa e indexa fontes do manifesto")
    p_ing.add_argument("--manifest", required=True)
    p_ing.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p_ing.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    p_ing.add_argument("--min-chars", type=int, default=80)
    p_ing.add_argument(
        "--sync-db",
        action="store_true",
        help="Sincroniza corpus com PostgreSQL se DATABASE_URL estiver setado",
    )

    p_dedupe = sub.add_parser("dedupe", help="Remove duplicatas do corpus JSONL")
    p_dedupe.add_argument("--corpus", default=str(DEFAULT_CORPUS))

    p_tok = sub.add_parser("train-tokenizer", help="Treina tokenizador BPE")
    p_tok.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p_tok.add_argument("--out", default=str(DEFAULT_TOKENIZER_DIR))
    p_tok.add_argument("--vocab-size", type=int, default=32_000)
    p_tok.add_argument("--min-frequency", type=int, default=2)
    p_tok.add_argument("--eval", action="store_true")

    p_model = sub.add_parser("train-model", help="Fine-tune causal / continual")
    p_model.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p_model.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER_DIR))
    p_model.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p_model.add_argument("--out", default=str(DEFAULT_MODEL_DIR))
    p_model.add_argument("--epochs", type=int, default=1)
    p_model.add_argument("--block-size", type=int, default=512)
    p_model.add_argument("--batch-size", type=int, default=2)
    p_model.add_argument("--lr", type=float, default=5e-5)
    p_model.add_argument("--max-steps", type=int, default=None)
    p_model.add_argument("--fp16", action="store_true")

    p_idx = sub.add_parser("build-index", help="Gera índice de embeddings (RAG)")
    p_idx.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p_idx.add_argument("--out", default=str(DEFAULT_EMBED_DIR))
    p_idx.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    p_idx.add_argument("--chunk-size", type=int, default=800)
    p_idx.add_argument("--overlap", type=int, default=120)
    p_idx.add_argument("--batch-size", type=int, default=64, help="Batch do encoder ST")
    p_idx.add_argument(
        "--backend",
        choices=["numpy", "pgvector", "both"],
        default="numpy",
    )
    p_idx.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Dimensão pgvector (default: VEDIC_EMBEDDING_DIM ou dim do modelo)",
    )

    p_search = sub.add_parser("search", help="Busca semântica no índice")
    p_search.add_argument("query", help="Consulta em linguagem natural")
    p_search.add_argument("--index", default=str(DEFAULT_EMBED_DIR))
    p_search.add_argument("--top-k", type=int, default=5, choices=range(1, 51), metavar="1-50")
    p_search.add_argument("--tradition", default=None)
    p_search.add_argument("--language", default=None)
    p_search.add_argument(
        "--backend",
        choices=["auto", "numpy", "pgvector"],
        default="auto",
    )
    p_search.add_argument("--prompt", action="store_true", help="Imprime prompt RAG")

    p_ask = sub.add_parser("ask", help="RAG + geração (xAI / local / extractive)")
    p_ask.add_argument("query", help="Pergunta")
    p_ask.add_argument("--index", default=str(DEFAULT_EMBED_DIR))
    p_ask.add_argument("--top-k", type=int, default=5, choices=range(1, 31), metavar="1-30")
    p_ask.add_argument("--tradition", default=None)
    p_ask.add_argument("--language", default=None)
    p_ask.add_argument(
        "--backend",
        choices=["auto", "numpy", "pgvector"],
        default="auto",
    )
    p_ask.add_argument(
        "--provider",
        choices=["auto", "xai", "local", "extractive"],
        default="auto",
    )
    p_ask.add_argument("--model", default=None)
    p_ask.add_argument("--no-hits", action="store_true")
    p_ask.add_argument("--prompt", action="store_true")

    p_db_init = sub.add_parser("db-init", help="Cria schema PostgreSQL/pgvector")
    p_db_init.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Dimensão vector(N) (default: VEDIC_EMBEDDING_DIM ou 384)",
    )
    p_db_sync = sub.add_parser("db-sync", help="Sincroniza corpus JSONL → PostgreSQL")
    p_db_sync.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p_db_sync.add_argument(
        "--no-purge",
        action="store_true",
        help="Não remove documentos órfãos ausentes no corpus",
    )
    sub.add_parser("db-check", help="Status da conexão PostgreSQL")
    p_art = sub.add_parser("artifacts", help="Backup/restauração de raw e artefatos (S3/MinIO)")
    art_sub = p_art.add_subparsers(dest="artifacts_action", required=True)
    art_sub.add_parser("status", help="Mostra backend de objetos (local ou S3)")
    p_push = art_sub.add_parser("push", help="Envia diretório local para o bucket S3")
    p_push.add_argument("--dir", default="artifacts", help="Diretório local (ex.: artifacts, data/raw)")
    p_push.add_argument("--prefix", default=None, help="Prefixo no bucket (default: nome do diretório)")
    p_push.add_argument("--bucket", default=None)
    p_push.add_argument("--dry-run", action="store_true")
    p_pull = art_sub.add_parser("pull", help="Baixa prefixo do bucket para diretório local")
    p_pull.add_argument("--prefix", required=True, help="Prefixo no bucket (ex.: artifacts)")
    p_pull.add_argument("--dir", default="artifacts", help="Diretório local de destino")
    p_pull.add_argument("--bucket", default=None)
    p_pull.add_argument("--dry-run", action="store_true")
    p_serve = sub.add_parser("serve", help="Sobe a API FastAPI")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)

    p_corpus = sub.add_parser(
        "corpus-status",
        help="Perfil + fingerprint do corpus (lock reproduzível)",
    )
    p_corpus.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p_corpus.add_argument(
        "--profile",
        default=None,
        help="Verifica min/max de um perfil (bootstrap, local, open-web, canonical-snapshot)",
    )
    p_corpus.add_argument(
        "--write-lock",
        default=None,
        help="Grava data/corpus.lock.json (ou path) com o fingerprint atual",
    )
    p_corpus.add_argument(
        "--verify-lock",
        default=None,
        help="Compara o corpus atual com um lock gravado",
    )

    p_deploy = sub.add_parser(
        "deploy-check",
        help="Checklist de senhas, tokens, CE e geração paga",
    )
    p_deploy.add_argument(
        "--mode",
        choices=["local", "prod"],
        default="local",
        help="prod é fail-closed (token de geração, senha, pipeline token)",
    )

    sub.add_parser(
        "reranker-status",
        help="Gate PROMOTE vs default off; se o dir local está pronto para opt-in",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        from vedic_pipeline.crawler.ingest import ingest_manifest

        stats = ingest_manifest(
            args.manifest,
            corpus_path=Path(args.corpus),
            raw_dir=Path(args.raw_dir),
            min_chars=args.min_chars,
        )
        if args.sync_db:
            from vedic_pipeline.storage.catalog import sync_corpus_to_db

            stats["db_sync"] = sync_corpus_to_db(Path(args.corpus))
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0 if stats.get("failed", 0) == 0 else 1

    if args.command == "dedupe":
        from vedic_pipeline.common.corpus import dedupe_corpus_file

        print(json.dumps(dedupe_corpus_file(Path(args.corpus)), ensure_ascii=False, indent=2))
        return 0

    if args.command == "train-tokenizer":
        from vedic_pipeline.train.tokenizer import (
            evaluate_tokenizer_compression,
            train_bpe_tokenizer,
        )

        out = train_bpe_tokenizer(
            corpus_path=Path(args.corpus),
            out_dir=Path(args.out),
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
        )
        payload = {"tokenizer_dir": str(out)}
        if args.eval:
            payload["compression"] = evaluate_tokenizer_compression(out, Path(args.corpus))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "train-model":
        from vedic_pipeline.train.model import train_causal_model

        out = train_causal_model(
            corpus_path=Path(args.corpus),
            out_dir=Path(args.out),
            base_model=args.base_model,
            tokenizer_dir=Path(args.tokenizer),
            epochs=args.epochs,
            block_size=args.block_size,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_steps=args.max_steps,
            fp16=args.fp16,
        )
        print(json.dumps({"model_dir": str(out)}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "build-index":
        result: dict = {"backend": args.backend}
        if args.backend in {"numpy", "both"}:
            from vedic_pipeline.search.embeddings import build_embedding_index

            result["numpy"] = build_embedding_index(
                corpus_path=Path(args.corpus),
                out_dir=Path(args.out),
                model_name=args.model,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                batch_size=args.batch_size,
            )
        if args.backend in {"pgvector", "both"}:
            from vedic_pipeline.storage.vectors import build_pgvector_index

            result["pgvector"] = build_pgvector_index(
                corpus_path=Path(args.corpus),
                model_name=args.model,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                batch_size=args.batch_size,
                embedding_dim=args.dim,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "search":
        from vedic_pipeline.llm.ask import retrieve_hits
        from vedic_pipeline.search.rag import build_rag_prompt

        hits, used = retrieve_hits(
            args.query,
            backend=args.backend,
            index_dir=Path(args.index),
            top_k=args.top_k,
            tradition=args.tradition,
            language=args.language,
        )
        out = {"query": args.query, "retrieval_backend": used, "hits": hits}
        if args.prompt:
            out["rag_prompt"] = build_rag_prompt(args.query, hits)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    if args.command == "ask":
        from vedic_pipeline.llm.ask import ask

        result = ask(
            args.query,
            backend=args.backend,
            index_dir=Path(args.index),
            top_k=args.top_k,
            tradition=args.tradition,
            language=args.language,
            provider=args.provider,
            model=args.model,
            include_hits=not args.no_hits,
            include_prompt=args.prompt,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "db-init":
        from vedic_pipeline.storage.db import init_schema

        print(json.dumps(init_schema(dim=args.dim), ensure_ascii=False, indent=2))
        return 0

    if args.command == "db-sync":
        from vedic_pipeline.storage.catalog import sync_corpus_to_db

        print(
            json.dumps(
                sync_corpus_to_db(
                    Path(args.corpus),
                    purge=not args.no_purge,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "db-check":
        from vedic_pipeline.storage.db import check_db

        print(json.dumps(check_db(), ensure_ascii=False, indent=2))
        return 0

    if args.command == "artifacts":
        from vedic_pipeline.storage import objects as obj_store

        if args.artifacts_action == "status":
            print(json.dumps(obj_store.status(), ensure_ascii=False, indent=2))
            return 0
        if args.artifacts_action == "push":
            prefix = args.prefix or Path(args.dir).name
            try:
                result = obj_store.push_dir(
                    args.dir, prefix, bucket=args.bucket, dry_run=args.dry_run
                )
            except RuntimeError as exc:
                print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
                return 2
            print(json.dumps({"ok": True, **result}, ensure_ascii=False, indent=2))
            return 0
        if args.artifacts_action == "pull":
            try:
                result = obj_store.pull_dir(
                    args.prefix, args.dir, bucket=args.bucket, dry_run=args.dry_run
                )
            except RuntimeError as exc:
                print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
                return 2
            print(json.dumps({"ok": True, **result}, ensure_ascii=False, indent=2))
            return 0
        parser.error(f"Ação desconhecida: {args.artifacts_action}")
        return 2

    if args.command == "corpus-status":
        from vedic_pipeline.common.corpus_status import (
            build_lock,
            corpus_status,
            status_exit_code,
            write_lock,
        )

        status = corpus_status(
            Path(args.corpus),
            profile=args.profile,
            lock_path=Path(args.verify_lock) if args.verify_lock else None,
        )
        if args.write_lock:
            lock = build_lock(status, profile=args.profile)
            dest = write_lock(Path(args.write_lock), lock)
            status["lock_written"] = str(dest)
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return status_exit_code(status)

    if args.command == "deploy-check":
        from vedic_pipeline.ops.deploy_check import deploy_check_exit_code, run_deploy_check

        result = run_deploy_check(mode=args.mode)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return deploy_check_exit_code(result)

    if args.command == "reranker-status":
        from vedic_pipeline.search.reranker_status import reranker_runtime_status

        print(json.dumps(reranker_runtime_status(), ensure_ascii=False, indent=2))
        return 0

    if args.command == "serve":
        import uvicorn

        from vedic_pipeline.api.app import create_app

        uvicorn.run(
            create_app(),
            host=args.host,
            port=args.port,
            reload=False,
            proxy_headers=True,
            forwarded_allow_ips="*",
        )
        return 0

    parser.error(f"Comando desconhecido: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
