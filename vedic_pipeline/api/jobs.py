"""Fila simples de trabalhos para operações pesadas do pipeline.

Operações como build-index/train/tokenize/ingest podem levar minutos e
estourar timeouts HTTP quando executadas na própria requisição. Este módulo
oferece uma fila em memória (ThreadPoolExecutor) com acompanhamento de
status e **persistência em disco** (`VEDIC_JOBS_DIR`, default `data/jobs`),
de modo que o histórico sobrevive a restarts. Para multi-worker/horizontal,
troque por fila externa (RQ/Celery) mantendo o mesmo formato de job.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from typing import Any

logger = logging.getLogger("vedic_pipeline.api.jobs")

_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}
_EXECUTOR: ThreadPoolExecutor | None = None
_MAX_RETAINED = 100


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def max_workers() -> int:
    raw = os.environ.get("VEDIC_JOB_WORKERS", "2").strip()
    try:
        return min(4, max(1, int(raw)))
    except ValueError:
        return 2


def jobs_dir() -> Path:
    raw = os.environ.get("VEDIC_JOBS_DIR", "data/jobs").strip()
    return Path(raw) if raw else Path("data/jobs")


def _executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    with _LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = ThreadPoolExecutor(max_workers=max_workers(), thread_name_prefix="vedic-job")
        return _EXECUTOR


def _job_path(job_id: str) -> Path:
    return jobs_dir() / f"{job_id}.json"


def _write_job_file(job: dict[str, Any]) -> None:
    path = _job_path(job["job_id"])
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:  # noqa: BLE001
        logger.warning("Falha ao persistir job %s: %s", job.get("job_id"), exc)


def _delete_job_file(job_id: str) -> None:
    with suppress(OSError):
        _job_path(job_id).unlink(missing_ok=True)


def load_jobs_from_disk() -> int:
    """Restaura jobs persistidos; marca queued/running como interrompidos.

    Chamado na criação da aplicação quando VEDIC_JOBS_RESTORE está habilitado
    (padrão no compose de produção). Retorna o número de jobs restaurados.
    """
    directory = jobs_dir()
    if not directory.is_dir():
        return 0
    restored = 0
    for path in sorted(directory.glob("*.json")):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(job, dict) or not job.get("job_id"):
            continue
        with _LOCK:
            if job.get("status") in {"queued", "running"}:
                job["status"] = "error"
                job["finished_at"] = _utc_now()
                job["error"] = "Interrompido por reinício do servidor"
                _write_job_file(job)
            if job["job_id"] not in _JOBS:
                _JOBS[job["job_id"]] = job
        restored += 1
    if restored:
        logger.info("Restaurados %d jobs de %s", restored, directory)
    return restored


def job_counts() -> dict[str, Any]:
    """Contagens para métricas (por status e por kind)."""
    by_status: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    with _LOCK:
        for j in _JOBS.values():
            s = str(j.get("status") or "unknown")
            k = str(j.get("kind") or "unknown")
            by_status[s] = by_status.get(s, 0) + 1
            by_kind[k] = by_kind.get(k, 0) + 1
    return {"total": sum(by_status.values()), "by_status": by_status, "by_kind": by_kind}


def submit_job(kind: str, fn: Callable[[], Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Enfileira fn e retorna o registro inicial do job (status=queued)."""
    job_id = uuid.uuid4().hex[:12]
    job: dict[str, Any] = {
        "job_id": job_id,
        "kind": kind,
        "status": "queued",
        "created_at": _utc_now(),
        "finished_at": None,
        "params": params or {},
        "result": None,
        "error": None,
    }
    with _LOCK:
        _JOBS[job_id] = job
        _write_job_file(job)
        # Evita crescimento ilimitado: remove os mais antigos já finalizados.
        if len(_JOBS) > _MAX_RETAINED:
            done = sorted(
                (j for j in _JOBS.values() if j["status"] in {"done", "error"}),
                key=lambda j: j.get("finished_at") or "",
            )
            for old in done[: len(_JOBS) - _MAX_RETAINED]:
                _JOBS.pop(old["job_id"], None)
                _delete_job_file(old["job_id"])
    _executor().submit(_run, job_id, fn)
    return public_job(job)


def _run(job_id: str, fn: Callable[[], Any]) -> None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        job["status"] = "running"
        _write_job_file(job)
    try:
        result = fn()
        with _LOCK:
            job = _JOBS.get(job_id)
            if job is not None:
                job["status"] = "done"
                job["finished_at"] = _utc_now()
                job["result"] = result if isinstance(result, dict) else {"result": result}
                _write_job_file(job)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Job %s falhou", job_id)
        with _LOCK:
            job = _JOBS.get(job_id)
            if job is not None:
                job["status"] = "error"
                job["finished_at"] = _utc_now()
                job["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                _write_job_file(job)


def get_job(job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        return public_job(job) if job else None


def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    with _LOCK:
        ordered = sorted(_JOBS.values(), key=lambda j: j.get("created_at") or "", reverse=True)
        return [public_job(j) for j in ordered[: max(1, min(limit, 100))]]


def public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "kind": job["kind"],
        "status": job["status"],
        "created_at": job.get("created_at"),
        "finished_at": job.get("finished_at"),
        "params": job.get("params") or {},
        "result": job.get("result"),
        "error": job.get("error"),
    }


def reset_jobs() -> None:
    """Limpa a fila em memória e em disco (uso em testes). Não cancela execuções."""
    with _LOCK:
        _JOBS.clear()
    directory = jobs_dir()
    if directory.is_dir():
        for path in directory.glob("*.json"):
            with suppress(OSError):
                path.unlink()
