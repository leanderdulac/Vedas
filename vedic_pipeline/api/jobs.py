"""Fila simples de trabalhos para operações pesadas do pipeline.

Operações como build-index/train/tokenize/ingest podem levar minutos e
estourar timeouts HTTP quando executadas na própria requisição. Este módulo
oferece uma fila em memória (ThreadPoolExecutor) com acompanhamento de
status — suficiente para instância única. Para multi-worker/horizontal,
troque por fila externa (RQ/Celery) mantendo o mesmo formato de job.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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


def _executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    with _LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = ThreadPoolExecutor(max_workers=max_workers(), thread_name_prefix="vedic-job")
        return _EXECUTOR


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
        # Evita crescimento ilimitado: remove os mais antigos já finalizados.
        if len(_JOBS) > _MAX_RETAINED:
            done = sorted(
                (j for j in _JOBS.values() if j["status"] in {"done", "error"}),
                key=lambda j: j.get("finished_at") or "",
            )
            for old in done[: len(_JOBS) - _MAX_RETAINED]:
                _JOBS.pop(old["job_id"], None)
    _executor().submit(_run, job_id, fn)
    return public_job(job)


def _run(job_id: str, fn: Callable[[], Any]) -> None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        job["status"] = "running"
    try:
        result = fn()
        with _LOCK:
            job = _JOBS.get(job_id)
            if job is not None:
                job["status"] = "done"
                job["finished_at"] = _utc_now()
                job["result"] = result if isinstance(result, dict) else {"result": result}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Job %s falhou", job_id)
        with _LOCK:
            job = _JOBS.get(job_id)
            if job is not None:
                job["status"] = "error"
                job["finished_at"] = _utc_now()
                job["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"


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
    """Limpa a fila (uso em testes). Não cancela trabalhos em execução."""
    with _LOCK:
        _JOBS.clear()
