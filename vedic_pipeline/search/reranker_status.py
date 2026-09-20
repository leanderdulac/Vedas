"""Estado do CE: gate PROMOTE ≠ ligado no default."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vedic_pipeline.common.constants import PROJECT_ROOT
from vedic_pipeline.search.reranker import (
    get_reranker_model_name,
    is_reranker_enabled,
    looks_like_filesystem_ref,
    resolve_reranker_model_source,
)

PROMOTE_PATH = PROJECT_ROOT / "fixtures" / "reranker_promote.json"


def load_promote_record(path: Path | None = None) -> dict[str, Any]:
    dest = Path(path) if path else PROMOTE_PATH
    if not dest.exists():
        return {
            "version": None,
            "gate": "UNKNOWN",
            "eligible_opt_in": False,
            "default_enabled": False,
            "weights_committed": False,
        }
    return json.loads(dest.read_text(encoding="utf-8"))


def reranker_runtime_status(promote_path: Path | None = None) -> dict[str, Any]:
    record = load_promote_record(promote_path)
    model = get_reranker_model_name()
    resolved = resolve_reranker_model_source(model)
    local_ref = looks_like_filesystem_ref(model) or looks_like_filesystem_ref(resolved)
    present = Path(resolved).is_dir() if local_ref else False
    enabled = is_reranker_enabled()
    eligible = bool(record.get("eligible_opt_in"))
    default_enabled = bool(record.get("default_enabled"))
    return {
        "enabled": enabled,
        "default_enabled": default_enabled,
        "eligible_opt_in": eligible,
        "gate": record.get("gate"),
        "version": record.get("version"),
        "gold": record.get("gold"),
        "gold_n": record.get("gold_n"),
        "hybrid_pass": record.get("hybrid_pass"),
        "ce_pass": record.get("ce_pass"),
        "model": model,
        "model_present": present,
        "weights_committed": bool(record.get("weights_committed")),
        "ready_to_opt_in": bool(eligible and present and not enabled and not default_enabled),
        "opt_in": record.get("opt_in") or {},
        "notes": record.get("notes"),
    }


def public_reranker_status() -> dict[str, Any]:
    """Recorte para /health: sem path de pesos, sem env de opt-in."""
    status = reranker_runtime_status()
    return {
        "enabled": status["enabled"],
        "default_enabled": status["default_enabled"],
        "eligible_opt_in": status["eligible_opt_in"],
        "gate": status["gate"],
        "version": status["version"],
        "model_present": status["model_present"],
        "ready_to_opt_in": status["ready_to_opt_in"],
    }
