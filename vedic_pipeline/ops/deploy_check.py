"""Checklist de deploy: senhas, tokens, CE opt-in e geração paga."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from vedic_pipeline.common.env import env_flag
from vedic_pipeline.search.reranker import (
    DEFAULT_RERANKER_MODEL,
    get_reranker_model_name,
    is_reranker_enabled,
    looks_like_filesystem_ref,
    resolve_reranker_model_source,
)

WEAK_PASSWORDS = frozenset(
    {
        "",
        "vedas",
        "postgres",
        "password",
        "changeme",
        "vedas-minio-dev-only",
        "troque-em-producao-com-32chars-minimo",
        "troque-por-senha-forte-alfanumerica-32chars",
    }
)


def _issue(level: str, code: str, message: str) -> dict[str, str]:
    return {"level": level, "code": code, "message": message}


def _password_from_env() -> str:
    return (os.environ.get("POSTGRES_PASSWORD") or "").strip()


def check_postgres_password(*, prod: bool) -> list[dict[str, str]]:
    password = _password_from_env()
    if not prod and not password:
        return []
    issues: list[dict[str, str]] = []
    weak = password in WEAK_PASSWORDS or (prod and len(password) < 16)
    if not weak:
        return []
    level = "fail" if prod else "warn"
    issues.append(
        _issue(
            level,
            "weak_postgres_password",
            "POSTGRES_PASSWORD ausente, default de exemplo ou com menos de 16 caracteres",
        )
    )
    return issues


def check_pipeline_token(*, prod: bool) -> list[dict[str, str]]:
    token = (os.environ.get("VEDIC_PIPELINE_API_TOKEN") or "").strip()
    if not prod:
        if not token:
            return [
                _issue(
                    "warn",
                    "pipeline_token_unset",
                    "VEDIC_PIPELINE_API_TOKEN vazio — /ingest e /train HTTP ficam 503 (CLI ok)",
                )
            ]
        return []
    if not token or len(token) < 16:
        return [
            _issue(
                "fail",
                "pipeline_token_missing",
                "Produção exige VEDIC_PIPELINE_API_TOKEN com pelo menos 16 caracteres",
            )
        ]
    return []


def check_generation_policy(*, prod: bool) -> list[dict[str, str]]:
    xai = bool((os.environ.get("XAI_API_KEY") or "").strip())
    token = bool((os.environ.get("VEDIC_GENERATION_API_TOKEN") or "").strip())
    require = env_flag("VEDIC_REQUIRE_GENERATION_TOKEN")
    issues: list[dict[str, str]] = []
    if prod and not require:
        issues.append(
            _issue(
                "fail",
                "require_generation_token_off",
                "Produção deve ter VEDIC_REQUIRE_GENERATION_TOKEN=true (compose.prod já define)",
            )
        )
    if xai and not token:
        level = "fail" if prod or require else "warn"
        issues.append(
            _issue(
                level,
                "open_paid_generation",
                "XAI_API_KEY sem VEDIC_GENERATION_API_TOKEN — geração paga fica aberta nesta instância",
            )
        )
    return issues


def check_reranker() -> list[dict[str, str]]:
    if not is_reranker_enabled():
        return []
    issues = [
        _issue(
            "warn",
            "reranker_opt_in",
            "VEDIC_ENABLE_RERANKER=true — opt-in consciente; default do projeto continua off",
        )
    ]
    model = get_reranker_model_name()
    resolved = resolve_reranker_model_source(model)
    present = Path(resolved).exists() if looks_like_filesystem_ref(resolved) else Path(resolved).exists()
    generic = model == DEFAULT_RERANKER_MODEL or not looks_like_filesystem_ref(model)
    if generic or not present:
        issues.append(
            _issue(
                "fail",
                "reranker_on_without_local_dir",
                "CE ligado sem diretório local de domínio (não use o MiniLM genérico em prod)",
            )
        )
    return issues


def check_ssrf_escape() -> list[dict[str, str]]:
    if env_flag("VEDIC_DISABLE_SSRF_DNS_CHECK"):
        return [
            _issue(
                "fail",
                "ssrf_dns_check_disabled",
                "VEDIC_DISABLE_SSRF_DNS_CHECK está setado; o check de DNS do crawler não deve ser desligado",
            )
        ]
    return []


def run_deploy_check(*, mode: str = "local") -> dict[str, Any]:
    prod = str(mode).strip().lower() == "prod"
    issues: list[dict[str, str]] = []
    issues.extend(check_postgres_password(prod=prod))
    issues.extend(check_pipeline_token(prod=prod))
    issues.extend(check_generation_policy(prod=prod))
    issues.extend(check_reranker())
    issues.extend(check_ssrf_escape())
    failures = [i for i in issues if i["level"] == "fail"]
    warnings = [i for i in issues if i["level"] == "warn"]
    return {
        "mode": "prod" if prod else "local",
        "ok": not failures,
        "fail_count": len(failures),
        "warn_count": len(warnings),
        "issues": issues,
    }


def deploy_check_exit_code(result: dict[str, Any]) -> int:
    return 0 if result.get("ok") else 1
