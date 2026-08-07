"""Gate de licenciamento — bloqueia fontes sem autorização explícita."""

from __future__ import annotations

from typing import Any

from vedic_pipeline.common.constants import ALLOWED_LICENSES


def validate_source(src: dict[str, Any]) -> tuple[bool, str]:
    """Retorna (ok, motivo)."""
    license_ = (src.get("license") or "").strip().lower()
    if not license_:
        return False, "licença ausente (bloqueado por política jurídica)"
    if license_ not in ALLOWED_LICENSES:
        return (
            False,
            f"licença '{license_}' não está na lista permitida: "
            f"{sorted(ALLOWED_LICENSES)}",
        )
    url = (src.get("url") or "").strip()
    if not url:
        return False, "URL ausente"
    if src.get("download") is False:
        return False, "download=false no manifesto (fonte apenas catalogada)"
    return True, "ok"
