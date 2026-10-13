"""Gate de licenciamento — bloqueia fontes sem autorização explícita."""

from __future__ import annotations

from typing import Any

from vedic_pipeline.common.constants import ALLOWED_LICENSES

# Exceções por fonte, nunca globais: uma licença fora de ALLOWED_LICENSES só
# passa para a URL exata listada aqui, com a mesma licença declarada no
# manifesto. Cada entrada registra quem aprovou e quando.
SOURCE_LICENSE_EXCEPTIONS: dict[str, dict[str, str]] = {
    # GRETIL, Bhāgavata Purāṇa em sânscrito (input de Ulrich Stiehl), CC BY-NC-SA
    # 4.0. Aprovado pelo dono do projeto em 2026-10-02 como texto sânscrito
    # principal do Bhāgavata. NC: o texto não pode ser usado comercialmente.
    "https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_bhAgavatapurANa.txt": {
        "license": "cc-by-nc-sa-4.0",
        "approved": "2026-10-02, dono do projeto (uso não comercial)",
    },
}


def license_exception(src: dict[str, Any]) -> dict[str, str] | None:
    """Exceção aprovada para esta fonte específica, ou None."""
    url = (src.get("url") or "").strip()
    license_ = (src.get("license") or "").strip().lower()
    entry = SOURCE_LICENSE_EXCEPTIONS.get(url)
    if entry and entry["license"] == license_:
        return entry
    return None


def validate_source(src: dict[str, Any]) -> tuple[bool, str]:
    """Retorna (ok, motivo)."""
    license_ = (src.get("license") or "").strip().lower()
    if not license_:
        return False, "licença ausente (bloqueado por política jurídica)"
    exception = None
    if license_ not in ALLOWED_LICENSES:
        exception = license_exception(src)
        if exception is None:
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
    if exception is not None:
        return True, f"ok (exceção por fonte: {license_}, aprovada {exception['approved']})"
    return True, "ok"
