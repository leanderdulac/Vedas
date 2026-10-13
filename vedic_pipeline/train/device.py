"""Seleção de device para treino/inferência (CPU / CUDA / MPS).

O Mac Apple Silicon (MPS) historicamente derrubava o processo com cargas
paralelas de modelos (SIGSEGV em versões antigas do torch). Por isso o
default é conservador: `auto` = CUDA se houver, senão CPU. Para usar MPS
(torch 2.14+ é estável para treino), defina `VEDIC_DEVICE=mps` ou
`VEDIC_ENABLE_MPS=1`.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("vedic_pipeline.train.device")


def detect_device() -> str:
    """Resolve o device conforme VEDIC_DEVICE / VEDIC_ENABLE_MPS.

    Ordem: env explícito > auto (cuda > cpu; mps só com flag).
    """
    explicit = (os.environ.get("VEDIC_DEVICE") or "").strip().lower()
    if explicit in {"cuda", "mps", "cpu"}:
        return explicit
    if explicit and explicit != "auto":
        logger.warning("VEDIC_DEVICE=%r desconhecido; usando auto", explicit)

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    mps_enabled = (os.environ.get("VEDIC_ENABLE_MPS") or "").strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }
    if mps_enabled and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def torch_device(device: str | None = None) -> Any:
    """Devolve o objeto torch.device correspondente (importa torch sob demanda)."""
    import torch

    resolved = device or detect_device()
    if resolved == "mps" and (
        not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available()
    ):
        logger.warning("MPS indisponível; caindo para CPU")
        resolved = "cpu"
    return torch.device(resolved)


def use_amp(device: str | None = None) -> bool:
    """AMP (fp16/bf16) só em CUDA/MPS."""
    resolved = device or detect_device()
    return resolved in {"cuda", "mps"}
