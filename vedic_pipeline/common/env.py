"""Flags de ambiente booleanas (opt-in explícito)."""

from __future__ import annotations

import os


def env_flag(name: str, default: str = "") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "on", "yes"}
