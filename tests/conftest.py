"""Fixtures globais dos testes."""

import pytest


@pytest.fixture(autouse=True)
def _spoken_recitation_by_default(monkeypatch):
    """O .env local pode ligar o canto (VEDIC_RECITATION_STYLE=chant); os testes
    não podem disparar o ACE-Step de verdade. Quem testa o canto liga por conta própria."""
    monkeypatch.setenv("VEDIC_RECITATION_STYLE", "spoken")
    monkeypatch.delenv("VEDIC_ACESTEP_DIR", raising=False)
