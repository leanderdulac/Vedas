"""Tradução de versos (sânscrito → PT/EN): cache, extrativo e endpoint."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from vedic_pipeline.api.app import create_app
from vedic_pipeline.llm import translate as tr


def _bundle() -> dict:
    return {
        "verse_id": "RV.1.1.1",
        "locator": "RV 1.1.1",
        "witnesses": [
            {"role": "sa", "title": "sa", "text": "अग्निमीळे पुरोहितं"},
            {"role": "iast", "title": "iast", "text": "agním īḷe puróhitaṃ"},
            {"role": "en", "title": "en", "text": "I Laud Agni, the chosen Priest"},
        ],
    }


class TranslationServiceTests(unittest.TestCase):
    def test_extractive_returns_references_not_fake_translation(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(tr, "TRANSLATION_DIR", Path(tmp)):
            out = tr.generate_translation(_bundle(), "pt", provider="extractive")
        self.assertEqual(out["provider"], "extractive")
        self.assertIsNone(out["translation"])
        self.assertEqual([r["role"] for r in out["references"]], ["iast", "en"])
        self.assertIn("sem llm", out["note"].lower())

    def test_generation_caches_by_text(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            tr, "TRANSLATION_DIR", Path(tmp)
        ), patch(
            "vedic_pipeline.llm.generate.generate_answer",
            return_value={"provider": "xai", "model": "grok-4.5", "answer": "Eu louvo Agni, o sacerdote escolhido."},
        ) as gen:
            out = tr.generate_translation(_bundle(), "pt", provider="xai", model="grok-4.5")
            self.assertEqual(out["translation"], "Eu louvo Agni, o sacerdote escolhido.")
            self.assertTrue((Path(tmp) / "RV.1.1.1_pt_bb996500f26d6a9e.json").exists() or any(Path(tmp).glob("*.json")))
            cached = tr.load_cached_translation(_bundle(), "pt")
            self.assertIsNotNone(cached)
            self.assertTrue(cached["cached"])
            gen.assert_called_once()

    def test_load_cached_missing_is_none(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(tr, "TRANSLATION_DIR", Path(tmp)):
            self.assertIsNone(tr.load_cached_translation(_bundle(), "pt"))


class TranslationApiTests(unittest.TestCase):
    def test_404_for_unknown_verse(self):
        with patch.dict(
            os.environ, {"VEDIC_GENERATION_API_TOKEN": "t", "XAI_API_KEY": "k"}
        ), patch("vedic_pipeline.api.verse_service.get_verse", return_value=None), TestClient(
            create_app()
        ) as client:
            r = client.post(
                "/api/v1/verses/NOPE.1.1/translate",
                json={"lang": "pt"},
                headers={"Authorization": "Bearer t"},
            )
            self.assertEqual(r.status_code, 404)

    def test_cached_read_is_open_generation_gated(self):
        bundle = _bundle()
        with patch.dict(
            os.environ, {"VEDIC_GENERATION_API_TOKEN": "t", "XAI_API_KEY": "k"}
        ), patch("vedic_pipeline.api.verse_service.get_verse", return_value=bundle), patch.object(
            tr, "load_cached_translation", return_value={"verse_id": "RV.1.1.1", "translation": "x", "cached": True}
        ), TestClient(create_app()) as client:
            # cache aberto mesmo sem Authorization
            r = client.post("/api/v1/verses/RV.1.1.1/translate", json={"lang": "pt"})
            self.assertEqual(r.status_code, 200, r.text)
            self.assertTrue(r.json().get("cached"))
            # geração nova exige token correto
            with patch.object(tr, "load_cached_translation", return_value=None):
                r = client.post("/api/v1/verses/RV.1.1.1/translate", json={"lang": "pt"})
                self.assertEqual(r.status_code, 401)


if __name__ == "__main__":
    unittest.main()
