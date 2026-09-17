"""Tradução de versos: cache em disco, modo extrativo e endpoint /translate."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from vedic_pipeline.llm.translate import (
    _cache_path,
    _source_text,
    generate_translation,
    load_cached_translation,
    translate_verse,
)


def _bundle(**overrides) -> dict:
    witnesses = [
        {"role": "sa", "title": "Bhagavad-gītā 2.47 (Sanskrit, DharmicData)", "text": "कर्मण्येवाधिकारस्ते"},
        {"role": "iast", "title": "Bhagavad-gītā 2.47 (IAST)", "text": "karmaṇy evādhikāras te"},
        {"role": "en", "title": "Bhagavad-gītā 2.47 (EN)", "text": "You have a right to your actions"},
    ]
    bundle = {
        "verse_id": "BG.2.47",
        "locator": "BG 2.47",
        "witnesses": witnesses,
    }
    bundle.update(overrides)
    return bundle


class SourceTextTests(unittest.TestCase):
    def test_prefers_sanskrit_then_iast_then_en(self):
        self.assertEqual(_source_text(_bundle()), ("sa", "कर्मण्येवाधिकारस्ते"))
        sa_only = _bundle(witnesses=[w for w in _bundle()["witnesses"] if w["role"] != "sa"])
        self.assertEqual(_source_text(sa_only), ("iast", "karmaṇy evādhikāras te"))
        self.assertEqual(_source_text(_bundle(witnesses=[])), ("", ""))
        self.assertEqual(_source_text(_bundle(witnesses=None)), ("", ""))


class CachePathTests(unittest.TestCase):
    def test_deterministic_and_safe(self):
        p1 = _cache_path(_bundle(), "pt")
        p2 = _cache_path(_bundle(), "pt")
        self.assertEqual(p1, p2)
        self.assertIn("BG.2.47_pt_", p1.name)
        self.assertNotIn("/", p1.stem)

    def test_raises_when_no_text(self):
        with self.assertRaises(ValueError):
            _cache_path(_bundle(witnesses=[]), "pt")


class GenerateTranslationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._patch = patch("vedic_pipeline.llm.translate.TRANSLATION_DIR", Path(self._tmp.name))
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def test_extractive_returns_reference_block(self):
        payload = generate_translation(_bundle(), "pt", provider="extractive")
        self.assertIsNone(payload["translation"])
        self.assertEqual(payload["provider"], "extractive")
        self.assertIn("LLM", payload["note"])
        roles = [w["role"] for w in payload["references"]]
        self.assertEqual(roles, ["iast", "en"])
        # Modo extrativo não grava cache.
        self.assertFalse(load_cached_translation(_bundle(), "pt"))

    def test_llm_generation_persists_cache(self):
        with patch(
            "vedic_pipeline.llm.generate.generate_answer",
            return_value={"provider": "xai", "model": "grok-test", "answer": "Aja na ação apenas."},
        ):
            payload = generate_translation(_bundle(), "pt", provider="xai")
        self.assertEqual(payload["translation"], "Aja na ação apenas.")
        self.assertEqual(payload["model"], "grok-test")
        self.assertFalse(payload["cached"])
        cached = load_cached_translation(_bundle(), "pt")
        self.assertIsNotNone(cached)
        self.assertEqual(cached["translation"], "Aja na ação apenas.")
        self.assertTrue(cached["cached"])

    def test_empty_answer_not_cached(self):
        with patch(
            "vedic_pipeline.llm.generate.generate_answer",
            return_value={"provider": "xai", "model": "grok-test", "answer": ""},
        ):
            payload = generate_translation(_bundle(), "pt", provider="xai")
        self.assertEqual(payload["translation"], "")
        self.assertFalse(load_cached_translation(_bundle(), "pt"))

    def test_translate_verse_not_found(self):
        with patch("vedic_pipeline.api.verse_service.get_verse", return_value=None), self.assertRaises(
            FileNotFoundError
        ):
            translate_verse("NOPE.1.1")


class TranslateEndpointTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._env = patch.dict(
            "os.environ",
            {
                "VEDIC_GENERATION_API_TOKEN": "gen-tok",
                "VEDIC_PIPELINE_API_TOKEN": "pipe-tok",
            },
        )
        self._env.start()

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def _client(self):
        from vedic_pipeline.api.app import create_app

        return TestClient(create_app())

    def test_404_unknown_verse(self):
        with patch("vedic_pipeline.llm.translate.TRANSLATION_DIR", Path(self._tmp.name)), patch(
            "vedic_pipeline.api.verse_service.get_verse", return_value=None
        ):
            res = self._client().post("/api/v1/verses/NOPE.1.1/translate", json={"lang": "pt"})
        self.assertEqual(res.status_code, 404)

    def test_cached_hit_is_open_generation_requires_auth(self):
        with patch("vedic_pipeline.llm.translate.TRANSLATION_DIR", Path(self._tmp.name)):
            cached_file = _cache_path(_bundle(), "pt")
            cached_file.parent.mkdir(parents=True, exist_ok=True)
            cached_file.write_text(
                '{"verse_id": "BG.2.47", "lang": "pt", "translation": "pré-gerada", "cached": false}',
                encoding="utf-8",
            )
            client = self._client()
            with patch("vedic_pipeline.api.verse_service.get_verse", return_value=_bundle()):
                # Leitura do cache é aberta — mesmo sem cabeçalho de geração
                # e mesmo pedindo provider pago: hit de cache não gera custo.
                res = client.post("/api/v1/verses/BG.2.47/translate", json={"lang": "pt"})
                self.assertEqual(res.status_code, 200, res.text)
                self.assertEqual(res.json()["translation"], "pré-gerada")
                self.assertTrue(res.json()["cached"])
                res = client.post("/api/v1/verses/BG.2.47/translate", json={"lang": "pt", "provider": "xai"})
                self.assertEqual(res.status_code, 200, res.text)
                self.assertTrue(res.json()["cached"])
                # Cache miss (en) + geração paga sem token → 401 antes de gerar.
                res = client.post("/api/v1/verses/BG.2.47/translate", json={"lang": "en", "provider": "xai"})
            self.assertEqual(res.status_code, 401, res.text)


if __name__ == "__main__":
    unittest.main()
