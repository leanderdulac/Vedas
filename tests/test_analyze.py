"""Vyākaraṇa interlinear: padas determinísticos, cache e endpoint /analyze."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from vedic_pipeline.api.verse_service import assemble_verse, padas_of
from vedic_pipeline.llm.analyze import (
    generate_analysis,
    load_cached_analysis,
)


def _bundle(**overrides) -> dict:
    witnesses = [
        {
            "role": "sa",
            "title": "Rigveda RV 1.1.1 (Sanskrit, DharmicData)",
            "text": "अग्निमीळे पुरोहितं । यज्ञस्य देवमृत्विजम् । होतारं रत्नधातमम् ॥",
        },
        {
            "role": "iast",
            "title": "Rigveda RV 1.1.1 (VedaWeb Zürich, ISO-15919)",
            "text": "agním īḷe puróhitaṃ | yajñásya devám ṛtvíjam | hótāraṃ ratnadhā́tamam ||",
        },
        {
            "role": "en",
            "title": "Rigveda RV 1.1 Agni (Griffith, sacred-texts)",
            "text": "I Laud Agni, the chosen Priest, God, minister of sacrifice",
        },
    ]
    bundle = {"verse_id": "RV.1.1.1", "locator": "RV 1.1.1", "witnesses": witnesses}
    bundle.update(overrides)
    return bundle


class PadasOfTests(unittest.TestCase):
    def test_splits_by_danda(self):
        padas = padas_of(_bundle())
        self.assertEqual(len(padas), 3)
        self.assertEqual(padas[0]["sa"], "अग्निमीळे पुरोहितं")
        self.assertEqual(padas[1]["sa"], "यज्ञस्य देवमृत्विजम्")
        self.assertEqual(padas[2]["sa"], "होतारं रत्नधातमम्")
        # iast (| ||) também segmenta em 3 → pares alinhados
        self.assertEqual(padas[0]["iast"], "agním īḷe puróhitaṃ")
        self.assertEqual(padas[2]["iast"], "hótāraṃ ratnadhā́tamam")

    def test_divergent_dandas_prefer_sa(self):
        b = _bundle(
            witnesses=[
                {"role": "sa", "title": "t", "text": "प प । प प । प प ॥"},
                {"role": "iast", "title": "t", "text": "a b | c d ||"},
            ]
        )
        padas = padas_of(b)
        self.assertEqual(len(padas), 3)
        self.assertEqual([p["iast"] for p in padas], ["", "", ""])

    def test_aligned_dandas_keep_iast(self):
        b = _bundle(
            witnesses=[
                {"role": "sa", "title": "t", "text": "पादः पादः । अन्तिमः पादः ॥"},
                {"role": "iast", "title": "t", "text": "pādaḥ pādaḥ | antimaḥ pādaḥ ||"},
            ]
        )
        padas = padas_of(b)
        self.assertEqual(len(padas), 2)
        self.assertEqual(padas[0]["sa"], "पादः पादः")
        self.assertEqual(padas[0]["iast"], "pādaḥ pādaḥ")
        self.assertEqual(padas[1]["iast"], "antimaḥ pādaḥ")

    def test_single_pada_without_danda(self):
        b = _bundle(
            witnesses=[
                {"role": "sa", "title": "t", "text": "कर्मण्येवाधिकारस्ते"},
            ]
        )
        padas = padas_of(b)
        self.assertEqual(len(padas), 1)
        self.assertEqual(padas[0]["sa"], "कर्मण्येवाधिकारस्ते")
        self.assertEqual(padas[0]["iast"], "")

    def test_empty_bundle(self):
        self.assertEqual(padas_of({"witnesses": []}), [])
        self.assertEqual(padas_of({}), [])

    def test_assemble_then_padas(self):
        rows = [
            {
                "verse_id": "RV.1.1.1",
                "locator": "RV 1.1.1",
                "work": "rigveda",
                "book": 1,
                "hymn": 1,
                "verse": 1,
                "language": "sa",
                "title": "Rigveda RV 1.1.1 (Sanskrit)",
                "text": "अग्निमीळे पुरोहितं । होतारं रत्नधातमम् ॥",
                "doc_id": "d1",
            }
        ]
        bundle = assemble_verse("RV.1.1.1", rows)
        assert bundle is not None
        padas = padas_of(bundle)
        self.assertEqual(len(padas), 2)

    def test_trims_embedded_next_verse(self):
        # chunks trazem o verso seguido do consecutivo marcado ([RV 1.1.2])
        b = _bundle(
            witnesses=[
                {
                    "role": "sa",
                    "title": "t",
                    "text": "[RV 1.1.1] प प । प प ॥\n\n[RV 1.1.2] क क । क क ॥",
                }
            ]
        )
        padas = padas_of(b)
        self.assertEqual([p["sa"] for p in padas], ["प प", "प प"])

    def test_drops_anukramani_rubric(self):
        # rubrica da edição védica: mesma linha do verso, com algarismos
        # devanāgarī e nome do ṛṣi sem acento, antes das partes acentuadas
        b = _bundle(
            witnesses=[
                {
                    "role": "sa",
                    "title": "t",
                    "text": "१-४ अथर्वा। वाचस्पतिः। ये॑ त्रिषप्ताः । वा॒चस्प॒तिर्बला॒ ॥",
                }
            ]
        )
        padas = padas_of(b)
        self.assertEqual(len(padas), 2)
        self.assertEqual(padas[0]["sa"], "ये॑ त्रिषप्ताः")
        self.assertEqual(padas[1]["sa"], "वा॒चस्प॒तिर्बला॒")


class AnalysisCacheTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._patch = patch("vedic_pipeline.llm.analyze.ANALYSIS_DIR", Path(self._tmp.name))
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def _padas(self):
        return padas_of(_bundle())

    def test_extractive_returns_padas_and_note(self):
        payload = generate_analysis(_bundle(), self._padas(), "pt", provider="extractive")
        self.assertEqual(payload["provider"], "extractive")
        self.assertEqual(payload["words"], [])
        self.assertEqual(len(payload["padas"]), 3)
        self.assertIn("LLM", payload["note"])

    def test_llm_generation_parses_json_and_caches(self):
        answer = (
            "Aqui está:\n```json\n"
            '{"words": ['
            '{"pada": 1, "form": "अग्निम्", "iast": "agni", "grammar": "acc. sg. m.", "gloss": "fogo"},'
            '{"pada": 1, "form": "īḷe", "iast": "īḷe", "grammar": "pres. 1sg. ātmanepada de il", "gloss": "louvo"}'
            "]}\n```"
        )
        with patch(
            "vedic_pipeline.llm.generate.generate_answer",
            return_value={"provider": "xai", "model": "grok-test", "answer": answer},
        ):
            payload = generate_analysis(_bundle(), self._padas(), "pt", provider="xai")
        self.assertEqual(len(payload["words"]), 2)
        self.assertEqual(payload["words"][0]["form"], "अग्निम्")
        self.assertFalse(payload["cached"])
        cached = load_cached_analysis(_bundle(), self._padas(), "pt")
        self.assertIsNotNone(cached)
        assert cached is not None
        self.assertEqual(len(cached["words"]), 2)
        self.assertTrue(cached["cached"])

    def test_invalid_llm_output_yields_note_not_cache(self):
        with patch(
            "vedic_pipeline.llm.generate.generate_answer",
            return_value={"provider": "xai", "model": "grok-test", "answer": "texto livre sem json"},
        ):
            payload = generate_analysis(_bundle(), self._padas(), "pt", provider="xai")
        self.assertEqual(payload["words"], [])
        self.assertIn("não retornou", payload["note"] or "")
        self.assertFalse(load_cached_analysis(_bundle(), self._padas(), "pt"))


class AnalyzeEndpointTests(unittest.TestCase):
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
        with patch("vedic_pipeline.llm.analyze.ANALYSIS_DIR", Path(self._tmp.name)), patch(
            "vedic_pipeline.api.verse_service.get_verse", return_value=None
        ):
            res = self._client().post("/api/v1/verses/NOPE.1.1/analyze", json={"lang": "pt"})
        self.assertEqual(res.status_code, 404)

    def test_cached_hit_is_open_generation_requires_auth(self):
        from vedic_pipeline.llm.analyze import _cache_path

        padas = padas_of(_bundle())
        with patch("vedic_pipeline.llm.analyze.ANALYSIS_DIR", Path(self._tmp.name)):
            path = _cache_path(_bundle(), padas, "pt")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                '{"verse_id": "RV.1.1.1", "padas": [], "words": [], "lang": "pt", "cached": false}',
                encoding="utf-8",
            )
            client = self._client()
            with patch("vedic_pipeline.api.verse_service.get_verse", return_value=_bundle()):
                res = client.post("/api/v1/verses/RV.1.1.1/analyze", json={"lang": "pt"})
                self.assertEqual(res.status_code, 200, res.text)
                self.assertTrue(res.json()["cached"])
                # cache miss (en) + geração paga sem token → 401 antes de gerar
                res = client.post("/api/v1/verses/RV.1.1.1/analyze", json={"lang": "en", "provider": "xai"})
            self.assertEqual(res.status_code, 401, res.text)


class PadasEndpointTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._env = patch.dict("os.environ", {"VEDIC_PIPELINE_API_TOKEN": "pipe-tok"})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def _client(self):
        from vedic_pipeline.api.app import create_app

        return TestClient(create_app())

    def test_padas_open_deterministic(self):
        client = self._client()
        with patch("vedic_pipeline.api.verse_service.get_verse", return_value=_bundle()):
            res = client.get("/api/v1/verses/RV.1.1.1/padas")
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertEqual(data["verse_id"], "RV.1.1.1")
        self.assertEqual(len(data["padas"]), 3)
        self.assertEqual(data["padas"][0]["sa"], "अग्निमीळे पुरोहितं")
        self.assertEqual(data["padas"][0]["iast"], "agním īḷe puróhitaṃ")

    def test_padas_404_and_422(self):
        client = self._client()
        with patch("vedic_pipeline.api.verse_service.get_verse", return_value=None):
            res = client.get("/api/v1/verses/NOPE.1.1/padas")
        self.assertEqual(res.status_code, 404)
        with patch("vedic_pipeline.api.verse_service.get_verse", return_value={"witnesses": []}):
            res = client.get("/api/v1/verses/EMPTY.1.1/padas")
        self.assertEqual(res.status_code, 422)


if __name__ == "__main__":
    unittest.main()
