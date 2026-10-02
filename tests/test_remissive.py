"""Índice remissivo e escolha do personagem ilustrado."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from vedic_pipeline.api.app import create_app
from vedic_pipeline.search.remissive import (
    annotate_search,
    figure_negative,
    figure_prompt,
    get_figure,
)


def _hit(text: str, locator: str = "RV 1.1.1") -> dict:
    return {"chunk_id": locator, "text": text, "locator": locator, "title": locator}


class RemissiveTests(unittest.TestCase):
    def test_query_names_the_portrait_even_if_hits_differ(self):
        aside = annotate_search("hymn to Agni", [_hit("Indra slew Vritra", "RV 1.32.1")])
        self.assertEqual(aside["figure"]["id"], "agni")
        ids = [e["id"] for e in aside["entries"]]
        self.assertEqual(ids[:2], ["agni", "indra"])
        indra = next(e for e in aside["entries"] if e["id"] == "indra")
        self.assertEqual(indra["locators"][0]["locator"], "RV 1.32.1")
        self.assertTrue(any(ref["id"] == "soma" for ref in aside["figure"]["see_also"]))

    def test_hit_frequency_picks_the_character_when_the_query_does_not(self):
        hits = [
            _hit("Indra and the Maruts", "RV 1.85.1"),
            _hit("Again Indra", "RV 1.32.2"),
        ]
        aside = annotate_search("storm and rain", hits)
        self.assertEqual(aside["figure"]["id"], "indra")
        self.assertEqual(aside["figure"]["count"], 2)

    def test_concepts_do_not_become_a_portrait(self):
        aside = annotate_search("dharma and the self", [_hit("Dharma holds the world")])
        self.assertIsNone(aside["figure"])
        self.assertIn("dharma", [e["id"] for e in aside["entries"]])

    def test_generic_words_do_not_invent_a_god(self):
        aside = annotate_search("fire and the self", [_hit("a bright flame")])
        self.assertIsNone(aside["figure"])
        self.assertEqual(aside["entries"], [])

    def test_inflected_and_devanagari_names_match(self):
        aside = annotate_search("अग्निम्", [_hit("agním īḷe")])
        self.assertEqual(aside["figure"]["id"], "agni")

    def test_short_names_do_not_match_inside_words(self):
        aside = annotate_search("drama and soma", [])
        self.assertEqual(aside["figure"]["id"], "soma")
        self.assertNotIn("rama", [e["id"] for e in aside["entries"]])

    def test_unknown_figure_has_no_portrait_prompt_target(self):
        self.assertIsNone(get_figure("dharma"))
        self.assertIsNone(get_figure("nope"))
        gloss = get_figure("agni")
        self.assertIsNotNone(gloss)
        prompt = figure_prompt(gloss)
        self.assertTrue(prompt.startswith("Agni"))
        for icon in ("two bearded heads", "seven flaming tongues", "red skin", "riding a ram", "ladle"):
            self.assertIn(icon, prompt)
        # "no X" no positivo atrai X: o que não deve aparecer vai no negativo.
        self.assertNotIn("no Latin", prompt)
        negative = figure_negative(gloss)
        self.assertIn("letters", negative)
        self.assertIn("blue skin", negative)

    def test_portrait_prompts_fit_the_clip_window(self):
        from vedic_pipeline.search.remissive import _GLOSSES

        for gloss in _GLOSSES:
            if get_figure(gloss.id) is None:
                continue
            prompt = figure_prompt(gloss)
            # CLIP corta em 77 tokens; ~1,4 token por palavra deixa folga com 50.
            self.assertLessEqual(len(prompt.split()), 50, gloss.id)
            self.assertTrue(prompt.startswith(gloss.visual.split(",")[0]), gloss.id)


class SearchAsideApiTests(unittest.TestCase):
    def test_search_payload_carries_the_index(self):
        hits = [_hit("Indra slew Vritra", "RV 1.32.1")]
        with patch("vedic_pipeline.llm.ask.retrieve_hits", return_value=(hits, "numpy")), TestClient(
            create_app()
        ) as client:
            res = client.post("/api/v1/search", json={"query": "hymn to Agni"})
        self.assertEqual(res.status_code, 200, res.text)
        body = res.json()
        self.assertEqual(body["figure"]["id"], "agni")
        self.assertIn("indra", [e["id"] for e in body["remissive"]])

    def test_figure_image_uses_the_named_cache(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VEDIC_GENERATION_API_TOKEN": ""}):
            dest = Path(tmp) / "agni.jpg"

            def _write(figure_id: str, prompt: str, *, negative: str | None = None, force: bool = False) -> Path:
                self.assertEqual(figure_id, "agni")
                self.assertIn("Agni", prompt)
                self.assertIn("letters", negative or "")
                dest.write_bytes(b"j" * 2000)
                return dest

            # O cache real em data/media/figures não pode decidir o teste.
            with patch("vedic_pipeline.llm.imagine.figure_image_path", return_value=dest), patch(
                "vedic_pipeline.llm.imagine.generate_figure_image", side_effect=_write
            ) as gen, TestClient(create_app()) as client:
                missing = client.get("/api/v1/figures/dharma/image")
                self.assertEqual(missing.status_code, 404)
                ok = client.get("/api/v1/figures/agni/image")
                self.assertEqual(ok.status_code, 200, ok.text)
                self.assertEqual(ok.headers["content-type"], "image/jpeg")
                gen.assert_called_once()


if __name__ == "__main__":
    unittest.main()
