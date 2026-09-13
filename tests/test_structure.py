"""Unidades canônicas de verso / sūtra."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vedic_pipeline.common.constants import PROJECT_ROOT
from vedic_pipeline.etl.chunking import chunk_records
from vedic_pipeline.etl.structure import (
    detect_work,
    parse_document_units,
    parse_gita,
    parse_int_or_roman,
    parse_rigveda,
    parse_verse_id,
    parse_yoga_sutra,
)

FIXTURES = PROJECT_ROOT / "fixtures" / "sample_texts"


class VerseIdParseTests(unittest.TestCase):
    def test_canonical_ids(self):
        rv = parse_verse_id("RV.10.129.1")
        self.assertEqual(rv["work"], "rigveda")
        self.assertEqual(rv["book"], 10)
        self.assertEqual(rv["hymn"], 129)
        self.assertEqual(rv["verse"], 1)
        bg = parse_verse_id("BG.2.47")
        self.assertEqual(bg["work"], "bhagavad-gita")
        self.assertEqual(bg["book"], 2)
        self.assertEqual(bg["verse"], 47)
        vs = parse_verse_id("VS.1.1")
        self.assertEqual(vs["work"], "yajurveda")
        self.assertEqual(vs["verse"], 1)


class RomanTests(unittest.TestCase):
    def test_roman_and_digits(self):
        self.assertEqual(parse_int_or_roman("X"), 10)
        self.assertEqual(parse_int_or_roman("xviii"), 18)
        self.assertEqual(parse_int_or_roman("4"), 4)
        self.assertIsNone(parse_int_or_roman("Agni"))


class DetectWorkTests(unittest.TestCase):
    def test_titles_and_urls(self):
        self.assertEqual(
            detect_work("Rigveda RV 1.1 (Griffith, sacred-texts)"),
            "rigveda",
        )
        self.assertEqual(
            detect_work(url="https://www.sacred-texts.com/hin/rigveda/rv10090.htm"),
            "rigveda",
        )
        self.assertEqual(
            detect_work("Bhagavad-Gita — The Song Celestial (Edwin Arnold)"),
            "bhagavad-gita",
        )
        self.assertEqual(
            detect_work("The Yoga Sutras of Patanjali (Charles Johnston)"),
            "yoga-sutra",
        )
        self.assertEqual(
            detect_work("Yajurveda VS 1 (Sanskrit, DharmicData)"),
            "yajurveda",
        )
        self.assertEqual(
            detect_work("Atharvaveda AV 1.1 (Sanskrit, DharmicData)"),
            "atharvaveda",
        )
        self.assertEqual(
            detect_work("Sāmaveda SV 1.1.1.1 (Griffith, DharmicData)"),
            "samaveda",
        )
        self.assertIsNone(detect_work("Mahabharata selected themes (English)"))


class RigvedaFixtureTests(unittest.TestCase):
    def setUp(self):
        self.text = Path(FIXTURES / "rigveda_selection_en.txt").read_text(encoding="utf-8")
        self.units = parse_document_units(
            {
                "title": "Rig Veda selected hymns (English)",
                "source_url": "fixtures/sample_texts/rigveda_selection_en.txt",
                "text": self.text,
            }
        )

    def test_canonical_hymns(self):
        ids = {u.verse_id: u for u in self.units}
        self.assertIn("RV.1.1.1", ids)
        self.assertTrue(ids["RV.1.1.1"].text.startswith("I Laud Agni"))
        self.assertEqual(ids["RV.1.1.1"].locator, "RV 1.1.1")
        self.assertIn("RV.10.129.1", ids)
        self.assertIn("non-existent", ids["RV.10.129.1"].text.lower())
        self.assertIn("RV.10.90.1", ids)
        self.assertEqual(ids["RV.1.32"].locator, "RV 1.32")
        self.assertIsNone(ids["RV.1.32"].verse)
        self.assertIn("Vritra", ids["RV.1.32"].text)

    def test_notes_are_skipped(self):
        self.assertTrue(all("Notes for pipeline" not in u.text for u in self.units))


class GriffithPageTests(unittest.TestCase):
    def test_single_hymn_page(self):
        text = """
        The Rig Veda
        Translation by Ralph T.H. Griffith
        [1896]

        HYMN I. Agni.

        1 I Laud Agni, the chosen Priest, God, minister of sacrifice,
        The hotar, lavishest of wealth.
        2 Worthy is Agni to be praised by living as by ancient seers.
        He shall bring hitherward the Gods.
        """
        units = parse_rigveda(text, book=1, hymn=1)
        self.assertEqual([u.verse_id for u in units], ["RV.1.1.1", "RV.1.1.2"])
        self.assertIn("hotar", units[0].text)
        self.assertEqual(units[0].heading, "Agni")


class GitaFixtureTests(unittest.TestCase):
    def test_numbered_study_fixture(self):
        text = Path(FIXTURES / "bhagavad_gita_en.txt").read_text(encoding="utf-8")
        units = parse_document_units(
            {
                "title": "Bhagavad-gita (English)",
                "text": text,
            }
        )
        ids = {u.verse_id: u for u in units}
        self.assertIn("BG.2.47", ids)
        self.assertIn("work only", ids["BG.2.47"].text.lower())
        self.assertEqual(ids["BG.2.47"].locator, "BG 2.47")
        self.assertIn("BG.18.66", ids)
        self.assertEqual(ids["BG.4.7"].book, 4)

    def test_sample_without_chapter_header(self):
        text = Path(FIXTURES / "gita_sample_en.txt").read_text(encoding="utf-8")
        units = parse_gita(text)
        ids = {u.verse_id for u in units}
        self.assertIn("BG.2.11", ids)
        self.assertIn("BG.2.47", ids)
        self.assertTrue(all(u.verse is not None for u in units))
        self.assertFalse(any("pipeline metadata" in u.text.lower() for u in units))

    def test_arnold_chapter_and_anchor(self):
        text = """
        CHAPTER II

        Krishna.
        Thou grievest where no grief should be! thou speak'st
        Words lacking wisdom! for the wise in heart
        Mourn not for those that live, nor those that die.

        But thou, want not! ask not! Find full reward
        Of doing right in right! Let right deeds be
        Thy motive, not the fruit which comes from them.

        HERE ENDETH CHAPTER II. OF THE BHAGAVAD-GITA,
        Entitled "Sankhya-Yog,"
        Or "The Book of Doctrines."
        """
        units = parse_gita(text)
        locators = {u.locator: u for u in units}
        self.assertTrue(any(u.verse_id == "BG.2.11" for u in units))
        self.assertTrue(any(u.verse_id == "BG.2.47" for u in units))
        untitled = [u for u in units if u.verse is None]
        self.assertTrue(all(u.locator == "BG 2" for u in untitled))
        self.assertIn("BG 2.47", locators)


class YogaSutraTests(unittest.TestCase):
    def test_sutra_1_2(self):
        text = Path(FIXTURES / "yoga_sutra_en.txt").read_text(encoding="utf-8")
        units = parse_yoga_sutra(text)
        ids = {u.verse_id: u for u in units}
        self.assertEqual(ids["YS.1.2"].locator, "YS 1.2")
        self.assertIn("stilling", ids["YS.1.2"].text.lower())
        self.assertEqual(ids["YS.2.46"].book, 2)


class ChunkingIntegrationTests(unittest.TestCase):
    def test_chunks_carry_verse_id(self):
        text = Path(FIXTURES / "rigveda_selection_en.txt").read_text(encoding="utf-8")
        chunks = list(
            chunk_records(
                [
                    {
                        "id": "doc-rv",
                        "title": "Rig Veda selected hymns (English)",
                        "source_url": "fixtures/sample_texts/rigveda_selection_en.txt",
                        "tradition": "vedic",
                        "language": "en",
                        "license": "public-domain",
                        "text": text,
                    }
                ]
            )
        )
        by_id = {c["verse_id"]: c for c in chunks if c.get("verse_id")}
        self.assertIn("RV.10.129.1", by_id)
        self.assertTrue(by_id["RV.10.129.1"]["text"].startswith("[RV 10.129"))
        self.assertEqual(by_id["RV.1.1.1"]["work"], "rigveda")

    def test_unstructured_keeps_character_windows(self):
        chunks = list(
            chunk_records(
                [
                    {
                        "id": "doc-mb",
                        "title": "Mahabharata selected themes (English)",
                        "text": "A" * 50 + " " + "B" * 50,
                    }
                ],
                chunk_size=40,
                overlap=5,
            )
        )
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(c.get("verse_id") is None for c in chunks))


class CatalogUnitsTests(unittest.TestCase):
    def test_get_document_exposes_units(self):
        from vedic_pipeline.api.catalog_service import get_document, invalidate_corpus_cache

        text = Path(FIXTURES / "rigveda_selection_en.txt").read_text(encoding="utf-8")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(
                json.dumps(
                    {
                        "id": "rv1",
                        "title": "Rig Veda selected hymns (English)",
                        "source_url": "fixtures/sample_texts/rigveda_selection_en.txt",
                        "tradition": "vedic",
                        "language": "en",
                        "text": text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            path = Path(handle.name)
        try:
            invalidate_corpus_cache(path)
            doc = get_document("rv1", corpus_path=path)
            assert doc is not None
            ids = {unit["verse_id"] for unit in doc["units"]}
            self.assertIn("RV.10.129.1", ids)
            self.assertEqual(doc["work"], "rigveda")
        finally:
            path.unlink(missing_ok=True)
            invalidate_corpus_cache(path)
