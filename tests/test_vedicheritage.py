"""Adaptador Vedic Heritage Portal: extração, registro e ingestão."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vedic_pipeline.common.sanskrit import has_devanagari
from vedic_pipeline.etl.vedicheritage import (
    VEDICHERITAGE_ATTRIBUTION,
    build_record,
    is_vedicheritage_url,
    looks_vedicheritage,
    main_devanagari,
    parse_vedicheritage_file,
)

FIXTURE = Path("fixtures/vedicheritage/isha_upanishad.html")


def _fixture_html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


class MainDevanagariTests(unittest.TestCase):
    def test_extracts_mantras_and_skips_noise(self):
        text = main_devanagari(_fixture_html())
        self.assertIn("पूर्णमदः", text)          # mantra presente
        self.assertIn("॥१८॥", text)               # numeração devanagárica
        self.assertTrue(has_devanagari(text))
        # Não deve conter navegação / footer / botões ASCII
        self.assertNotIn("Home", text)
        self.assertNotIn("Copyright", text)
        self.assertNotIn("Upanishads", text)

    def test_preserves_svara_marks(self):
        text = main_devanagari(_fixture_html())
        self.assertIn("ई॒शा वा॒स्य॑", text)       # svarita/anudātta mantidos

    def test_modals_do_not_duplicate_text(self):
        # O bloco .modal repete o mesmo mantra; deve ser excluído.
        text = main_devanagari(_fixture_html())
        # "ॐ ईशा..." do modal é prefixo truncado ("ॐ ई॒शा वा॒स्य॑मि॒द सर्वं॒");
        # garante que o texto completo extraído vem do bloco principal e não
        # do modal (que tem apenas a primeira franga).
        self.assertIn("धन॑म्", text)
        self.assertLessEqual(text.count("॥१॥"), 1)

    def test_looks_vedicheritage_and_url(self):
        self.assertTrue(looks_vedicheritage(_fixture_html()))
        self.assertFalse(looks_vedicheritage("<html><body>only ascii, nothing special</body></html>"))
        self.assertEqual(main_devanagari("<html><body>only ascii</body></html>"), "")


class BuildRecordTests(unittest.TestCase):
    def test_build_record(self):
        rec = build_record(
            "https://vedicheritage.gov.in/upanishads/ishavasyopanishad/",
            _fixture_html(),
            title="Ishavasyopanishad of Vajasneyi Kanva Samhita",
        )
        self.assertIsNotNone(rec)
        self.assertEqual(rec["language"], "sa")
        self.assertEqual(rec["license"], "gov-ind")
        self.assertEqual(rec["tradition"], "upanishad")
        self.assertEqual(rec["source_url"], "https://vedicheritage.gov.in/upanishads/ishavasyopanishad/")
        self.assertIn("पूर्णमदः", rec["text"])
        self.assertEqual(rec["title"], "Ishavasyopanishad of Vajasneyi Kanva Samhita")
        self.assertEqual(rec["attribution"], VEDICHERITAGE_ATTRIBUTION)
        self.assertGreater(rec["char_count"], 0)

    def test_non_devanagari_html_yields_none(self):
        self.assertIsNone(build_record("https://example.com/x.html", "<html><body>hello</body></html>"))

    def test_url_classification(self):
        self.assertTrue(is_vedicheritage_url("https://vedicheritage.gov.in/upanishads/ishavasyopanishad/"))
        self.assertTrue(is_vedicheritage_url("http://www.vedicheritage.gov.in/samhitas/rigveda/"))
        self.assertFalse(is_vedicheritage_url("https://example.com/" ))


class IngestTests(unittest.TestCase):
    def test_ingest_manifest_vedicheritage(self):
        from vedic_pipeline.crawler.ingest import ingest_manifest

        with tempfile.TemporaryDirectory() as tmp:
            corpus_path = Path(tmp) / "corpus.jsonl"
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "sources": [
                            {
                                "url": "fixtures/vedicheritage/isha_upanishad.html",
                                "title": "Ishavasyopanishad",
                                "tradition": "upanishad",
                                "language": "sa",
                                "license": "gov-ind",
                                "source_class": "vedicheritage",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            stats = ingest_manifest(manifest_path, corpus_path=corpus_path)

            self.assertEqual(stats["accepted"], 1)
            self.assertEqual(stats["added"], 1)
            records = [
                json.loads(line)
                for line in corpus_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["license"], "gov-ind")
            self.assertEqual(records[0]["language"], "sa")
            self.assertIn("पूर्णमदः", records[0]["text"])

    def test_is_vedicheritage_url_import_roundtrip(self):
        # parse_vedicheritage_file lê o fixture e devolve um registro.
        recs = parse_vedicheritage_file(FIXTURE, source={"url": "x", "title": "Ishavasyopanishad"})
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["license"], "gov-ind")


if __name__ == "__main__":
    unittest.main()
