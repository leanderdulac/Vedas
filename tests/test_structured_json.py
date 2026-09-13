"""Ingestão de JSON estruturado (DharmicData)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vedic_pipeline.common.constants import PROJECT_ROOT
from vedic_pipeline.crawler.ingest import ingest_manifest
from vedic_pipeline.etl.chunking import chunk_records
from vedic_pipeline.etl.structure import parse_document_units
from vedic_pipeline.etl.structured_json import (
    compact_gita_chapter,
    compact_zurich_rows,
    expand_structured_file,
    expand_structured_payload,
    split_sukta_text,
)

SAMPLES = PROJECT_ROOT / "fixtures" / "structured"


class SuktaSplitTests(unittest.TestCase):
    def test_nasadiya_markers(self):
        heading, verses = split_sukta_text(
            (SAMPLES / "rigveda_10_129_sa.sample.json").read_text(encoding="utf-8")
        )
        # file is JSON; use the inner text
        payload = json.loads((SAMPLES / "rigveda_10_129_sa.sample.json").read_text(encoding="utf-8"))
        heading, verses = split_sukta_text(payload[0]["text"])
        self.assertIn("प्रजापति", heading or "")
        self.assertEqual([n for n, _ in verses], [1, 2, 3])
        self.assertTrue(verses[0][1].startswith("नासदासीन्"))


class CompactTests(unittest.TestCase):
    def test_gita_drops_commentaries(self):
        compact = compact_gita_chapter(
            {
                "BhagavadGitaChapter": [
                    {
                        "chapter": 2,
                        "verse": 47,
                        "text": "कर्मण्येवाधिकारस्ते",
                        "commentaries": {"Swami Ramsukhdas": "copyrighted"},
                        "translations": {"swami sivananda": "Thy right is to work only"},
                    }
                ]
            }
        )
        self.assertEqual(compact["verses"][0]["text"], "कर्मण्येवाधिकारस्ते")
        self.assertNotIn("commentaries", compact["verses"][0])
        self.assertNotIn("Thy right", json.dumps(compact))

    def test_expand_sample_files(self):
        gita = expand_structured_file(SAMPLES / "gita_chapter_2_sa.sample.json")
        self.assertEqual(len(gita), 1)
        units = parse_document_units(gita[0])
        ids = {u.verse_id for u in units}
        self.assertIn("BG.2.47", ids)
        self.assertIn("BG.2.11", ids)
        self.assertEqual(gita[0]["language"], "sa")

        rv = expand_structured_file(SAMPLES / "rigveda_10_129_sa.sample.json")
        self.assertEqual(len(rv), 1)
        self.assertIn("RV 10.129", rv[0]["title"])
        units = parse_document_units(rv[0])
        ids = {u.verse_id for u in units}
        self.assertIn("RV.10.129.1", ids)
        self.assertIn("RV.10.129.3", ids)
        chunks = list(chunk_records(rv))
        self.assertTrue(any(c.get("verse_id") == "RV.10.129.1" for c in chunks))

    def test_yajurveda_atharvaveda_samaveda_samples(self):
        yv = expand_structured_file(SAMPLES / "yajurveda_adhyaya_1_sa.sample.json")
        self.assertEqual(len(yv), 1)
        ids = {u.verse_id for u in parse_document_units(yv[0])}
        self.assertIn("VS.1.1", ids)
        self.assertIn("VS.1.2", ids)
        self.assertEqual(yv[0]["license"], "odbl")

        av = expand_structured_file(SAMPLES / "atharvaveda_1_1_sa.sample.json")
        self.assertEqual(len(av), 1)
        self.assertIn("AV 1.1", av[0]["title"])
        av_ids = {u.verse_id for u in parse_document_units(av[0])}
        self.assertIn("AV.1.1.1", av_ids)
        self.assertIn("AV.1.1.2", av_ids)

        sv = expand_structured_file(SAMPLES / "samaveda_hymn_sample.json")
        self.assertEqual(len(sv), 1)
        self.assertEqual(sv[0]["language"], "en")
        sv_ids = {u.verse_id for u in parse_document_units(sv[0])}
        self.assertIn("SV.1.1.1.1.1", sv_ids)
        self.assertIn("SV.1.1.1.1.2", sv_ids)

    def test_yajurveda_double_danda(self):
        heading, verses = split_sukta_text(
            "इषे त्वा ।। १।।\nवसोः पवित्रमसि ।। २ ।।"
        )
        self.assertEqual([n for n, _ in verses], [1, 2])
        self.assertTrue(verses[0][1].startswith("इषे"))

    def test_zurich_rows_group_padas(self):
        compact = compact_zurich_rows(
            [
                ("01.001.01", "a", "agním īḷe puróhitaṃ"),
                ("01.001.01", "a", "agním īḷe puróhitaṃ"),
                ("01.001.01", "b", "yajñásya devám r̥tvíjam |"),
                ("01.001.01", "c", "hótāraṃ ratnadhā́tamam ‖"),
                ("01.129.01", "a", "násad āsīn nó sád āsīt tadânīm"),
            ]
        )
        self.assertEqual(set(compact), {1})
        hymns = {s["sukta"]: s for s in compact[1]["suktas"]}
        self.assertIn(1, hymns)
        self.assertIn(129, hymns)
        verse1 = hymns[1]["verses"][0]["text"]
        self.assertIn("agním īḷe", verse1)
        self.assertIn("yajñásya", verse1)

        docs = expand_structured_payload(compact[1])
        titles = {d["title"] for d in docs}
        self.assertTrue(any("VedaWeb Zürich" in t for t in titles))
        nasadiya = next(d for d in docs if "RV 1.129" in d["title"])
        units = parse_document_units(nasadiya)
        self.assertIn("RV.1.129.1", {u.verse_id for u in units})
        self.assertEqual(docs[0]["license"], "cc-by")


class IngestExpandTests(unittest.TestCase):
    def test_manifest_expands_json(self):
        manifest = {
            "sources": [
                {
                    "title": "Gita sample structured",
                    "url": str(SAMPLES / "gita_chapter_2_sa.sample.json"),
                    "tradition": "vaishnava",
                    "language": "sa",
                    "license": "public-domain",
                    "download": True,
                    "source_class": "structured-json",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            man = root / "m.json"
            man.write_text(json.dumps(manifest), encoding="utf-8")
            stats = ingest_manifest(man, corpus_path=root / "corpus.jsonl", raw_dir=root / "raw")
            self.assertEqual(stats["added"], 1)
            self.assertEqual(stats["failed"], 0)
