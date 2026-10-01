"""SLM LoRA: filtro de qualidade do corpus + dry-run (sem download/torch)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.train_slm_lora as slm


def _mk_corpus(rows: list[dict]) -> Path:
    import json

    fd, name = tempfile.mkstemp(suffix=".jsonl")
    with open(fd, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return Path(name)


class FilterTests(unittest.TestCase):
    def test_noise_ocr_excluded_by_default(self):
        rows = [
            {"id": "a", "title": "The Markandeya Purana (Pargiter, 1904, Cornell scan OCR)", "text": "x" * 60, "language": "en"},
            {"id": "b", "title": "Mārkaṇḍeya Purāṇa (Devanāgarī, scan OCR)", "text": "y" * 60, "language": "sa"},
            {"id": "c", "title": "Rigveda RV 1.1 (Griffith)", "text": "z" * 60, "language": "en"},
        ]
        kept = slm.filter_records(rows)
        self.assertEqual([r["title"] for r in kept], ["Rigveda RV 1.1 (Griffith)"])

    def test_keep_noise_flag(self):
        rows = [{"id": "a", "title": "X scan OCR", "text": "x" * 60, "language": "en"}]
        self.assertEqual(len(slm.filter_records(rows)), 0)
        self.assertEqual(len(slm.filter_records(rows, exclude_noise=False)), 1)

    def test_short_texts_dropped(self):
        rows = [{"id": "a", "title": "ok", "text": "curto", "language": "en"}]
        self.assertEqual(slm.filter_records(rows), [])


class DryRunTests(unittest.TestCase):
    def test_dry_run_reports_dataset_without_download(self):
        corpus = _mk_corpus(
            [
                {"id": "a", "title": "Upanishad clean", "text": "अग्निमीळे पुरोहितं " * 10, "language": "sa"},
                {"id": "b", "title": "Gita clean", "text": "duty and action " * 10, "language": "en"},
            ]
        )
        try:
            with patch("sys.argv", ["train_slm_lora.py", "--corpus", str(corpus), "--dry-run"]):
                code = slm.main()
        finally:
            corpus.unlink(missing_ok=True)
        self.assertEqual(code, 0)

    def test_build_texts_max_chars(self):
        corpus = _mk_corpus([{"id": "a", "title": "t", "text": "x" * 100, "language": "en"}])
        try:
            # cap de 50 chars: o único doc (100 chars) estoura o teto e nada entra
            self.assertEqual(slm.build_texts(corpus, max_chars=50), [])
            # teto generoso: doc entra inteiro
            texts = slm.build_texts(corpus, max_chars=200)
        finally:
            corpus.unlink(missing_ok=True)
        self.assertEqual(len(texts), 1)
        self.assertEqual(len(texts[0]), 100)


if __name__ == "__main__":
    unittest.main()
