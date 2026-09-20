"""Pares de fine-tune do reranker e CLIs em dry-run (sem embeddings / sem CE)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from vedic_pipeline.common.constants import PROJECT_ROOT
from vedic_pipeline.train.rerank_pairs import (
    DEFAULT_TINY_CANDIDATES,
    assign_splits,
    build_pairs,
    index_is_ready,
    load_candidates_payload,
    load_gold_queries,
    pairs_from_hits,
    reranker_forced_off,
    retrieve_candidates,
    summarize_pairs,
    title_matches,
    write_jsonl,
)

SCRIPTS = PROJECT_ROOT / "scripts"


class PairLogicTests(unittest.TestCase):
    def test_title_match_nasadiya(self):
        expect = ["10.129", "Nasadiya", "Rig Veda selected"]
        self.assertTrue(title_matches("Rigveda RV 10.129 (Griffith)", expect))
        self.assertFalse(title_matches("Rigveda RV 10.125 (Griffith)", expect))

    def test_pairs_from_hits_label_positives_and_hard_negatives(self):
        pairs = pairs_from_hits(
            query="Nasadiya hymn",
            query_id="nasadiya",
            expect_title_any=["10.129", "Nasadiya"],
            hits=[
                {"title": "Rigveda RV 10.129", "text": "Then was not non-existent", "score": 0.9},
                {"title": "Rigveda RV 10.125", "text": "I am the queen", "score": 0.8},
                {"title": "empty", "text": "   ", "score": 0.7},
            ],
            split="train",
            max_negatives=8,
        )
        self.assertEqual(len(pairs), 2)
        by_title = {p["doc_title"]: p for p in pairs}
        self.assertEqual(by_title["Rigveda RV 10.129"]["label"], 1)
        self.assertEqual(by_title["Rigveda RV 10.125"]["label"], 0)
        self.assertEqual(pairs[0]["query_id"], "nasadiya")
        self.assertIn("split", pairs[0])

    def test_tiny_fixture_builds_labeled_jsonl(self):
        gold = load_gold_queries(DEFAULT_TINY_CANDIDATES)
        candidates = load_candidates_payload(DEFAULT_TINY_CANDIDATES)
        pairs = build_pairs(gold, candidates, holdout_ratio=0.5, max_negatives=4)
        summary = summarize_pairs(pairs)
        self.assertGreaterEqual(summary["positives"], 2)
        self.assertGreaterEqual(summary["negatives"], 2)
        self.assertEqual(summary["queries"], 2)
        nasadiya = [p for p in pairs if p["query_id"] == "nasadiya"]
        titles = {p["doc_title"] for p in nasadiya if p["label"] == 0}
        self.assertTrue(any("10.125" in t for t in titles))

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "pairs.jsonl"
            written = write_jsonl(dest, pairs)
            self.assertEqual(written, len(pairs))
            row = json.loads(dest.read_text(encoding="utf-8").splitlines()[0])
            for key in ("query", "text", "label", "query_id", "doc_title", "split"):
                self.assertIn(key, row)

    def test_assign_splits_holdout(self):
        splits = assign_splits([f"q{i}" for i in range(10)], holdout_ratio=0.2)
        self.assertEqual(sum(1 for s in splits.values() if s == "eval"), 2)
        self.assertEqual(sum(1 for s in splits.values() if s == "train"), 8)

    def test_index_missing_raises_without_embeddings(self):
        self.assertFalse(index_is_ready(Path("/tmp/vedas-no-index")))
        with self.assertRaises(FileNotFoundError):
            retrieve_candidates(
                [{"id": "x", "query": "ātman"}],
                index_dir=Path("/tmp/vedas-no-index"),
                backend="numpy",
            )

    def test_reranker_forced_off_restores_env(self):
        import os
        from unittest.mock import patch

        from vedic_pipeline.search.reranker import invalidate_reranker, is_reranker_enabled

        with patch.dict(os.environ, {"VEDIC_ENABLE_RERANKER": "true"}):
            invalidate_reranker()
            self.assertTrue(is_reranker_enabled())
            with reranker_forced_off():
                invalidate_reranker()
                self.assertFalse(is_reranker_enabled())
            invalidate_reranker()
            self.assertTrue(is_reranker_enabled())


class ScriptDryRunTests(unittest.TestCase):
    def test_build_rerank_pairs_help_and_dry_run(self):
        help_proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "build_rerank_pairs.py"), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_proc.returncode, 0, help_proc.stderr)
        self.assertIn("JSONL", help_proc.stdout)

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "pairs.jsonl"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "build_rerank_pairs.py"),
                    "--dry-run",
                    "--out",
                    str(dest),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            self.assertTrue(dest.exists())
            lines = dest.read_text(encoding="utf-8").strip().splitlines()
            self.assertGreaterEqual(len(lines), 4)
            payload = json.loads(proc.stdout)
            self.assertTrue(payload["dry_run"])
            self.assertGreaterEqual(payload["positives"], 1)

        missing = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "build_rerank_pairs.py"),
                "--gold",
                str(PROJECT_ROOT / "fixtures" / "smoke_queries.json"),
                "--index",
                "/tmp/vedas-no-index",
                "--out",
                str(Path(tempfile.gettempdir()) / "vedas-pairs-should-not-write.jsonl"),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(missing.returncode, 2, missing.stderr)

    def test_train_reranker_help_and_dry_run(self):
        help_proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "train_reranker.py"), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_proc.returncode, 0, help_proc.stderr)
        self.assertIn("CrossEncoder", help_proc.stdout)

        with tempfile.TemporaryDirectory() as tmp:
            pairs = Path(tmp) / "pairs.jsonl"
            pairs.write_text(
                json.dumps(
                    {
                        "query": "ātman",
                        "text": "The Self",
                        "label": 1,
                        "query_id": "isha-self",
                        "doc_title": "Isha",
                        "split": "train",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            out = Path(tmp) / "reranker"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "train_reranker.py"),
                    "--dry-run",
                    "--pairs",
                    str(pairs),
                    "--out",
                    str(out),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            meta = json.loads((out / "train_meta.json").read_text(encoding="utf-8"))
            self.assertTrue(meta["dry_run"])
            self.assertFalse(meta["trained"])
            self.assertEqual(meta["positives"], 1)


if __name__ == "__main__":
    unittest.main()
