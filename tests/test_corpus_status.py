"""Perfil, fingerprint e lock do corpus."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vedic_pipeline.common.corpus import rewrite_corpus
from vedic_pipeline.common.corpus_status import (
    build_lock,
    corpus_status,
    fingerprint_records,
    load_profiles,
    matched_profiles,
    status_exit_code,
    summarize_corpus,
    verify_lock,
    verify_profile,
    write_lock,
)


def _write_corpus(path: Path, n: int) -> None:
    rows = [
        {
            "id": f"doc-{i}",
            "title": f"Title {i}",
            "text": f"śloka {i} " * 8,
            "tradition": "vedic",
            "language": "en",
        }
        for i in range(n)
    ]
    rewrite_corpus(path, rows)


class CorpusStatusTests(unittest.TestCase):
    def test_profiles_file_has_reproducible_bootstrap(self):
        payload = load_profiles()
        self.assertEqual(payload["default_profile"], "bootstrap")
        bootstrap = payload["profiles"]["bootstrap"]
        self.assertTrue(bootstrap["reproducible"])
        self.assertEqual(bootstrap["min_documents"], 11)
        self.assertFalse(payload["profiles"]["canonical-snapshot"]["reproducible"])

    def test_fingerprint_stable_and_order_independent(self):
        a = [{"id": "b", "text": "two"}, {"id": "a", "text": "one"}]
        b = [{"id": "a", "text": "one"}, {"id": "b", "text": "two"}]
        self.assertEqual(fingerprint_records(a), fingerprint_records(b))
        self.assertNotEqual(fingerprint_records(a), fingerprint_records([{"id": "a", "text": "outro"}]))

    def test_bootstrap_profile_matches_eleven_docs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.jsonl"
            _write_corpus(path, 11)
            summary = summarize_corpus(path)
            self.assertEqual(summary["documents"], 11)
            self.assertIn("bootstrap", matched_profiles(summary))
            verify = verify_profile(summary, "bootstrap")
            self.assertTrue(verify["ok"])
            self.assertEqual(status_exit_code({"corpus_exists": True, "verify_profile": verify}), 0)

    def test_missing_corpus_is_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = corpus_status(Path(tmp) / "missing.jsonl")
            self.assertFalse(status["corpus_exists"])
            self.assertEqual(status_exit_code(status), 2)

    def test_lock_roundtrip_detects_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.jsonl"
            _write_corpus(path, 3)
            summary = summarize_corpus(path)
            lock = build_lock(summary, profile="local")
            lock_path = write_lock(Path(tmp) / "corpus.lock.json", lock)
            self.assertTrue(verify_lock(summary, lock)["ok"])
            _write_corpus(path, 4)
            drifted = summarize_corpus(path)
            check = verify_lock(drifted, json.loads(lock_path.read_text(encoding="utf-8")))
            self.assertFalse(check["ok"])
            self.assertTrue(any("fingerprint" in item for item in check["issues"]))

    def test_cli_corpus_and_reranker_status(self):
        from vedic_pipeline.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.jsonl"
            lock = Path(tmp) / "lock.json"
            _write_corpus(path, 11)
            self.assertEqual(
                main(
                    [
                        "corpus-status",
                        "--corpus",
                        str(path),
                        "--profile",
                        "bootstrap",
                        "--write-lock",
                        str(lock),
                    ]
                ),
                0,
            )
            self.assertTrue(lock.exists())
            self.assertEqual(
                main(
                    [
                        "corpus-status",
                        "--corpus",
                        str(path),
                        "--verify-lock",
                        str(lock),
                    ]
                ),
                0,
            )
        self.assertEqual(main(["reranker-status"]), 0)
        self.assertEqual(main(["deploy-check", "--mode", "local"]), 0)
