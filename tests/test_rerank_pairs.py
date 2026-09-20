"""Pares de fine-tune do reranker e CLIs em dry-run (sem embeddings / sem CE)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from vedic_pipeline.common.constants import PROJECT_ROOT
from vedic_pipeline.train.rerank_pairs import (
    DEFAULT_TINY_CANDIDATES,
    assign_splits,
    build_pairs,
    hymn_in_blob,
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
        self.assertFalse(title_matches("Rig Veda selected hymns (Griffith)", expect))
        self.assertTrue(
            title_matches("Rig Veda selected hymns (Griffith)", expect, prefer_strict=False)
        )

    def test_hymn_boundaries_10_5_vs_10_50(self):
        self.assertTrue(hymn_in_blob("Rigveda RV 10.5 (Griffith)", "10.5"))
        self.assertFalse(hymn_in_blob("Rigveda RV 10.50 (Griffith)", "10.5"))
        self.assertFalse(hymn_in_blob("Rigveda RV 10.125 (Griffith)", "10.5"))
        self.assertTrue(hymn_in_blob("Rigveda RV 2.38 (Griffith)", "2.38"))

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

    def test_nasadiya_hard_negatives_kept_when_present(self):
        pairs = pairs_from_hits(
            query="In the beginning there was neither non-existence nor existence then Nasadiya",
            query_id="nasadiya",
            expect_title_any=["10.129", "Nasadiya", "Rig Veda selected"],
            hits=[
                {"title": "Rigveda RV 10.129 Nasadiya", "text": "Then was not non-existent"},
                {"title": "Rig Veda selected hymns", "text": "Agni priest of the sacrifice"},
                {"title": "Rigveda RV 10.125", "text": "I am the queen"},
                {"title": "Rigveda RV 10.5", "text": "two mothers"},
                {"title": "Rigveda RV 2.38", "text": "Savitar"},
                {"title": "Rigveda RV 10.50", "text": "unrelated hymn"},
            ],
            split="train",
            max_negatives=0,
        )
        positives = [p for p in pairs if p["label"] == 1]
        negatives = [p for p in pairs if p["label"] == 0]
        self.assertEqual(len(positives), 1)
        self.assertIn("10.129", positives[0]["doc_title"])
        neg_titles = {p["doc_title"] for p in negatives}
        self.assertTrue(any("10.125" in t for t in neg_titles))
        self.assertIn("Rigveda RV 10.5", neg_titles)
        self.assertTrue(any("2.38" in t for t in neg_titles))
        self.assertFalse(any("10.50" in t for t in neg_titles))
        self.assertFalse(any("selected hymns" in t for t in {p["doc_title"] for p in positives}))

    def test_assign_splits_pins_nasadiya_to_train(self):
        splits = assign_splits(["isha-self", "nasadiya", "nasadiya-deva"], holdout_ratio=0.5)
        self.assertEqual(splits["nasadiya"], "train")
        self.assertEqual(splits["nasadiya-deva"], "train")
        self.assertEqual(splits["isha-self"], "eval")

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
        self.assertIn("--device", help_proc.stdout)
        self.assertIn("mps", help_proc.stdout)

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
            self.assertIn("device", meta)
            self.assertEqual(meta["n_train"], 1)
            self.assertEqual(meta["base_model"], "cross-encoder/ms-marco-MiniLM-L-6-v2")

    def test_eval_reranker_smoke_help_and_missing_local_model(self):
        help_proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "eval_reranker_smoke.py"), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_proc.returncode, 0, help_proc.stderr)
        self.assertIn("Cross-Encoder", help_proc.stdout)

        missing = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "eval_reranker_smoke.py"),
                "--model",
                "artifacts/reranker-does-not-exist",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(missing.returncode, 2, missing.stderr + missing.stdout)
        self.assertIn("não encontrado", missing.stderr)


class DetectDeviceTests(unittest.TestCase):
    def test_detect_device_prefers_mps_when_cuda_unavailable(self):
        from vedic_pipeline.train.reranker import detect_device

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        fake_torch.backends.mps.is_available.return_value = True
        with patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertEqual(detect_device(), "mps")

    def test_detect_device_cpu_when_neither_cuda_nor_mps(self):
        from vedic_pipeline.train.reranker import detect_device

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        fake_torch.backends.mps.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertEqual(detect_device(), "cpu")

    def test_detect_device_prefers_cuda_over_mps(self):
        from vedic_pipeline.train.reranker import detect_device

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.backends.mps.is_available.return_value = True
        with patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertEqual(detect_device(), "cuda")

    def test_resolve_device_auto_and_explicit(self):
        from vedic_pipeline.train.reranker import resolve_device

        with patch("vedic_pipeline.train.reranker.detect_device", return_value="mps"):
            self.assertEqual(resolve_device(None), "mps")
            self.assertEqual(resolve_device("auto"), "mps")
            self.assertEqual(resolve_device(""), "mps")
        self.assertEqual(resolve_device("cpu"), "cpu")
        self.assertEqual(resolve_device("CUDA"), "cuda")

    def test_train_reranker_honors_explicit_device(self):
        from vedic_pipeline.train.reranker import train_reranker

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "query": "ātman",
                        "text": "The Self",
                        "label": 1,
                        "split": "train",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            code, meta = train_reranker(
                pairs_path=path,
                out_dir=Path(tmp) / "out",
                dry_run=True,
                device="mps",
            )
            self.assertEqual(code, 0)
            self.assertEqual(meta["device"], "mps")


class TrainPairsShapeTests(unittest.TestCase):
    def test_load_pairs_accepts_passage_and_skips_holdout(self):
        from vedic_pipeline.train.reranker import load_pairs, select_train_pairs, train_reranker

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.jsonl"
            rows = [
                {
                    "query": "Nasadiya",
                    "passage": "Then was not non-existent RV 10.129",
                    "label": "positive",
                    "query_id": "nasadiya",
                    "split": "train",
                },
                {
                    "query": "Nasadiya",
                    "text": "I am the queen 10.125",
                    "label": 0,
                    "query_id": "nasadiya",
                    "split": "holdout",
                },
                {
                    "query": "Isha",
                    "text": "The Self",
                    "label": True,
                    "query_id": "isha-self",
                    "split": "eval",
                },
            ]
            path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                encoding="utf-8",
            )
            loaded = load_pairs(path)
            self.assertEqual(len(loaded), 3)
            self.assertEqual(loaded[0]["text"], "Then was not non-existent RV 10.129")
            self.assertEqual(loaded[0]["label"], 1.0)
            train, skipped, fallback = select_train_pairs(loaded)
            self.assertEqual(len(train), 1)
            self.assertEqual(len(skipped), 2)
            self.assertFalse(fallback)
            code, meta = train_reranker(
                pairs_path=path,
                out_dir=Path(tmp) / "out",
                dry_run=True,
            )
            self.assertEqual(code, 0)
            self.assertEqual(meta["n_train"], 1)
            self.assertEqual(meta["n_skipped_eval_split"], 2)
            self.assertIn("device", meta)
            self.assertIn("base_model", meta)

    def test_fit_cross_encoder_uses_train_split_only(self):
        from vedic_pipeline.train.reranker import fit_cross_encoder

        mock = MagicMock()
        pairs = [
            {"query": "q", "text": "pos", "label": 1.0, "split": "train"},
            {"query": "q", "text": "hold", "label": 0.0, "split": "holdout"},
            {"query": "q", "text": "eval", "label": 0.0, "split": "eval"},
        ]
        with patch("vedic_pipeline.train.reranker._fit_with_trainer") as fit:
            used = fit_cross_encoder(
                base_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                pairs=pairs,
                out_dir=Path("/tmp/vedas-reranker-test"),
                epochs=1,
                batch_size=2,
                max_length=32,
                learning_rate=2e-5,
                model=mock,
            )
        fit.assert_called_once()
        self.assertEqual([p["split"] for p in used], ["train"])
        self.assertEqual(fit.call_args.kwargs["train_rows"], used)


if __name__ == "__main__":
    unittest.main()
