"""Gate A/B do Cross-Encoder de domínio vs híbrido (sem índice / sem CE)."""

from __future__ import annotations

import unittest

from vedic_pipeline.search.rerank_eval import (
    compare_smoke_suites,
    find_nasadiya_row,
    gate_exit_code,
    is_nasadiya_item,
)


def _suite(rows: list[dict], *, passed: int | None = None) -> dict:
    return {
        "passed": passed if passed is not None else sum(1 for r in rows if r.get("ok")),
        "failed": sum(1 for r in rows if not r.get("ok")),
        "total": len(rows),
        "results": rows,
    }


class RerankEvalGateTests(unittest.TestCase):
    def test_is_nasadiya_from_id_or_query(self):
        self.assertTrue(is_nasadiya_item({"id": "nasadiya", "query": "creation"}))
        self.assertTrue(
            is_nasadiya_item(
                {
                    "id": "q",
                    "query": "In the beginning there was neither non-existence then Nasadiya",
                }
            )
        )
        self.assertFalse(is_nasadiya_item({"id": "isha-self", "query": "What is the Self"}))

    def test_promote_when_ce_matches_hybrid_including_nasadiya(self):
        rows = [
            {
                "id": "nasadiya",
                "query": "Nasadiya hymn",
                "ok": True,
                "top_titles": ["Rigveda RV 10.129"],
            },
            {"id": "isha-self", "query": "Isha", "ok": True, "top_titles": ["Isha Upanishad"]},
        ]
        comparison = compare_smoke_suites(_suite(rows), _suite(rows), model="artifacts/reranker")
        self.assertTrue(comparison["promote"])
        self.assertEqual(comparison["fail_reasons"], [])
        self.assertEqual(gate_exit_code(comparison), 0)
        self.assertTrue(comparison["nasadiya"]["hybrid_ok"])
        self.assertTrue(comparison["nasadiya"]["ce_ok"])
        nas_row = next(r for r in comparison["per_query"] if r["nasadiya"])
        self.assertEqual(nas_row["id"], "nasadiya")
        self.assertEqual(nas_row["ce_top_titles"], ["Rigveda RV 10.129"])

    def test_fail_when_ce_worse_overall(self):
        hybrid = _suite(
            [
                {"id": "nasadiya", "query": "Nasadiya", "ok": True, "top_titles": ["10.129"]},
                {"id": "isha-self", "query": "Isha", "ok": True, "top_titles": ["Isha"]},
            ]
        )
        ce = _suite(
            [
                {"id": "nasadiya", "query": "Nasadiya", "ok": True, "top_titles": ["10.129"]},
                {"id": "isha-self", "query": "Isha", "ok": False, "top_titles": ["Mahabharata"]},
            ]
        )
        comparison = compare_smoke_suites(hybrid, ce, model="x")
        self.assertFalse(comparison["promote"])
        self.assertIn("ce_worse_overall", comparison["fail_reasons"])
        self.assertNotIn("nasadiya_regressed", comparison["fail_reasons"])
        self.assertEqual(gate_exit_code(comparison), 1)

    def test_fail_when_nasadiya_regresses_while_hybrid_passes(self):
        hybrid = _suite(
            [
                {
                    "id": "nasadiya",
                    "query": "neither existence then Nasadiya",
                    "ok": True,
                    "top_titles": ["Rigveda RV 10.129"],
                },
                {"id": "gita-dharma", "query": "Gita", "ok": True, "top_titles": ["Gita"]},
            ]
        )
        ce = _suite(
            [
                {
                    "id": "nasadiya",
                    "query": "neither existence then Nasadiya",
                    "ok": False,
                    "top_titles": ["Rigveda RV 10.125", "Rigveda RV 10.5"],
                },
                {"id": "gita-dharma", "query": "Gita", "ok": True, "top_titles": ["Gita"]},
            ]
        )
        comparison = compare_smoke_suites(hybrid, ce, model="ms-marco")
        self.assertFalse(comparison["promote"])
        self.assertIn("nasadiya_failed", comparison["fail_reasons"])
        self.assertIn("nasadiya_regressed", comparison["fail_reasons"])
        self.assertIn("ce_worse_overall", comparison["fail_reasons"])
        self.assertEqual(comparison["nasadiya"]["ce_top_titles"][0], "Rigveda RV 10.125")
        self.assertTrue(find_nasadiya_row(ce["results"]))

    def test_equal_rates_still_fail_if_ce_misses_nasadiya(self):
        rows_fail = [
            {
                "id": "nasadiya",
                "query": "Nasadiya",
                "ok": False,
                "top_titles": ["Rigveda RV 10.125"],
            },
            {"id": "isha-self", "query": "Isha", "ok": True, "top_titles": ["Isha"]},
        ]
        comparison = compare_smoke_suites(_suite(rows_fail), _suite(rows_fail), model="domain-v3")
        self.assertFalse(comparison["promote"])
        self.assertIn("nasadiya_failed", comparison["fail_reasons"])
        self.assertNotIn("ce_worse_overall", comparison["fail_reasons"])
        self.assertEqual(gate_exit_code(comparison), 1)

    def test_ci_gold_without_nasadiya_only_checks_pass_rate(self):
        rows = [
            {"id": "isha-self", "query": "Isha", "ok": True, "top_titles": ["Isha"]},
            {"id": "gita-dharma", "query": "Gita", "ok": True, "top_titles": ["Gita"]},
        ]
        comparison = compare_smoke_suites(_suite(rows), _suite(rows), model="artifacts/reranker")
        self.assertTrue(comparison["promote"])
        self.assertIsNone(comparison["nasadiya"])


if __name__ == "__main__":
    unittest.main()
