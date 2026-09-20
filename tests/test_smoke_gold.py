"""Schema do gold expandido de smoke/eval (`fixtures/smoke_queries.json`)."""

from __future__ import annotations

import json
import unittest

from vedic_pipeline.common.constants import PROJECT_ROOT
from vedic_pipeline.search.hybrid import extract_query_hymn_ids
from vedic_pipeline.train.rerank_pairs import load_gold_queries

GOLD_PATH = PROJECT_ROOT / "fixtures" / "smoke_queries.json"
CI_PATH = PROJECT_ROOT / "fixtures" / "smoke_queries_ci.json"
EXPECTED_GOLD_COUNT = 37
STABLE_IDS = (
    "isha-self",
    "atman-brahman",
    "gita-dharma",
    "nasadiya",
    "purusha-sukta",
    "agni-rv1",
    "yoga-citta",
    "rama-exile",
    "mahabharata-war",
    "manu-dharma",
)
REQUIRED_NEW_IDS = (
    "katha-nachiketas",
    "gita-2-47",
    "yoga-1-2",
    "rama-sita-abduction",
    "bhishma-arrows",
    "ishavasya-sa",
    "shvetashvatara-rudra",
    "prasna-six",
    "mundaka-two-birds",
)
REQUIRED_BEYOND_IDS = (
    "wrong-nasadiya-10-125",
    "wrong-purusha-10-9",
    "rv-10-5-exact",
    "gayatri-mantra",
    "hiranyagarbha",
    "vak-sukta",
    "nasadiya-devanagari-only",
    "gayatri-devanagari",
    "isha-iast-vs-dev",
    "brihad-neti",
    "katha-vs-kaushitaki",
    "mandukya-om",
    "samaveda-melody",
    "yajur-shukla",
    "pt-nasadiya",
    "pt-gita-acao",
    "pt-yoga-definicao",
    "sita-abduction-wrong-epic",
)


class SmokeGoldSchemaTests(unittest.TestCase):
    def test_expanded_gold_has_unique_ids_and_schema(self):
        queries = load_gold_queries(GOLD_PATH)
        self.assertEqual(len(queries), EXPECTED_GOLD_COUNT)
        ids = [str(q.get("id") or "") for q in queries]
        self.assertTrue(all(ids))
        self.assertEqual(len(ids), len(set(ids)), "gold ids must be unique")
        idset = set(ids)
        self.assertTrue(set(STABLE_IDS).issubset(idset), "original 10 ids must stay")
        self.assertTrue(set(REQUIRED_NEW_IDS).issubset(idset))
        self.assertTrue(set(REQUIRED_BEYOND_IDS).issubset(idset))
        self.assertEqual(
            ids[:19],
            list(STABLE_IDS + REQUIRED_NEW_IDS),
            "first 19 gold ids must stay append-only",
        )
        self.assertEqual(ids[19:], list(REQUIRED_BEYOND_IDS))
        for query in queries:
            self.assertTrue(str(query.get("query") or "").strip())
            self.assertGreaterEqual(int(query.get("top_k") or 0), 1)
            expect = list(query.get("expect_title_any") or [])
            self.assertTrue(expect, f"{query.get('id')} missing expect_title_any")
            self.assertGreaterEqual(int(query.get("min_hits") or 1), 1)
            self.assertFalse(
                any(p.strip().lower() == "rig" for p in expect),
                f"{query.get('id')} must not use loose expect token 'Rig'",
            )

    def test_gold_meta_count_matches_queries(self):
        payload = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["meta"]["count"], EXPECTED_GOLD_COUNT)
        self.assertEqual(payload["meta"]["version"], 4)
        self.assertEqual(len(payload["queries"]), EXPECTED_GOLD_COUNT)

    def test_ci_gold_is_fast_subset(self):
        ci = load_gold_queries(CI_PATH)
        gold = load_gold_queries(GOLD_PATH)
        self.assertLess(len(ci), len(gold))
        self.assertEqual(len(ci), 4)
        self.assertLessEqual(len(ci), 8, "CI gold must stay a small recorte")

    def test_locator_and_devanagari_queries(self):
        by_id = {q["id"]: q for q in load_gold_queries(GOLD_PATH)}
        self.assertIn("2.47", extract_query_hymn_ids(by_id["gita-2-47"]["query"]))
        isha = by_id["ishavasya-sa"]["query"]
        self.assertTrue(any("\u0900" <= ch <= "\u097f" for ch in isha))
        self.assertIn("Isha", isha)
        nasadiya_deva = by_id["nasadiya-devanagari-only"]["query"]
        self.assertTrue(any("\u0900" <= ch <= "\u097f" for ch in nasadiya_deva))
        self.assertIn("3.62", extract_query_hymn_ids(by_id["gayatri-mantra"]["query"]))
        self.assertIn("10.129", extract_query_hymn_ids(by_id["wrong-nasadiya-10-125"]["query"]))

    def test_reranker_stays_default_off(self):
        example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn("VEDIC_ENABLE_RERANKER=false", example)
        self.assertNotRegex(example, r"(?m)^VEDIC_ENABLE_RERANKER=true")
        self.assertIn("37/37", example)
        self.assertNotIn("re-rodar o gate", example)
        source = (
            PROJECT_ROOT / "vedic_pipeline" / "search" / "reranker.py"
        ).read_text(encoding="utf-8")
        self.assertIn('os.environ.get("VEDIC_ENABLE_RERANKER", "false")', source)

    def test_promote_gate_docs_record_gold37(self):
        gate = (PROJECT_ROOT / "docs" / "ce_promote_gate.md").read_text(encoding="utf-8")
        decision = (PROJECT_ROOT / "docs" / "reranker_decision.md").read_text(encoding="utf-8")
        self.assertIn("37/37", gate)
        self.assertIn("PROMOTE", gate)
        self.assertIn("VEDIC_ENABLE_RERANKER=true", gate)
        self.assertIn("VEDIC_RERANKER_MODEL=artifacts/reranker_domain_v5", gate)
        self.assertIn("reranker_domain_v4", gate)
        self.assertNotIn("re-rodar o gate", gate)
        self.assertIn("37/37", decision)
        self.assertIn("PROMOTE", decision)
        self.assertIn("reranker_domain_v5", decision)
        self.assertIn("reranker_domain_v4", decision)
        self.assertNotIn("re-rodar o gate", decision)


if __name__ == "__main__":
    unittest.main()
