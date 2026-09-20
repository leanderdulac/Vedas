"""Schema do gold expandido de smoke/eval (`fixtures/smoke_queries.json`)."""

from __future__ import annotations

import unittest

from vedic_pipeline.common.constants import PROJECT_ROOT
from vedic_pipeline.search.hybrid import extract_query_hymn_ids
from vedic_pipeline.train.rerank_pairs import load_gold_queries

GOLD_PATH = PROJECT_ROOT / "fixtures" / "smoke_queries.json"
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


class SmokeGoldSchemaTests(unittest.TestCase):
    def test_expanded_gold_has_unique_ids_and_schema(self):
        queries = load_gold_queries(GOLD_PATH)
        self.assertGreaterEqual(len(queries), 19)
        ids = [str(q.get("id") or "") for q in queries]
        self.assertTrue(all(ids))
        self.assertEqual(len(ids), len(set(ids)), "gold ids must be unique")
        idset = set(ids)
        self.assertTrue(set(STABLE_IDS).issubset(idset), "original 10 ids must stay")
        self.assertTrue(set(REQUIRED_NEW_IDS).issubset(idset))
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

    def test_locator_and_devanagari_queries(self):
        by_id = {q["id"]: q for q in load_gold_queries(GOLD_PATH)}
        self.assertIn("2.47", extract_query_hymn_ids(by_id["gita-2-47"]["query"]))
        isha = by_id["ishavasya-sa"]["query"]
        self.assertTrue(any("\u0900" <= ch <= "\u097f" for ch in isha))
        self.assertIn("Isha", isha)


if __name__ == "__main__":
    unittest.main()
