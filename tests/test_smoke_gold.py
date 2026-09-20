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
    "gayatri-rv362",
    "asya-vamasya",
    "chandogya-tattvamasi",
    "brihadaranyaka-neti",
    "katha-nachiketa",
    "mandukya-om",
    "gita-247",
    "yoga-12-paraphrase",
    "rama-sita-abduction",
    "bhishma-arrows",
    "nasadiya-deva",
    "isha-opening-deva",
)


class SmokeGoldSchemaTests(unittest.TestCase):
    def test_expanded_gold_has_unique_ids_and_schema(self):
        queries = load_gold_queries(GOLD_PATH)
        self.assertGreaterEqual(len(queries), 20)
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

    def test_locator_queries_expose_hymn_ids(self):
        by_id = {q["id"]: q for q in load_gold_queries(GOLD_PATH)}
        self.assertIn("10.129", extract_query_hymn_ids(by_id["nasadiya-deva"]["query"]))
        self.assertIn("3.62", extract_query_hymn_ids(by_id["gayatri-rv362"]["query"]))
        self.assertIn("1.164", extract_query_hymn_ids(by_id["asya-vamasya"]["query"]))
        self.assertIn("2.47", extract_query_hymn_ids(by_id["gita-247"]["query"]))
        self.assertTrue(any("\u0900" <= ch <= "\u097f" for ch in by_id["isha-opening-deva"]["query"]))


if __name__ == "__main__":
    unittest.main()
