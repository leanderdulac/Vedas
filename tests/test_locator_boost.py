"""Boost de locator/hino: Nasadiya 10.129, Purusha 10.90, RV X.Y explícito."""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from vedic_pipeline.search.hybrid import (
    apply_locator_hymn_boost,
    extract_query_hymn_ids,
    hybrid_rerank,
    hymn_id_in_blob,
)
from vedic_pipeline.search.reranker import invalidate_reranker


class HymnIdExtractTests(unittest.TestCase):
    def test_nasadiya_maps_to_10_129(self):
        self.assertEqual(
            extract_query_hymn_ids(
                "In the beginning there was neither non-existence nor existence then Nasadiya"
            ),
            ["10.129"],
        )
        self.assertEqual(extract_query_hymn_ids("nāsadīya sūkta"), ["10.129"])
        self.assertEqual(
            extract_query_hymn_ids("नासदासीन् नो सदासीत् तदानीं"),
            ["10.129"],
        )
        self.assertEqual(
            extract_query_hymn_ids("नासदासीन् नो सदासीत् तदानीं 10.129"),
            ["10.129"],
        )

    def test_purusha_sukta_maps_to_10_90(self):
        self.assertEqual(
            extract_query_hymn_ids("Purusha Sukta cosmic person sacrifice"),
            ["10.90"],
        )

    def test_explicit_rv_and_hymn_numbers(self):
        self.assertIn("10.129", extract_query_hymn_ids("Rig Veda RV 10.129 creation"))
        self.assertEqual(
            extract_query_hymn_ids("Agni priest of the sacrifice Rig Veda hymn 1.1"),
            ["1.1"],
        )

    def test_isha_has_no_hymn_id(self):
        self.assertEqual(
            extract_query_hymn_ids("What is the Self according to the Isha Upanishad?"),
            [],
        )

    def test_hymn_10_5_does_not_match_10_50(self):
        self.assertTrue(hymn_id_in_blob("Rigveda RV 10.5 (Griffith)", "10.5"))
        self.assertFalse(hymn_id_in_blob("Rigveda RV 10.50 (Griffith)", "10.5"))
        self.assertFalse(hymn_id_in_blob("Rigveda RV 10.125 (Griffith)", "10.5"))


class LocatorHymnBoostTests(unittest.TestCase):
    def test_nasadiya_title_beats_generic_ce_decoys(self):
        hits = hybrid_rerank(
            "In the beginning there was neither non-existence nor existence then Nasadiya",
            [
                {
                    "chunk_id": "125",
                    "doc_id": "rv-125",
                    "title": "Rigveda RV 10.125 (Griffith)",
                    "locator": "RV 10.125",
                    "text": "I am the queen, the gatherer-up of treasures.",
                    "score": 0.95,
                },
                {
                    "chunk_id": "5",
                    "doc_id": "rv-5",
                    "title": "Rigveda RV 10.5 (Griffith)",
                    "locator": "RV 10.5",
                    "text": "The child of two mothers, of two fathers.",
                    "score": 0.90,
                },
                {
                    "chunk_id": "129",
                    "doc_id": "rv-129",
                    "title": "Rigveda RV 10.129 Nasadiya (Griffith)",
                    "locator": "RV 10.129",
                    "text": "Then was not non-existent nor existent.",
                    "score": 0.35,
                },
            ],
            top_k=3,
            max_per_doc=2,
            use_cross_encoder=False,
        )
        self.assertEqual(hits[0]["chunk_id"], "129")
        self.assertGreater(hits[0].get("_hymn_boost") or 0, 0)

    def test_roman_x_129_in_body_of_selected_hymns(self):
        pool = [
            {
                "chunk_id": "sel",
                "title": "Rig Veda selected hymns",
                "locator": "",
                "text": "HYMN X.129 — Nasadiya Sukta (Creation Hymn)\nThen was not non-existent.",
                "score": 0.2,
            },
            {
                "chunk_id": "125",
                "title": "Rigveda RV 10.125",
                "locator": "RV 10.125",
                "text": "I am the queen",
                "score": 0.9,
            },
        ]
        apply_locator_hymn_boost("Nasadiya creation hymn", pool)
        by_id = {c["chunk_id"]: c for c in pool}
        self.assertGreater(by_id["sel"]["score"], by_id["125"]["score"])

    def test_boost_after_ce_restores_nasadiya(self):
        mock_ce = MagicMock()

        def predict(pairs):
            scores = []
            for _query, text in pairs:
                blob = str(text)
                if "queen" in blob:
                    scores.append(6.0)
                elif "two mothers" in blob:
                    scores.append(5.0)
                else:
                    scores.append(-4.0)
            return scores

        mock_ce.predict.side_effect = predict
        invalidate_reranker()
        with (
            patch.dict(os.environ, {"VEDIC_ENABLE_RERANKER": "true", "VEDIC_RERANKER_MODEL": "mock"}),
            patch("vedic_pipeline.search.reranker.load_cross_encoder", return_value=mock_ce),
        ):
            invalidate_reranker()
            hits = hybrid_rerank(
                "In the beginning there was neither non-existence nor existence then Nasadiya",
                [
                    {
                        "chunk_id": "125",
                        "doc_id": "rv-125",
                        "title": "Rigveda RV 10.125 (Griffith)",
                        "locator": "RV 10.125",
                        "text": "I am the queen, the gatherer-up of treasures.",
                        "score": 0.95,
                    },
                    {
                        "chunk_id": "5",
                        "doc_id": "rv-5",
                        "title": "Rigveda RV 10.5 (Griffith)",
                        "locator": "RV 10.5",
                        "text": "The child of two mothers, of two fathers.",
                        "score": 0.90,
                    },
                    {
                        "chunk_id": "129",
                        "doc_id": "rv-129",
                        "title": "Rigveda RV 10.129 Nasadiya (Griffith)",
                        "locator": "RV 10.129",
                        "text": "Then was not non-existent nor existent.",
                        "score": 0.35,
                    },
                ],
                top_k=3,
                max_per_doc=2,
                use_cross_encoder=True,
            )
        invalidate_reranker()
        self.assertEqual(hits[0]["chunk_id"], "129")
        mock_ce.predict.assert_called()

    def test_devanagari_nasadiya_boosts_10_129(self):
        hits = hybrid_rerank(
            "नासदासीन् नो सदासीत् तदानीं 10.129",
            [
                {
                    "chunk_id": "125",
                    "doc_id": "rv-125",
                    "title": "Rigveda RV 10.125 (Griffith)",
                    "locator": "RV 10.125",
                    "text": "I am the queen, the gatherer-up of treasures.",
                    "score": 0.95,
                },
                {
                    "chunk_id": "129",
                    "doc_id": "rv-129",
                    "title": "Rigveda RV 10.129 Nasadiya (Griffith)",
                    "locator": "RV 10.129",
                    "text": "Then was not non-existent nor existent.",
                    "score": 0.35,
                },
            ],
            top_k=2,
            use_cross_encoder=False,
        )
        self.assertEqual(hits[0]["chunk_id"], "129")
        self.assertGreater(hits[0].get("_hymn_boost") or 0, 0)

    def test_purusha_sukta_boosts_10_90(self):
        hits = hybrid_rerank(
            "Purusha Sukta cosmic person sacrifice",
            [
                {
                    "chunk_id": "noise",
                    "doc_id": "rv-other",
                    "title": "Rigveda RV 9.22",
                    "locator": "RV 9.22",
                    "text": "soma juice",
                    "score": 0.88,
                },
                {
                    "chunk_id": "90",
                    "doc_id": "rv-90",
                    "title": "Rigveda RV 10.90 Purusha (Griffith)",
                    "locator": "RV 10.90",
                    "text": "A thousand heads hath Purusha",
                    "score": 0.40,
                },
            ],
            top_k=2,
            use_cross_encoder=False,
        )
        self.assertEqual(hits[0]["chunk_id"], "90")


if __name__ == "__main__":
    unittest.main()
