"""Boost de locator/hino: nomes canônicos (Nasadiya, Gāyatrī, …) e RV X.Y."""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from vedic_pipeline.search.hybrid import (
    LOCATOR_HYMN_INJECT_PER_ID,
    apply_locator_hymn_boost,
    chunk_matches_hymn,
    collect_locator_hymn_injections,
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

    def test_gayatri_maps_to_3_62(self):
        self.assertEqual(extract_query_hymn_ids("Gayatri mantra Savitr"), ["3.62"])
        self.assertEqual(
            extract_query_hymn_ids("Gayatri mantra Savitr Rig Veda"),
            ["3.62"],
        )
        self.assertEqual(extract_query_hymn_ids("gāyatrī sūkta"), ["3.62"])
        self.assertEqual(extract_query_hymn_ids("गायत्री मन्त्र"), ["3.62"])

    def test_gayatri_verse_devanagari_and_iast_extract_3_62(self):
        """Devanāgarī/IAST pāda (no word 'Gayatri') still resolves to RV 3.62."""
        self.assertEqual(
            extract_query_hymn_ids("तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि"),
            ["3.62"],
        )
        self.assertEqual(extract_query_hymn_ids("तत् सवितुर् वरेण्यं"), ["3.62"])
        self.assertEqual(
            extract_query_hymn_ids("tat savitur vareṇyaṃ bhargo devasya dhīmahi"),
            ["3.62"],
        )
        self.assertEqual(extract_query_hymn_ids("tatsavitur varenyam"), ["3.62"])
        self.assertEqual(extract_query_hymn_ids("tat-savitur varenyam"), ["3.62"])

    def test_hiranyagarbha_maps_to_10_121(self):
        self.assertEqual(extract_query_hymn_ids("Hiranyagarbha golden womb"), ["10.121"])
        self.assertEqual(extract_query_hymn_ids("hiraṇyagarbha sūkta"), ["10.121"])
        self.assertEqual(extract_query_hymn_ids("हिरण्यगर्भः समवर्तताग्रे"), ["10.121"])

    def test_vak_sukta_maps_to_10_125(self):
        self.assertEqual(extract_query_hymn_ids("Vak Sukta hymn of speech"), ["10.125"])
        self.assertEqual(extract_query_hymn_ids("Vāc Sūkta"), ["10.125"])
        self.assertEqual(extract_query_hymn_ids("वाक् सूक्त"), ["10.125"])

    def test_explicit_rv_and_hymn_numbers(self):
        self.assertIn("10.129", extract_query_hymn_ids("Rig Veda RV 10.129 creation"))
        self.assertEqual(
            extract_query_hymn_ids("Agni priest of the sacrifice Rig Veda hymn 1.1"),
            ["1.1"],
        )
        self.assertEqual(extract_query_hymn_ids("Rigveda RV 10.125"), ["10.125"])

    def test_named_sukta_beats_conflicting_explicit_id(self):
        self.assertEqual(
            extract_query_hymn_ids("Nasadiya creation hymn RV 10.125"),
            ["10.129"],
        )
        self.assertNotIn(
            "10.125",
            extract_query_hymn_ids("Nasadiya Sukta 10.125"),
        )
        self.assertEqual(
            extract_query_hymn_ids("Purusha Sukta 10.9"),
            ["10.90"],
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
        self.assertEqual(extract_query_hymn_ids("Rigveda 10.5"), ["10.5"])
        self.assertEqual(extract_query_hymn_ids("Rigveda 10.50"), ["10.50"])


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

    def test_nasadiya_wrong_number_boosts_10_129_not_10_125(self):
        hits = hybrid_rerank(
            "Nasadiya creation hymn RV 10.125",
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
        decoy = next(h for h in hits if h["chunk_id"] == "125")
        self.assertIsNone(decoy.get("_hymn_match"))

    def test_gayatri_boosts_3_62(self):
        hits = hybrid_rerank(
            "Gayatri mantra Savitr",
            [
                {
                    "chunk_id": "noise",
                    "doc_id": "rv-other",
                    "title": "Rigveda RV 10.125 (Griffith)",
                    "locator": "RV 10.125",
                    "text": "I am the queen",
                    "score": 0.88,
                },
                {
                    "chunk_id": "62",
                    "doc_id": "rv-362",
                    "title": "Rigveda RV 3.62 (Griffith)",
                    "locator": "RV 3.62",
                    "text": "May we attain that excellent glory of Savitar the God",
                    "score": 0.40,
                },
            ],
            top_k=2,
            use_cross_encoder=False,
        )
        self.assertEqual(hits[0]["chunk_id"], "62")

    def test_gayatri_does_not_boost_chandogya_iii_12(self):
        """Mac beyond-gold: gayatri-mantra top-1 was Chandogya III.12, not RV 3.62."""
        hits = hybrid_rerank(
            "Gayatri mantra Savitr",
            [
                {
                    "chunk_id": "ch-iii12",
                    "doc_id": "chandogya",
                    "title": "Chandogya Upanishad III.12 (Müller, SBE01, sacred-texts)",
                    "locator": "Chandogya III.12",
                    "text": "Gayatri is everything whatsoever here exists. Gayatri is speech.",
                    "score": 0.95,
                },
                {
                    "chunk_id": "62",
                    "doc_id": "rv-362",
                    "title": "Rigveda RV 3.62 (Griffith)",
                    "locator": "RV 3.62",
                    "text": "May we attain that excellent glory of Savitar the God",
                    "score": 0.35,
                },
            ],
            top_k=2,
            use_cross_encoder=False,
        )
        self.assertEqual(hits[0]["chunk_id"], "62")
        self.assertGreater(hits[0].get("_hymn_boost") or 0, 0)
        decoy = next(h for h in hits if h["chunk_id"] == "ch-iii12")
        self.assertIsNone(decoy.get("_hymn_match"))

    def test_hiranyagarbha_beats_generic_selected_hymns(self):
        """Mac beyond-gold: hiranyagarbha top-1 was a generic RV anthology, not 10.121."""
        hits = hybrid_rerank(
            "Hiranyagarbha golden womb",
            [
                {
                    "chunk_id": "sel",
                    "doc_id": "rv-selected",
                    "title": "Rig Veda selected hymns",
                    "locator": "",
                    "text": "A compilation of famous suktas including Hiranyagarbha in passing.",
                    "score": 0.95,
                },
                {
                    "chunk_id": "121",
                    "doc_id": "rv-121",
                    "title": "Rigveda RV 10.121 (Griffith)",
                    "locator": "RV 10.121",
                    "text": "In the beginning rose Hiranyagarbha, born Only Lord of all created beings.",
                    "score": 0.35,
                },
            ],
            top_k=2,
            use_cross_encoder=False,
        )
        self.assertEqual(hits[0]["chunk_id"], "121")
        self.assertGreater(hits[0].get("_hymn_boost") or 0, 0)
        decoy = next(h for h in hits if h["chunk_id"] == "sel")
        self.assertIsNone(decoy.get("_hymn_match"))

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


class LocatorHymnRecallInjectionTests(unittest.TestCase):
    """Boost (PR #5) cannot help if the hymn never entered the candidate pool."""

    def test_gayatri_english_injects_3_62_when_missing_from_hits(self):
        chandogya = {
            "chunk_id": "ch-iii12",
            "doc_id": "chandogya",
            "title": "Chandogya Upanishad III.12 (Müller, SBE01, sacred-texts)",
            "locator": "Chandogya III.12",
            "text": "Gayatri is everything whatsoever here exists. Gayatri mantra Savitr.",
            "score": 0.95,
        }
        anthology = {
            "chunk_id": "sel",
            "doc_id": "rv-selected",
            "title": "Rig Veda selected hymns",
            "locator": "",
            "text": "An anthology of famous Gayatri mantra Savitr passages.",
            "score": 0.90,
        }
        # No lexical overlap with the query except the RV id in title/locator.
        rv362 = {
            "chunk_id": "62",
            "doc_id": "rv-362",
            "title": "Griffith 3.62",
            "locator": "RV 3.62.10",
            "text": "May we attain that excellent glory of the God.",
            "score": 0.10,
        }
        filler = {
            "chunk_id": "1",
            "doc_id": "rv-1",
            "title": "Griffith 1.1",
            "locator": "RV 1.1.1",
            "text": "I laud the priest, the household priest.",
            "score": 0.20,
        }
        hits = hybrid_rerank(
            "Gayatri mantra Savitr Rig Veda",
            [chandogya, anthology],
            all_chunks=[chandogya, anthology, filler, rv362],
            top_k=8,
            max_per_doc=2,
            use_cross_encoder=False,
        )
        self.assertTrue(
            any(chunk_matches_hymn(h, "3.62", text_too=False) for h in hits),
            f"RV 3.62 missing after injection; ids={[h['chunk_id'] for h in hits]}",
        )
        matched = next(h for h in hits if chunk_matches_hymn(h, "3.62", text_too=False))
        self.assertEqual(matched["chunk_id"], "62")
        self.assertGreater(matched.get("_hymn_boost") or 0, 0)

    def test_injection_skips_chandogya_name_only(self):
        chandogya = {
            "chunk_id": "ch-iii12",
            "doc_id": "chandogya",
            "title": "Chandogya Upanishad III.12 (Müller, SBE01, sacred-texts)",
            "locator": "Chandogya III.12",
            "text": "Gayatri is everything whatsoever here exists. Gayatri is speech.",
            "score": 0.95,
        }
        noise = {
            "chunk_id": "noise",
            "doc_id": "rv-other",
            "title": "Rigveda RV 10.125 (Griffith)",
            "locator": "RV 10.125",
            "text": "I am the queen",
            "score": 0.40,
        }
        injected = collect_locator_hymn_injections(
            "Gayatri mantra Savitr",
            [chandogya, noise],
        )
        self.assertEqual(injected, [])
        hits = hybrid_rerank(
            "Gayatri mantra Savitr",
            [chandogya, noise],
            all_chunks=[chandogya, noise],
            top_k=2,
            use_cross_encoder=False,
        )
        decoy = next(h for h in hits if h["chunk_id"] == "ch-iii12")
        self.assertIsNone(decoy.get("_hymn_match"))
        self.assertFalse(decoy.get("_locator_injected"))

    def test_injection_caps_chunks_per_id(self):
        chunks = [
            {
                "chunk_id": f"c{i}",
                "doc_id": f"doc-{i}",
                "title": f"Griffith 3.62.{i}",
                "locator": f"RV 3.62.{i}",
                "text": "verse body",
            }
            for i in range(12)
        ]
        injected = collect_locator_hymn_injections("Gayatri mantra", chunks)
        self.assertEqual(len(injected), LOCATOR_HYMN_INJECT_PER_ID)
        self.assertTrue(all(chunk_matches_hymn(c, "3.62", text_too=False) for c in injected))

    def test_hiranyagarbha_injects_10_121_when_missing_from_hits(self):
        anthology = {
            "chunk_id": "sel",
            "doc_id": "rv-selected",
            "title": "Rig Veda selected hymns",
            "locator": "",
            "text": "A compilation of famous suktas including Hiranyagarbha in passing.",
            "score": 0.95,
        }
        rv121 = {
            "chunk_id": "121",
            "doc_id": "rv-121",
            "title": "Griffith 10.121",
            "locator": "RV 10.121.1",
            "text": "In the beginning rose the embryo, born Only Lord of all created beings.",
            "score": 0.10,
        }
        hits = hybrid_rerank(
            "Hiranyagarbha golden womb",
            [anthology],
            all_chunks=[anthology, rv121],
            top_k=8,
            use_cross_encoder=False,
        )
        self.assertTrue(any(h["chunk_id"] == "121" for h in hits))
        matched = next(h for h in hits if h["chunk_id"] == "121")
        self.assertGreater(matched.get("_hymn_boost") or 0, 0)


if __name__ == "__main__":
    unittest.main()
