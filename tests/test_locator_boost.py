"""Boost de locator/hino: nomes canônicos (Nasadiya, Gāyatrī, …) e RV X.Y."""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from vedic_pipeline.search.hybrid import (
    LOCATOR_INJECT_PER_HYMN,
    LOCATOR_INJECT_PER_WORK,
    WORK_ISHA,
    WORK_KATHA,
    WORK_SAMAVEDA,
    WORK_YAJUR_VS,
    apply_locator_hymn_boost,
    apply_locator_work_boost,
    extract_query_hymn_ids,
    extract_query_work_keys,
    hybrid_rerank,
    hymn_id_in_blob,
    locator_hymn_injections,
    locator_work_injections,
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
        self.assertEqual(extract_query_hymn_ids("gāyatrī sūkta"), ["3.62"])
        self.assertEqual(extract_query_hymn_ids("गायत्री मन्त्र"), ["3.62"])

    def test_gayatri_verse_fingerprints_map_to_3_62(self):
        self.assertEqual(
            extract_query_hymn_ids("तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि"),
            ["3.62"],
        )
        self.assertEqual(
            extract_query_hymn_ids("tat savitur vareṇyaṃ bhargo devasya dhīmahi"),
            ["3.62"],
        )
        self.assertEqual(
            extract_query_hymn_ids("tatsaviturvarenyam bhargo devasya dhimahi"),
            ["3.62"],
        )

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
    def test_gayatri_injects_3_62_absent_from_semantic_and_lexical(self):
        """Boost cannot help if 3.62 never enters the pool (Mac live fail)."""
        chandogya = {
            "chunk_id": "inject-ch-iii12",
            "doc_id": "chandogya-inject",
            "title": "Chandogya Upanishad III.12 (Müller, SBE01, sacred-texts)",
            "locator": "Chandogya III.12",
            "text": "Gayatri is everything whatsoever here exists. Gayatri is speech.",
            "score": 0.95,
        }
        anthology = {
            "chunk_id": "inject-anth",
            "doc_id": "anthology-inject",
            "title": "Vedic anthologies on the Gayatri metre",
            "locator": "",
            "text": "Many Upanishads discuss the Gayatri as a metre, not RV 10.90.",
            "score": 0.90,
        }
        # Unique chunk_id: token cache is process-global and other tests reuse "62".
        rv362 = {
            "chunk_id": "inject-rv-3-62",
            "doc_id": "rv-362-inject",
            "title": "Mandala III Griffith 3.62",
            "locator": "RV 3.62.10",
            "text": "May we attain that excellent glory of the God",
            "score": 0.01,
        }
        hits = hybrid_rerank(
            "Gayatri mantra Savitr Rig Veda",
            [chandogya, anthology],
            all_chunks=[chandogya, anthology, rv362],
            top_k=3,
            use_cross_encoder=False,
        )
        ids = [h["chunk_id"] for h in hits]
        self.assertIn("inject-rv-3-62", ids)
        injected = next(h for h in hits if h["chunk_id"] == "inject-rv-3-62")
        self.assertEqual(injected.get("_hymn_injected"), "3.62")
        self.assertGreater(injected.get("_hymn_boost") or 0, 0)
        self.assertEqual(hits[0]["chunk_id"], "inject-rv-3-62")
        decoy = next(h for h in hits if h["chunk_id"] == "inject-ch-iii12")
        self.assertIsNone(decoy.get("_hymn_match"))
        self.assertIsNone(decoy.get("_hymn_injected"))

    def test_injection_skips_name_only_chandogya_and_caps_per_id(self):
        decoys = [
            {
                "chunk_id": f"ch-{i}",
                "title": "Chandogya Upanishad III.12 Gayatri commentary",
                "locator": "Chandogya III.12",
                "text": "Gayatri is speech",
            }
            for i in range(6)
        ]
        verses = [
            {
                "chunk_id": f"v-{n}",
                "title": f"Rigveda RV 3.62.{n} (Griffith, sacred-texts)",
                "locator": f"RV 3.62.{n}",
                "text": "verse body",
            }
            for n in range(1, 13)
        ]
        hymn_level = {
            "chunk_id": "hymn",
            "title": "Rigveda RV 3.62 (Griffith, sacred-texts)",
            "locator": "RV 3.62",
            "text": "whole hymn",
        }
        injected = locator_hymn_injections(
            "Gayatri mantra Savitr Rig Veda",
            decoys + verses + [hymn_level],
        )
        self.assertEqual(len(injected), LOCATOR_INJECT_PER_HYMN)
        self.assertTrue(all(c.get("_hymn_injected") == "3.62" for c in injected))
        self.assertEqual(injected[0]["chunk_id"], "hymn")
        self.assertFalse(any(str(c["chunk_id"]).startswith("ch-") for c in injected))

    def test_devanagari_verse_injection_uses_extracted_3_62(self):
        rv362 = {
            "chunk_id": "62",
            "title": "Rigveda RV 3.62 (Griffith, sacred-texts)",
            "locator": "RV 3.62",
            "text": "May we attain that excellent glory of Savitar the God",
        }
        injected = locator_hymn_injections(
            "तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि",
            [rv362],
        )
        self.assertEqual([c["chunk_id"] for c in injected], ["62"])


class NamedWorkExtractTests(unittest.TestCase):
    def test_isha_opening_fingerprint_maps_to_isha(self):
        self.assertEqual(
            extract_query_work_keys(
                "īśāvāsyam idaṃ sarvaṃ yat kiñca jagatyāṃ jagat"
            ),
            [WORK_ISHA],
        )
        self.assertEqual(
            extract_query_work_keys("ईशावास्यमिदं सर्वं यत्किञ्च जगत्यां जगत्"),
            [WORK_ISHA],
        )
        self.assertEqual(
            extract_query_work_keys("isavasya idam sarvam yat kinca jagatyam jagat"),
            [WORK_ISHA],
        )

    def test_isha_name_maps_to_isha_but_not_a_hymn_id(self):
        self.assertEqual(
            extract_query_work_keys("What is the Self according to the Isha Upanishad?"),
            [WORK_ISHA],
        )
        self.assertEqual(
            extract_query_hymn_ids("īśāvāsyam idaṃ sarvaṃ yat kiñca jagatyāṃ jagat"),
            [],
        )

    def test_isha_shared_sanskrit_words_are_not_enough(self):
        self.assertEqual(extract_query_work_keys("idam sarvam jagat in the Rigveda"), [])

    def test_nachiketas_maps_to_katha_even_with_kaushitaki_label(self):
        self.assertEqual(
            extract_query_work_keys("Nachiketas meets Death Kaushitaki Upanishad"),
            [WORK_KATHA],
        )
        self.assertEqual(
            extract_query_work_keys("नचिकेतस् and Yama"),
            [WORK_KATHA],
        )
        self.assertEqual(
            extract_query_work_keys("Kaushitaki Upanishad on prana"),
            [],
        )

    def test_samaveda_collection_signals(self):
        self.assertEqual(
            extract_query_work_keys("Sama Veda chant melody of Rig verses"),
            [WORK_SAMAVEDA],
        )
        self.assertEqual(extract_query_work_keys("Sāmaveda recitation"), [WORK_SAMAVEDA])
        self.assertEqual(extract_query_work_keys("सामवेद गान"), [WORK_SAMAVEDA])
        self.assertEqual(extract_query_work_keys("chant melody of Rig verses"), [])

    def test_shukla_yajur_vajasaneyi_signals(self):
        self.assertEqual(
            extract_query_work_keys("Shukla Yajur Veda Vajasaneyi Samhita"),
            [WORK_YAJUR_VS],
        )
        self.assertEqual(extract_query_work_keys("White Yajur Veda"), [WORK_YAJUR_VS])
        self.assertEqual(extract_query_work_keys("Yajurveda VS 40"), [WORK_YAJUR_VS])
        self.assertEqual(extract_query_work_keys("Yajur Veda in general"), [])


class NamedWorkLocatorTests(unittest.TestCase):
    def test_isha_fingerprint_beats_rv_10_58(self):
        """Mac beyond-gold: isha-iast-vs-dev top-1 was Rigveda RV 10.58."""
        hits = hybrid_rerank(
            "īśāvāsyam idaṃ sarvaṃ yat kiñca jagatyāṃ jagat",
            [
                {
                    "chunk_id": "rv-1058",
                    "doc_id": "rv-1058",
                    "title": "Rigveda RV 10.58 (Griffith)",
                    "locator": "RV 10.58",
                    "text": "Thy spirit, that went far away to Yama, to Vivasvan's son — "
                    "we cause to stay, that it may live and dwell here, idam sarvam jagat.",
                    "score": 0.95,
                },
                {
                    "chunk_id": "isha-1",
                    "doc_id": "isha",
                    "title": "Isha Upanishad (Müller)",
                    "locator": "Isha 1",
                    "text": "All this is for habitation by the Lord, whatsoever is individual "
                    "universe of movement in the universal motion.",
                    "score": 0.30,
                },
            ],
            top_k=2,
            use_cross_encoder=False,
        )
        self.assertEqual(hits[0]["chunk_id"], "isha-1")
        self.assertGreater(hits[0].get("_work_boost") or 0, 0)
        decoy = next(h for h in hits if h["chunk_id"] == "rv-1058")
        self.assertIsNone(decoy.get("_work_match"))
        self.assertIsNone(decoy.get("_hymn_match"))

    def test_isha_does_not_boost_rv_sharing_a_few_sanskrit_words(self):
        pool = [
            {
                "chunk_id": "rv-1058",
                "title": "Rigveda RV 10.58 (Griffith)",
                "locator": "RV 10.58",
                "text": "idam sarvam yat kinca jagatyam jagat",
                "score": 0.9,
            },
            {
                "chunk_id": "isha-1",
                "title": "Īśā Upanishad",
                "locator": "Isha 1",
                "text": "īśāvāsyam idaṃ sarvam",
                "score": 0.2,
            },
        ]
        apply_locator_work_boost("īśāvāsyam idaṃ sarvaṃ yat kiñca jagatyāṃ jagat", pool)
        by_id = {c["chunk_id"]: c for c in pool}
        self.assertGreater(by_id["isha-1"].get("_work_boost") or 0, 0)
        self.assertIsNone(by_id["rv-1058"].get("_work_match"))

    def test_nachiketas_beats_kaushitaki_label_and_chandogya(self):
        """Adversarial: story is Kaṭha; query wrongly names Kauṣītaki."""
        hits = hybrid_rerank(
            "Nachiketas meets Death Kaushitaki Upanishad",
            [
                {
                    "chunk_id": "ch-death",
                    "doc_id": "chandogya",
                    "title": "Chandogya Upanishad I.2 (Müller, SBE01, sacred-texts)",
                    "locator": "Chandogya I.2",
                    "text": "Death and the gods contend; the Upanishad names breath.",
                    "score": 0.95,
                },
                {
                    "chunk_id": "katha-1",
                    "doc_id": "katha",
                    "title": "Katha Upanishad — seção 1 (Müller, SBE15, sacred-texts)",
                    "locator": "Katha 1",
                    "text": "Nachiketas went to the house of Death and Yama granted three boons.",
                    "score": 0.28,
                },
            ],
            top_k=2,
            use_cross_encoder=False,
        )
        self.assertEqual(hits[0]["chunk_id"], "katha-1")
        self.assertEqual(hits[0].get("_work_match"), WORK_KATHA)
        decoy = next(h for h in hits if h["chunk_id"] == "ch-death")
        self.assertIsNone(decoy.get("_work_match"))

    def test_samaveda_title_beats_chandogya_i6(self):
        """Mac beyond-gold: samaveda-melody top-1 was Chandogya I.6."""
        hits = hybrid_rerank(
            "Sama Veda chant melody of Rig verses",
            [
                {
                    "chunk_id": "ch-i6",
                    "doc_id": "chandogya-i6",
                    "title": "Chandogya Upanishad I.6 (Müller, SBE01, sacred-texts)",
                    "locator": "Chandogya I.6",
                    "text": "This earth is the Rc, fire is the Saman. The Saman is sung on the Rc.",
                    "score": 0.96,
                },
                {
                    "chunk_id": "sv-1111",
                    "doc_id": "sv-1111",
                    "title": "Sāmaveda SV 1.1.1.1 (Griffith, DharmicData)",
                    "locator": "SV 1.1.1.1",
                    "text": "Come, Agni, praised with song, to feast and sacrificial offering.",
                    "score": 0.22,
                },
            ],
            top_k=2,
            use_cross_encoder=False,
        )
        self.assertEqual(hits[0]["chunk_id"], "sv-1111")
        self.assertEqual(hits[0].get("_work_match"), WORK_SAMAVEDA)
        decoy = next(h for h in hits if h["chunk_id"] == "ch-i6")
        self.assertIsNone(decoy.get("_work_match"))

    def test_yajur_vs_title_beats_muller_upanishads(self):
        """Mac beyond-gold: yajur-shukla top-1 was a generic Müller Upanishad."""
        hits = hybrid_rerank(
            "Shukla Yajur Veda Vajasaneyi Samhita",
            [
                {
                    "chunk_id": "muller-up",
                    "doc_id": "sbe-up",
                    "title": "The Upanishads (Müller, SBE01, sacred-texts)",
                    "locator": "",
                    "text": "The White Yajur Veda is mentioned among the Vedas in this anthology.",
                    "score": 0.94,
                },
                {
                    "chunk_id": "vs-1",
                    "doc_id": "yv-vs-1",
                    "title": "Yajurveda VS 1 (Sanskrit, DharmicData)",
                    "locator": "VS 1",
                    "text": "इषे त्वोर्जे त्वा वायव स्थ देवो वः सविता प्रार्पयतु श्रेष्ठतमाय कर्मणे",
                    "score": 0.20,
                },
            ],
            top_k=2,
            use_cross_encoder=False,
        )
        self.assertEqual(hits[0]["chunk_id"], "vs-1")
        self.assertEqual(hits[0].get("_work_match"), WORK_YAJUR_VS)
        decoy = next(h for h in hits if h["chunk_id"] == "muller-up")
        self.assertIsNone(decoy.get("_work_match"))


class NamedWorkRecallInjectionTests(unittest.TestCase):
    def test_isha_injects_title_chunk_absent_from_semantic(self):
        rv = {
            "chunk_id": "inject-rv-1058",
            "doc_id": "rv-1058-inject",
            "title": "Rigveda RV 10.58 (Griffith)",
            "locator": "RV 10.58",
            "text": "idam sarvam jagat far away to Yama",
            "score": 0.95,
        }
        isha = {
            "chunk_id": "inject-isha",
            "doc_id": "isha-inject",
            "title": "Isha Upanishad (English, fixture)",
            "locator": "Isha 1",
            "text": "All this is for habitation by the Lord",
            "score": 0.01,
        }
        hits = hybrid_rerank(
            "īśāvāsyam idaṃ sarvaṃ yat kiñca jagatyāṃ jagat",
            [rv],
            all_chunks=[rv, isha],
            top_k=2,
            use_cross_encoder=False,
        )
        self.assertEqual(hits[0]["chunk_id"], "inject-isha")
        self.assertEqual(hits[0].get("_work_injected"), WORK_ISHA)
        self.assertGreater(hits[0].get("_work_boost") or 0, 0)
        self.assertIsNone(next(h for h in hits if h["chunk_id"] == "inject-rv-1058").get("_work_match"))

    def test_katha_injects_despite_kaushitaki_token(self):
        chandogya = {
            "chunk_id": "inject-ch-k",
            "title": "Chandogya Upanishad III.1 (Müller, SBE01, sacred-texts)",
            "locator": "Chandogya III.1",
            "text": "Death and the honey-doctrine",
            "score": 0.9,
        }
        katha = {
            "chunk_id": "inject-katha",
            "title": "Katha Upanishad (English, sacred-texts SBE15)",
            "locator": "Katha 1",
            "text": "Nachiketas son of Vajasravasa",
            "score": 0.01,
        }
        injected = locator_work_injections(
            "Nachiketas meets Death Kaushitaki Upanishad",
            [chandogya, katha],
        )
        self.assertEqual([c["chunk_id"] for c in injected], ["inject-katha"])
        self.assertEqual(injected[0].get("_work_injected"), WORK_KATHA)

    def test_samaveda_injection_skips_chandogya_and_caps(self):
        decoys = [
            {
                "chunk_id": f"ch-saman-{i}",
                "title": "Chandogya Upanishad I.6 Saman commentary",
                "locator": "Chandogya I.6",
                "text": "The Saman is sung upon the Rig verses",
            }
            for i in range(6)
        ]
        verses = [
            {
                "chunk_id": f"sv-{n}",
                "title": f"Sāmaveda SV 1.1.1.{n} (Griffith, DharmicData)",
                "locator": f"SV 1.1.1.{n}",
                "text": "chant body",
            }
            for n in range(1, 13)
        ]
        injected = locator_work_injections(
            "Sama Veda chant melody of Rig verses",
            decoys + verses,
        )
        self.assertEqual(len(injected), LOCATOR_INJECT_PER_WORK)
        self.assertEqual(LOCATOR_INJECT_PER_WORK, LOCATOR_INJECT_PER_HYMN)
        self.assertTrue(all(c.get("_work_injected") == WORK_SAMAVEDA for c in injected))
        self.assertFalse(any(str(c["chunk_id"]).startswith("ch-") for c in injected))

    def test_yajur_vs_injection_skips_muller_anthology(self):
        muller = {
            "chunk_id": "inject-muller",
            "title": "The Upanishads (Müller, SBE15)",
            "locator": "",
            "text": "Vajasaneyi and the White Yajur are discussed here",
        }
        vs = {
            "chunk_id": "inject-vs-40",
            "title": "Yajurveda VS 40 (Sanskrit, DharmicData)",
            "locator": "VS 40",
            "text": "ईशावास्यमिदं सर्वम्",
        }
        injected = locator_work_injections(
            "Shukla Yajur Veda Vajasaneyi Samhita",
            [muller, vs],
        )
        self.assertEqual([c["chunk_id"] for c in injected], ["inject-vs-40"])
        self.assertEqual(injected[0].get("_work_injected"), WORK_YAJUR_VS)


if __name__ == "__main__":
    unittest.main()
