"""Consultas de entidade ("Narada Muni"): normalização, recall por obra e cobertura."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vedic_pipeline.search import lexical_index
from vedic_pipeline.search.entity import (
    coverage_note,
    diversify_by_work,
    missing_reference_works,
    name_alternates,
    parse_entity_query,
    work_cap,
    work_key,
)
from vedic_pipeline.search.hybrid import _lexical_scores_scan, hybrid_rerank, lexical_scores
from vedic_pipeline.search.rag import build_rag_prompt


def _c(cid: str, title: str, text: str, doc: str | None = None, score: float | None = None) -> dict:
    ch = {"chunk_id": cid, "doc_id": doc or cid, "title": title, "text": text}
    if score is not None:
        ch["score"] = score
    return ch


def _corpus() -> list[dict]:
    mbh = "The Mahabharata Volume {} (Ganguli, Gutenberg)"
    return [
        _c("m1", mbh.format(1), "The celestial Rishi Narada came to the court of Yudhishthira.", "mbh1"),
        _c("m2", mbh.format(1), "Narada spoke to the king about the sabha of Indra. Narada smiled.", "mbh1"),
        _c("m3", mbh.format(2), "Matali met the great Rishi Narada on his way to Varuna.", "mbh2"),
        _c("m4", mbh.format(3), "Suka worshipped Narada with the Arghya.", "mbh3"),
        _c("m5", mbh.format(4), "Marutta said, I have been told by Narada, wandering on his way.", "mbh4"),
        _c("ch", "Chandogya Upanishad VII.1 (Müller, SBE01)", "Nârada approached Sanatkumâra and said, Teach me, Sir! Nârada knows the Rig-veda."),
        _c("ra", "The Ramayan of Valmiki — English verse (Griffith)", "To sainted Nárad, prince of those whose lore in words of wisdom lies, Valmiki spoke."),
        _c("mu1", "Vishnu Purana — Book II, página 63", "The muni sat in meditation; every muni praised the lord."),
        _c("mu2", "Markandeya Purana (Pargiter)", "A muni of great austerity dwelt by the river; the muni wept."),
        _c("rv", "Rigveda RV 1.1 Agni (Griffith)", "I laud Agni, the chosen Priest, God, minister of sacrifice."),
    ]


class ParseEntityQueryTests(unittest.TestCase):
    def test_honorifics_and_question_words_are_stripped(self):
        cases = {
            "Narada Muni": ("narada",),
            "Nārada": ("narada",),
            "Quem foi Nārada?": ("narada",),
            "Devarshi Narada": ("narada",),
            "who is Sri Narada Muni": ("narada",),
            "नारद": ("narada",),
        }
        for query, names in cases.items():
            with self.subTest(query=query):
                ent = parse_entity_query(query)
                self.assertIsNotNone(ent)
                self.assertEqual(ent.names, names)
        self.assertEqual(parse_entity_query("Narada Muni").honorifics, ("muni",))
        self.assertEqual(parse_entity_query("Narada Muni").display, "Narada")

    def test_scoped_or_long_queries_are_not_entities(self):
        for query in (
            "Nasadiya Sukta",
            "RV 8.13",
            "Narada no Ramayana",
            "o que é o Atman segundo as Upanishads",
            "Kurukshetra war Mahabharata Bhishma",
            "Om",
            "Muni",
            "",
        ):
            with self.subTest(query=query):
                self.assertIsNone(parse_entity_query(query))

    def test_entity_mode_can_be_disabled(self):
        with patch.dict(os.environ, {"VEDIC_ENTITY_MODE": "false"}):
            self.assertIsNone(parse_entity_query("Narada"))

    def test_spelling_alternates(self):
        self.assertIn("narad", name_alternates("narada"))  # Griffith: Nárad
        self.assertIn("narada", name_alternates("Naarada"))
        self.assertIn("vasistha", name_alternates("vasishtha"))
        self.assertEqual(name_alternates("agni"), [])


class WorkKeyTests(unittest.TestCase):
    def test_volumes_books_and_hymns_collapse_to_the_work(self):
        cases = {
            "The Mahabharata Volume 3 (Ganguli, Gutenberg #15476)": "mahabharata",
            "Vishnu Purana — Book I, página 50 (Wilson, 1840, sacred-texts)": "vishnu purana",
            "Rigveda RV 8.13 (Griffith, sacred-texts)": "rigveda",
            "Rigveda Shakala Samhita — Mandala 8, Sukta 13": "rigveda",
            "Chandogya Upanishad VII.1 (Müller, SBE01, sacred-texts)": "chandogya upanishad",
            "Bhagavad-gītā 10 (Sanskrit, DharmicData)": "bhagavad gita",
            "Atharvaveda Shaunaka Samhita — Kanda 12, Sukta 4": "atharvaveda",
        }
        for title, key in cases.items():
            with self.subTest(title=title):
                self.assertEqual(work_key(title), key)


class DiversifyByWorkTests(unittest.TestCase):
    def test_cap_per_work_then_fill(self):
        hits = [_c(f"m{i}", f"The Mahabharata Volume {i % 4 + 1}", "x", f"mbh{i % 4}", 2.0 - i * 0.01) for i in range(8)]
        hits += [_c("ch", "Chandogya Upanishad VII.1", "x", score=1.0), _c("gi", "Bhagavad-Gita", "x", score=0.9)]
        out = diversify_by_work(hits, top_k=5, max_per_doc=2)
        works = [work_key(h["title"]) for h in out]
        self.assertEqual(works.count("mahabharata"), work_cap(5))
        self.assertIn("chandogya upanishad", works)
        self.assertIn("bhagavad gita", works)
        self.assertEqual(out[0]["chunk_id"], "m0")  # top-1 preservado
        # Sem outras obras, completa com o que sobrou.
        out = diversify_by_work(hits[:8], top_k=6, max_per_doc=2)
        self.assertEqual(len(out), 6)


class EntityRetrievalTests(unittest.TestCase):
    def setUp(self):
        lexical_index._BY_CHUNK.clear()

    def tearDown(self):
        lexical_index._BY_CHUNK.clear()

    def test_alternates_score_in_scan_and_index(self):
        chunks = _corpus()
        alts = {"narada": ["narad"]}
        scan = _lexical_scores_scan("narada", chunks, alternates=alts)
        ramayan = next(i for i, c in enumerate(chunks) if c["chunk_id"] == "ra")
        self.assertGreater(scan[ramayan], 0)
        self.assertEqual(_lexical_scores_scan("narada", chunks)[ramayan], 0)
        with tempfile.TemporaryDirectory() as tmp:
            lexical_index.save_lexical_index(Path(tmp), chunks)
            fast = lexical_scores("narada", chunks, alternates=alts)
        for got, expected in zip(fast, scan, strict=True):
            self.assertAlmostEqual(got, expected, places=5)

    def test_entity_query_covers_each_work_and_ignores_the_honorific(self):
        chunks = _corpus()
        semantic = [dict(c, score=0.9 - i * 0.01) for i, c in enumerate(chunks) if c["doc_id"].startswith("mbh")]
        semantic += [dict(chunks[7], score=0.95), dict(chunks[8], score=0.94)]  # "muni" sem Nārada
        hits = hybrid_rerank("Narada Muni", semantic, all_chunks=chunks, top_k=6, use_cross_encoder=False)
        ids = [h["chunk_id"] for h in hits]
        self.assertIn("ch", ids)  # Chāndogya VII (Nârada)
        self.assertIn("ra", ids)  # Rāmāyaṇa (Nárad)
        mbh = [h for h in hits if work_key(h["title"]) == "mahabharata"]
        self.assertLessEqual(len(mbh), work_cap(6))
        mention_ids = {"m1", "m2", "m3", "m4", "m5", "ch", "ra"}
        first_non_mention = next((i for i, cid in enumerate(ids) if cid not in mention_ids), len(ids))
        self.assertTrue(all(cid in mention_ids for cid in ids[:first_non_mention]))
        self.assertGreaterEqual(first_non_mention, 5)

    def test_non_entity_queries_keep_the_doc_diversification(self):
        chunks = _corpus()
        hits = hybrid_rerank("Agni chosen Priest minister of sacrifice", [], all_chunks=chunks, top_k=3, use_cross_encoder=False)
        self.assertEqual(hits[0]["chunk_id"], "rv")


class CoverageAndPromptTests(unittest.TestCase):
    def test_coverage_note_lists_works_and_missing_references(self):
        chunks = _corpus()
        note = coverage_note(parse_entity_query("Narada"), chunks)
        self.assertIn("Mahabharata (5)", note)
        self.assertIn("Chandogya Upanishad (1)", note)
        self.assertIn("Ramayan Of Valmiki (1)", note)
        missing = missing_reference_works(chunks)
        self.assertIn("Bhāgavata Purāṇa", missing)
        self.assertIn("Nārada Bhakti Sūtra", missing)
        self.assertNotIn("Mahābhārata", missing)
        self.assertIn("Bhāgavata Purāṇa", note)

    def test_entity_prompt_asks_for_each_tradition_and_what_is_missing(self):
        hits = [_c("ch", "Chandogya Upanishad VII.1", "Nârada approached Sanatkumâra")]
        plain = build_rag_prompt("Narada", hits)
        self.assertNotIn("5 a 9 parágrafos", plain["user"])
        rich = build_rag_prompt("Narada", hits, entity_note="NOTA-COBERTURA", max_context_chars=20000)
        self.assertIn("NOTA-COBERTURA", rich["user"])
        self.assertIn("5 a 9 parágrafos", rich["user"])
        self.assertIn("não estão no acervo", rich["user"])

    def test_entity_plan_raises_k_and_tokens(self):
        from vedic_pipeline.llm.ask import _entity_plan

        ent, k, tokens = _entity_plan("Narada Muni", 10, 1800, True)
        self.assertIsNotNone(ent)
        self.assertGreaterEqual(k, 14)
        self.assertGreaterEqual(tokens, 2600)
        ent, k, tokens = _entity_plan("Krishna explains dharma to Arjuna Song Celestial", 10, 1800, True)
        self.assertIsNone(ent)
        self.assertEqual((k, tokens), (10, 1800))


if __name__ == "__main__":
    unittest.main()
