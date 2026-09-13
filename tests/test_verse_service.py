"""Estudo do verso: testemunhas alinhadas e explicação extrativa."""

from __future__ import annotations

import unittest

from vedic_pipeline.api.verse_service import assemble_verse, recitation_text, witness_role


class WitnessRoleTests(unittest.TestCase):
    def test_editions(self):
        self.assertEqual(witness_role({"title": "Rigveda RV 1.1 (VedaWeb Zürich)", "language": "sa"}), "iast")
        self.assertEqual(witness_role({"title": "Bhagavad-gītā 2 (Sanskrit, DharmicData)", "language": "sa"}), "sa")
        self.assertEqual(witness_role({"title": "Rigveda RV 1.1 Agni (Griffith)", "language": "en"}), "en")


class AssembleVerseTests(unittest.TestCase):
    def test_aligns_sa_iast_en(self):
        rows = [
            {
                "verse_id": "RV.1.1.1",
                "locator": "RV 1.1.1",
                "work": "rigveda",
                "book": 1,
                "hymn": 1,
                "verse": 1,
                "verse_end": 1,
                "language": "sa",
                "title": "Rigveda RV 1.1 (Sanskrit, DharmicData)",
                "text": "अग्निमीळे पुरोहितं",
                "doc_id": "sa1",
            },
            {
                "verse_id": "RV.1.1.1",
                "locator": "RV 1.1.1",
                "work": "rigveda",
                "book": 1,
                "hymn": 1,
                "verse": 1,
                "verse_end": 1,
                "language": "sa",
                "title": "Rigveda RV 1.1 (VedaWeb Zürich, ISO-15919)",
                "text": "agním īḷe puróhitaṃ",
                "doc_id": "vw1",
            },
            {
                "verse_id": "RV.1.1.1",
                "locator": "RV 1.1.1",
                "work": "rigveda",
                "book": 1,
                "hymn": 1,
                "verse": 1,
                "verse_end": 1,
                "language": "en",
                "title": "Rigveda RV 1.1 Agni (Griffith, sacred-texts)",
                "text": "I Laud Agni, the chosen Priest",
                "doc_id": "en1",
            },
        ]
        bundle = assemble_verse("RV.1.1.1", rows)
        self.assertIsNotNone(bundle)
        roles = [w["role"] for w in bundle["witnesses"]]
        self.assertEqual(roles, ["sa", "iast", "en"])
        self.assertTrue(bundle["has_sanskrit"])
        self.assertIn("अग्नि", recitation_text(bundle) or "")


class ImaginePromptTests(unittest.TestCase):
    def test_visual_prompt_uses_english_witness(self):
        from vedic_pipeline.llm.imagine import visual_prompt

        prompt = visual_prompt(
            {
                "locator": "RV 1.1.1",
                "witnesses": [
                    {"role": "sa", "text": "अग्निमीळे पुरोहितं"},
                    {"role": "en", "text": "I Laud Agni, the chosen Priest"},
                ],
            }
        )
        self.assertIn("RV 1.1.1", prompt)
        self.assertIn("Agni", prompt)
        self.assertNotIn("अग्नि", prompt)

    def test_list_cached_media_reads_disk(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from vedic_pipeline.llm import imagine

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "BG.2.47.jpg").write_bytes(b"x" * 2000)
            (root / "BG.2.47.mp4").write_bytes(b"y" * 2000)
            (root / "tiny.jpg").write_bytes(b"no")
            with patch.object(imagine, "MEDIA_DIR", root):
                listed = imagine.list_cached_media()
        self.assertEqual(listed["images"], ["BG.2.47"])
        self.assertEqual(listed["videos"], ["BG.2.47"])


class ChatHistoryTests(unittest.TestCase):
    def test_history_is_threaded(self):
        from vedic_pipeline.llm.generate import _chat_messages

        msgs = _chat_messages(
            "sys",
            "o que ele pede em seguida?",
            [{"role": "user", "content": "explique BG 2.47"}, {"role": "assistant", "content": "adhikāra na ação"}],
        )
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["content"], "explique BG 2.47")
        self.assertEqual(msgs[-1]["content"], "o que ele pede em seguida?")
