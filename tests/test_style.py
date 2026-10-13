"""Pós-processamento editorial dos textos gerados (vedic_pipeline.common.style)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from vedic_pipeline.common import style
from vedic_pipeline.common.style import clean_prose, edit_prose, speech_text

RAW = """**Resposta direta:** Nas Upaniṣads, o **ātman** é o Si-mesmo (Self) interior — sutil demais para o raciocínio [7].

### Fundamentação com o corpus
1. **Inacessível ao argumento**
   Ele é o *puruṣa* que “jaz no corpo” [10].

---

### Nuances e limites do corpus recuperado
- O material é parcial em RV 1.1.1–2.

Em resumo, o ātman é o Si sutil. 🙏"""


class CleanProseTests(unittest.TestCase):
    def test_strips_markdown_labels_and_cliches(self):
        out = clean_prose(RAW)
        for marker in ("**", "###", "---", "Resposta direta", "Fundamentação com o corpus",
                       "Nuances e limites", "Em resumo", "🙏", "—", "Self", "\n- ", "1. "):
            self.assertNotIn(marker, out, marker)
        self.assertTrue(out.startswith("Nas Upaniṣads, o ātman é o Si interior, sutil demais"))
        self.assertIn("O ātman é o Si sutil.", out)  # recapitaliza após cortar "Em resumo,"
        self.assertIn("o puruṣa que “jaz no corpo” [10].", out)

    def test_preserves_citations_ranges_devanagari_and_iast(self):
        text = "Agni [1] é louvado em RV 1.1.1–2: अग्निमीळे पुरोहितं (agním īḷe)."
        self.assertEqual(clean_prose(text), text)

    def test_citation_placement_and_merge(self):
        self.assertEqual(clean_prose("Ele ensina. [7]\nOutro [5][8][1]."), "Ele ensina [7].\nOutro [5, 8, 1].")

    def test_idempotent(self):
        once = clean_prose(RAW)
        self.assertEqual(clean_prose(once), once)

    def test_dash_rules(self):
        self.assertEqual(clean_prose("Agni — o fogo — é o hotṛ."), "Agni, o fogo, é o hotṛ.")
        self.assertEqual(clean_prose("- **agním** — Agni no acusativo"), "agním: Agni no acusativo")
        self.assertEqual(clean_prose("fim do hino —."), "fim do hino.")

    def test_lexical_slips_and_offer_closing(self):
        out = clean_prose("O himno é solemne. Se quiser, posso detalhar mais.")
        self.assertEqual(out, "O hino é solene.")

    def test_inference_label_removed(self):
        out = clean_prose("Inferência legítima (marcada como tal): o ātman é sujeito.")
        self.assertEqual(out, "O ātman é sujeito.")

    def test_empty_and_english(self):
        self.assertEqual(clean_prose(None), "")
        self.assertEqual(clean_prose("**Answer:** In summary, the Self is subtle.", lang="en"),
                         "The Self is subtle.")

    def test_speech_text_has_no_symbols_or_locator_lines(self):
        out = speech_text("**RV 1.1.1**\nLouvo Agni [1], o *purohita*.")
        self.assertEqual(out, "Louvo Agni, o purohita.")


class EditProseTests(unittest.TestCase):
    LONG = ("Agni é o sacerdote do rito [1] em RV 1.1.1. " * 20).strip()

    def test_disabled_or_short_or_not_xai_skips_llm(self):
        with patch("vedic_pipeline.llm.generate.generate_answer") as gen:
            self.assertEqual(edit_prose("curto", provider="xai"), "curto")
            self.assertEqual(edit_prose(self.LONG, provider="extractive"), self.LONG)
            with patch.dict("os.environ", {"VEDIC_TEXT_EDITOR": "0"}):
                self.assertEqual(edit_prose(self.LONG, provider="xai"), self.LONG)
            gen.assert_not_called()

    def test_accepts_faithful_edit(self):
        edited = self.LONG.replace("é o sacerdote", "é o sacerdote principal")
        with patch.dict("os.environ", {"VEDIC_TEXT_EDITOR": "1"}), patch(
            "vedic_pipeline.llm.generate.generate_answer", return_value={"answer": edited}
        ):
            self.assertEqual(edit_prose(self.LONG, provider="xai"), edited)

    def test_rejects_edit_that_drops_citations_or_shrinks(self):
        with patch.dict("os.environ", {"VEDIC_TEXT_EDITOR": "1"}):
            no_cite = self.LONG.replace("[1]", "")
            with patch("vedic_pipeline.llm.generate.generate_answer", return_value={"answer": no_cite}):
                self.assertEqual(edit_prose(self.LONG, provider="xai"), self.LONG)
            with patch("vedic_pipeline.llm.generate.generate_answer", return_value={"answer": "Agni [1] RV 1.1.1."}):
                self.assertEqual(edit_prose(self.LONG, provider="xai"), self.LONG)
            with patch("vedic_pipeline.llm.generate.generate_answer", side_effect=RuntimeError("x")):
                self.assertEqual(edit_prose(self.LONG, provider="xai"), self.LONG)

    def test_prompts_carry_style_guide(self):
        from vedic_pipeline.api.verse_service import EXPLAIN_SYSTEM_PT
        from vedic_pipeline.search.rag import VEDIC_SYSTEM_PROMPT, build_rag_prompt

        self.assertIn(style.STYLE_GUIDE_PT, VEDIC_SYSTEM_PROMPT)
        self.assertIn(style.STYLE_GUIDE_PT, EXPLAIN_SYSTEM_PT)
        self.assertNotIn("Estruture", VEDIC_SYSTEM_PROMPT)
        prompt = build_rag_prompt("O que é ātman?", [{"text": "x", "title": "t"}])
        self.assertNotIn("marque inferências", prompt["user"])


if __name__ == "__main__":
    unittest.main()
