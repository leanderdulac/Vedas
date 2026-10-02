"""ETL de livros inteiros (scan OCR, Gutenberg, Sacred Texts) → registros por seção."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vedic_pipeline.etl.books import (
    book_title,
    chapter_heading_number,
    clean_ocr_text,
    expand_book_file,
    is_running_head,
    join_paragraphs,
    ocr_roman,
    repair_chapter_numbers,
    sacred_texts_html_to_text,
    split_dutt_bhagavata,
    split_dutt_harivamsa,
    split_sturdy_narada_bhakti,
    strip_gutenberg_boilerplate,
)
from vedic_pipeline.search.entity import parse_entity_query, titled_works, work_key

PARA = (
    "Suta said: the divine sage Narada, playing upon his lute and singing the glories of Hari, "
    "came to the hermitage of Vyasa on the bank of the Sarasvati, and the son of Satyavati "
    "received him with due honours, offering him water to wash his feet and a seat."
)


def _dutt_scan() -> str:
    page = f"{PARA}\n{PARA}\n\n"
    return (
        "j 'V',\n\n_>  m -I*'  ■'■  i\n\nSRIMADBHAGABATAM\n\nBOOK  I.\n\n"
        f"CHAPTER  I.  V\n\n{page}"
        "82\n\nSR  I M A D B H A G A V A TA  M.\n\n"
        f"CHAPTER  IV.\n\n{page}"
        "The  end  of  Book  I.\n\n"
        "Chapter  I. — Suka  describes  how  should  one  fix  his  mind  upon\n"
        "the  Divine  Person  by  chanting. — P.  II.\n\n"
        f"6HAPTER  II.\n\n{page}"
        f"CHAP  I EU  IX.\n\n{page}"
    )


class OcrCleanupTests(unittest.TestCase):
    def test_running_heads_with_ocr_noise(self):
        heads = ("SRIMADBHAGAVATAM",)
        for line in ("SRIMADBHAGABATAM", "SR  I M AD  B HA  GAB  AT  A M", "82 SRIMADBHAGAVATAM.", "i6 NARADA sOtRA,"):
            with self.subTest(line=line):
                expected = "NARADA" not in line.upper()
                self.assertEqual(is_running_head(line, heads), expected)
        self.assertTrue(is_running_head("i6 NARADA sOtRA,", ("NARADA SUTRA",)))
        self.assertFalse(is_running_head("The divine sage Narada said:", heads))

    def test_hyphenation_and_page_break_are_joined(self):
        text = "the sage was devo-\ntion itself and he\n\nwent away singing.\n\nA new paragraph."
        self.assertEqual(
            join_paragraphs(text),
            "the sage was devotion itself and he went away singing.\n\nA new paragraph.",
        )

    def test_clean_drops_page_numbers_junk_and_toc(self):
        raw = (
            "Chapter  XXV. — Kapila  describes  before  his  mother  all  the  charac-\n"
            "teristics of  Bhakti, — P.  1 19.\n\n"
            f"{PARA[:80]}\n\n124\n\nSRIMADBHAGAVATAM.\n\n■■ ¥ |{{ ~\n\n{PARA[80:]}\n\n"
            "0 king, hear this."
        )
        out = clean_ocr_text(raw, running_heads=("SRIMADBHAGAVATAM",))
        self.assertNotIn("Kapila", out)
        self.assertNotIn("124", out)
        self.assertNotIn("SRIMAD", out)
        self.assertIn(PARA[75:95], out.replace("\n\n", " "))
        self.assertTrue(out.endswith("O king, hear this."))

    def test_ocr_roman_numerals(self):
        cases = {"XVH": 17, "V 1 1 1": 8, "XLII": 13, "XXV ML": 28, "XU": 12, "XfX": 19, "Vli": 7, "1": 1}
        for raw, n in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(ocr_roman(raw), n)

    def test_chapter_heading_detection(self):
        self.assertEqual(chapter_heading_number("CHAPTER  I.  V"), 1)
        self.assertEqual(chapter_heading_number("6HAPTER  VI."), 6)
        self.assertEqual(chapter_heading_number("CHAP  I EU  XIX."), 19)
        self.assertIs(chapter_heading_number("Chapter XXV. — Kapila describes"), False)
        self.assertIs(chapter_heading_number("The chapter ends here and more"), False)
        self.assertIsNone(chapter_heading_number("CHAPTER  ib"))

    def test_repair_keeps_sequence_inside_known_chapter_count(self):
        # Livro VI: "XVIII" lido duas vezes e "XIX" no fim → 17, 18, 19
        self.assertEqual(repair_chapter_numbers([2, 6, 14, 18, 18, 19], 19), [2, 6, 14, 17, 18, 19])
        self.assertEqual(repair_chapter_numbers([None, 40, 3], 19), [1, 2, 3])


class SplitterTests(unittest.TestCase):
    def test_dutt_bhagavata_books_and_chapter_ranges(self):
        secs = split_dutt_bhagavata(_dutt_scan())
        self.assertEqual(
            [s.locator for s in secs],
            ["Book 1, ch. 1–3", "Book 1, ch. 4–19", "Book 2, ch. 2–8", "Book 2, ch. 9–10"],
        )
        joined = " ".join(s.text for s in secs)
        self.assertNotIn("SRIMAD", joined)
        self.assertNotIn("Suka  describes", joined)
        self.assertEqual(len({s.anchor for s in secs}), len(secs))

    def test_sturdy_groups_sutras_with_commentary(self):
        comment = "This aphorism teaches that love is its own reward and seeks nothing in return. " * 10
        raw = (
            "This is a digital copy of a book that was preserved for generations on library shelves "
            "... at |http : //books . google . com/\n\nIn a former Age, O Saint, I was born.\n\n"
            "INTRODUCTION.\n\nThere is always plenty of opposition.\n\n"
            "1. We will now explain Love (bhakti).\n\n2. Its nature is extreme devotion.\n\n"
            f"{comment}\n\n3. Love is immortal.\n\n4. Obtaining which man becomes\n\nperfect.\n\n"
            f"5. And obtaining which he desires nothing.\n\n{comment}\n\nAPPENDIX.\n\nAN INDIAN YOGI IN LONDON."
        )
        secs = split_sturdy_narada_bhakti(raw)
        self.assertEqual([s.locator for s in secs], ["front matter and introduction", "sūtras 1–2", "sūtras 3–84"])
        self.assertNotIn("google", " ".join(s.text for s in secs).lower())
        self.assertNotIn("YOGI IN LONDON", " ".join(s.text for s in secs))

    def test_harivamsa_chapters_parva_and_typo_repair(self):
        body = "Vaishampayana said:—Narada then repaired to the residence of Mahendra (1). " * 5
        raw = (
            "header\n*** START OF THE PROJECT GUTENBERG EBOOK HARIVAMSHA ***\n"
            "      CHAPTER I. AN ACCOUNT OF THE PRIMEVAL CREATION .....\n\n"
            "THE PRELUDE.\n\n" + ("Prelude text about the glory of Hari and of _Narayana_. " * 6) + "\n\n"
            f"CHAPTER I. AN ACCOUNT OF THE PRIMEVAL CREATION\n\n{body}\n\n"
            f"CHAPTER LVIII. A TYPO FOR THE SECOND\n\n{body}\n\n"
            f"CHAPTER III. KRISHNA AND NARADA WITH A HEADING THAT\nWRAPS TO A SECOND LINE.\n\n{body}\n\n"
            "BHAVISHYA PARVA OR THE BOOK OF FUTURE.\n\n"
            f"CHAPTER I. AN ACCOUNT OF JANAMEJAYA’S FAMILY.\n\n{body}\n\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK HARIVAMSHA ***\nlicense"
        )
        secs = split_dutt_harivamsa(raw)
        self.assertEqual(
            [s.locator for s in secs],
            [
                "Prelude",
                "ch. 1: An Account of the Primeval Creation",
                "ch. 2: A Typo for the Second",
                "ch. 3: Krishna and Narada with a Heading That Wraps to a Second Line",
                "Bhaviṣya Parva, ch. 1: An Account of Janamejaya’s Family",
            ],
        )
        self.assertNotIn("_", secs[0].text)
        self.assertNotIn("license", secs[-1].text)

    def test_gutenberg_boilerplate(self):
        raw = "x\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nbody\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\ny"
        self.assertEqual(strip_gutenberg_boilerplate(raw), "body")

    def test_sacred_texts_page(self):
        html = (
            '<html><head><title>t</title></head><body><a href="../index.htm">Sacred Texts</a> '
            '<a href="sbe3325.htm">Previous</a><p>The Minor Law Books (SBE33), by Julius Jolly, [1889], at sacred-texts.com</p>'
            '<p><a name="page_100"><font size="1" color="green">p. 100</font></a></p>'
            '<p>* 248. <a href="#fn_248"><font size="1">248</font></a> (Let him cause a Brahman to swear by) truth, '
            '(a Vai<i>s</i>ya) by his cows, first<span class="margnote"><font>Supposed origin.</font></span> seeds or gold.</p>'
            '<hr><p><a name="fn_248"></a><a href="sbe3326.htm#fr_248">100:248</a> A. Vish<i>n</i>u IX, 24.</p>'
            '<p>Next: <a href="sbe3327.htm">20. The Ordeal by Balance</a></p></body></html>'
        )
        text = sacred_texts_html_to_text(html)
        self.assertIn("(a Vaisya) by his cows, first seeds or gold.", text)
        self.assertIn("[n. 248] A. Vishnu IX, 24.", text)
        for noise in ("p. 100", "Previous", "Next:", "sacred-texts.com", "Supposed origin"):
            self.assertNotIn(noise, text)


class RecordTests(unittest.TestCase):
    def test_records_titles_urls_and_work_grouping(self):
        src = {
            "url": "https://archive.org/download/x/x_djvu.txt",
            "citation_url": "https://archive.org/details/x",
            "title": "Bhāgavata Purāṇa",
            "work_title": "Bhāgavata Purāṇa",
            "translator": "M. N. Dutt",
            "year": "1896",
            "tradition": "purana",
            "language": "en",
            "license": "public-domain",
            "license_note": "PD",
            "etl": "book",
            "split": "dutt-bhagavata",
            "clean": "ocr",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scan.txt"
            path.write_text(_dutt_scan(), encoding="utf-8")
            recs = expand_book_file(path, src)
        self.assertEqual(recs[0]["title"], "Bhāgavata Purāṇa — Book 1, ch. 1–3 (M. N. Dutt, 1896)")
        self.assertEqual(recs[0]["source_url"], "https://archive.org/details/x#book-1-ch-1")
        self.assertEqual(recs[0]["retrieved_from"], src["url"])
        self.assertEqual(recs[0]["license_note"], "PD")
        self.assertEqual(len({r["source_url"] for r in recs}), len(recs))
        # todos os livros e capítulos contam como uma obra no teto por obra
        self.assertEqual({work_key(r["title"]) for r in recs}, {"bhagavata purana"})
        self.assertEqual(work_key(book_title({"work_title": "Harivaṃśa", "translator": "M. N. Dutt", "year": "1897"}, "Bhaviṣya Parva, ch. 3: X")), "harivamsa")

    def test_titled_works_are_reported_apart(self):
        chunks = [
            {"title": "Nārada Smṛti — Title II (Deposits) (J. Jolly, SBE 33, 1889)", "text": "A deposit is ..."},
            {"title": "Nārada Bhakti Sūtra — sūtras 1–3 (E. T. Sturdy, 1896)", "text": "Love is immortal."},
            {"title": "Harivaṃśa — ch. 126: The Colloquy Between Narada and Indra (M. N. Dutt, 1897)", "text": "Narada said"},
        ]
        self.assertEqual(titled_works(parse_entity_query("Narada Muni"), chunks), ["narada bhakti sutra", "narada smrti"])

    def test_manifest_is_valid_and_gretil_enters_by_exception(self):
        from vedic_pipeline.crawler.licenses import validate_source

        root = Path(__file__).resolve().parents[1]
        sources = json.loads((root / "fixtures" / "sources_narada_2026_10.json").read_text(encoding="utf-8"))["sources"]
        ok = [s for s in sources if validate_source(s)[0]]
        self.assertEqual(len(ok), len(sources))
        self.assertEqual(len(ok), 61)
        gretil = [s for s in sources if s["language"] == "sa"]
        self.assertEqual(len(gretil), 1)
        self.assertTrue(gretil[0]["download"])
        self.assertIn("exceção por fonte", validate_source(gretil[0])[1])
        self.assertIn("NonCommercial", gretil[0]["license_note"])
        bhp = [s for s in ok if s.get("work_title") == "Bhāgavata Purāṇa"]
        self.assertEqual(len(bhp), 4)  # Dutt I–VII, Dutt VIII–IX, Subba Rau X–XII, GRETIL
        for s in ok:
            self.assertEqual(s["etl"], "book")
            self.assertTrue(s.get("license_note"))
        smrti = [s for s in ok if s["work_title"] == "Nārada Smṛti"]
        self.assertEqual(len({s["citation_url"] for s in smrti}), 55)
        self.assertFalse(any("sbe335" in s["citation_url"] and int(s["citation_url"][-6:-4]) > 57 for s in smrti))


if __name__ == "__main__":
    unittest.main()
