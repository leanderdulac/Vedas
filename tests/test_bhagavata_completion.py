"""Bhāgavata completo: GRETIL (sânscrito), Subba Rau (X–XII, re-OCR), Dutt VIII–IX e a exceção de licença."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from vedic_pipeline.crawler.licenses import (
    ALLOWED_LICENSES,
    SOURCE_LICENSE_EXCEPTIONS,
    validate_source,
)
from vedic_pipeline.etl.books import (
    _sr_heading_number,
    _sr_trim_line,
    expand_book_file,
    repair_by_anchors,
    split_gretil_bhagavata,
    split_subbarau_bhagavata,
)
from vedic_pipeline.search.entity import work_key

GRETIL_URL = "https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_bhAgavatapurANa.txt"


def _src(**kw):
    base = {
        "url": GRETIL_URL,
        "title": "Bhāgavata Purāṇa",
        "work_title": "Bhāgavata Purāṇa",
        "tradition": "purana",
        "language": "sa",
        "license": "cc-by-nc-sa-4.0",
        "license_note": "CC BY-NC-SA 4.0",
        "download": True,
        "source_class": "gretil",
    }
    base.update(kw)
    return base


class LicenseExceptionTests(unittest.TestCase):
    def test_exception_is_per_source_not_global(self):
        self.assertNotIn("cc-by-nc-sa-4.0", ALLOWED_LICENSES)
        self.assertIn(GRETIL_URL, SOURCE_LICENSE_EXCEPTIONS)
        ok, msg = validate_source(_src())
        self.assertTrue(ok)
        self.assertIn("exceção por fonte", msg)

    def test_other_nc_source_stays_blocked(self):
        ok, _ = validate_source(_src(url="https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_viSNupurANa.txt"))
        self.assertFalse(ok)

    def test_exception_requires_the_approved_licence(self):
        ok, _ = validate_source(_src(license="cc-by-nc-nd-4.0"))
        self.assertFalse(ok)


GRETIL_TEXT = """Bhāgavatapurāṇa

# Header
## Licence: CC BY-NC-SA

# Text

bhp_01.01.001/0 śrī-sūta uvāca
janmādy asya yato 'nvayād itarataś $ cārtheṣv abhijñaḥ svarāṭ & janmādy asya yato 'nvayād itarataś $ cārtheṣv abhijñaḥ svarāṭ & tene brahma hṛdā ya ādi-kavaye % muhyanti yat sūrayaḥ // bhp_01.01.001 //
dharmaḥ projjhita-kaitavo 'tra paramo $ nirmatsarāṇāṃ satāṃ & vedyaṃ vāstavam atra vastu % śivadaṃ tāpa-trayonmūlanam // bhp_01.01.002 //
naimiṣe 'nimiṣa-kṣetre $ ṛṣayaḥ śaunakādayaḥ & satraṃ svargāya lokāya % sahasra-samam āsata // bhp_01.01.004 //
sūta uvāca
yaṃ pravrajantam anupetam apeta-kṛtyaṃ $ dvaipāyano viraha-kātara ājuhāva & putreti // bhp_01.02.002 //
nārāyaṇaṃ namaskṛtya $ naraṃ caiva narottamam & devīṃ sarasvatīṃ vyāsaṃ % tato jayam udīrayet // bhp_01.02.004 //
mislabelled verse from the next chapter $ still in chapter two & text // bhp_01.03.022 //
second mislabelled verse $ also here & text // bhp_01.03.023 //
real chapter three begins $ here & text // bhp_01.03.001 //
"""


class GretilTests(unittest.TestCase):
    def test_one_section_per_adhyaya_with_verse_refs(self):
        secs = split_gretil_bhagavata(GRETIL_TEXT)
        self.assertEqual([s.locator for s in secs], ["Skandha 1, adhyāya 1", "Skandha 1, adhyāya 2", "Skandha 1, adhyāya 3"])
        self.assertEqual(secs[0].anchor, "bhp-1-1")
        first = secs[0].text
        self.assertIn("// 1.1.1 //", first)
        self.assertEqual(first.count("janmādy asya"), 1)  # metade repetida do GRETIL sai
        self.assertNotIn("bhp_", first)
        for mark in "$%&":
            self.assertNotIn(mark, first)
        # rótulo trocado (1.3.22) fica no capítulo onde o texto está
        self.assertIn("mislabelled verse from the next chapter", secs[1].text)
        self.assertIn("// 1.2.22 //", secs[1].text)
        self.assertTrue(secs[2].text.startswith("real chapter three"))

    def test_records_count_as_same_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bhp.txt"
            path.write_text(GRETIL_TEXT, encoding="utf-8")
            recs = expand_book_file(path, _src(etl="book", split="gretil-bhagavata", credit="sânscrito, GRETIL", min_chars=10))
        self.assertEqual(recs[0]["title"], "Bhāgavata Purāṇa — Skandha 1, adhyāya 1 (sânscrito, GRETIL)")
        self.assertEqual(recs[0]["language"], "sa")
        self.assertEqual({work_key(r["title"]) for r in recs}, {"bhagavata purana"})


class AnchorRepairTests(unittest.TestCase):
    def test_repair_by_anchors_trusts_the_consistent_numerals(self):
        self.assertEqual(repair_by_anchors([1, 2, 3, 4, 5], 10), [1, 2, 3, 4, 5])
        # "8" lido no lugar de "3" e um ilegível: a sequência coerente prevalece
        self.assertEqual(repair_by_anchors([1, 2, 8, None, 5, 6], 10), [1, 2, 3, 4, 5, 6])
        # título perdido: o salto vira faixa no recorte, não renumera os seguintes
        self.assertEqual(repair_by_anchors([1, 2, 4, 5], 6), [1, 2, 4, 5])
        out = repair_by_anchors([1, 40, 3], 5)
        self.assertEqual(out, [1, 2, 3])


class SelectAndKeepBooksTests(unittest.TestCase):
    def _scan(self) -> str:
        para = (
            "Suta said: the divine sage Narada, playing upon his lute and singing the glories of Hari, "
            "came to the hermitage of Vyasa on the bank of the Sarasvati and was received with honours."
        )
        page = f"{para}\n{para}\n\n"
        return (
            "SRIMADBHAGABATAM\n\nBOOK  I.\n\n"
            f"CHAPTER  I.\n\n{page}CHAPTER  IV.\n\n{page}"
            "The  end  of  Book  I.\n\n"
            f"CHAPTER  II.\n\n{page}CHAPTER  IX.\n\n{page}"
        )

    def _expand(self, **kw):
        src = {
            "url": "https://archive.org/download/x/x_djvu.txt",
            "citation_url": "https://archive.org/details/x",
            "work_title": "Bhāgavata Purāṇa",
            "translator": "M. N. Dutt",
            "year": "1895",
            "tradition": "purana",
            "language": "en",
            "license": "public-domain",
            "etl": "book",
            "split": "dutt-bhagavata",
            "clean": "ocr",
            "min_chars": 50,
        }
        src.update(kw)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scan.txt"
            path.write_text(self._scan(), encoding="utf-8")
            return expand_book_file(path, src)

    def test_keep_books(self):
        recs = self._expand(keep_books=[2])
        self.assertEqual([r["locator"] for r in recs], ["Book 2, ch. 2–8", "Book 2, ch. 9–10"])

    def test_select_relabels_and_filters(self):
        every = self._expand()
        anchors = [r["source_url"].split("#")[1] for r in every]
        self.assertEqual(anchors, ["book-1-ch-1", "book-1-ch-4", "book-2-ch-2", "book-2-ch-9"])
        recs = self._expand(select={"book-1-ch-4": "Book 8, ch. 23", "book-2-ch-9": None})
        self.assertEqual([r["locator"] for r in recs], ["Book 8, ch. 23", "Book 2, ch. 9–10"])
        self.assertEqual(recs[0]["source_url"], "https://archive.org/details/x#book-8-ch-23")


SUBBA_RAU = """[[page 1]]
CONTENTS.
TENTH SKANDHA.
1. Introductory. 1
2. Balarama's birth. 9
3. Avatar of Sri Krishna. 15

[[page 2]]
TENTH SKANDHA.
ADHYAYA I.
Introductory. The Avatar of Sri Krishna.
| 1. The King said:— Please describe the glorious deeds of Krishna in detail, O sage. | i
2. Sri Suka said:— Hear then, O king, how the Lord descended on earth for the good of all. a
3. When the earth was oppressed by the demons she went to Brahma in the form of a cow.
4. Brahma went with the gods to the shore of the milk ocean and praised the Lord.

[[page 3]]
Sx. 10. Apu. 1.] SRIMAD BHAGAVATAM 3
5. The Lord said that He would be born in the house of Vasudeva for the good of the world.
ADHYAYA 2.
Balarama's birth. Brahma and other gods pray.
1. Sri Suka said:— Kamsa, aided by mighty asuras, began to oppress the Yadus everywhere.
2. Some of the Yadus fled to other countries and lived there in fear of Kamsa.

[[page 4]]
SRIMAD BHAGAVATAM. [Sk. 10. Apu. 2.
3. Yogamaya was sent by the Lord to transfer the seventh child of Devaki to Rohini.
4. Then the Lord Himself entered the mind of Vasudeva and was borne by Devaki.
[84th Adhyaya ends in D. Reading.]
5. The gods came and praised the Lord who was in the womb of Devaki.
Thus ends the twelfth Skandha.
"""


class SubbaRauTests(unittest.TestCase):
    def test_heading_numbers(self):
        self.assertEqual(_sr_heading_number("ADHYAYA 41."), 41)
        self.assertEqual(_sr_heading_number("ADHYAYA I7. We"), 17)
        self.assertIsNone(_sr_heading_number("ADHYAYA."))
        self.assertIs(_sr_heading_number("[84th Adhyaya ends in D. Reading.]"), False)
        self.assertIs(_sr_heading_number("The Lord said: this is a long verse line about ADHYAYA"), False)

    def test_trim_line_drops_scan_margins(self):
        self.assertEqual(_sr_trim_line("| 7. Accordingly commanded by Siva, Maya able to | i 2"), "7. Accordingly commanded by Siva, Maya able to")
        self.assertEqual(_sr_trim_line("i 6. Sri Suka said:—When"), "6. Sri Suka said:—When")
        self.assertEqual(_sr_trim_line("He went to a"), "He went to a")

    def test_split_skips_contents_and_keeps_chapters(self):
        secs = split_subbarau_bhagavata(SUBBA_RAU)
        # o último capítulo do recorte vai até o fim do skandha
        self.assertEqual([s.locator for s in secs], ["Book 10, ch. 1", "Book 10, ch. 2–90"])
        self.assertNotIn("CONTENTS", secs[0].text)
        self.assertIn("Please describe the glorious deeds", secs[0].text)
        self.assertNotIn("SRIMAD", secs[1].text)
        self.assertIn("The gods came and praised", secs[1].text)


class OcrWiringTests(unittest.TestCase):
    def test_pdf_with_ocr_spec_uses_ocr_text(self):
        src = {
            "url": "https://archive.org/download/in.ernet.dli.2015.273815/x_text.pdf",
            "citation_url": "https://archive.org/details/in.ernet.dli.2015.273815",
            "work_title": "Bhāgavata Purāṇa",
            "translator": "S. Subba Rau",
            "year": "1928",
            "tradition": "purana",
            "language": "en",
            "license": "public-domain",
            "etl": "book",
            "split": "subbarau-bhagavata",
            "ocr": {"engine": "tesseract", "lang": "eng", "first_page": 1, "last_page": 4},
            "keep_books": [10],
            "min_chars": 50,
        }
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "x_text.pdf"
            pdf.write_bytes(b"%PDF-1.4 fake")
            with mock.patch("vedic_pipeline.etl.ocr.ocr_pdf", return_value=SUBBA_RAU) as fake:
                recs = expand_book_file(pdf, src)
        fake.assert_called_once()
        self.assertEqual([r["locator"] for r in recs], ["Book 10, ch. 1", "Book 10, ch. 2–90"])
        self.assertEqual(recs[0]["title"], "Bhāgavata Purāṇa — Book 10, ch. 1 (S. Subba Rau, 1928)")

    def test_ocr_cache_is_reused(self):
        from vedic_pipeline.etl.ocr import ocr_cache_path, ocr_pdf

        spec = {"engine": "tesseract", "lang": "eng", "first_page": 1, "last_page": 2}
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "a.pdf"
            pdf.write_bytes(b"%PDF")
            cache = ocr_cache_path(pdf, spec)
            self.assertEqual(cache.name, "a.pdf.tesseract-eng-1-2.txt")
            cache.write_text("[[page 1]]\ncached", encoding="utf-8")
            self.assertEqual(ocr_pdf(pdf, spec), "[[page 1]]\ncached")


if __name__ == "__main__":
    unittest.main()
