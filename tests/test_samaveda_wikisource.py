"""ETL do Sāmaveda Kauthuma (sa.wikisource): parsing, pareamento e payload."""

from __future__ import annotations

import unittest

from vedic_pipeline.etl import samaveda_wikisource as svws

RAW_DASATI_POEM = """
{{cabeçalho}}
<poem><span style="font-size:large">अग्न आ याहि वीतये<ref>nota</ref> गृणानो हव्यदातये । नि होता सत्सि बर्हिषि ॥ १ ॥</span></poem>

<poem><span>त्वमग्ने यज्ञानां होता विश्वेषां हितः । देवेभिर्मानुषे जने ॥ २ ॥</span></poem>
"""

RAW_DASATI_TABLE = """<tr><td><p>अग्न आ याहि वीतये|गृणानो हव्यदातये|| ३ || <td>१अ</p></tr>
<tr><td><p>त्वमग्ने यज्ञानां होता|विश्वेषां हितः|| ४ || <td>१छ्</p></tr>
"""

RAW_DASATI_OPEN = """सोमं राजानं वरुणमग्निमन्वारभामहे।
आदित्यं विष्णुं सूर्यं ब्रह्मानं च बृहस्पतिम्॥ ९१

इत एत उदारुहन्दिवः पृष्ठान्या रुहन्।
प्र भूर्जयो यथा पथोद्यामङ्गिरसो ययुः॥ ९२
"""

RAW_ARDHA = """२
एष देवो अमर्त्यः पर्णवीरिव दीयते । अभि द्रोणान्यासदं ॥ १२५६ ॥
एष विप्रैरभिष्टुतोऽपो देवो वि गाहते । दधद्रत्नानि दाशुषे ॥ १२५७ ॥
(३)
एष विश्वानि वार्या शूरो यन्निव सत्वभिः । पवमानः सिषासति ॥ १२५८ ॥
"""


def _griffith(hymns: list[dict]) -> dict:
    return {"work": "samaveda", "language": "en", "hymns": hymns}


class PageTitleTests(unittest.TestCase):
    def test_all_leaf_pages_counts(self):
        pages = svws.all_leaf_pages()
        kinds = [k for k, _, _ in pages]
        self.assertEqual(kinds.count("dasati"), 59)
        self.assertEqual(kinds.count("aranya"), 5)
        self.assertEqual(kinds.count("mahanamnya"), 1)
        self.assertEqual(kinds.count("ardha"), 22)
        self.assertEqual(len(pages), 87)

    def test_raw_url_encodes_devanagari(self):
        url = svws.raw_url(svws.page_title("mahanamnya"))
        self.assertIn("action=raw", url)
        self.assertNotIn(" ", url)
        self.assertTrue(url.startswith("https://sa.wikisource.org/wiki/%"))

    def test_invalid_kind(self):
        with self.assertRaises(ValueError):
            svws.page_title("outro", 1)


class ParsingTests(unittest.TestCase):
    def test_deva_int(self):
        self.assertEqual(svws.deva_int("१२३"), 123)
        self.assertIsNone(svws.deva_int("०"))
        self.assertIsNone(svws.deva_int("abc"))
        self.assertIsNone(svws.deva_int(""))

    def test_parse_poem_format(self):
        secs = svws.parse_wikisource_sections(RAW_DASATI_POEM)
        self.assertEqual(len(secs), 1)
        marker, arcas = secs[0]
        self.assertIsNone(marker)
        self.assertEqual([n for n, _ in arcas], [1, 2])
        self.assertTrue(arcas[0][1].endswith("॥ १ ॥"))
        self.assertNotIn("ref", arcas[0][1])

    def test_parse_table_format(self):
        secs = svws.parse_wikisource_sections(RAW_DASATI_TABLE)
        _, arcas = secs[0]
        self.assertEqual([n for n, _ in arcas], [3, 4])
        # daṇḍas ASCII viram devanāgarī e o nº global fecha o arca
        self.assertIn(" । ", arcas[0][1])
        self.assertTrue(arcas[0][1].endswith("॥ ३ ॥"))

    def test_parse_open_arca_without_closing_danda(self):
        secs = svws.parse_wikisource_sections(RAW_DASATI_OPEN)
        _, arcas = secs[0]
        self.assertEqual([n for n, _ in arcas], [91, 92])
        self.assertTrue(arcas[0][1].endswith("॥ ९१ ॥"))

    def test_parse_ardha_markers(self):
        secs = svws.parse_wikisource_sections(RAW_ARDHA)
        self.assertEqual([(m, len(s)) for m, s in secs], [(2, 2), (3, 1)])
        # marcador entre parênteses também vira número de seção
        self.assertEqual(secs[1][0], 3)


class IdentityTests(unittest.TestCase):
    def test_dasati_chapter_restart(self):
        self.assertEqual(svws.hymn_identity_sa("dasati", (1, 1)), (1, 1, 1, 1))
        self.assertEqual(svws.hymn_identity_sa("dasati", (1, 5)), (1, 1, 1, 5))
        self.assertEqual(svws.hymn_identity_sa("dasati", (1, 6)), (1, 1, 2, 1))
        self.assertEqual(svws.hymn_identity_sa("dasati", (3, 4)), (1, 3, 1, 4))
        self.assertEqual(svws.hymn_identity_sa("dasati", (3, 7)), (1, 3, 2, 2))

    def test_sa_only_chapters(self):
        self.assertEqual(svws.hymn_identity_sa("aranya", (3,)), (1, 1, 3, 3))
        self.assertEqual(svws.hymn_identity_sa("mahanamnya", ()), (1, 1, 4, 1))

    def test_ardha_hymn_from_section(self):
        self.assertEqual(svws.hymn_identity_sa("ardha", (9, 2)), (2, 9, 2, 0))


class PairingTests(unittest.TestCase):
    def test_pair_ardha_match_mismatch_saonly(self):
        sections = [
            (1, [(10, "a ॥ १० ॥")]),
            (2, [(11, "b ॥ ११ ॥"), (12, "c ॥ १२ ॥")]),
            (4, [(14, "d ॥ १४ ॥")]),
        ]
        counts = {1: 1, 2: 3, 5: 2}
        paired, notes = svws.pair_ardha_sections(sections, counts)
        self.assertEqual([(h, len(s)) for h, s in paired], [(1, 1), (-4, 1)])
        # nota 0 = seção 2 pulada (contagem), nota 1 = seção 4 sa-only,
        # nota 2 = hino 5 Griffith sem seção no wikisource
        self.assertEqual(len(notes), 3)
        self.assertIn("pulada", notes[0])
        self.assertIn("sa-only", notes[1])
        self.assertIn("5", notes[2])

    def test_pair_ardha_markerless_skipped(self):
        paired, notes = svws.pair_ardha_sections([(None, [(1, "x ॥ १ ॥")])], {})
        self.assertEqual(paired, [])
        self.assertEqual(len(notes), 1)

    def test_pair_dasati(self):
        arcas = [(1, "a ॥ १ ॥"), (2, "b ॥ २ ॥")]
        paired, notes = svws.pair_dasati(arcas, 2)
        self.assertEqual([(h, len(s)) for h, s in paired], [(0, 2)])
        self.assertEqual(notes, [])

        paired, notes = svws.pair_dasati(arcas, None)
        self.assertEqual(paired, [])
        self.assertIn("ausente", notes[0])

        paired, notes = svws.pair_dasati(arcas, 3)
        self.assertEqual(paired, [])
        self.assertIn("vs Griffith 3", notes[0])


class BuildHymnsTests(unittest.TestCase):
    def test_build_hymns_full(self):
        fetched = [
            ("dasati", (1, 1), RAW_DASATI_POEM),
            ("ardha", (1, 1), RAW_ARDHA),
            ("aranya", (1,), RAW_DASATI_POEM),
            ("mahanamnya", (), RAW_DASATI_POEM),
        ]
        griffith = _griffith(
            [
                {"part": 1, "book": 1, "chapter": 1, "hymn": 1, "verses": [{"n": 1, "text": "x"}] * 2},
                {"part": 2, "book": 1, "chapter": 1, "hymn": 2, "verses": [{"n": 1, "text": "y"}] * 2},
                {"part": 2, "book": 1, "chapter": 1, "hymn": 4, "verses": [{"n": 1, "text": "z"}]},
            ]
        )
        compact, notes = svws.build_hymns(fetched, griffith)
        self.assertEqual(compact["work"], "samaveda")
        self.assertEqual(compact["language"], "sa")
        self.assertEqual(compact["license"], "cc-by-sa")
        keys = [(h["part"], h["book"], h["chapter"], h["hymn"]) for h in compact["hymns"]]
        # dasati pareada; ardha s2 pareada; s3 diverge (1 arca vs 1 verso do
        # hino 3? não existe hino 3 → sa-only); āraṇya e mahānāmnya sa-only
        self.assertIn((1, 1, 1, 1), keys)
        self.assertIn((2, 1, 1, 2), keys)
        self.assertIn((2, 1, 1, 3), keys)  # seção 3 sai sa-only com o nº do marcador
        self.assertIn((1, 1, 3, 1), keys)
        self.assertIn((1, 1, 4, 1), keys)
        # URLs únicos por hino (o ingest dedupe por source_url)
        urls = [h.get("source_url") for h in compact["hymns"]]
        self.assertEqual(len(urls), len(set(urls)))
        ardha_url = next(
            u
            for (p, _, _, _), u in zip(keys, urls, strict=True)
            if p == 2 and "#section-" in u
        )
        self.assertIn("#section-", ardha_url)
        # hino 4 Griffith fica sem seção correspondente (s4 não existe no raw)
        self.assertTrue(any("4" in n for n in notes))

    def test_build_hymns_sa_only_aranya_uses_arcas(self):
        # regressão: āraṇya recebia a tupla (marker, arcas) em vez dos arcas
        fetched = [("aranya", (1,), RAW_DASATI_POEM)]
        compact, _ = svws.build_hymns(fetched, _griffith([]))
        self.assertEqual(len(compact["hymns"]), 1)
        hymn = compact["hymns"][0]
        self.assertEqual([v["n"] for v in hymn["verses"]], [1, 2])
        self.assertEqual(hymn["arca_start"], 1)

    def test_hymn_entry_numbering_relative(self):
        entry = svws._hymn_entry(2, 4, 1, 22, [(1107, "a ॥ ११०७ ॥"), (1108, "b ॥ ११०८ ॥")])
        self.assertEqual([v["n"] for v in entry["verses"]], [1, 2])
        self.assertEqual(entry["arca_start"], 1107)


if __name__ == "__main__":
    unittest.main()
