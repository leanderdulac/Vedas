"""Fontes licenciadas (BBT, permissão pendente): coletor, flag e gate do índice.

As fixtures são sintéticas; nenhum texto real da BBT entra no repositório.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vedic_pipeline.crawler.licensed_sources import (
    LICENSED_SOURCES,
    LICENSED_SOURCES_ENV,
    is_restricted,
    licensed_sources_enabled,
    load_licensed_records,
    records_for_index,
    sb_record_to_corpus,
    visible_mask,
)
from vedic_pipeline.crawler.licenses import validate_source
from vedic_pipeline.etl import vedabase_sb as vb
from vedic_pipeline.search.entity import work_key

# Página sintética com a mesma estrutura de classes do Vedabase.
FAKE_VERSE_PAGE = """
<html><body><nav><a href="/pt-br/library/sb/9/9/9/">x</a></nav>
<h1 id="h" class="text-center">ŚB 9.9.2-3</h1>
<div class="av-devanagari"><h2 class="hidden">Devanagari</h2><div id="a"><div class="c">
क ख ग ।<br/>घ ङ ॥ २ ॥</div></div></div>
<div class="av-verse_text"><h2 class="hidden">Verse text</h2><div id="b"><div class="c">
ka kha ga<br /><em>gha ṅa</em></div></div></div>
<div class="av-synonyms"><h2>Synonyms</h2><div><span>SINONIMO-SINTETICO</span></div></div>
<div class="av-translation"><h2>Translation</h2><div id="d"><div class="c"><strong>Tradução
sintética &amp; de teste.</strong></div></div></div>
<a href="/pt-br/library/sb/9/9/1/">anterior</a>
<div class="av-purport"><h2>Purport</h2><div><p>SIGNIFICADO-SINTETICO</p></div></div>
</body></html>
"""

FAKE_CHAPTER_PAGE = """
<a href="/pt-br/library/sb/9/9/">cap</a><a href="/pt-br/library/sb/9/8/">anterior</a>
<a href="/pt-br/library/sb/9/9/1/">1</a><a href="/pt-br/library/sb/9/9/10/">10</a>
<a href="/pt-br/library/sb/9/9/2-3/">2-3</a><a href="/pt-br/library/sb/9/9/advanced-view">av</a>
<a href="/pt-br/search/synonyms/?original=ka">s</a>
"""


class VedabaseParserTests(unittest.TestCase):
    def test_cache_keeps_only_verse_fields(self) -> None:
        frags = vb.extract_fragments(FAKE_VERSE_PAGE)
        cached = vb.fragments_to_cache_html(frags)
        self.assertNotIn("SINONIMO-SINTETICO", cached)
        self.assertNotIn("SIGNIFICADO-SINTETICO", cached)
        self.assertNotIn("anterior", cached)
        rec = vb.record_from_fragments("/pt-br/library/sb/9/9/2-3/", vb.cache_html_to_fragments(cached))
        self.assertEqual(rec["locator"], "SB 9.9.2-3")
        self.assertEqual((rec["canto"], rec["chapter"], rec["verse_start"], rec["verse_end"]), (9, 9, 2, 3))
        self.assertEqual(rec["devanagari"], "क ख ग ।\nघ ङ ॥ २ ॥")
        self.assertEqual(rec["transliteration"], "ka kha ga\ngha ṅa")
        self.assertEqual(rec["translation_pt"], "Tradução\nsintética & de teste.")
        self.assertEqual(rec["url"], "https://vedabase.io/pt-br/library/sb/9/9/2-3/")
        self.assertNotIn("missing", rec)

    def test_missing_fields_are_flagged(self) -> None:
        rec = vb.record_from_fragments("/pt-br/library/sb/1/1/1/", {"devanagari": "<div>क</div>"})
        self.assertEqual(rec["missing"], ["transliteration", "translation_pt"])

    def test_child_links_numeric_and_ordered(self) -> None:
        links = vb.child_links(FAKE_CHAPTER_PAGE, "/pt-br/library/sb/9/9/")
        self.assertEqual(
            links,
            ["/pt-br/library/sb/9/9/1/", "/pt-br/library/sb/9/9/2-3/", "/pt-br/library/sb/9/9/10/"],
        )

    def test_build_jsonl_from_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cf = out / "cache" / "sb" / "9" / "9" / "2-3.html"
            cf.parent.mkdir(parents=True)
            cf.write_text(vb.fragments_to_cache_html(vb.extract_fragments(FAKE_VERSE_PAGE)), "utf-8")
            summary = vb.build_jsonl(out)
            self.assertEqual(summary["records"], 1)
            rows = [json.loads(x) for x in (out / "sb_ptbr.jsonl").read_text("utf-8").splitlines()]
            self.assertEqual(rows[0]["locator"], "SB 9.9.2-3")


class LicensedSourceGateTests(unittest.TestCase):
    def _write_source(self, root: Path) -> None:
        src = root / LICENSED_SOURCES["vedabase-sb-ptbr"]["local_path"]
        src.parent.mkdir(parents=True)
        rec = vb.record_from_fragments("/pt-br/library/sb/9/9/2-3/", vb.extract_fragments(FAKE_VERSE_PAGE))
        src.write_text(json.dumps(rec, ensure_ascii=False) + "\n", "utf-8")

    def test_flag_off_by_default(self) -> None:
        self.assertFalse(licensed_sources_enabled({}))
        self.assertFalse(licensed_sources_enabled({LICENSED_SOURCES_ENV: "0"}))
        self.assertTrue(licensed_sources_enabled({LICENSED_SOURCES_ENV: "1"}))

    def test_registry_records_licence_and_blocks_generic_download(self) -> None:
        entry = LICENSED_SOURCES["vedabase-sb-ptbr"]
        self.assertEqual(entry["license_note"], "BBT — permission pending, private use")
        self.assertFalse(entry["download"])
        self.assertTrue(entry["local_path"].startswith("data/raw/"))
        ok, _ = validate_source(entry)
        self.assertFalse(ok)

    def test_records_only_with_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_source(root)
            self.assertEqual(load_licensed_records(root, enabled=False), [])
            recs = load_licensed_records(root, enabled=True)
            self.assertEqual(len(recs), 1)
            self.assertTrue(is_restricted(recs[0]))
            public = [{"id": "x", "text": "t", "license": "public-domain"}]
            self.assertEqual(records_for_index(public, root), public)  # flag fora do ambiente

    def test_same_work_as_dutt_in_entity_mode(self) -> None:
        rec = vb.record_from_fragments("/pt-br/library/sb/9/9/2-3/", vb.extract_fragments(FAKE_VERSE_PAGE))
        corpus = sb_record_to_corpus(rec)
        self.assertEqual(work_key(corpus["title"]), work_key("Bhāgavata Purāṇa — Book 1, ch. 1–3 (M. N. Dutt, 1896)"))
        self.assertEqual(corpus["locator"], "SB 9.9.2-3")
        self.assertIn("Tradução: Tradução", corpus["text"])
        self.assertEqual(corpus["language"], "pt")

    def test_chunks_inherit_restriction(self) -> None:
        from vedic_pipeline.etl.chunking import chunk_records

        rec = vb.record_from_fragments("/pt-br/library/sb/9/9/2-3/", vb.extract_fragments(FAKE_VERSE_PAGE))
        chunks = list(chunk_records([sb_record_to_corpus(rec)]))
        self.assertTrue(chunks and all(is_restricted(c) for c in chunks))
        self.assertEqual(visible_mask(chunks, enabled=False), [False] * len(chunks))
        self.assertEqual(visible_mask(chunks, enabled=True), [True] * len(chunks))

    def test_index_loader_hides_restricted_chunks_when_off(self) -> None:
        import os
        from unittest import mock

        from vedic_pipeline.search.embeddings import load_embedding_index

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            chunks = [
                {"chunk_id": "a", "doc_id": "a", "text": "agni hotar", "license": "public-domain", "title": "RV"},
                {"chunk_id": "b", "doc_id": "b", "text": "nárada muni", "license": "bbt-permission-pending",
                 "title": "Bhāgavata Purāṇa — SB 1.1.1 (BBT, pt-br)"},
            ]
            vecs = np.eye(2, 4, dtype=np.float32)
            np.save(d / "embeddings.npy", vecs)
            (d / "chunks.jsonl").write_text("\n".join(json.dumps(c) for c in chunks) + "\n", "utf-8")
            with mock.patch.dict(os.environ, {LICENSED_SOURCES_ENV: ""}):
                idx = load_embedding_index(d)
            self.assertEqual([c["chunk_id"] for c in idx["chunks"]], ["a"])
            self.assertEqual(idx["vectors"].shape[0], 1)
            with mock.patch.dict(os.environ, {LICENSED_SOURCES_ENV: "1"}):
                idx = load_embedding_index(d)
            self.assertEqual(len(idx["chunks"]), 2)


if __name__ == "__main__":
    unittest.main()
