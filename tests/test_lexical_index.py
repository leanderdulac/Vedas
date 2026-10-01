"""O índice lexical persistente reproduz o BM25 que varre os chunks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vedic_pipeline.search import lexical_index
from vedic_pipeline.search.hybrid import _lexical_scores_scan, lexical_scores


def _chunks() -> list[dict]:
    return [
        {
            "chunk_id": "c-atman",
            "title": "Isha",
            "text": "ātman is the self that the upanishad names",
        },
        {
            "chunk_id": "c-dharma",
            "title": "Manu",
            "text": "dharma and duty hold the world",
        },
        {
            "chunk_id": "c-purusha",
            "title": "Purusha Sukta",
            "text": "purusha alone, the cosmic person",
        },
        {
            "chunk_id": "c-empty",
            "title": "Blank",
            "text": "",
        },
    ]


class LexicalIndexTests(unittest.TestCase):
    def setUp(self):
        lexical_index._BY_CHUNK.clear()

    def tearDown(self):
        lexical_index._BY_CHUNK.clear()

    def test_fast_path_matches_the_scan_including_expansion(self):
        chunks = _chunks()
        with tempfile.TemporaryDirectory() as tmp:
            lexical_index.save_lexical_index(Path(tmp), chunks)
            for query in ("self", "dharma", "yoga sutra", "ātman", ""):
                with self.subTest(query=query):
                    scan = _lexical_scores_scan(query, chunks)
                    fast = lexical_scores(query, chunks)
                    self.assertEqual(len(fast), len(scan))
                    for got, expected in zip(fast, scan, strict=True):
                        self.assertAlmostEqual(got, expected, places=5)
            subset = [chunks[1], chunks[0]]
            scan = _lexical_scores_scan("self", subset)
            fast = lexical_scores("self", subset)
            for got, expected in zip(fast, scan, strict=True):
                self.assertAlmostEqual(got, expected, places=5)

    def test_reload_keeps_the_same_scores(self):
        chunks = _chunks()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            first = lexical_index.save_lexical_index(directory, chunks)
            lexical_index._BY_CHUNK.clear()
            loaded = lexical_index.load_lexical_index(
                directory, expected_fingerprint=lexical_index.fingerprint(chunks)
            )
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(
                first.score("self", [0, 1, 2]),
                loaded.score("self", [0, 1, 2]),
            )

    def test_stale_fingerprint_is_rebuilt(self):
        chunks = _chunks()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            lexical_index.save_lexical_index(directory, chunks)
            changed = [dict(chunks[0], text=chunks[0]["text"] + " extra"), *chunks[1:]]
            lexical_index._BY_CHUNK.clear()
            self.assertIsNone(
                lexical_index.load_lexical_index(
                    directory, expected_fingerprint=lexical_index.fingerprint(changed)
                )
            )
            rebuilt = lexical_index.ensure_lexical_index(directory, changed)
            self.assertEqual(rebuilt.n_docs, 4)
            self.assertGreater(rebuilt.score("extra", [0, 1, 2])[0], 0.0)

    def test_unknown_chunks_fall_back_to_the_scan(self):
        ad_hoc = [{"title": "Nota", "text": "agni the priest"}]
        self.assertEqual(
            lexical_scores("agni", ad_hoc),
            _lexical_scores_scan("agni", ad_hoc),
        )


if __name__ == "__main__":
    unittest.main()
