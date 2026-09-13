import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from vedic_pipeline.api.app import create_app
from vedic_pipeline.api.schemas import AskRequest, BuildIndexRequest, SearchRequest
from vedic_pipeline.etl.chunking import chunk_text


class ChunkTests(unittest.TestCase):
    def test_invalid_window(self):
        for size, overlap in [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11)]:
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                chunk_text('text', size, overlap)

    def test_short_and_empty(self):
        self.assertEqual(chunk_text('  '), [])
        self.assertEqual(chunk_text(' abc ', 10, 2), ['abc'])

    def test_overlap_preserves_continuous_text(self):
        self.assertEqual(chunk_text('abcdefghijklmnop', 10, 3), ['abcdefghij', 'hijklmnop'])

    def test_early_boundary_cannot_stall(self):
        # A subprocess timeout catches regressions without hanging the test suite.
        result = subprocess.run([sys.executable, '-c',
            "from vedic_pipeline.etl.chunking import chunk_text; "
            "pieces = chunk_text('aaaa bbbb cccc dddd', 10, 9); "
            "assert pieces[-1].endswith('dddd'); assert len(pieces) < 20"],
            timeout=5, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class SchemaTests(unittest.TestCase):
    def test_query_validation(self):
        for cls in [SearchRequest, AskRequest]:
            self.assertEqual(cls(query='  atman  ').query, 'atman')
            for query in ['  \n', 'x' * 4001]:
                with self.assertRaises(ValidationError):
                    cls(query=query)
            with self.assertRaises(ValidationError):
                cls(query='atman', backend='invalid')

    def test_build_validation(self):
        for kwargs in [dict(chunk_size=100, overlap=100), dict(backend='invalid')]:
            with self.assertRaises(ValidationError):
                BuildIndexRequest(**kwargs)


class ApiTests(unittest.TestCase):
    def test_pipeline_disabled_without_token(self):
        with patch.dict(os.environ, {'VEDIC_PIPELINE_API_TOKEN': ''}), TestClient(create_app()) as client:
            for route in ['/ingest', '/tokenize', '/train', '/build-index', '/db/init', '/db/sync']:
                self.assertEqual(client.post(route, json={'manifest': 'unused'}).status_code, 503)
            self.assertEqual(client.get('/api/v1/documents').status_code, 200)

    def test_pipeline_requires_bearer_and_accepts_configured_token(self):
        with patch.dict(os.environ, {'VEDIC_PIPELINE_API_TOKEN': 'test-only-token'}), \
             patch('vedic_pipeline.storage.db.init_schema', return_value={'ok': True}) as operation, \
             TestClient(create_app()) as client:
            for value in ['', 'Bearer wrong', 'Basic test-only-token']:
                self.assertEqual(client.post('/db/init', headers={'Authorization': value}).status_code, 401)
            operation.assert_not_called()
            response = client.post('/db/init', headers={'Authorization': 'Bearer test-only-token'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {'ok': True})
            operation.assert_called_once()

    def test_spa_blocks_traversal_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dist = root / 'dist'
            dist.mkdir()
            (dist / 'index.html').write_text('SPA')
            (root / 'private.txt').write_text('PRIVATE')
            (dist / 'escape.txt').symlink_to(root / 'private.txt')
            (dist / 'public.txt').write_text('PUBLIC')
            with patch('vedic_pipeline.api.app.FRONTEND_DIST', dist), TestClient(create_app()) as client:
                for url in ['/%2e%2e%2fprivate.txt', '/escape.txt']:
                    response = client.get(url)
                    self.assertEqual(response.status_code, 404)
                    self.assertNotIn('PRIVATE', response.text)
                self.assertEqual(client.get('/public.txt').text, 'PUBLIC')
                self.assertEqual(client.get('/biblioteca').text, 'SPA')
                self.assertEqual(client.get('/api/v1/missing').status_code, 404)


class RetrievalTests(unittest.TestCase):
    def test_hyphenated_titles_match_spaced_queries(self):
        from vedic_pipeline.search.hybrid import tokenize
        self.assertEqual(tokenize('Bhagavad-gita'), tokenize('Bhagavad Gita'))
        self.assertEqual(tokenize('Yoga-sutras'), tokenize('Yoga sutras'))
        self.assertEqual(tokenize('इ॒षे'), tokenize('इषे'))

    def test_phrase_boost_beats_semantic_near_misses(self):
        from vedic_pipeline.search.hybrid import hybrid_rerank

        hits = hybrid_rerank(
            "कर्मण्येवाधिकारस्ते मा फलेषु",
            [
                {
                    "chunk_id": "noise",
                    "doc_id": "rv",
                    "title": "Rigveda RV 9.22 (Sanskrit, DharmicData)",
                    "text": "इन्द्राय सोमं मधुमन्तम",
                    "score": 0.81,
                },
                {
                    "chunk_id": "gita",
                    "doc_id": "bg",
                    "title": "Bhagavad-gītā 2 (Sanskrit, DharmicData)",
                    "text": "2.47 कर्मण्येवाधिकारस्ते मा फलेषु कदाचन।",
                    "score": 0.40,
                },
            ],
            top_k=2,
            max_per_doc=2,
            use_cross_encoder=False,
        )
        self.assertEqual(hits[0]["chunk_id"], "gita")

    def test_lexical_expansion_respects_filters(self):
        from vedic_pipeline.llm.ask import retrieve_hits
        chunks = [
            dict(chunk_id='allowed', doc_id='1', text='atman self', tradition='upanishad', language='en'),
            dict(chunk_id='wrong-language', doc_id='2', text='atman self', tradition='upanishad', language='sa'),
            dict(chunk_id='wrong-tradition', doc_id='3', text='atman self', tradition='yoga', language='en'),
        ]
        with patch('vedic_pipeline.llm.ask.retrieve', return_value=[]), \
             patch('vedic_pipeline.llm.ask.get_index', return_value={'chunks': chunks}):
            hits, _ = retrieve_hits('atman', backend='numpy', tradition='upanishad', language='en')
            self.assertEqual([hit['chunk_id'] for hit in hits], ['allowed'])
            hits, _ = retrieve_hits('atman', backend='numpy', language='pt')
            self.assertEqual(hits, [])


if __name__ == '__main__':
    unittest.main()
