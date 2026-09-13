import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from vedic_pipeline.api.app import create_app
from vedic_pipeline.api.schemas import SearchRequest

ASK_ROUTES = ['/ask', '/api/v1/ask', '/ask/stream', '/api/v1/ask/stream']


class RequestPolicyTests(unittest.TestCase):
    def test_arbitrary_paths_are_rejected_before_retrieval(self):
        with patch('vedic_pipeline.llm.ask.retrieve_hits') as search, TestClient(create_app()) as client:
            for route in ['/search', '/api/v1/search', *ASK_ROUTES]:
                for directory in ['/etc', '../../', 'artifacts/other-index']:
                    with self.subTest(route=route, directory=directory):
                        self.assertEqual(client.post(route, json={'query': 'atman', 'index_dir': directory}).status_code, 422)
            search.assert_not_called()

    def test_configured_index_and_aliases(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'VEDIC_API_INDEX_DIR': directory}):
            self.assertEqual(SearchRequest(query='atman').index_dir, str(Path(directory).resolve()))
            self.assertEqual(SearchRequest(query='atman', index_dir=directory + '/.').index_dir, str(Path(directory).resolve()))

    def test_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / 'index'
            index.mkdir()
            alias = root / 'alias'
            alias.symlink_to(root)
            with patch.dict(os.environ, {'VEDIC_API_INDEX_DIR': str(index)}), TestClient(create_app()) as client:
                self.assertEqual(client.post('/search', json={'query': 'atman', 'index_dir': str(alias)}).status_code, 422)

    def test_local_xai_allowed_without_generation_token(self):
        with patch.dict(os.environ, {'VEDIC_GENERATION_API_TOKEN': '', 'XAI_API_KEY': 'test-key', 'XAI_MODEL': 'configured-model'}), \
             patch('vedic_pipeline.llm.ask.ask', return_value={'answer': 'test'}) as ask, \
             TestClient(create_app()) as client:
            self.assertEqual(client.post('/ask', json={'query': 'atman', 'provider': 'auto'}).status_code, 200)
            self.assertEqual(ask.call_args.kwargs['provider'], 'xai')
            self.assertEqual(ask.call_args.kwargs['model'], 'configured-model')

    def test_auth_and_model_selection_are_checked_before_streaming(self):
        with patch.dict(os.environ, {'VEDIC_GENERATION_API_TOKEN': 'test-token', 'XAI_API_KEY': 'test-key', 'XAI_MODEL': 'configured-model'}), \
             patch('vedic_pipeline.llm.ask.ask') as ask, patch('vedic_pipeline.llm.ask.ask_stream_events') as stream, \
             TestClient(create_app()) as client:
            for route in ASK_ROUTES:
                self.assertEqual(client.post(route, json={'query': 'atman'}).status_code, 401)
                self.assertEqual(client.post(route, json={'query': 'atman'}, headers={'Authorization': 'Bearer wrong'}).status_code, 401)
                self.assertEqual(client.post(route, json={'query': 'atman', 'model': 'unapproved'}, headers={'Authorization': 'Bearer test-token'}).status_code, 422)
            ask.assert_not_called()
            stream.assert_not_called()

    def test_authorized_requests_use_server_model(self):
        with patch.dict(os.environ, {'VEDIC_GENERATION_API_TOKEN': 'test-token', 'XAI_API_KEY': 'test-key', 'XAI_MODEL': 'configured-model'}), \
             patch('vedic_pipeline.llm.ask.ask', return_value={'answer': 'test'}) as ask, \
             patch('vedic_pipeline.llm.ask.ask_stream_events', return_value=iter([{'event': 'done', 'data': {'answer': 'test'}}])) as stream, \
             TestClient(create_app()) as client:
            headers = {'Authorization': 'Bearer test-token'}
            self.assertEqual(client.post('/ask', json={'query': 'atman'}, headers=headers).status_code, 200)
            self.assertEqual(ask.call_args.kwargs['model'], 'configured-model')
            self.assertEqual(ask.call_args.kwargs['provider'], 'xai')
            response = client.post('/ask/stream', json={'query': 'atman'}, headers=headers)
            self.assertIn('event: done', response.text)
            self.assertEqual(stream.call_args.kwargs['model'], 'configured-model')

    def test_extractive_stays_public_even_with_xai_configured(self):
        with patch.dict(os.environ, {'VEDIC_GENERATION_API_TOKEN': '', 'XAI_API_KEY': 'test-key'}), \
             patch('vedic_pipeline.llm.ask.ask', return_value={'answer': 'test'}) as ask, TestClient(create_app()) as client:
            self.assertEqual(client.post('/ask', json={'query': 'atman', 'provider': 'extractive'}).status_code, 200)
            self.assertEqual(ask.call_args.kwargs['provider'], 'extractive')
            self.assertEqual(client.post('/ask', json={'query': 'atman', 'provider': 'extractive', 'model': '/private/model'}).status_code, 422)


if __name__ == '__main__':
    unittest.main()
