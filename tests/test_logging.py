"""Testes unitários para o módulo de logging estruturado em JSON."""

from __future__ import annotations

import json
import logging
import unittest

from vedic_pipeline.common.logging import JsonFormatter


class LoggingTests(unittest.TestCase):
    def test_json_formatter_outputs_valid_json(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="vedic_pipeline.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=20,
            msg="Mensagem de teste estruturada",
            args=(),
            exc_info=None,
        )
        record.status = 200
        record.path = "/api/v1/search"

        output = formatter.format(record)
        data = json.loads(output)

        self.assertEqual(data["level"], "INFO")
        self.assertEqual(data["logger"], "vedic_pipeline.test")
        self.assertEqual(data["message"], "Mensagem de teste estruturada")
        self.assertEqual(data["status"], 200)
        self.assertEqual(data["path"], "/api/v1/search")
        self.assertIn("timestamp", data)


if __name__ == "__main__":
    unittest.main()
