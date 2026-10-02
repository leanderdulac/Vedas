"""Checklist de deploy (senhas, geração paga, CE)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vedic_pipeline.ops.deploy_check import deploy_check_exit_code, run_deploy_check


class DeployCheckTests(unittest.TestCase):
    def test_local_default_is_ok_with_warnings(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("VEDIC_") and k not in {"XAI_API_KEY", "POSTGRES_PASSWORD"}}
        with patch.dict(os.environ, env, clear=True):
            result = run_deploy_check(mode="local")
        self.assertTrue(result["ok"])
        self.assertEqual(deploy_check_exit_code(result), 0)

    def test_prod_fails_on_weak_password_and_open_generation(self):
        env = {
            "POSTGRES_PASSWORD": "vedas",
            "VEDIC_PIPELINE_API_TOKEN": "",
            "XAI_API_KEY": "xai-test",
            "VEDIC_GENERATION_API_TOKEN": "",
            "VEDIC_REQUIRE_GENERATION_TOKEN": "false",
            "VEDIC_ENABLE_RERANKER": "false",
        }
        with patch.dict(os.environ, env, clear=False):
            result = run_deploy_check(mode="prod")
        codes = {i["code"] for i in result["issues"] if i["level"] == "fail"}
        self.assertFalse(result["ok"])
        self.assertIn("weak_postgres_password", codes)
        self.assertIn("pipeline_token_missing", codes)
        self.assertIn("open_paid_generation", codes)
        self.assertIn("require_generation_token_off", codes)

    def test_prod_passes_with_strong_secrets(self):
        env = {
            "POSTGRES_PASSWORD": "a" * 32,
            "VEDIC_PIPELINE_API_TOKEN": "pipeline-token-16+",
            "XAI_API_KEY": "xai-test",
            "VEDIC_GENERATION_API_TOKEN": "generation-token-16+",
            "VEDIC_REQUIRE_GENERATION_TOKEN": "true",
            "VEDIC_ENABLE_RERANKER": "false",
        }
        with patch.dict(os.environ, env, clear=False):
            result = run_deploy_check(mode="prod")
        self.assertTrue(result["ok"], result["issues"])
        self.assertEqual(deploy_check_exit_code(result), 0)

    def test_reranker_on_without_local_dir_fails(self):
        with patch.dict(
            os.environ,
            {
                "VEDIC_ENABLE_RERANKER": "true",
                "VEDIC_RERANKER_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            },
            clear=False,
        ):
            result = run_deploy_check(mode="local")
        codes = {i["code"] for i in result["issues"] if i["level"] == "fail"}
        self.assertIn("reranker_on_without_local_dir", codes)

    def test_reranker_on_with_local_dir_is_warn_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text("{}", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "VEDIC_ENABLE_RERANKER": "true",
                    "VEDIC_RERANKER_MODEL": tmp,
                    "VEDIC_PIPELINE_API_TOKEN": "",
                    "XAI_API_KEY": "",
                    "POSTGRES_PASSWORD": "",
                },
                clear=False,
            ):
                result = run_deploy_check(mode="local")
        fails = [i for i in result["issues"] if i["level"] == "fail"]
        warns = {i["code"] for i in result["issues"] if i["level"] == "warn"}
        self.assertTrue(result["ok"], fails)
        self.assertIn("reranker_opt_in", warns)

    def test_ssrf_disable_flag_fails(self):
        with patch.dict(os.environ, {"VEDIC_DISABLE_SSRF_DNS_CHECK": "true"}, clear=False):
            result = run_deploy_check(mode="local")
        self.assertFalse(result["ok"])
        self.assertIn("ssrf_dns_check_disabled", {i["code"] for i in result["issues"]})
