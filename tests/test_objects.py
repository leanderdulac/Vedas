"""Object store S3/MinIO (raw + artefatos) — testes sem rede via client fake."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from vedic_pipeline.storage import objects as obj


class FakePaginator:
    def __init__(self, store: dict[str, bytes]):
        self.store = store

    def paginate(self, Bucket: str, Prefix: str = ""):
        contents = [{"Key": k} for k in sorted(self.store) if k.startswith(Prefix)]
        yield {"Contents": contents}


class FakeS3Client:
    """Fake mínimo do subset boto3 usado (upload/download/list/head/create)."""

    def __init__(self):
        self.buckets: set[str] = set()
        self.store: dict[str, bytes] = {}

    def head_bucket(self, Bucket: str):
        if Bucket not in self.buckets:
            raise RuntimeError("404 Not Found")

    def create_bucket(self, Bucket: str):
        if Bucket in self.buckets:
            raise RuntimeError("409 BucketAlreadyOwnedByYou")
        self.buckets.add(Bucket)

    def upload_file(self, filename: str, bucket: str, key: str):
        self.store[key] = Path(filename).read_bytes()

    def download_file(self, bucket: str, key: str, filename: str):
        Path(filename).write_bytes(self.store[key])

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        return FakePaginator(self.store)


class ObjectStoreTests(unittest.TestCase):
    def test_build_key_normalizes_and_blocks_traversal(self):
        self.assertEqual(obj.build_key("artifacts", "embeddings/a.npy"), "artifacts/embeddings/a.npy")
        self.assertEqual(obj.build_key("", "a/b.txt"), "a/b.txt")
        self.assertEqual(obj.build_key("/pfx/", "\\win\\p.txt"), "pfx/win/p.txt")
        for bad in ("../escape.txt", "a/../../x", ""):
            with self.assertRaises(ValueError, msg=bad):
                obj.build_key("pfx", bad)

    def test_config_defaults_to_local(self):
        with patch.dict(os.environ, {}, clear=False):
            for var in ("VEDIC_S3_BUCKET", "AWS_S3_BUCKET", "VEDIC_S3_ENABLED"):
                os.environ.pop(var, None)
            cfg = obj.get_s3_config()
            self.assertFalse(cfg.enabled)
            self.assertFalse(obj.is_s3_enabled())
            st = obj.status()
            self.assertEqual(st["backend"], "local")
            self.assertNotIn("secret", str(st).lower())

    def test_config_enabled_by_bucket_and_explicit_off_wins(self):
        with patch.dict(os.environ, {"VEDIC_S3_BUCKET": "vedas", "VEDIC_S3_ENDPOINT": "http://localhost:9000"}):
            self.assertTrue(obj.is_s3_enabled())
            self.assertEqual(obj.get_s3_config().bucket, "vedas")
        with patch.dict(os.environ, {"VEDIC_S3_BUCKET": "vedas", "VEDIC_S3_ENABLED": "0"}):
            self.assertFalse(obj.is_s3_enabled())

    def test_status_never_exposes_secret(self):
        with patch.dict(
            os.environ,
            {"VEDIC_S3_BUCKET": "vedas", "VEDIC_S3_SECRET_KEY": "segredo-supersecreto"},
        ):
            dumped = str(obj.status())
            self.assertNotIn("segredo-supersecreto", dumped)

    def test_push_pull_roundtrip_with_fake(self):
        fake = FakeS3Client()
        with TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            (src / "sub").mkdir(parents=True)
            (src / "a.txt").write_text("alpha")
            (src / "sub" / "b.txt").write_text("beta")
            res = obj.push_dir(src, "artifacts", client=fake, bucket="vedas")
            self.assertEqual(res["uploaded"], 2)
            self.assertEqual(res["bytes"], len(b"alpha") + len(b"beta"))
            self.assertTrue(res["bucket_created"])
            self.assertEqual(sorted(fake.store), ["artifacts/a.txt", "artifacts/sub/b.txt"])

            dst = Path(tmp) / "dst"
            pulled = obj.pull_dir("artifacts", dst, client=fake, bucket="vedas")
            self.assertEqual(pulled["downloaded"], 2)
            self.assertEqual((dst / "a.txt").read_text(), "alpha")
            self.assertEqual((dst / "sub" / "b.txt").read_text(), "beta")

    def test_push_dry_run_writes_nothing(self):
        fake = FakeS3Client()
        with TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            src.mkdir()
            (src / "a.txt").write_text("x")
            res = obj.push_dir(src, "p", client=fake, bucket="b", dry_run=True)
            self.assertTrue(res["dry_run"])
            self.assertEqual(res["uploaded"], 1)
            self.assertEqual(fake.store, {})

    def test_pull_rejects_escape_keys(self):
        fake = FakeS3Client()
        fake.store["p/../evil.txt"] = b"x"
        with TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            obj.pull_dir("p", Path(tmp) / "dst", client=fake, bucket="b")

    def test_push_without_backend_raises_helpful_error(self):
        with TemporaryDirectory() as tmp:
            src = Path(tmp)
            (src / "a.txt").write_text("x")
            with patch.dict(os.environ, {}, clear=False):
                for var in ("VEDIC_S3_BUCKET", "AWS_S3_BUCKET"):
                    os.environ.pop(var, None)
                with self.assertRaises(RuntimeError) as ctx:
                    obj.push_dir(src, "p")
                self.assertIn("VEDIC_S3_BUCKET", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
