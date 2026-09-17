"""Object store S3-compatível (MinIO/AWS) para raw e artefatos, com fallback local.

Operação é local-first: o pipeline lê/escreve no filesystem como antes.
Este módulo adiciona backup/restauração versionável de diretórios inteiros
(`data/raw`, `artifacts/...`) para um bucket S3 — útil para multi-container
e disaster recovery.

Configuração via env (nenhum segredo é logado):

    VEDIC_S3_BUCKET      bucket alvo (define habilitação; ex.: vedas)
    VEDIC_S3_ENDPOINT    endpoint S3/MinIO (ex.: http://localhost:9000)
    VEDIC_S3_PREFIX      prefixo base das chaves (default: "")
    VEDIC_S3_REGION      região (default: us-east-1)
    VEDIC_S3_ACCESS_KEY  (fallback: AWS_ACCESS_KEY_ID / MINIO_ROOT_USER)
    VEDIC_S3_SECRET_KEY  (fallback: AWS_SECRET_ACCESS_KEY / MINIO_ROOT_PASSWORD)

``boto3`` é dependência opcional (extra ``s3``): só é importado quando um
push/pull real é executado. Sem bucket configurado, os comandos informam
que o backend está desabilitado em vez de falhar com traceback obscuro.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("vedic_pipeline.storage.objects")


@dataclass(frozen=True)
class S3Config:
    enabled: bool
    endpoint: str | None
    bucket: str
    prefix: str
    region: str


def get_s3_config() -> S3Config:
    """Lê a configuração S3 do ambiente (sem expor segredos no objeto)."""
    bucket = (
        os.environ.get("VEDIC_S3_BUCKET") or os.environ.get("AWS_S3_BUCKET") or ""
    ).strip()
    endpoint = (
        os.environ.get("VEDIC_S3_ENDPOINT")
        or os.environ.get("AWS_S3_ENDPOINT_URL")
        or os.environ.get("AWS_ENDPOINT_URL_S3")
        or ""
    ).strip() or None
    prefix = (os.environ.get("VEDIC_S3_PREFIX") or "").strip().strip("/")
    region = (os.environ.get("VEDIC_S3_REGION") or os.environ.get("AWS_REGION") or "us-east-1").strip()
    explicit = (os.environ.get("VEDIC_S3_ENABLED") or "").strip().lower()
    if explicit in {"0", "false", "off", "no"}:
        enabled = False
    elif explicit in {"1", "true", "on", "yes"}:
        enabled = bool(bucket)
    else:
        enabled = bool(bucket)
    return S3Config(enabled=enabled, endpoint=endpoint, bucket=bucket, prefix=prefix, region=region or "us-east-1")


def get_s3_credentials() -> tuple[str | None, str | None]:
    access = (
        os.environ.get("VEDIC_S3_ACCESS_KEY")
        or os.environ.get("AWS_ACCESS_KEY_ID")
        or os.environ.get("MINIO_ROOT_USER")
    )
    secret = (
        os.environ.get("VEDIC_S3_SECRET_KEY")
        or os.environ.get("AWS_SECRET_ACCESS_KEY")
        or os.environ.get("MINIO_ROOT_PASSWORD")
    )
    return (access or None, secret or None)


def is_s3_enabled() -> bool:
    return get_s3_config().enabled


def status() -> dict[str, Any]:
    """Status público do backend (sem segredos)."""
    cfg = get_s3_config()
    access, _secret = get_s3_credentials()
    return {
        "enabled": cfg.enabled,
        "endpoint": cfg.endpoint,
        "bucket": cfg.bucket,
        "prefix": cfg.prefix,
        "region": cfg.region,
        "credentials_set": bool(access),
        "backend": "s3" if cfg.enabled else "local",
    }


def require_boto3() -> Any:
    try:
        import boto3  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "boto3 não instalado. Instale o extra S3: pip install -e '.[s3]' "
            "ou pip install boto3>=1.28"
        ) from exc
    return boto3


def make_client(config: S3Config | None = None) -> Any:
    """Cria o client S3 (boto3). Levanta RuntimeError amigável sem boto3/bucket."""
    cfg = config or get_s3_config()
    if not cfg.enabled:
        raise RuntimeError(
            "Backend S3 desabilitado. Defina VEDIC_S3_BUCKET (e VEDIC_S3_ENDPOINT "
            "para MinIO) ou use os arquivos locais."
        )
    boto3 = require_boto3()
    access, secret = get_s3_credentials()
    kwargs: dict[str, Any] = {"region_name": cfg.region}
    if cfg.endpoint:
        kwargs["endpoint_url"] = cfg.endpoint
    if access:
        kwargs["aws_access_key_id"] = access
    if secret:
        kwargs["aws_secret_access_key"] = secret
    return boto3.client("s3", **kwargs)


def build_key(prefix: str, relative: str) -> str:
    """Mapeia caminho relativo local → chave S3 (sempre com `/`)."""
    rel = relative.replace("\\", "/").lstrip("/")
    if not rel or rel in {".", "./"}:
        raise ValueError(f"caminho relativo inválido: {relative!r}")
    parts = [p for p in f"{prefix.strip().strip('/')}/{rel}".split("/") if p not in {"", "."}]
    safe: list[str] = []
    for part in parts:
        if part == "..":
            raise ValueError(f"caminho fora da base: {relative!r}")
        safe.append(part)
    return "/".join(safe)


def iter_local_files(base: Path) -> list[Path]:
    base = Path(base)
    if not base.exists():
        return []
    return sorted(p for p in base.rglob("*") if p.is_file() and not p.is_symlink())


def ensure_bucket(client: Any, bucket: str) -> bool:
    """Cria o bucket se ausente. Retorna True se criou, False se já existia.

    Fora de us-east-1, o AWS exige ``CreateBucketConfiguration``; o MinIO
    ignora. Como o client fake dos testes só aceita ``Bucket``, o kwarg extra
    é adicionado apenas quando a região difere do default.
    """
    try:
        client.head_bucket(Bucket=bucket)
        return False
    except Exception:
        pass
    region = get_s3_config().region
    kwargs: dict[str, Any] = {"Bucket": bucket}
    if region and region != "us-east-1":
        kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
    try:
        client.create_bucket(**kwargs)
        return True
    except Exception as exc:  # noqa: BLE001
        message = str(exc).lower()
        if "exists" in message or "already" in message or "409" in message:
            return False
        raise


def push_dir(
    local_dir: str | Path,
    prefix: str = "",
    *,
    client: Any = None,
    bucket: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Envia um diretório local para o bucket. Retorna contagens (sem segredos)."""
    base = Path(local_dir)
    files = iter_local_files(base)
    s3 = client or make_client()
    target_bucket = bucket or get_s3_config().bucket
    if not target_bucket:
        raise RuntimeError("Bucket S3 não configurado (VEDIC_S3_BUCKET)")
    created = False if dry_run else ensure_bucket(s3, target_bucket)
    uploaded = 0
    total_bytes = 0
    keys: list[str] = []
    for path in files:
        rel = path.relative_to(base).as_posix()
        key = build_key(prefix, rel)
        size = path.stat().st_size
        keys.append(key)
        if not dry_run:
            s3.upload_file(str(path), target_bucket, key)
        uploaded += 1
        total_bytes += size
    logger.info("push s3://%s/%s: %d arquivos (%d bytes, dry_run=%s)", target_bucket, prefix or "-", uploaded, total_bytes, dry_run)
    return {
        "backend": "s3",
        "bucket": target_bucket,
        "prefix": prefix,
        "uploaded": uploaded,
        "bytes": total_bytes,
        "bucket_created": created,
        "dry_run": dry_run,
        "keys": keys[:100],
        "truncated_keys": max(0, len(keys) - 100),
    }


def list_remote_keys(prefix: str = "", *, client: Any = None, bucket: str | None = None) -> list[str]:
    s3 = client or make_client()
    target_bucket = bucket or get_s3_config().bucket
    if not target_bucket:
        raise RuntimeError("Bucket S3 não configurado (VEDIC_S3_BUCKET)")
    norm = prefix.strip().strip("/")
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=target_bucket, Prefix=f"{norm}/" if norm else ""):
        for obj in page.get("Contents", []) or []:
            key = obj.get("Key") or ""
            if key and not key.endswith("/"):
                keys.append(key)
    return sorted(keys)


def pull_dir(
    prefix: str,
    local_dir: str | Path,
    *,
    client: Any = None,
    bucket: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Baixa um prefixo do bucket para o diretório local."""
    dest = Path(local_dir)
    keys = list_remote_keys(prefix, client=client, bucket=bucket)
    norm = prefix.strip().strip("/")
    downloaded = 0
    total_bytes = 0
    s3 = client or make_client()
    target_bucket = bucket or get_s3_config().bucket
    for key in keys:
        rel = key[len(norm) + 1 :] if norm and key.startswith(f"{norm}/") else key
        if not rel:
            continue
        target = (dest / rel).resolve()
        if not str(target).startswith(str(dest.resolve())):
            raise ValueError(f"chave S3 fora da base: {key!r}")
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(target_bucket, key, str(target))
            total_bytes += target.stat().st_size
        downloaded += 1
    logger.info("pull s3://%s/%s: %d arquivos (%d bytes, dry_run=%s)", target_bucket, prefix or "-", downloaded, total_bytes, dry_run)
    return {
        "backend": "s3",
        "bucket": target_bucket,
        "prefix": prefix,
        "downloaded": downloaded,
        "bytes": total_bytes,
        "dry_run": dry_run,
    }
