"""Perfil + fingerprint do corpus: o que este ambiente realmente tem."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from vedic_pipeline.common.constants import DEFAULT_CORPUS, PROJECT_ROOT
from vedic_pipeline.common.corpus import content_fingerprint, load_corpus, utc_now_iso

PROFILES_PATH = PROJECT_ROOT / "fixtures" / "corpus_profiles.json"


def load_profiles(path: Path | None = None) -> dict[str, Any]:
    dest = Path(path) if path else PROFILES_PATH
    if not dest.exists():
        return {"default_profile": "bootstrap", "profiles": {}}
    return json.loads(dest.read_text(encoding="utf-8"))


def fingerprint_records(records: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for rec in records:
        rid = str(rec.get("id") or rec.get("source_url") or rec.get("title") or "")
        fp = rec.get("fingerprint") or content_fingerprint(rec.get("text") or "")
        lines.append(f"{rid}|{fp}")
    lines.sort()
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def summarize_corpus(corpus_path: Path | None = None) -> dict[str, Any]:
    path = Path(corpus_path) if corpus_path else DEFAULT_CORPUS
    exists = path.exists()
    records = load_corpus(path) if exists else []
    total_chars = 0
    by_tradition: dict[str, int] = {}
    by_language: dict[str, int] = {}
    for rec in records:
        total_chars += int(rec.get("char_count") or len(rec.get("text") or ""))
        tradition = str(rec.get("tradition") or "unknown").lower()
        language = str(rec.get("language") or "und").lower()
        by_tradition[tradition] = by_tradition.get(tradition, 0) + 1
        by_language[language] = by_language.get(language, 0) + 1
    return {
        "corpus_path": str(path),
        "corpus_exists": exists,
        "documents": len(records),
        "total_chars": total_chars,
        "fingerprint": fingerprint_records(records) if records else "",
        "by_tradition": by_tradition,
        "by_language": by_language,
    }


def profile_matches(summary: dict[str, Any], profile: dict[str, Any]) -> bool:
    docs = int(summary.get("documents") or 0)
    minimum = profile.get("min_documents")
    maximum = profile.get("max_documents")
    if minimum is not None and docs < int(minimum):
        return False
    return maximum is None or docs <= int(maximum)


def matched_profiles(
    summary: dict[str, Any],
    profiles: dict[str, Any] | None = None,
) -> list[str]:
    payload = profiles if profiles is not None else load_profiles()
    items = payload.get("profiles") or {}
    return [key for key, spec in items.items() if profile_matches(summary, spec)]


def verify_profile(
    summary: dict[str, Any],
    profile_id: str,
    profiles: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = profiles if profiles is not None else load_profiles()
    spec = (payload.get("profiles") or {}).get(profile_id)
    issues: list[str] = []
    if spec is None:
        issues.append(f"perfil desconhecido: {profile_id}")
        return {"ok": False, "profile": profile_id, "issues": issues}
    docs = int(summary.get("documents") or 0)
    minimum = spec.get("min_documents")
    maximum = spec.get("max_documents")
    expected = spec.get("expected_documents")
    if not summary.get("corpus_exists"):
        issues.append("corpus ausente")
    if minimum is not None and docs < int(minimum):
        issues.append(f"documents {docs} < min {minimum}")
    if maximum is not None and docs > int(maximum):
        issues.append(f"documents {docs} > max {maximum}")
    if expected is not None and docs != int(expected):
        issues.append(f"documents {docs} != expected {expected} (snapshot; não é falha de bootstrap)")
    ok = not any(item.startswith("documents") and "!=" not in item for item in issues) and "corpus ausente" not in issues
    if expected is not None and docs != int(expected) and spec.get("reproducible") is False:
        # Snapshot é informativo — não falha o verify de perfil não-reproduzível.
        ok = "corpus ausente" not in issues and (
            minimum is None or docs >= int(minimum)
        )
    return {
        "ok": ok,
        "profile": profile_id,
        "reproducible": bool(spec.get("reproducible")),
        "issues": issues,
        "spec": {k: spec[k] for k in spec if k != "notes"},
        "notes": spec.get("notes"),
    }


def build_lock(summary: dict[str, Any], *, profile: str | None = None) -> dict[str, Any]:
    return {
        "created_at": utc_now_iso(),
        "profile": profile,
        "corpus_path": summary.get("corpus_path"),
        "documents": summary.get("documents"),
        "total_chars": summary.get("total_chars"),
        "fingerprint": summary.get("fingerprint"),
    }


def write_lock(path: Path, lock: dict[str, Any]) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return dest


def read_lock(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_lock(summary: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    if not summary.get("corpus_exists"):
        issues.append("corpus ausente")
    if (summary.get("fingerprint") or "") != (lock.get("fingerprint") or ""):
        issues.append("fingerprint diverge do lock")
    if int(summary.get("documents") or 0) != int(lock.get("documents") or 0):
        issues.append(
            f"documents {summary.get('documents')} != lock {lock.get('documents')}"
        )
    return {
        "ok": not issues,
        "issues": issues,
        "lock_fingerprint": lock.get("fingerprint"),
        "current_fingerprint": summary.get("fingerprint"),
    }


def corpus_status(
    corpus_path: Path | None = None,
    *,
    profile: str | None = None,
    lock_path: Path | None = None,
) -> dict[str, Any]:
    summary = summarize_corpus(corpus_path)
    profiles = load_profiles()
    matched = matched_profiles(summary, profiles)
    out: dict[str, Any] = {
        **summary,
        "default_profile": profiles.get("default_profile"),
        "matched_profiles": matched,
        "reproducible": any(
            (profiles.get("profiles") or {}).get(key, {}).get("reproducible") for key in matched
        ),
    }
    if profile:
        out["verify_profile"] = verify_profile(summary, profile, profiles)
    if lock_path and Path(lock_path).exists():
        out["verify_lock"] = verify_lock(summary, read_lock(Path(lock_path)))
    return out


def public_corpus_profile(corpus_path: Path | None = None) -> dict[str, Any]:
    """Recorte seguro para /health (sem path, sem fingerprint completo)."""
    status = corpus_status(corpus_path)
    return {
        "documents": status.get("documents"),
        "matched_profiles": status.get("matched_profiles"),
        "reproducible": bool(status.get("reproducible")),
        "default_profile": status.get("default_profile"),
    }


def status_exit_code(status: dict[str, Any]) -> int:
    if status.get("verify_lock") and not status["verify_lock"].get("ok"):
        return 1
    verify = status.get("verify_profile")
    if verify and not verify.get("ok"):
        return 1
    if not status.get("corpus_exists"):
        return 2
    return 0
