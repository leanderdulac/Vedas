#!/usr/bin/env python3
"""Importa a camada Zürich do VedaWeb (CC BY 4.0), ISO-15919 por hino.

Uso:
  python scripts/import_vedaweb.py
  python scripts/import_vedaweb.py --no-ingest
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vedic_pipeline.common.constants import DEFAULT_CORPUS, DEFAULT_RAW_DIR
from vedic_pipeline.crawler.ingest import ingest_manifest
from vedic_pipeline.etl.structured_json import VEDAWEB_ATTR, VEDAWEB_REPO
from vedic_pipeline.etl.vedaweb import compact_zurich_xlsx

XLSX_URL = (
    "https://raw.githubusercontent.com/VedaWebProject/vedaweb-data/main/rigveda/versions/zurich.xlsx"
)
OUT_DIR = ROOT / "data" / "structured"


def fetch(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"Já existe: {dest}")
        return dest
    print(f"Baixando {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "VedaKnowledge/2.0"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        dest.write_bytes(resp.read())
    print(f"  {dest.stat().st_size} bytes → {dest}")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description="Importa Ṛgveda Zürich (VedaWeb, CC BY 4.0)")
    parser.add_argument("--xlsx", default="")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--no-ingest", action="store_true")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    xlsx = Path(args.xlsx) if args.xlsx else DEFAULT_RAW_DIR / "vedaweb" / "zurich.xlsx"
    fetch(XLSX_URL, xlsx)

    compact_by_book = compact_zurich_xlsx(xlsx)
    sources: list[dict] = []
    for mandala in sorted(compact_by_book):
        compact = compact_by_book[mandala]
        dest = out / f"vedaweb_zurich_mandala_{mandala:02d}.json"
        dest.write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        n_hymns = len(compact.get("suktas") or [])
        n_verses = sum(len(s.get("verses") or []) for s in compact.get("suktas") or [])
        print(f"Mandala {mandala}: {n_hymns} hinos, {n_verses} ṛcas → {dest}")
        sources.append(
            {
                "title": f"Rigveda mandala {mandala} (VedaWeb Zürich, ISO-15919)",
                "url": str(dest.relative_to(ROOT)),
                "tradition": "vedic",
                "language": "sa",
                "license": "cc-by",
                "download": True,
                "source_class": "structured-json",
                "tags": ["rigveda", "vedaweb", "iso-15919"],
                "attribution": VEDAWEB_ATTR,
            }
        )

    manifest = {
        "meta": {
            "description": "Camada Zürich do VedaWeb (CC BY 4.0). Sem camadas NC.",
            "attribution": VEDAWEB_ATTR,
            "source": VEDAWEB_REPO,
        },
        "sources": sources,
    }
    man_path = out / "sources_vedaweb.json"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Manifesto: {man_path} ({len(sources)} mandalas)")

    if args.no_ingest:
        return 0

    stats = ingest_manifest(
        man_path,
        corpus_path=Path(args.corpus),
        raw_dir=DEFAULT_RAW_DIR,
        min_chars=40,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0 if stats.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
