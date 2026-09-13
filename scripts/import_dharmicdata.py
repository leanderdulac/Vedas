#!/usr/bin/env python3
"""Baixa JSON do DharmicData, descarta comentários e ingere sânscrito numerado.

Uso:
  python scripts/import_dharmicdata.py
  python scripts/import_dharmicdata.py --gita-chapters 2 --rigveda-mandalas 10
  python scripts/import_dharmicdata.py --gita-chapters 1-18 --rigveda-mandalas 1-10 --yajurveda --atharvaveda --samaveda
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
from vedic_pipeline.etl.structured_json import (
    DHARMICDATA_ATTR,
    compact_atharvaveda_kaanda,
    compact_gita_chapter,
    compact_rigveda_mandala,
    compact_samaveda,
    compact_yajurveda_samhita,
)

RAW_BASE = "https://raw.githubusercontent.com/bhavykhatri/DharmicData/main"
OUT_DIR = ROOT / "data" / "structured"


def parse_span(spec: str) -> list[int]:
    values: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            values.extend(range(int(a), int(b) + 1))
        else:
            values.append(int(part))
    return values


def fetch_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": "VedaKnowledge/2.0"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def source_entry(title: str, dest: Path, *, tradition: str, language: str, tags: list[str]) -> dict:
    return {
        "title": title,
        "url": str(dest.relative_to(ROOT)),
        "tradition": tradition,
        "language": language,
        "license": "odbl",
        "download": True,
        "source_class": "structured-json",
        "tags": tags,
        "attribution": DHARMICDATA_ATTR,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Importa textos estruturados do DharmicData (ODbL)")
    parser.add_argument("--gita-chapters", default="", help="ex.: 2 ou 1-18; vazio = pular")
    parser.add_argument("--rigveda-mandalas", default="", help="ex.: 10 ou 1-10; vazio = pular")
    parser.add_argument("--yajurveda", action="store_true")
    parser.add_argument("--atharvaveda", action="store_true")
    parser.add_argument("--samaveda", action="store_true")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--no-ingest", action="store_true")
    args = parser.parse_args()

    if not any(
        [
            args.gita_chapters,
            args.rigveda_mandalas,
            args.yajurveda,
            args.atharvaveda,
            args.samaveda,
        ]
    ):
        args.gita_chapters = "2"
        args.rigveda_mandalas = "10"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sources: list[dict] = []

    if args.gita_chapters:
        for chapter in parse_span(args.gita_chapters):
            url = f"{RAW_BASE}/SrimadBhagvadGita/bhagavad_gita_chapter_{chapter}.json"
            print(f"Gītā {chapter}: {url}")
            compact = compact_gita_chapter(fetch_json(url))
            dest = out / f"gita_chapter_{chapter:02d}_sa.json"
            dest.write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"  {len(compact.get('verses') or [])} ślokas → {dest}")
            sources.append(
                source_entry(
                    f"Bhagavad-gītā {chapter} (Sanskrit, DharmicData)",
                    dest,
                    tradition="vaishnava",
                    language="sa",
                    tags=["gita", "dharmicdata", "sanskrit"],
                )
            )

    if args.rigveda_mandalas:
        for mandala in parse_span(args.rigveda_mandalas):
            url = f"{RAW_BASE}/Rigveda/rigveda_mandala_{mandala}.json"
            print(f"Ṛgveda mandala {mandala}: {url}")
            compact = compact_rigveda_mandala(fetch_json(url))
            dest = out / f"rigveda_mandala_{mandala:02d}_sa.json"
            dest.write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"  {len(compact.get('suktas') or [])} sūktas → {dest}")
            sources.append(
                source_entry(
                    f"Rigveda mandala {mandala} (Sanskrit, DharmicData)",
                    dest,
                    tradition="vedic",
                    language="sa",
                    tags=["rigveda", "dharmicdata", "sanskrit"],
                )
            )

    if args.yajurveda:
        url = f"{RAW_BASE}/Yajurveda/vajasneyi_madhyadina_samhita.json"
        print(f"Yajurveda Mādhyandina: {url}")
        compact = compact_yajurveda_samhita(fetch_json(url))
        dest = out / "yajurveda_madhyandina_sa.json"
        dest.write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"  {len(compact.get('adhyayas') or [])} adhyāyas → {dest}")
        sources.append(
            source_entry(
                "Yajurveda Vājasaneyi Mādhyandina (Sanskrit, DharmicData)",
                dest,
                tradition="vedic",
                language="sa",
                tags=["yajurveda", "dharmicdata", "sanskrit"],
            )
        )

    if args.atharvaveda:
        for kaanda in range(1, 21):
            url = f"{RAW_BASE}/AtharvaVeda/atharvaveda_kaanda_{kaanda}.json"
            print(f"Atharvaveda kāṇḍa {kaanda}: {url}")
            compact = compact_atharvaveda_kaanda(fetch_json(url))
            dest = out / f"atharvaveda_kaanda_{kaanda:02d}_sa.json"
            dest.write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"  {len(compact.get('suktas') or [])} sūktas → {dest}")
            sources.append(
                source_entry(
                    f"Atharvaveda kāṇḍa {kaanda} (Sanskrit, DharmicData)",
                    dest,
                    tradition="vedic",
                    language="sa",
                    tags=["atharvaveda", "dharmicdata", "sanskrit"],
                )
            )

    if args.samaveda:
        url = f"{RAW_BASE}/Samaveda/Samaveda.json"
        print(f"Sāmaveda Griffith: {url}")
        compact = compact_samaveda(fetch_json(url))
        dest = out / "samaveda_griffith_en.json"
        dest.write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"  {len(compact.get('hymns') or [])} hinos → {dest}")
        sources.append(
            source_entry(
                "Sāmaveda (Griffith, DharmicData)",
                dest,
                tradition="vedic",
                language="en",
                tags=["samaveda", "dharmicdata", "griffith"],
            )
        )

    manifest = {
        "meta": {
            "description": "Textos estruturados extraídos do DharmicData (ODbL). Sem comentários modernos.",
            "attribution": DHARMICDATA_ATTR,
        },
        "sources": sources,
    }
    man_path = out / "sources_dharmicdata.json"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Manifesto: {man_path} ({len(sources)} entradas)")

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
