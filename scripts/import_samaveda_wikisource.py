#!/usr/bin/env python3
"""Baixa o Sāmaveda Kauthuma em sânscrito de sa.wikisource.org (CC BY-SA 4.0),
pareia com Griffith (EN) e grava o compacto sa + entrada no manifesto.

Uso:
  python scripts/import_samaveda_wikisource.py                 # tudo (83 páginas)
  python scripts/import_samaveda_wikisource.py --no-ingest
  python scripts/import_samaveda_wikisource.py --only-ardha 2.1
  python scripts/import_samaveda_wikisource.py --only-dasati 1.1.1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vedic_pipeline.common.constants import DEFAULT_CORPUS, DEFAULT_RAW_DIR
from vedic_pipeline.crawler.ingest import ingest_manifest
from vedic_pipeline.etl import samaveda_wikisource as svws

OUT_DIR = ROOT / "data" / "structured"
GRIFFITH_PATH = OUT_DIR / "samaveda_griffith_en.json"
MANIFEST_PATH = OUT_DIR / "sources_dharmicdata.json"

UA = "VedaKnowledgePipeline/2.0 (+research; authorized sources only)"


def fetch_raw(url: str, timeout: int = 60, attempts: int = 3) -> str:
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
            if len(body.strip()) < 200:
                raise RuntimeError(f"resposta suspeita de vazia ({len(body)} bytes)")
            return body
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(f"  tentativa {attempt} falhou: {exc}")
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"fetch falhou após {attempts} tentativas: {url} ({last_exc})")


def source_entry(title: str, dest: Path) -> dict:
    return {
        "title": title,
        "url": str(dest.relative_to(ROOT)),
        "tradition": "vedic",
        "language": "sa",
        "license": "cc-by-sa",
        "download": True,
        "source_class": "structured-json",
        "tags": ["samaveda", "wikisource", "sanskrit"],
        "attribution": svws._ATTRIBUTION,
    }


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


def _last_two(spec: str) -> tuple[int, int]:
    tokens = [int(x) for x in spec.split(".")]
    return tokens[-2], tokens[-1]


def select_pages(args: argparse.Namespace) -> list[tuple[str, str, tuple[int, ...]]]:
    pages = svws.all_leaf_pages()
    if args.only_dasati:
        p, d = _last_two(args.only_dasati)
        return [pg for pg in pages if pg[0] == "dasati" and pg[2] == (p, d)]
    if args.only_ardha:
        p, a = _last_two(args.only_ardha)
        return [pg for pg in pages if pg[0] == "ardha" and pg[2] == (p, a)]
    return pages


def update_manifest(dest: Path) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    rel = str(dest.relative_to(ROOT))
    entry = source_entry("Sāmaveda Kauthuma (Sanskrit, sa.wikisource)", dest)
    manifest["sources"] = [s for s in manifest.get("sources") or [] if s.get("url") != rel]
    manifest["sources"].append(entry)
    manifest["sources"].sort(key=lambda s: s.get("title") or "")
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Importa Sāmaveda Kauthuma sa de sa.wikisource (CC BY-SA)")
    parser.add_argument("--only-dasati", help="uma daśatí do chanda, ex.: 1.1.1")
    parser.add_argument("--only-ardha", help="um ardha do uttarārcika, ex.: 2.1.1")
    parser.add_argument("--delay", type=float, default=0.7, help="pausa entre fetches (s)")
    parser.add_argument("--out", default=str(OUT_DIR / "samaveda_wikisource_sa.json"))
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--no-ingest", action="store_true")
    args = parser.parse_args()

    pages = select_pages(args)
    if not pages:
        parser.error("nenhuma página selecionada")
    griffith = json.loads(GRIFFITH_PATH.read_text(encoding="utf-8"))

    fetched: list[tuple[str, tuple[int, ...], str]] = []
    for i, (kind, title, nums) in enumerate(pages, start=1):
        url = svws.raw_url(title)
        print(f"[{i}/{len(pages)}] {kind} {nums}: {url}")
        fetched.append((kind, nums, fetch_raw(url)))
        if i < len(pages):
            time.sleep(args.delay)

    compact, notes = svws.build_hymns(fetched, griffith)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    n_verses = sum(len(h["verses"]) for h in compact["hymns"])
    print(f"{len(compact['hymns'])} hinos sa, {n_verses} arcas → {dest}")

    if notes:
        print(f"\n{len(notes)} aviso(s) de pareamento (divergências honestas, nada inventado):")
        for note in notes:
            print(f"  - {note}")
        notes_path = dest.with_suffix(".notes.txt")
        notes_path.write_text("\n".join(notes) + "\n", encoding="utf-8")
        print(f"  avisos gravados em {notes_path}")

    if args.no_ingest:
        return 0

    update_manifest(dest)
    stats = ingest_manifest(
        MANIFEST_PATH,
        corpus_path=Path(args.corpus),
        raw_dir=DEFAULT_RAW_DIR,
        min_chars=40,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0 if stats.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
