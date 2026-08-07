#!/usr/bin/env python3
"""
Gera manifesto com hinos do Ṛgveda (Griffith / sacred-texts.com), domínio público.

Uso:
  python scripts/generate_rigveda_manifest.py --out fixtures/sources_rigveda_full.json
  python scripts/bulk_ingest_open.py --manifest fixtures/sources_rigveda_full.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Contagem aproximada de hinos por mandala (Ṛgveda Śākala)
HYMNS_PER_BOOK = {
    1: 191,
    2: 43,
    3: 62,
    4: 58,
    5: 87,
    6: 75,
    7: 104,
    8: 103,
    9: 114,
    10: 191,
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="fixtures/sources_rigveda_full.json")
    p.add_argument("--books", default="1-10", help="ex: 1,10 ou 1-10")
    args = p.parse_args()

    books: list[int] = []
    for part in args.books.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            books.extend(range(int(a), int(b) + 1))
        else:
            books.append(int(part))

    sources = []
    for book in books:
        n = HYMNS_PER_BOOK.get(book, 0)
        for h in range(1, n + 1):
            fn = f"rv{book:02d}{h:03d}.htm"
            sources.append(
                {
                    "title": f"Rigveda RV {book}.{h} (Griffith, sacred-texts)",
                    "url": f"https://www.sacred-texts.com/hin/rigveda/{fn}",
                    "tradition": "vedic",
                    "language": "en",
                    "license": "public-domain",
                    "download": True,
                    "source_class": "sacred-texts",
                    "tags": ["rigveda", f"mandala-{book}"],
                }
            )

    out = {
        "meta": {
            "description": "Ṛgveda completo (Griffith) via sacred-texts — public domain",
            "hymns": len(sources),
            "books": books,
        },
        "sources": sources,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(sources)} hymns -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
