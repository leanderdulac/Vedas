#!/usr/bin/env python3
"""Gera o manifesto SBE/Purāṇas (sacred-texts) com URLs reais verificadas.

Cobre o que falta no portfólio:
  - Chāndogya Upanishad integral (SBE01, Müller) — 154 páginas
  - SBE15 integral: Kaṭha, Muṇḍaka, Taittirīya, Bṛhadāraṇyaka, Śvetāśvatara,
    Praśna, Maitrāyaṇa (substitui as entradas de página única)
  - Viṣṇu Purāṇa integral (Wilson, 1840) — livros I–VI
  - Garuḍa Purāṇa (Wood/Subrahmanyam, 1911)

O crawler canonicaliza www.sacred-texts.com → archive.sacred-texts.com.
Uso: python scripts/build_sbe_manifest.py [--out fixtures/sources_sbe_complete.json]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://www.sacred-texts.com/hin"


def sbe(n: int) -> str:
    return f"{BASE}/sbe01/sbe01{n:03d}.htm"


def sbe15(n: int) -> str:
    return f"{BASE}/sbe15/sbe15{n:03d}.htm"


def vp(n: int) -> str:
    return f"{BASE}/vp/vp{n:03d}.htm"


def gpu(n: int) -> str:
    return f"{BASE}/gpu/gpu{n:02d}.htm"


def src(url: str, title: str, tradition: str, language: str = "en") -> dict:
    return {
        "url": url,
        "title": title,
        "tradition": tradition,
        "language": language,
        "license": "public-domain",
        "source_class": "sacred-texts-sbe",
    }


def build_sources() -> list[dict]:
    sources: list[dict] = []

    # --- Chāndogya Upanishad (SBE01): prapāṭhakas I–VIII ------------------
    chandogya = [
        ("I", 22, 34, 13),
        ("II", 35, 58, 24),
        ("III", 59, 77, 19),
        ("IV", 78, 94, 17),
        ("V", 95, 118, 24),
        ("VI", 119, 134, 16),
        ("VII", 135, 160, 26),
        ("VIII", 161, 175, 15),
    ]
    for roman, start, end, expected in chandogya:
        pages = list(range(start, end + 1))
        assert len(pages) == expected, (roman, len(pages))
        for i, page in enumerate(pages, 1):
            sources.append(
                src(
                    sbe(page),
                    f"Chandogya Upanishad {roman}.{i} (Müller, SBE01, sacred-texts)",
                    "upanishad",
                )
            )

    # --- SBE15: obras integrais -------------------------------------------
    sbe15_works = [
        ("Katha", 10, 15),
        ("Mundaka", 16, 21),
        ("Taittiriya", 22, 52),
        ("Brihadaranyaka", 53, 99),
        ("Svetasvatara", 100, 105),
        ("Prasna", 106, 111),
        ("Maitrayana", 112, 118),
    ]
    for work, start, end in sbe15_works:
        seq = 0
        for page in range(start, end + 1):
            if work == "Brihadaranyaka" and page == 98:
                continue  # duplicata da tradução de Hume
            seq += 1
            sources.append(
                src(
                    sbe15(page),
                    f"{work} Upanishad — seção {seq} (Müller, SBE15, sacred-texts)",
                    "upanishad",
                )
            )

    # --- Viṣṇu Purāṇa (Wilson): livros I–VI -------------------------------
    vp_books = [
        ("I", 35, 57),
        ("II", 58, 74),
        ("III", 75, 92),
        ("IV", 93, 117),
        ("V", 118, 155),
        ("VI", 156, 163),
    ]
    for book, start, end in vp_books:
        for page in range(start, end + 1):
            sources.append(
                src(
                    vp(page),
                    f"Vishnu Purana — Book {book}, página {page} (Wilson, 1840, sacred-texts)",
                    "purana",
                )
            )

    # --- Garuḍa Purāṇa (Wood/Subrahmanyam, 1911) --------------------------
    for page in range(3, 19):  # gpu03..gpu18 (capítulos I–XVI)
        sources.append(
            src(
                gpu(page),
                f"Garuda Purana — capítulo {page - 2} (Wood & Subrahmanyam, 1911, sacred-texts)",
                "purana",
            )
        )

    return sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="fixtures/sources_sbe_complete.json",
        help="Caminho de saída do manifesto",
    )
    args = parser.parse_args()

    sources = build_sources()
    payload = {
        "_meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "note": (
                "Manifesto gerado a partir dos índices verificados de "
                "archive.sacred-texts.com (SBE01, SBE15, vp, gpu). Traduções em "
                "domínio público (Müller 1879/1884, Wilson 1840, Wood 1911). "
                "Regenere com: python scripts/build_sbe_manifest.py"
            ),
            "text_sources": len(sources),
        },
        "sources": sources,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"gravadas {len(sources)} fontes de texto em {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
