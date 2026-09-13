"""Camada Zürich do VedaWeb (CC BY 4.0): texto ISO-15919 por ṛc, sem morfologia.

As demais camadas do TEI/CSV (Aufrecht, Padapāṭha, traduções) são CC BY-NC-SA
e ficam de fora do corpus.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from vedic_pipeline.etl.structured_json import compact_zurich_rows

_SSML = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _col_row(ref: str) -> tuple[str, int]:
    letters = "".join(ch for ch in ref if ch.isalpha())
    digits = "".join(ch for ch in ref if ch.isdigit())
    return letters, int(digits or 0)


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    out: list[str] = []
    for si in root.findall("m:si", _SSML):
        out.append("".join(node.text or "" for node in si.findall(".//m:t", _SSML)))
    return out


def _cell_text(cell: ET.Element, shared: list[str]) -> str | None:
    value = cell.find("m:v", _SSML)
    if value is None or value.text is None:
        return None
    if cell.get("t") == "s":
        idx = int(value.text)
        if 0 <= idx < len(shared):
            return shared[idx]
        return None
    return value.text


def iter_zurich_pada_rows(path: str | Path) -> list[tuple[str, str, str]]:
    """Lê stelle / pada / texto (colunas A–C) e devolve uma linha por pada."""
    seen: set[tuple[str, str]] = set()
    rows: list[tuple[str, str, str]] = []
    with zipfile.ZipFile(path) as archive:
        shared = _shared_strings(archive)
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        for row in root.findall("m:sheetData/m:row", _SSML):
            stelle = pada = text = None
            for cell in row.findall("m:c", _SSML):
                col, _r = _col_row(cell.get("r") or "")
                value = _cell_text(cell, shared)
                if col == "A":
                    stelle = value
                elif col == "B":
                    pada = value
                elif col == "C":
                    text = value
            if not stelle or stelle.startswith("belege"):
                continue
            key = (stelle, (pada or "a").strip().lower()[:1])
            if key in seen or not text:
                continue
            seen.add(key)
            rows.append((stelle, key[1], text))
    return rows


def compact_zurich_xlsx(path: str | Path) -> dict[int, dict]:
    return compact_zurich_rows(iter_zurich_pada_rows(path))
