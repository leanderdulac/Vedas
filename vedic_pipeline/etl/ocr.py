"""OCR local de scans em PDF (pdftoppm + tesseract), com cache em disco.

Para scans cujo texto do archive.org saiu inutilizável (ex.: OCR rodado com o
modelo de Devanāgarī num livro em inglês). O manifesto pede com
``"ocr": {"engine": "tesseract", "lang": "eng", "first_page": 1, "last_page": 900}``;
o texto sai em ``<pdf>.<engine>-<lang>-<first>-<last>.txt`` ao lado do PDF,
com um marcador ``[[page N]]`` antes de cada página, e é reaproveitado nas
ingestões seguintes.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

PAGE_MARK = "[[page {n}]]"


def ocr_cache_path(pdf: Path, spec: dict[str, Any]) -> Path:
    engine = spec.get("engine", "tesseract")
    lang = spec.get("lang", "eng")
    return pdf.with_name(f"{pdf.name}.{engine}-{lang}-{spec['first_page']}-{spec['last_page']}.txt")


def _ocr_page(pdf: Path, page: int, lang: str, dpi: int, workdir: Path) -> str:
    stem = workdir / f"p{page}"
    subprocess.run(
        ["pdftoppm", "-f", str(page), "-l", str(page), "-r", str(dpi), "-gray", "-png", "-singlefile", str(pdf), str(stem)],
        check=True,
        capture_output=True,
    )
    png = stem.with_suffix(".png")
    out = subprocess.run(
        ["tesseract", str(png), "-", "-l", lang, "--psm", "6"],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_THREAD_LIMIT": "1"},
    ).stdout
    png.unlink(missing_ok=True)
    return out


def ocr_pdf(pdf: Path | str, spec: dict[str, Any], *, workers: int | None = None) -> str:
    """Texto OCR das páginas pedidas (com cache)."""
    pdf = Path(pdf)
    cache = ocr_cache_path(pdf, spec)
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    if spec.get("engine", "tesseract") != "tesseract":
        raise ValueError(f"motor de OCR não suportado: {spec.get('engine')!r}")
    for tool in ("pdftoppm", "tesseract"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"OCR pedido no manifesto, mas `{tool}` não está instalado")
    first, last = int(spec["first_page"]), int(spec["last_page"])
    lang, dpi = spec.get("lang", "eng"), int(spec.get("dpi", 300))
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(workers) as pool:
        pages = list(range(first, last + 1))
        texts = list(pool.map(lambda p: _ocr_page(pdf, p, lang, dpi, Path(tmp)), pages))
    text = "\n".join(f"{PAGE_MARK.format(n=p)}\n{t}" for p, t in zip(pages, texts, strict=True))
    cache.write_text(text, encoding="utf-8")
    return text
