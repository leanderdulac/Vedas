"""Baixa a tanpura do canto: "Acoustic Tanpura 10" de sankalp (Freesound #155272).

Licença CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/): uso livre com
crédito, que o app mostra na UI, no cabeçalho HTTP e nos metadados do MP3.
O arquivo fica fora do git (data/assets/ está no .gitignore). Usa a prévia HQ
pública (OGG) do Freesound, que não exige login.

    python scripts/fetch_tanpura.py [--dest data/assets/tanpura/...]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://cdn.freesound.org/previews/155/155272_1859932-hq.ogg"
PAGE = "https://freesound.org/people/sankalp/sounds/155272/"
SHA256 = "b5d2335ce963c2084ad775e8402b41ebfb7878a90e67e30dd853f3dc91abe9fe"
DEFAULT_DEST = Path("data/assets/tanpura/sankalp_155272_acoustic_tanpura_10.ogg")
CREDIT = '"Acoustic Tanpura 10" by sankalp (freesound.org/s/155272), CC BY 4.0'


def fetch(dest: Path) -> Path:
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == SHA256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "vedas-fetch-tanpura/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (URL fixa)
        data = resp.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"checksum inesperado ({digest}); confira {PAGE}")
    dest.write_bytes(data)
    (dest.parent / "CREDITS.txt").write_text(f"{CREDIT}\n{PAGE}\n", encoding="utf-8")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = ap.parse_args()
    print(fetch(args.dest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
