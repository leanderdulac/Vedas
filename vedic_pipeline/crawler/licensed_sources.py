"""Fontes licenciadas de uso privado (permissão pendente), atrás de uma flag.

Uma fonte daqui:

* nunca é baixada pelo pipeline genérico (``download: False`` — o
  ``validate_source`` a recusa) nem vai para o Git: o texto fica só em
  ``data/raw/...`` (gitignored), coletado por um script próprio;
* fica FORA do ``data/corpus.jsonl``: entra no índice só quando
  ``VEDIC_ENABLE_LICENSED_SOURCES=1`` está ligado na hora do rebuild
  (``scripts/rebuild_index_safe.py`` ou ``build-index``);
* se um índice construído com a flag ligada for servido com ela desligada,
  ``load_embedding_index`` descarta os trechos dela ao carregar.

Para ligar (depois da permissão): ``VEDIC_ENABLE_LICENSED_SOURCES=1`` no
``.env``, ``python scripts/rebuild_index_safe.py`` com a mesma variável, e
reiniciar a API.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from vedic_pipeline.common.corpus import content_fingerprint, stable_id

logger = logging.getLogger("vedic_pipeline.licensed_sources")

LICENSED_SOURCES_ENV = "VEDIC_ENABLE_LICENSED_SOURCES"

LICENSED_SOURCES: dict[str, dict[str, Any]] = {
    "vedabase-sb-ptbr": {
        "title": "Śrīmad-Bhāgavatam — tradução em português (Bhaktivedanta Book Trust)",
        "work_title": "Bhāgavata Purāṇa",
        "url": "https://vedabase.io/pt-br/library/sb/",
        "license": "bbt-permission-pending",
        "license_note": "BBT — permission pending, private use",
        "rights_holder": "Bhaktivedanta Book Trust (BBT)",
        "permission": "pendente: o dono do projeto vai pedir autorização diretamente à BBT",
        "download": False,
        "private": True,
        "enabled_by": LICENSED_SOURCES_ENV,
        "local_path": "data/raw/vedabase_sb_ptbr/sb_ptbr.jsonl",
        "collector": "python -m vedic_pipeline.etl.vedabase_sb crawl --out data/raw/vedabase_sb_ptbr",
        "fields": ["devanagari", "transliteration", "translation_pt"],
        "excluded": "sinônimos palavra a palavra, significados (purports), prefácios e introduções",
        "language": "pt",
        "tradition": "purana",
    },
}

RESTRICTED_LICENSES = frozenset(s["license"] for s in LICENSED_SOURCES.values())

_TRUE = {"1", "true", "yes", "on", "sim"}


def licensed_sources_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Flag global; desligada por padrão."""
    value = (env if env is not None else os.environ).get(LICENSED_SOURCES_ENV, "")
    return value.strip().lower() in _TRUE


def is_restricted(item: Mapping[str, Any]) -> bool:
    """Registro ou chunk de fonte licenciada (pela marca ou pela licença)."""
    if item.get("restricted_source") in LICENSED_SOURCES:
        return True
    return str(item.get("license") or "").strip().lower() in RESTRICTED_LICENSES


def visible_mask(items: Iterable[Mapping[str, Any]], *, enabled: bool | None = None) -> list[bool]:
    on = licensed_sources_enabled() if enabled is None else enabled
    return [on or not is_restricted(it) for it in items]


# ------------------------------------------------------------ Bhāgavata BBT

def sb_record_to_corpus(rec: Mapping[str, Any], source_id: str = "vedabase-sb-ptbr") -> dict[str, Any]:
    """Um verso do coletor (``sb_ptbr.jsonl``) → registro do corpus.

    O título começa com "Bhāgavata Purāṇa — " para contar como a mesma obra
    que o Dutt e o GRETIL no modo entidade (``work_key``).
    """
    src = LICENSED_SOURCES[source_id]
    locator = str(rec["locator"])
    ref = locator.removeprefix("SB ").strip()
    parts = [f"Śrīmad-Bhāgavatam {ref}"]
    for key in ("devanagari", "transliteration"):
        if rec.get(key):
            parts.append(str(rec[key]).strip())
    if rec.get("translation_pt"):
        parts.append("Tradução: " + str(rec["translation_pt"]).strip())
    text = "\n\n".join(parts)
    return {
        "id": stable_id(source_id, locator),
        "text": text,
        "source_url": rec.get("url") or src["url"],
        "title": f"{src['work_title']} — {locator} (BBT, pt-br)",
        "work_title": src["work_title"],
        "locator": locator,
        "canto": rec.get("canto"),
        "chapter": rec.get("chapter"),
        "verse": rec.get("verse"),
        "tradition": src["tradition"],
        "language": src["language"],
        "license": src["license"],
        "license_note": src["license_note"],
        "restricted_source": source_id,
        "translator": "Bhaktivedanta Book Trust",
        "fingerprint": content_fingerprint(text),
        "char_count": len(text),
    }


_CONVERTERS = {"vedabase-sb-ptbr": sb_record_to_corpus}


def load_licensed_records(
    root: Path,
    *,
    enabled: bool | None = None,
    sources: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Registros das fontes licenciadas presentes em disco; [] com a flag desligada."""
    on = licensed_sources_enabled() if enabled is None else enabled
    if not on:
        return []
    out: list[dict[str, Any]] = []
    for sid in sources or LICENSED_SOURCES:
        path = Path(root) / LICENSED_SOURCES[sid]["local_path"]
        if not path.exists():
            logger.warning("Fonte licenciada %s ligada, mas %s não existe", sid, path)
            continue
        convert = _CONVERTERS[sid]
        n = 0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if not rec.get("translation_pt") and not rec.get("devanagari"):
                    continue
                out.append(convert(rec, sid))
                n += 1
        logger.info("Fonte licenciada %s: %d registros (%s)", sid, n, LICENSED_SOURCES[sid]["license_note"])
    return out


def records_for_index(corpus_records: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    """Corpus público + fontes licenciadas (só com a flag ligada)."""
    extra = load_licensed_records(root)
    if not extra:
        return corpus_records
    seen = {r.get("id") for r in corpus_records}
    return corpus_records + [r for r in extra if r["id"] not in seen]
