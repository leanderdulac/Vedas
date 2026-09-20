"""Geração de pares fracos para fine-tune do CrossEncoder.

Positivos: hits cujo título casa com marcadores estritos de ``expect_title_any``
(hino ``10.129`` / Nasadiya), não o rótulo amplo "Rig Veda".
Negativos hard de Nasadiya: 10.125, 10.5, 2.38 quando presentes nos candidatos.

Pode ler candidatos pré-computados (JSON) — caminho de CI / dry-run —
ou recuperar via híbrido local quando o índice existir.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from vedic_pipeline.common.constants import DEFAULT_EMBED_DIR, PROJECT_ROOT

logger = logging.getLogger("vedic_pipeline.train.rerank_pairs")

DEFAULT_GOLD = PROJECT_ROOT / "fixtures" / "smoke_queries.json"
DEFAULT_TINY_CANDIDATES = PROJECT_ROOT / "fixtures" / "rerank" / "candidates_tiny.json"
DEFAULT_PAIRS_OUT = PROJECT_ROOT / "data" / "rerank" / "pairs.jsonl"

PAIR_FIELDS = ("query", "text", "label", "query_id", "doc_title", "split")

# Hinos que o CE genérico (ms-marco MiniLM) empurrou no lugar de RV 10.129.
NASADIYA_HARD_NEG_HYMNS = ("10.125", "10.5", "2.38")
PIN_TRAIN_QUERY_IDS = frozenset({"nasadiya"})

_HYMN_NUM_RE = re.compile(r"^\d+\.\d+$")
_HYMN_IN_PATTERN_RE = re.compile(r"\d+\.\d+")
_NASADIYA_NAME_RE = re.compile(r"nasadiya|n[aā]sad[iī]ya", re.I)


def hymn_in_blob(blob: str, hymn: str) -> bool:
    """Casa 10.5 em 'RV 10.5' sem casar 10.50 / 10.125."""
    escaped = re.escape(hymn.strip())
    return re.search(rf"(?<![\d.]){escaped}(?!\d)", blob or "", flags=re.I) is not None


def is_strict_title_marker(pattern: str) -> bool:
    """Número de hino (10.129) ou nome Nasadiya — não o rótulo amplo 'Rig Veda'."""
    text = (pattern or "").strip()
    if not text:
        return False
    if _HYMN_IN_PATTERN_RE.search(text):
        return True
    return bool(_NASADIYA_NAME_RE.search(text))


def marker_matches_title(title: str, marker: str) -> bool:
    marker = (marker or "").strip()
    if not marker:
        return False
    if _HYMN_NUM_RE.fullmatch(marker):
        return hymn_in_blob(title or "", marker)
    return marker.lower() in (title or "").lower()


def title_matches(title: str, patterns: Iterable[str], *, prefer_strict: bool = True) -> bool:
    """Positivos preferem marcadores estritos (10.129 / Nasadiya) a 'Rig Veda'."""
    markers = [str(p).strip() for p in patterns if str(p).strip()]
    if not markers:
        return False
    strict = [m for m in markers if is_strict_title_marker(m)]
    if prefer_strict and strict:
        return any(marker_matches_title(title, m) for m in strict)
    return any(marker_matches_title(title, m) for m in markers)


def is_nasadiya_query(query_id: str, query: str = "") -> bool:
    blob = f"{query_id} {query}".lower()
    return "nasadiya" in blob or "nāsadīya" in blob


def hit_title_blob(hit: dict[str, Any]) -> str:
    return " ".join(
        str(hit.get(key) or "") for key in ("title", "doc_title", "locator", "heading")
    )


def is_nasadiya_hard_negative(query_id: str, query: str, hit: dict[str, Any]) -> bool:
    if not is_nasadiya_query(query_id, query):
        return False
    blob = hit_title_blob(hit)
    return any(hymn_in_blob(blob, hymn) for hymn in NASADIYA_HARD_NEG_HYMNS)


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_gold_queries(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, list):
        return list(payload)
    queries = payload.get("queries")
    if not isinstance(queries, list):
        raise ValueError(f"Gold set sem lista 'queries': {path}")
    return queries


def load_candidates_payload(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Aceita ``{query_id: [hits]}`` ou ``{queries: [{id, hits}]}``."""
    payload = load_json(path)
    if isinstance(payload, dict) and "queries" in payload:
        by_id: dict[str, list[dict[str, Any]]] = {}
        for item in payload.get("queries") or []:
            qid = str(item.get("id") or item.get("query_id") or "")
            hits = item.get("hits") or item.get("candidates") or []
            if qid:
                by_id[qid] = list(hits)
        return by_id
    if isinstance(payload, dict):
        return {str(key): list(value) for key, value in payload.items()}
    raise ValueError(f"Candidatos JSON inválidos: {path}")


def assign_splits(
    query_ids: Iterable[str],
    holdout_ratio: float = 0.2,
    pin_train_ids: Iterable[str] = PIN_TRAIN_QUERY_IDS,
) -> dict[str, str]:
    """Holdout por query_id; Nasadiya fica no treino (hard negatives do CE genérico)."""
    ids = sorted({str(qid) for qid in query_ids})
    pinned = {str(qid) for qid in pin_train_ids if str(qid) in ids}
    if len(ids) < 2 or holdout_ratio <= 0:
        return {qid: "train" for qid in ids}
    eligible = [qid for qid in ids if qid not in pinned]
    if not eligible:
        return {qid: "train" for qid in ids}
    n_hold = max(1, int(round(len(ids) * holdout_ratio)))
    n_hold = min(n_hold, len(eligible))
    hold = set(eligible[-n_hold:])
    return {qid: ("eval" if qid in hold else "train") for qid in ids}


def pair_record(
    *,
    query: str,
    text: str,
    label: int,
    query_id: str,
    doc_title: str,
    split: str,
    pair_kind: str = "",
) -> dict[str, Any]:
    rec = {
        "query": query,
        "text": text,
        "label": int(label),
        "query_id": query_id,
        "doc_title": doc_title,
        "split": split,
    }
    if pair_kind:
        rec["pair_kind"] = pair_kind
    return rec


def pairs_from_hits(
    *,
    query: str,
    query_id: str,
    expect_title_any: list[str],
    hits: list[dict[str, Any]],
    split: str = "train",
    max_negatives: int = 8,
) -> list[dict[str, Any]]:
    """Emite pares 0/1 a partir de hits híbridos já recuperados.

    Positivos: marcadores estritos (10.129 / Nasadiya) quando existem no gold;
    'Rig Veda' só entra se não houver âncora de hino. Hard negatives de Nasadiya
    (10.125, 10.5, 2.38) entram sempre que aparecerem nos candidatos.
    """
    positives: list[dict[str, Any]] = []
    hard_negatives: list[dict[str, Any]] = []
    other_negatives: list[dict[str, Any]] = []
    for hit in hits:
        text = str(hit.get("text") or "").strip()
        if not text:
            continue
        title = str(hit.get("title") or hit.get("doc_title") or "")
        hard_neg = is_nasadiya_hard_negative(query_id, query, hit)
        if hard_neg:
            kind, label = "hard_negative", 0
        elif title_matches(title, expect_title_any, prefer_strict=True):
            kind, label = "strict_positive", 1
        else:
            kind, label = "negative", 0
        rec = pair_record(
            query=query,
            text=text,
            label=label,
            query_id=query_id,
            doc_title=title,
            split=split,
            pair_kind=kind,
        )
        if label == 1:
            positives.append(rec)
        elif hard_neg:
            hard_negatives.append(rec)
        else:
            other_negatives.append(rec)
    kept_other = other_negatives[: max(0, max_negatives)]
    return positives + hard_negatives + kept_other


def build_pairs(
    queries: list[dict[str, Any]],
    candidates_by_id: dict[str, list[dict[str, Any]]],
    *,
    holdout_ratio: float = 0.2,
    max_negatives: int = 8,
) -> list[dict[str, Any]]:
    splits = assign_splits((q.get("id") or q.get("query") or "") for q in queries)
    pairs: list[dict[str, Any]] = []
    for spec in queries:
        qid = str(spec.get("id") or spec.get("query") or "")
        query = str(spec.get("query") or "")
        expect = [str(p) for p in (spec.get("expect_title_any") or [])]
        hits = candidates_by_id.get(qid) or spec.get("hits") or []
        if not query or not hits:
            logger.info("Sem candidatos para query_id=%s — pulando", qid)
            continue
        pairs.extend(
            pairs_from_hits(
                query=query,
                query_id=qid,
                expect_title_any=expect,
                hits=list(hits),
                split=splits.get(qid, "train"),
                max_negatives=max_negatives,
            )
        )
    return pairs


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with dest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def summarize_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    n_pos = sum(1 for p in pairs if int(p.get("label") or 0) == 1)
    n_neg = len(pairs) - n_pos
    by_split: dict[str, int] = {}
    by_query: dict[str, int] = {}
    for pair in pairs:
        split = str(pair.get("split") or "train")
        qid = str(pair.get("query_id") or "")
        by_split[split] = by_split.get(split, 0) + 1
        by_query[qid] = by_query.get(qid, 0) + 1
    return {
        "pairs": len(pairs),
        "positives": n_pos,
        "negatives": n_neg,
        "by_split": by_split,
        "queries": len(by_query),
    }


@contextmanager
def reranker_forced_off() -> Iterator[None]:
    """Garante híbrido sem CE ao minerar candidatos."""
    from vedic_pipeline.search.reranker import reranker_runtime

    with reranker_runtime(enabled=False):
        yield


def index_is_ready(index_dir: Path) -> bool:
    root = Path(index_dir)
    return (root / "embeddings.npy").exists() and (root / "chunks.jsonl").exists()


def retrieve_candidates(
    queries: list[dict[str, Any]],
    *,
    index_dir: Path = DEFAULT_EMBED_DIR,
    backend: str = "numpy",
    top_n: int = 40,
) -> dict[str, list[dict[str, Any]]]:
    """Recupera top-N híbrido (CE off). Exige índice local."""
    if not index_is_ready(index_dir) and backend != "pgvector":
        raise FileNotFoundError(
            f"Índice numpy ausente em {index_dir}. "
            "Passe --candidates com JSON pré-computado ou rode build-index."
        )

    from vedic_pipeline.llm.ask import retrieve_hits

    by_id: dict[str, list[dict[str, Any]]] = {}
    with reranker_forced_off():
        for spec in queries:
            qid = str(spec.get("id") or spec.get("query") or "")
            query = str(spec.get("query") or "")
            if not qid or not query:
                continue
            hits, _used = retrieve_hits(
                query,
                backend=backend,
                index_dir=Path(index_dir),
                top_k=top_n,
                tradition=spec.get("tradition") or None,
                language=spec.get("language") or None,
            )
            by_id[qid] = hits
    return by_id
