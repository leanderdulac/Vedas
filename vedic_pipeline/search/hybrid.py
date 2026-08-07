"""Recuperação híbrida: semântica + lexical (RRF) + expansão de consulta."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Optional

# Sinônimos / equivalentes úteis para literatura védica (en-sa)
QUERY_EXPANSIONS: dict[str, list[str]] = {
    "self": ["atman", "ātman", "atma", "purusha", "soul"],
    "atman": ["self", "ātman", "soul", "purusha"],
    "brahman": ["absolute", "imperishable", "brahma", "supreme"],
    "dharma": ["duty", "righteousness", "law", "rita", "ṛta"],
    "karma": ["action", "work", "deed"],
    "yoga": ["union", "discipline", "samadhi", "meditation"],
    "bhakti": ["devotion", "love", "worship"],
    "jnana": ["knowledge", "wisdom", "vidya"],
    "moksha": ["liberation", "release", "kaivalya", "freedom"],
    "krishna": ["vasudeva", "madhava", "lord", "bhagavan"],
    "rama": ["raghava", "dasharathi"],
    "agni": ["fire", "priest of sacrifice"],
    "indra": ["vritra", "vajra"],
    "om": ["aum", "pranava", "ॐ"],
    "aum": ["om", "pranava", "ॐ"],
    "gita": ["bhagavad", "bhagavad-gita", "song"],
    "upanishad": ["vedanta", "aranyaka"],
    "veda": ["rig", "rigveda", "yajur", "sama", "atharva"],
    "purusha": ["person", "cosmic person", "self"],
    "ishvara": ["lord", "god", "isvara"],
    "samadhi": ["absorption", "yoga", "meditation"],
    "ahimsa": ["non-injury", "nonviolence"],
    "samsara": ["cycle", "rebirth", "becoming"],
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\wāīūṛṝḷḹṃḥśṣñṅṭḍṇā-]+", (text or "").lower(), flags=re.UNICODE)


def expand_query(query: str) -> list[str]:
    """Gera variantes da pergunta para recall mais alto."""
    base = query.strip()
    variants = [base]
    tokens = tokenize(base)
    extra_terms: list[str] = []
    for t in tokens:
        for exp in QUERY_EXPANSIONS.get(t, []):
            extra_terms.append(exp)
    if extra_terms:
        variants.append(base + " " + " ".join(dict.fromkeys(extra_terms)))
        # variante só com termos expandidos + originais chave
        variants.append(" ".join(dict.fromkeys(tokens + extra_terms)))
    # dedupe preservando ordem
    seen: set[str] = set()
    out: list[str] = []
    for v in variants:
        k = v.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(v.strip())
    return out


def lexical_scores(query: str, chunks: list[dict[str, Any]]) -> list[float]:
    """BM25-light sobre tokens unicode."""
    q_tokens = tokenize(query)
    if not q_tokens or not chunks:
        return [0.0] * len(chunks)

    docs = [tokenize(c.get("text") or "") for c in chunks]
    df: Counter[str] = Counter()
    for d in docs:
        df.update(set(d))
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / max(n, 1)
    k1, b = 1.5, 0.75
    scores: list[float] = []
    for d in docs:
        tf = Counter(d)
        dl = len(d) or 1
        s = 0.0
        for t in q_tokens:
            if t not in tf:
                # tenta expansões do termo
                alts = QUERY_EXPANSIONS.get(t, [])
                found = False
                for a in alts:
                    if a in tf:
                        t = a
                        found = True
                        break
                if not found:
                    continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            freq = tf[t]
            s += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    return scores


def rrf_fuse(
    ranked_lists: list[list[int]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion sobre listas de índices de chunk."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


_EPIC_TITLE = re.compile(
    r"mahabharata|mahābhārata|ramayan|rāmāyaṇ|ganguli|valmiki|vālmīki",
    re.I,
)
_HYMN_TITLE = re.compile(
    r"rigveda\s+rv\s+\d|rig\s*veda|sukta|sūkta|upanishad|upaniṣad|g[iī]t[aā]|yoga.?s[uū]tra",
    re.I,
)
_EPIC_QUERY_HINTS = frozenset(
    {
        "mahabharata",
        "mahābhārata",
        "ramayana",
        "rāmāyaṇa",
        "rama",
        "rāma",
        "sita",
        "sītā",
        "arjuna",
        "krishna",
        "kṛṣṇa",
        "bhishma",
        "bhīṣma",
        "kurukshetra",
        "pandava",
        "kaurava",
        "hanuman",
        "lakshmana",
        "valmiki",
        "ganguli",
    }
)


def diversify_by_doc(
    hits: list[dict[str, Any]],
    top_k: int,
    max_per_doc: int = 2,
) -> list[dict[str, Any]]:
    """Limita chunks por doc_id para o top-k não virar um único épico."""
    out: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for h in hits:
        did = str(h.get("doc_id") or h.get("title") or h.get("chunk_id") or "")
        if counts[did] >= max_per_doc:
            continue
        counts[did] += 1
        out.append(h)
        if len(out) >= top_k:
            break
    return out


def apply_title_and_size_boost(query: str, pool: list[dict[str, Any]]) -> None:
    """Ajusta scores in-place: título, hinos curtos e penalidade leve de épicos."""
    q_tokens = set(tokenize(query))
    wants_epic = bool(q_tokens & _EPIC_QUERY_HINTS)
    for c in pool:
        title = c.get("title") or ""
        text = c.get("text") or ""
        t_tokens = set(tokenize(title))
        boost = 0.0
        if q_tokens:
            boost += 0.14 * (len(q_tokens & t_tokens) / len(q_tokens))
        if _HYMN_TITLE.search(title):
            boost += 0.07
        # preferir trechos curtos (śāstra/hino) em queries não-épicas
        n = len(text)
        if n and n < 2500:
            boost += 0.04
        elif n > 8000 and not wants_epic:
            boost -= 0.03
        if _EPIC_TITLE.search(title) and not wants_epic:
            boost -= 0.08
        c["score"] = round(float(c.get("score") or 0.0) + boost, 4)


def hybrid_rerank(
    query: str,
    semantic_hits: list[dict[str, Any]],
    all_chunks: Optional[list[dict[str, Any]]] = None,
    top_k: int = 8,
    semantic_weight: float = 0.65,
    max_per_doc: int = 2,
) -> list[dict[str, Any]]:
    """
    Combina ranking semântico (já filtrado) com score lexical nos mesmos hits
    e, se all_chunks for dado, amplia candidatos lexicais.

    Pós-processamento: boost de título/hino + diversificação por doc_id
    (evita top-k monopolizado pelo Mahābhārata).
    """
    pool: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    def add(ch: dict[str, Any]) -> None:
        cid = str(ch.get("chunk_id") or id(ch))
        if cid in seen_ids:
            return
        seen_ids.add(cid)
        pool.append(dict(ch))

    for h in semantic_hits:
        add(h)

    if all_chunks:
        # top lexicais do corpus chunkado
        scores = lexical_scores(query, all_chunks)
        order = sorted(range(len(all_chunks)), key=lambda i: scores[i], reverse=True)
        for i in order[: max(top_k * 4, 20)]:
            if scores[i] <= 0:
                break
            ch = dict(all_chunks[i])
            ch["_lex_score"] = scores[i]
            add(ch)

    if not pool:
        return []

    # ranking semântico: ordem de entrada dos semantic_hits
    sem_score_map: dict[str, float] = {}
    for i, h in enumerate(semantic_hits):
        cid = str(h.get("chunk_id") or "")
        sem_score_map[cid] = float(h.get("score") or max(0.0, 1.0 - i * 0.02))

    lex = lexical_scores(query, pool)
    for i, ch in enumerate(pool):
        ch["_lex_score"] = lex[i]
        cid = str(ch.get("chunk_id") or "")
        ch["_sem_score"] = sem_score_map.get(cid, 0.0)

    # normaliza
    max_lex = max((c["_lex_score"] for c in pool), default=1.0) or 1.0
    max_sem = max((c["_sem_score"] for c in pool), default=1.0) or 1.0

    for c in pool:
        ns = c["_sem_score"] / max_sem
        nl = c["_lex_score"] / max_lex
        c["score"] = round(semantic_weight * ns + (1 - semantic_weight) * nl, 4)
        c.pop("_lex_score", None)
        c.pop("_sem_score", None)

    apply_title_and_size_boost(query, pool)
    pool.sort(key=lambda x: float(x.get("score") or 0), reverse=True)
    # pool maior antes de diversificar
    return diversify_by_doc(pool, top_k=top_k, max_per_doc=max_per_doc)
