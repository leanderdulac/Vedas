"""Recuperação híbrida: semântica + lexical (RRF) + expansão de consulta."""

from __future__ import annotations

import math
import re
import threading
from collections import Counter
from typing import Any

from vedic_pipeline.common.sanskrit import (
    fold_for_search,
    get_sanskrit_variants,
    iast_to_ascii,
)
from vedic_pipeline.search.reranker import rerank_chunks

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
    folded = fold_for_search(text)
    return re.findall(r"[\wāīūṛṝḷḹṃḥśṣñṅṭḍṇ]+", folded, flags=re.UNICODE)


def expand_query(query: str) -> list[str]:
    """Gera variantes da pergunta para recall mais alto, incluindo transliterações sânscritas."""
    base = query.strip()
    variants = [base]

    # Transliteração e variações sânscritas da query inteira
    for s_var in get_sanskrit_variants(base):
        if s_var not in variants:
            variants.append(s_var)

    tokens = tokenize(base)
    extra_terms: list[str] = []
    for t in tokens:
        for exp in QUERY_EXPANSIONS.get(t, []):
            extra_terms.append(exp)
        # Expande termos sânscritos específicos do token
        for s_token in get_sanskrit_variants(t):
            if s_token != t and s_token not in extra_terms:
                extra_terms.append(s_token)
            # Também verifica se a variante tem expansão em QUERY_EXPANSIONS
            for exp in QUERY_EXPANSIONS.get(s_token, []):
                if exp not in extra_terms:
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


_CHUNK_TOKEN_CACHE: dict[str, list[str]] = {}
_CHUNK_TOKEN_CACHE_LOCK = threading.Lock()


def get_chunk_tokens(chunk: dict[str, Any]) -> list[str]:
    """Retorna tokens do chunk com cache em memória.

    Indexa também o metadado (título/obra/localizador), nas formas nativa e
    transliterada (IAST->ASCII), para que uma query em alfabeto latino
    ("bhagavad", "gita", "yoga sutra") case com chunks cujo texto é Devanāgarī
    e, por isso, não responderia ao BM25 puro sobre o corpo.
    """
    cid = chunk.get("chunk_id")
    if cid:
        with _CHUNK_TOKEN_CACHE_LOCK:
            cached = _CHUNK_TOKEN_CACHE.get(cid)
        if cached is not None:
            return cached

    meta_bits = [str(chunk.get(key) or "") for key in ("title", "work", "locator", "heading")]
    meta = " ".join(b for b in meta_bits if b).strip()
    meta_ascii = iast_to_ascii(meta) if meta else ""
    text = chunk.get("text") or ""
    # metadado aparece duas vezes (original + ascii) => df por token não muda
    # (set() deduplica no cálculo), mas expõe a variante que casa com a query.
    combined = f"{meta}\n{meta_ascii}\n{text}" if meta else text
    tokens = tokenize(combined)
    if cid:
        with _CHUNK_TOKEN_CACHE_LOCK:
            if len(_CHUNK_TOKEN_CACHE) > 50_000:
                _CHUNK_TOKEN_CACHE.clear()
            _CHUNK_TOKEN_CACHE[cid] = tokens
    return tokens


def lexical_scores(query: str, chunks: list[dict[str, Any]]) -> list[float]:
    """BM25-light sobre tokens unicode com cache de tokenização."""
    q_tokens = tokenize(query)
    if not q_tokens or not chunks:
        return [0.0] * len(chunks)

    docs = [get_chunk_tokens(c) for c in chunks]
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
    all_chunks: list[dict[str, Any]] | None = None,
    top_k: int = 8,
    semantic_weight: float = 0.65,
    max_per_doc: int = 2,
    use_cross_encoder: bool = True,
) -> list[dict[str, Any]]:
    """
    Combina ranking semântico (já filtrado) com score lexical nos mesmos hits
    e, se all_chunks for dado, amplia candidatos lexicais.

    Pós-processamento: boost de título/hino + Cross-Encoder + diversificação por doc_id
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
    # Candidatos trazidos só pela via léxica (não no topo semântico) não podem
    # ser aniquilados por nota semântica 0: usam um piso = menor nota semântica
    # presente. Sem isso, um match forte de título/texto (ex.: query em latim
    # x chunk em Devanāgarī) nunca supera o ruído semântico de candidatos RV.
    sem_floor = min(sem_score_map.values()) if sem_score_map else 0.0
    for i, ch in enumerate(pool):
        ch["_lex_score"] = lex[i]
        cid = str(ch.get("chunk_id") or "")
        ch["_sem_score"] = sem_score_map.get(cid, sem_floor)

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

    if use_cross_encoder and pool:
        top_slice_size = max(top_k * 3, 12)
        top_candidates = pool[:top_slice_size]
        reranked_top = rerank_chunks(query, top_candidates, top_k=top_slice_size)
        pool = reranked_top + pool[top_slice_size:]

    apply_phrase_boost(query, pool)
    pool.sort(key=lambda x: float(x.get("score") or 0), reverse=True)

    # pool maior antes de diversificar
    return diversify_by_doc(pool, top_k=top_k, max_per_doc=max_per_doc)


def apply_phrase_boost(query: str, pool: list[dict[str, Any]]) -> None:
    """Sobe trechos que contêm a consulta (sem acentos de recitação)."""
    needle = fold_for_search(query)
    compact = re.sub(r"\s+", "", needle)
    if len(compact) < 8:
        return
    for chunk in pool:
        hay = fold_for_search(chunk.get("text") or "")
        hay_compact = re.sub(r"\s+", "", hay)
        if needle in hay or compact in hay_compact:
            chunk["score"] = round(float(chunk.get("score") or 0.0) + 1.25, 4)
