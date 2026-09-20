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
from vedic_pipeline.search.reranker import is_reranker_enabled, rerank_chunks

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

# Mandala 1–10 em romano (Griffith: "HYMN X.129").
_MANDALA_ROMAN = {
    1: "I",
    2: "II",
    3: "III",
    4: "IV",
    5: "V",
    6: "VI",
    7: "VII",
    8: "VIII",
    9: "IX",
    10: "X",
}

_NAMED_HYMN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"n[aā]sad[iī]ya|नासदीय|नासदासीन्", re.I), "10.129"),
    (
        re.compile(r"puru[sṣś]h?a\s+s[uū]kta|\bpurusha\s+sukta|\bpurusa\s+sukta", re.I),
        "10.90",
    ),
    (re.compile(r"g[aā]yat(?:h)?r[iī]|गायत्री", re.I), "3.62"),
    # Verse fingerprints (no “Gāyatrī” token): tat savitur… / तत्सवितुर्…
    (
        re.compile(r"tatsavitur|tat\s*savitur|तत्\s*सवितुर्", re.I),
        "3.62",
    ),
    (re.compile(r"hira[nṇ]yagarbha|हिरण्यगर्भ", re.I), "10.121"),
    (
        re.compile(
            r"\bv[aā][kcç]\s+s[uū]kta|\bvak\s+sukta|\bvac\s+sukta|\bv[aā]k\b|\bvāc\b"
            r"|वाक्\s*सूक्त|वाच्\s*सूक्त|वाक\s*सूक्त|वाक्सूक्त",
            re.I,
        ),
        "10.125",
    ),
)

# Só nomes distintivos no *chunk*. Gāyatrī/Hiraṇyagarbha/Vāk na query
# extraem o id (3.62 / 10.121 / 10.125); casar o nome no texto também
# subiria Chandogya III.12 (metro Gāyatrī) ou antologias genéricas.
_NAMED_IN_BLOB: dict[str, re.Pattern[str]] = {
    "10.129": re.compile(r"n[aā]sad[iī]ya|नासदीय|नासदासीन्", re.I),
    "10.90": re.compile(r"puru[sṣś]h?a\s+s[uū]kta|\bpurusha\s+sukta|\bpurusa\s+sukta", re.I),
}

# "RV 10.129", "hymn 1.1", ou id solto "10.129" / "10.90".
_EXPLICIT_HYMN_RE = re.compile(
    r"(?:(?:\brv\b|ṛgveda|rigveda|rig\s*veda|hymn|s[uū]kta)\s*)(\d{1,2}\.\d{1,3})"
    r"|\b(\d{1,2}\.\d{1,3})\b",
    re.I,
)

# PoC local: +~1.4 no título restaurou Nasadiya sob pressão do CE genérico.
LOCATOR_HYMN_TITLE_BOOST = 1.45
LOCATOR_HYMN_TEXT_BOOST = 1.05
# Recall: chunks cujo título/locator casa o id, mesmo fora do top léxico/denso.
LOCATOR_INJECT_PER_HYMN = 4
LOCATOR_INJECT_PER_WORK = LOCATOR_INJECT_PER_HYMN
LOCATOR_WORK_TITLE_BOOST = LOCATOR_HYMN_TITLE_BOOST
# Antologia que cita o hino/obra no corpo não pode ganhar do título específico
# (phrase boost 1.25 + text hymn 1.05). Demote depois do phrase boost.
LOCATOR_ANTHOLOGY_DEMOTE = 1.35
LOCATOR_SHORT_DEITY_BOOST = 1.05

WORK_ISHA = "isha-upanishad"
WORK_KATHA = "katha-upanishad"
WORK_SAMAVEDA = "samaveda"
WORK_YAJUR_VS = "yajurveda-vs"
WORK_BRIHADARANYAKA = "brihadaranyaka-upanishad"
WORK_MANDUKYA = "mandukya-upanishad"
WORK_RAMAYANA = "ramayana"
WORK_YOGA_SUTRA = "yoga-sutra"

# Antologias genéricas: título de coleção, não da obra/hino pedido.
_ANTHOLOGY_TITLE = re.compile(
    r"selected\s+hymns"
    r"|principal\s+upanishads"
    r"|thirteen\s+principal"
    r"|english\s+core"
    r"|antholog"
    r"|selected\s+(?:upanishads|verses|texts|suktas)"
    r"|(?:^|\s)the\s+upanishads\s*[\(:]"
    r"|vedic\s+antholog",
    re.I,
)

# Query curta de divindade RV: Agni → hinos RV Agni, não Mahābhārata.
_SHORT_RV_DEITY = {
    "agni": re.compile(r"\bagni\b|अग्नि", re.I),
}
_SHORT_RV_DEITY_TITLE = {
    "agni": re.compile(
        r"(?:rigveda|ṛgveda|rig\s*veda|\brv\s+\d).{0,120}\bagni\b"
        r"|\bagni\b.{0,120}(?:rigveda|ṛgveda|rig\s*veda|\brv\s+\d)",
        re.I,
    ),
}
_SHORT_DEITY_EXTRA = frozenset(
    {"fire", "god", "deity", "deva", "devata", "hymn", "sukta", "sūkta"}
)

# Query → obra/coleção. Intenção nomeada vence rótulo decoy (Nachiketas > Kauṣītaki).
_NAMED_WORK_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Īśā opening: IAST / ASCII / Devanāgarī. Não casa só idaṃ/jagat (isso sobe RV).
    (
        re.compile(
            r"ī[sś]āvāsy|[iī]sha?v[aā]sy|isavasy|ईशावास्य",
            re.I,
        ),
        WORK_ISHA,
    ),
    (
        re.compile(
            r"\b[iī][sśṣ]h?[aā]\s*upani[sṣś]h?ad|\bisa\s+upanishad|ishavasyopanishad"
            r"|ईशावास्योपनिष",
            re.I,
        ),
        WORK_ISHA,
    ),
    # História de Nachiketas = Kaṭha, mesmo se a query nomear Kauṣītaki.
    (re.compile(r"nachiketas?|n[aā]ciketas?|नचिकेत", re.I), WORK_KATHA),
    (
        re.compile(r"\bka[tṭ]ha\s*upani[sṣś]h?ad|कठोपनिष", re.I),
        WORK_KATHA,
    ),
    (re.compile(r"s[aā]ma\s*[- ]?\s*veda|s[aā]maveda|सामवेद", re.I), WORK_SAMAVEDA),
    (
        re.compile(
            r"shukla\s+yajur|śukla\s+yajur|white\s+yajur"
            r"|v[aā]jasaneyi|vajasneyi|yajurveda\s+vs\b|yajur\s+veda\s+vs\b"
            r"|शुक्ल\s*यजुर्|वाजसनेयि",
            re.I,
        ),
        WORK_YAJUR_VS,
    ),
    # neti neti é o sinal canónico da Bṛhadāraṇyaka (não da antologia).
    (
        re.compile(
            r"b[rṛ]i?had[aā]ra[nṇ]yaka|बृहदारण्यक",
            re.I,
        ),
        WORK_BRIHADARANYAKA,
    ),
    (re.compile(r"\bneti[\s\-]*neti\b|नेति[\s\-]*नेति", re.I), WORK_BRIHADARANYAKA),
    # Māṇḍūkya pelo nome; Om/AUM sozinho é demasiado genérico.
    (
        re.compile(r"m[aā][nṇ][dḍ][uū]kya|माण्डूक्य|मांडूक्य", re.I),
        WORK_MANDUKYA,
    ),
    # Rapto de Sītā / Rāvaṇa → Rāmāyaṇa mesmo se a query nomear o Mahābhārata.
    (
        re.compile(
            r"(?:\bs[iī]t[aā]\b|सीता).{0,80}(?:abduct|kidnap|r[aā]va[nṇ]a|रावण|rapto|raptad)"
            r"|(?:abduct|kidnap|r[aā]va[nṇ]a|रावण|rapto|raptad).{0,80}(?:\bs[iī]t[aā]\b|सीता)",
            re.I,
        ),
        WORK_RAMAYANA,
    ),
    (
        re.compile(r"r[aā]m[aā]ya[nṇ]|रामायण|v[aā]lm[iī]ki|वाल्मीकि", re.I),
        WORK_RAMAYANA,
    ),
    # Definição de yoga (EN/PT) ou Yoga-sūtra / Patañjali explícitos.
    (
        re.compile(
            r"yoga.?s[uū]tra|yogasutra|pata[nñ]jali|patañjali|योगसूत्र|पतञ्जलि"
            r"|yogaś\s+citta|citta.?v[rṛ]tti.?nirodha",
            re.I,
        ),
        WORK_YOGA_SUTRA,
    ),
    (
        re.compile(
            r"defini(?:tion|ção|cao|çao|cão)\s+(?:of\s+|de\s+|do\s+)?yoga"
            r"|yoga\s+defini(?:tion|ção|cao|çao|cão)"
            r"|o\s+que\s+[eé]\s+(?:o\s+)?yoga",
            re.I,
        ),
        WORK_YOGA_SUTRA,
    ),
)

# Título/locator da *obra*, não comentário/antologia que só discute a coleção.
_WORK_TITLE_MATCH: dict[str, re.Pattern[str]] = {
    WORK_ISHA: re.compile(
        r"ishavasy|īśāvāsy|ईशावास्य"
        r"|\b[iī][sśṣ]h?[aā]\s*upani[sṣś]h?ad"
        r"|\bisa\s+upanishad|\bisha\b|īśā",
        re.I,
    ),
    WORK_KATHA: re.compile(
        r"\bka[tṭ]ha\s*upani[sṣś]h?ad|\bka[tṭ]ha\b|कठोपनिष|कठ",
        re.I,
    ),
    WORK_SAMAVEDA: re.compile(
        r"s[aā]maveda\s+sv\b|\bs[aā]ma[- ]?veda\b|सामवेद|\bsv\s+\d",
        re.I,
    ),
    WORK_YAJUR_VS: re.compile(
        r"yajurveda\s+vs\b|yajur\s+veda\s+vs\b|\bvs\s+\d"
        r"|v[aā]jasaneyi|vajasneyi|shukla\s+yajur|white\s+yajur"
        r"|शुक्ल\s*यजुर्|वाजसनेयि",
        re.I,
    ),
    WORK_BRIHADARANYAKA: re.compile(
        r"b[rṛ]i?had[aā]ra[nṇ]yaka|बृहदारण्यक",
        re.I,
    ),
    WORK_MANDUKYA: re.compile(
        r"m[aā][nṇ][dḍ][uū]kya|माण्डूक्य|मांडूक्य",
        re.I,
    ),
    WORK_RAMAYANA: re.compile(
        r"r[aā]m[aā]ya[nṇ]|रामायण|v[aā]lm[iī]ki|वाल्मीकि",
        re.I,
    ),
    WORK_YOGA_SUTRA: re.compile(
        r"yoga.?s[uū]tra|yogasutra|pata[nñ]jali|patañjali|योगसूत्र|पतञ्जलि",
        re.I,
    ),
}

_WORK_TITLE_EXCLUDE: dict[str, re.Pattern[str]] = {
    WORK_ISHA: re.compile(r"\brigveda\b|\brig\s*veda\b|\brv\s+\d", re.I),
    WORK_KATHA: re.compile(
        r"chandogya|chāndogya|kaushitaki|kau[sṣ][iī]taki|kausitaki",
        re.I,
    ),
    WORK_SAMAVEDA: re.compile(
        r"chandogya|chāndogya|upanishad|upani[sṣ]ad|antholog",
        re.I,
    ),
    WORK_YAJUR_VS: re.compile(
        r"müller|muller|upanishad|upani[sṣ]ad|antholog"
        r"|krishna\s+yajur|kṛṣṇa\s+yajur",
        re.I,
    ),
    WORK_BRIHADARANYAKA: re.compile(
        r"principal\s+upanishads|english\s+core|antholog|selected\s+hymns"
        r"|markandeya|mārkaṇḍeya",
        re.I,
    ),
    WORK_MANDUKYA: re.compile(
        r"principal\s+upanishads|english\s+core|antholog|selected\s+hymns"
        r"|mu[nṇ][dḍ]aka|मुण्डक",
        re.I,
    ),
    WORK_RAMAYANA: re.compile(
        r"mahabharata|mahābhārata|ganguli|bhishma|bhīṣma|kurukshetra",
        re.I,
    ),
    WORK_YOGA_SUTRA: re.compile(
        r"markandeya|mārkaṇḍeya|pur[aā][nṇ]a|mah[aā]bh[aā]rata"
        r"|upanishad|upani[sṣ]ad|selected\s+hymns|antholog",
        re.I,
    ),
}

# Preferir o prefixo canônico da coleção (Sāmaveda SV / Yajurveda VS) ao injetar.
_WORK_TITLE_PREFERRED: dict[str, re.Pattern[str]] = {
    WORK_ISHA: re.compile(r"upani[sṣś]h?ad|ईशावास्य|\bisha\b|īśā", re.I),
    WORK_KATHA: re.compile(r"upani[sṣś]h?ad|कठ", re.I),
    WORK_SAMAVEDA: re.compile(r"s[aā]maveda\s+sv\b|\bsv\s+\d", re.I),
    WORK_YAJUR_VS: re.compile(r"yajurveda\s+vs\b|\bvs\s+\d|v[aā]jasaneyi|vajasneyi", re.I),
    WORK_BRIHADARANYAKA: re.compile(r"b[rṛ]i?had[aā]ra[nṇ]yaka|बृहदारण्यक", re.I),
    WORK_MANDUKYA: re.compile(r"m[aā][nṇ][dḍ][uū]kya|माण्डूक्य", re.I),
    WORK_RAMAYANA: re.compile(r"r[aā]m[aā]ya[nṇ]|v[aā]lm[iī]ki|रामायण", re.I),
    WORK_YOGA_SUTRA: re.compile(r"yoga.?s[uū]tra|yogasutra|pata[nñ]jali", re.I),
}


def hymn_id_in_blob(blob: str, hymn: str) -> bool:
    """Casa 10.5 em 'RV 10.5' sem casar 10.50 / 10.125."""
    escaped = re.escape((hymn or "").strip())
    if not escaped:
        return False
    return re.search(rf"(?<![\d.]){escaped}(?!\d)", blob or "", flags=re.I) is not None


def hymn_id_variants(hymn: str) -> list[str]:
    raw = (hymn or "").strip()
    if not raw or "." not in raw:
        return [raw] if raw else []
    book_s, num = raw.split(".", 1)
    variants = [raw]
    try:
        roman = _MANDALA_ROMAN.get(int(book_s))
    except ValueError:
        roman = None
    if roman:
        variants.append(f"{roman}.{num}")
    return variants


def extract_query_hymn_ids(query: str) -> list[str]:
    """Hinos pedidos na query: nomes canônicos e RV X.Y explícito.

    Nomes: Nasadiya→10.129, Purusha Sukta→10.90, Gāyatrī→3.62
    (também tat savitur… / तत्सवितुर्…), Hiraṇyagarbha→10.121,
    Vāk/Vāc Sūkta→10.125. Se um nome casa e um id explícito discorda,
    o nome vence — o id conflitante não entra no boost.
    """
    text = query or ""
    found: list[str] = []
    seen: set[str] = set()
    named: set[str] = set()

    def add(hymn: str) -> None:
        key = hymn.strip()
        if key and key not in seen:
            seen.add(key)
            found.append(key)

    for pattern, hymn in _NAMED_HYMN_PATTERNS:
        if pattern.search(text):
            named.add(hymn)
            add(hymn)
    for match in _EXPLICIT_HYMN_RE.finditer(text):
        explicit = (match.group(1) or match.group(2) or "").strip()
        if named and explicit not in named:
            continue
        add(explicit)
    return found


def _chunk_locator_blob(chunk: dict[str, Any]) -> str:
    return " ".join(
        str(chunk.get(key) or "") for key in ("title", "locator", "heading", "work")
    )


def chunk_matches_hymn(chunk: dict[str, Any], hymn: str, *, text_too: bool = True) -> str | None:
    """Retorna 'title' | 'text' conforme onde o id (ou nome distintivo) aparece.

    Gāyatrī → 3.62 casa o id RV, não a palavra 'Gayatri' em Chandogya III.12.
    """
    loc_blob = _chunk_locator_blob(chunk)
    text_blob = str(chunk.get("text") or "") if text_too else ""
    named = _NAMED_IN_BLOB.get(hymn)
    if named and named.search(loc_blob):
        return "title"
    for variant in hymn_id_variants(hymn):
        if hymn_id_in_blob(loc_blob, variant):
            return "title"
    if not text_too:
        return None
    if named and named.search(text_blob):
        return "text"
    for variant in hymn_id_variants(hymn):
        if hymn_id_in_blob(text_blob, variant):
            return "text"
    return None


def _hymn_level_locator(chunk: dict[str, Any], hymn: str) -> bool:
    """True se o blob de título/locator traz o id sem verso seguinte (3.62 vs 3.62.10)."""
    blob = _chunk_locator_blob(chunk)
    if not hymn_id_in_blob(blob, hymn):
        return False
    return re.search(rf"(?<![\d.]){re.escape(hymn)}\.\d", blob or "", flags=re.I) is None


def locator_hymn_injections(
    query: str,
    all_chunks: list[dict[str, Any]],
    *,
    exclude_ids: set[str] | None = None,
    per_hymn: int = LOCATOR_INJECT_PER_HYMN,
) -> list[dict[str, Any]]:
    """Chunks do corpus cujo título/locator casa os RV ids extraídos da query.

    Recall, não só re-score: Gāyatrī frequentemente perde o pool para Chandogya
    III.12 / antologias. Casa só título/locator (barato) com as mesmas regras
    pós-PR5 — id RV para 3.62 / 10.121 / 10.125, sem boost pelo nome no decoy.
    Prefere o hino inteiro (RV 3.62) a um verso avulso; limita a ``per_hymn``.
    """
    hymns = extract_query_hymn_ids(query)
    if not hymns or not all_chunks:
        return []
    skip = set(exclude_ids or ())
    injected: list[dict[str, Any]] = []
    for hymn in hymns:
        ranked: list[tuple[int, int, dict[str, Any]]] = []
        for idx, ch in enumerate(all_chunks):
            cid = str(ch.get("chunk_id") or id(ch))
            if cid in skip:
                continue
            if chunk_matches_hymn(ch, hymn, text_too=False) != "title":
                continue
            if _chunk_is_generic_anthology(ch):
                continue
            hymn_level = 0 if _hymn_level_locator(ch, hymn) else 1
            ranked.append((hymn_level, idx, ch))
        ranked.sort(key=lambda row: (row[0], row[1]))
        for _level, _idx, ch in ranked[: max(0, per_hymn)]:
            cid = str(ch.get("chunk_id") or id(ch))
            skip.add(cid)
            copy = dict(ch)
            copy["_hymn_injected"] = hymn
            injected.append(copy)
    return injected


def _chunk_is_generic_anthology(chunk: dict[str, Any]) -> bool:
    """True se o metadado é coleção/antologia, não o título da obra pedida."""
    return bool(_ANTHOLOGY_TITLE.search(_chunk_locator_blob(chunk)))


def _chunk_is_specific_hymn_title(chunk: dict[str, Any], hymn: str) -> bool:
    if _chunk_is_generic_anthology(chunk):
        return False
    return chunk_matches_hymn(chunk, hymn, text_too=False) == "title"


def _chunk_is_specific_work_title(chunk: dict[str, Any], work: str) -> bool:
    if _chunk_is_generic_anthology(chunk):
        return False
    return chunk_matches_work(chunk, work)


def pool_has_specific_locator_hit(query: str, pool: list[dict[str, Any]]) -> bool:
    """True se o pool já tem o hino/obra nomeado no título (não antologia)."""
    hymns = extract_query_hymn_ids(query)
    works = extract_query_work_keys(query)
    for chunk in pool:
        if any(_chunk_is_specific_hymn_title(chunk, hymn) for hymn in hymns):
            return True
        if any(_chunk_is_specific_work_title(chunk, work) for work in works):
            return True
    return False


def apply_locator_hymn_boost(query: str, pool: list[dict[str, Any]]) -> None:
    """Sobe candidatos cujo título/locator/texto traz o hino pedido na query.

    Corre no híbrido e de novo *depois* do CE (quando ligado): o MiniLM genérico
    e o CE de domínio v1–v3 ainda empurram RV 10.125 / 10.5 no lugar de 10.129.
    Nome canônico tem peso cheio; id explícito conflitante não entra no conjunto.
    Com o hino real no pool, antologias que só citam o sūkta no corpo não sobem.
    """
    hymns = extract_query_hymn_ids(query)
    if not hymns or not pool:
        return
    has_specific = any(
        _chunk_is_specific_hymn_title(chunk, hymn) for chunk in pool for hymn in hymns
    )
    for chunk in pool:
        if has_specific and _chunk_is_generic_anthology(chunk):
            continue
        best = 0.0
        matched: str | None = None
        for hymn in hymns:
            where = chunk_matches_hymn(chunk, hymn)
            if where == "title":
                best = max(best, LOCATOR_HYMN_TITLE_BOOST)
                matched = hymn
            elif where == "text":
                best = max(best, LOCATOR_HYMN_TEXT_BOOST)
                matched = matched or hymn
        if best:
            chunk["score"] = round(float(chunk.get("score") or 0.0) + best, 4)
            chunk["_hymn_boost"] = best
            chunk["_hymn_match"] = matched


def extract_query_work_keys(query: str) -> list[str]:
    """Obras/coleções pedidas na query.

    Mesma política do hino nomeado: o sinal canónico vence o rótulo decoy.
    Nachiketas → Kaṭha mesmo se a query disser Kauṣītaki; Sītā+abduction →
    Rāmāyaṇa mesmo com rótulo Mahābhārata; neti neti → Bṛhadāraṇyaka, não a
    antologia Principal Upanishads; definição de yoga (PT/EN) → Yoga-sūtra.
    """
    text = query or ""
    found: list[str] = []
    seen: set[str] = set()
    for pattern, work in _NAMED_WORK_PATTERNS:
        if work in seen:
            continue
        if pattern.search(text):
            seen.add(work)
            found.append(work)
    return found


def chunk_matches_work(chunk: dict[str, Any], work: str) -> bool:
    """True se título/locator/heading/work identificam a obra, não um decoy.

    Só metadado (não o corpo): Chandogya I.6 fala de sāman mas o título
    não é Sāmaveda SV…; RV 10.58 não é Īśā só porque o verso partilha palavras.
    """
    blob = _chunk_locator_blob(chunk)
    exclude = _WORK_TITLE_EXCLUDE.get(work)
    if exclude and exclude.search(blob):
        return False
    match = _WORK_TITLE_MATCH.get(work)
    return bool(match and match.search(blob))


def _work_injection_rank(chunk: dict[str, Any], work: str) -> int:
    blob = _chunk_locator_blob(chunk)
    preferred = _WORK_TITLE_PREFERRED.get(work)
    if preferred and preferred.search(blob):
        return 0
    return 1


def locator_work_injections(
    query: str,
    all_chunks: list[dict[str, Any]],
    *,
    exclude_ids: set[str] | None = None,
    per_work: int = LOCATOR_INJECT_PER_WORK,
) -> list[dict[str, Any]]:
    """Injeta chunks cujo título/locator casa a obra extraída da query.

    Mesmo teto que o hino (PR #6): a coleção pode ter milhares de SV/VS;
    bastam poucos candidatos certos no pool para o boost ganhar do comentário.
    """
    works = extract_query_work_keys(query)
    if not works or not all_chunks:
        return []
    skip = set(exclude_ids or ())
    injected: list[dict[str, Any]] = []
    for work in works:
        ranked: list[tuple[int, int, dict[str, Any]]] = []
        for idx, ch in enumerate(all_chunks):
            cid = str(ch.get("chunk_id") or id(ch))
            if cid in skip:
                continue
            if not chunk_matches_work(ch, work):
                continue
            if _chunk_is_generic_anthology(ch):
                continue
            ranked.append((_work_injection_rank(ch, work), idx, ch))
        ranked.sort(key=lambda row: (row[0], row[1]))
        for _level, _idx, ch in ranked[: max(0, per_work)]:
            cid = str(ch.get("chunk_id") or id(ch))
            skip.add(cid)
            copy = dict(ch)
            copy["_work_injected"] = work
            injected.append(copy)
    return injected


def apply_locator_work_boost(query: str, pool: list[dict[str, Any]]) -> None:
    """Sobe candidatos cujo título/locator é a obra pedida (não o decoy)."""
    works = extract_query_work_keys(query)
    if not works or not pool:
        return
    for chunk in pool:
        if _chunk_is_generic_anthology(chunk):
            continue
        matched: str | None = None
        for work in works:
            if chunk_matches_work(chunk, work):
                matched = work
                break
        if matched:
            chunk["score"] = round(
                float(chunk.get("score") or 0.0) + LOCATOR_WORK_TITLE_BOOST,
                4,
            )
            chunk["_work_boost"] = LOCATOR_WORK_TITLE_BOOST
            chunk["_work_match"] = matched


def extract_short_rv_deity(query: str) -> str | None:
    """Divindade RV em query curta (Agni), sem hino/obra já extraídos.

    ``dharma`` e outros termos ambíguos não entram: não há sinal seguro.
    """
    text = query or ""
    tokens = tokenize(text)
    if not tokens or len(tokens) > 3:
        return None
    if extract_query_hymn_ids(text) or extract_query_work_keys(text):
        return None
    folded_tokens = {fold_for_search(t) for t in tokens}
    for deity, pattern in _SHORT_RV_DEITY.items():
        if not pattern.search(text):
            continue
        leftover = folded_tokens - {deity} - _SHORT_DEITY_EXTRA
        if leftover:
            continue
        return deity
    return None


def chunk_matches_short_rv_deity(chunk: dict[str, Any], deity: str) -> bool:
    title_re = _SHORT_RV_DEITY_TITLE.get(deity)
    if not title_re:
        return False
    blob = _chunk_locator_blob(chunk)
    if _EPIC_TITLE.search(blob) or _chunk_is_generic_anthology(chunk):
        return False
    return bool(title_re.search(blob))


def locator_deity_injections(
    query: str,
    all_chunks: list[dict[str, Any]],
    *,
    exclude_ids: set[str] | None = None,
    per_deity: int = LOCATOR_INJECT_PER_HYMN,
) -> list[dict[str, Any]]:
    """Injeta hinos RV da divindade curta (teto igual ao hino nomeado)."""
    deity = extract_short_rv_deity(query)
    if not deity or not all_chunks:
        return []
    skip = set(exclude_ids or ())
    ranked: list[tuple[int, dict[str, Any]]] = []
    for idx, ch in enumerate(all_chunks):
        cid = str(ch.get("chunk_id") or id(ch))
        if cid in skip:
            continue
        if not chunk_matches_short_rv_deity(ch, deity):
            continue
        ranked.append((idx, ch))
    injected: list[dict[str, Any]] = []
    for _idx, ch in ranked[: max(0, per_deity)]:
        cid = str(ch.get("chunk_id") or id(ch))
        skip.add(cid)
        copy = dict(ch)
        copy["_deity_injected"] = deity
        injected.append(copy)
    return injected


def apply_short_rv_deity_boost(query: str, pool: list[dict[str, Any]]) -> None:
    """Sobe hinos RV da divindade (Agni) contra épicos, só em query curta."""
    deity = extract_short_rv_deity(query)
    if not deity or not pool:
        return
    for chunk in pool:
        if chunk_matches_short_rv_deity(chunk, deity):
            chunk["score"] = round(
                float(chunk.get("score") or 0.0) + LOCATOR_SHORT_DEITY_BOOST,
                4,
            )
            chunk["_deity_boost"] = LOCATOR_SHORT_DEITY_BOOST
            chunk["_deity_match"] = deity


def apply_anthology_demotion(query: str, pool: list[dict[str, Any]]) -> None:
    """Rebaixa antologia genérica quando o hino/obra específico já está no pool.

    Principal Upanishads / Rig Veda selected hymns citam neti neti ou Nasadiya
    no corpo e ganham phrase-boost; o título específico deve ficar no top-1.
    Sem hit específico no pool, a antologia não é rebaixada (ainda é recall).
    """
    if not pool:
        return
    if not extract_query_hymn_ids(query) and not extract_query_work_keys(query):
        return
    if not pool_has_specific_locator_hit(query, pool):
        return
    for chunk in pool:
        if chunk.get("_anthology_demote"):
            continue
        if not _chunk_is_generic_anthology(chunk):
            continue
        chunk["score"] = round(
            float(chunk.get("score") or 0.0) - LOCATOR_ANTHOLOGY_DEMOTE,
            4,
        )
        chunk["_anthology_demote"] = LOCATOR_ANTHOLOGY_DEMOTE


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
    e, se all_chunks for dado, amplia candidatos lexicais e injeta chunks
    cujo título/locator casa um RV id ou uma obra/coleção extraída da query
    (antes do boost).

    Pós-processamento: boost de título/hino + locator/RV X.Y + obra nomeada
    (Īśā, Kaṭha, Sāmaveda, Śukla Yajur, Bṛhadāraṇyaka, Māṇḍūkya, Rāmāyaṇa,
    Yoga-sūtra) + demote de antologia quando o título específico está no
    pool + Cross-Encoder (locator de novo após o CE) + diversificação por
    doc_id.
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
        # Recall por locator: o boost só reordena o que já entrou no pool.
        for ch in locator_hymn_injections(query, all_chunks, exclude_ids=seen_ids):
            add(ch)
        for ch in locator_work_injections(query, all_chunks, exclude_ids=seen_ids):
            add(ch)
        for ch in locator_deity_injections(query, all_chunks, exclude_ids=seen_ids):
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
    apply_locator_hymn_boost(query, pool)
    apply_locator_work_boost(query, pool)
    apply_short_rv_deity_boost(query, pool)
    pool.sort(key=lambda x: float(x.get("score") or 0), reverse=True)

    if use_cross_encoder and is_reranker_enabled() and pool:
        top_slice_size = max(top_k * 3, 12)
        top_candidates = pool[:top_slice_size]
        reranked_top = rerank_chunks(query, top_candidates, top_k=top_slice_size)
        pool = reranked_top + pool[top_slice_size:]
        # CE genérico/domínio ainda inverte Nasadiya → 10.125/10.5; reaplicar o locator.
        apply_locator_hymn_boost(query, pool)
        apply_locator_work_boost(query, pool)
        apply_short_rv_deity_boost(query, pool)

    apply_phrase_boost(query, pool)
    apply_anthology_demotion(query, pool)
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
