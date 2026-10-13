"""Consultas de entidade: "Narada Muni", "Quem foi Vyāsa?", "Nārada".

Uma pergunta que só nomeia um personagem (ou conceito) pede cobertura, não
o trecho mais parecido: o nome aparece no Mahābhārata, nos Purāṇas, num
Upaniṣad e num hino, e a resposta deve passar por cada tradição. Sem isso o
top-k vira um único épico (o Mahābhārata tem milhares de chunks em quatro
documentos) e a resposta sai incompleta.

Este módulo é genérico, sem listas de personagens:

- ``parse_entity_query`` reconhece a consulta curta que só nomeia alguém,
  tira títulos (Muni, Ṛṣi, Devarṣi, Śrī…) e palavras de pergunta (PT/EN);
- ``name_alternates`` gera grafias que o corpus usa para o mesmo nome
  (Griffith escreve "Nárad", sem o -a final; "Naarada" → "narada");
- ``work_key`` agrupa chunks por obra (os quatro volumes do Mahābhārata são
  uma obra; RV 8.13 e RV 9.104 também);
- ``diversify_by_work`` monta o top-k em rodadas: primeiro o melhor trecho
  de cada obra, depois o segundo, até o teto por obra;
- ``corpus_coverage`` conta, no índice lexical, em quantos trechos de cada
  obra o nome aparece, e ``missing_reference_works`` diz quais obras de
  referência não estão no acervo — a resposta pode dizer o que falta.
"""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from vedic_pipeline.common.sanskrit import fold_for_search, iast_to_ascii

# Títulos e tratamentos que acompanham o nome sem identificá-lo. "Muni" na
# consulta somava BM25 a qualquer trecho com "muni" e afastava o personagem.
HONORIFICS: frozenset[str] = frozenset(
    {
        "muni", "munis", "rishi", "rishis", "risi", "rsi", "rshi", "rṣi",
        "maharshi", "maharsi", "maharishi", "maharisi",
        "devarshi", "devarsi", "devarishi", "devarisi", "devarṣi",
        "brahmarshi", "brahmarsi", "rajarshi", "rajarsi",
        "sri", "shri", "shree", "srila", "srimad", "sriman",
        "bhagavan", "bhagwan", "swami", "svami", "goswami", "gosvami",
        "acharya", "acarya", "guru", "baba", "maharaj", "maharaja",
        "sage", "saint", "seer", "sabio", "santo", "vidente",
        "lord", "senhor", "god", "deus", "deusa", "goddess", "deva", "devata",
    }
)

# Palavras de pergunta e ligação (PT/EN) que não fazem parte do nome.
QUESTION_WORDS: frozenset[str] = frozenset(
    {
        "quem", "foi", "era", "e", "eh", "o", "a", "os", "as", "um", "uma", "de", "do", "da", "dos",
        "das", "sobre", "fale", "falar", "conte", "contar", "me", "mim", "explique", "quero", "saber",
        "historia", "personagem", "papel", "nos", "nas", "no", "na", "em", "que", "qual", "quais",
        "segundo", "who", "was", "is", "are", "the", "an", "about", "tell", "of", "in", "what",
        "explain", "story", "character", "role", "describe", "according", "to", "and",
    }
)

# Gênero ou coleção na consulta ("Atman segundo as Upanishads") é pergunta
# com escopo, não um nome: segue a busca normal.
GENRE_WORDS: frozenset[str] = frozenset(
    {
        "veda", "vedas", "rigveda", "samaveda", "yajurveda", "atharvaveda", "upanishad",
        "upanishads", "upanisad", "upanisads", "purana", "puranas", "sutra", "sutras", "gita",
        "itihasa", "epic", "epico", "hymn", "hino", "sukta", "mantra", "verse", "verso", "sloka",
        "shloka",
    }
)

MAX_NAME_TOKENS = 3
MAX_QUERY_TOKENS = 8
MIN_NAME_LEN = 3

# Ajustes do modo entidade (sobrescrevíveis por env, sem mudar o default
# das buscas que não são de entidade).
ENTITY_TOP_K = 14
ENTITY_MAX_PER_WORK = 3
ENTITY_INJECT_PER_WORK = 3
ENTITY_INJECT_MAX_WORKS = 14
ENTITY_CONTEXT_CHARS = 15000
ENTITY_MAX_TOKENS = 2600
ENTITY_MENTION_BOOST = 0.15
ENTITY_MENTION_STEP = 0.04
ENTITY_MENTION_CAP = 5
ENTITY_NO_MENTION_PENALTY = 0.3

_WORD = re.compile(r"[a-z]+")


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, "") or default)
    except ValueError:
        return default
    return value if value > 0 else default


def entity_top_k() -> int:
    return _env_int("VEDIC_ENTITY_TOP_K", ENTITY_TOP_K)


def entity_context_chars() -> int:
    return _env_int("VEDIC_ENTITY_CONTEXT_CHARS", ENTITY_CONTEXT_CHARS)


def entity_mode_enabled() -> bool:
    raw = (os.environ.get("VEDIC_ENTITY_MODE") or "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def fold_ascii(text: str) -> str:
    """Nārada / Nárada / Nârada / NARADA → narada."""
    return fold_for_search(iast_to_ascii(text or ""))


def name_alternates(token: str) -> list[str]:
    """Grafias do mesmo nome que o corpus usa (sem a forma original)."""
    base = fold_ascii(token)
    out: list[str] = []

    def add(form: str) -> None:
        if form and form != base and form not in out and len(form) >= MIN_NAME_LEN:
            out.append(form)

    collapsed = re.sub(r"([aeiou])\1+", r"\1", base)  # Naarada → narada
    add(collapsed)
    for form in (base, collapsed):
        # Griffith (Rāmāyaṇa) corta o -a final: Nárad, Vasishtha → Vasishth não.
        if len(form) >= 6 and form.endswith("a") and form[-2] not in "aeiou":
            add(form[:-1])
        if "sh" in form:
            add(form.replace("sh", "s"))
    return out


@dataclass(frozen=True)
class EntityQuery:
    query: str
    names: tuple[str, ...]          # dobrados: ("narada",)
    display: str                    # como o usuário escreveu, sem títulos: "Narada"
    honorifics: tuple[str, ...] = field(default_factory=tuple)

    def lexical_query(self) -> str:
        return " ".join(self.names)

    def alternates(self) -> dict[str, list[str]]:
        return {n: name_alternates(n) for n in self.names if name_alternates(n)}

    def all_forms(self) -> set[str]:
        forms = set(self.names)
        for alts in self.alternates().values():
            forms.update(alts)
        return forms


def parse_entity_query(query: str) -> EntityQuery | None:
    """EntityQuery se a consulta só nomeia alguém/algo (1–3 tokens úteis).

    Pergunta com hino ou obra nomeados ("Nārada no Rāmāyaṇa", "RV 8.13")
    segue o caminho do localizador, não este.
    """
    if not entity_mode_enabled():
        return None
    text = (query or "").strip()
    if not text or re.search(r"\d", text):
        return None
    from vedic_pipeline.search.hybrid import extract_query_hymn_ids, extract_query_work_keys

    if extract_query_hymn_ids(text) or extract_query_work_keys(text):
        return None
    # Divide por espaço/pontuação (não por \w: as matras do Devanāgarī não são \w).
    raw_tokens = [t for t in re.split(r"[\s,.;:!?¿¡()\[\]{}\"'“”‘’/\\|–—-]+", text) if t]
    if not raw_tokens or len(raw_tokens) > MAX_QUERY_TOKENS:
        return None
    names: list[str] = []
    display: list[str] = []
    honor: list[str] = []
    for raw in raw_tokens:
        folded = fold_ascii(raw)
        if not _WORD.fullmatch(folded or ""):
            # Devanāgarī: a variante IAST vem do expand_query; aqui dobra p/ ASCII.
            from vedic_pipeline.common.sanskrit import devanagari_to_iast, has_devanagari

            if has_devanagari(raw):
                folded = fold_ascii(devanagari_to_iast(raw))
            if not _WORD.fullmatch(folded or ""):
                return None
        if folded in HONORIFICS:
            honor.append(folded)
            continue
        if folded in QUESTION_WORDS:
            continue
        if folded in GENRE_WORDS:
            return None
        names.append(folded)
        display.append(raw)
    if not names or len(names) > MAX_NAME_TOKENS:
        return None
    if any(len(n) < MIN_NAME_LEN for n in names):
        return None
    return EntityQuery(
        query=text,
        names=tuple(dict.fromkeys(names)),
        display=" ".join(display),
        honorifics=tuple(honor),
    )


# --------------------------------------------------------------------- obras

_SAMHITA_TAIL = re.compile(r"\s+(?:shakala|sakala|shaunaka|saunaka|kanva|kauthuma)?\s*samhita\b.*$")


@lru_cache(maxsize=8192)
def work_key(title: str) -> str:
    """Obra de um título de chunk, sem volume/livro/página/hino.

    "The Mahabharata Volume 3 (Ganguli…)" → "mahabharata";
    "Vishnu Purana — Book I, página 50" → "vishnu purana";
    "Rigveda RV 8.13 (Griffith…)" e "Rigveda Shakala Samhita — Mandala 8" → "rigveda";
    "Chandogya Upanishad VII.1 (Müller…)" → "chandogya upanishad".
    """
    t = fold_ascii(title or "")
    t = re.sub(r"\(.*?\)", " ", t)
    t = re.split(r"\s[—–-]\s|,|:|\|", t)[0]
    t = re.sub(r"^\s*the\s+", "", t)
    t = re.sub(r"\b(?:rv|av|sv|vs|bg)\b.*$", "", t)
    t = re.sub(
        r"\b(?:volume|vol|book|canto|chapter|mandala|sukta|kanda|parva|adhyaya|section|secao|pagina|part)\b.*$",
        "",
        t,
    )
    t = re.sub(r"\s+(?:[ivxlc]+(?:\.\d+)*|\d+(?:\.\d+)*)\s*$", "", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\b(?:english|sanskrit|selected hymns|excerpts|opening|samples)\b.*$", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = _SAMHITA_TAIL.sub("", t).strip()
    t = re.sub(r"\brig veda\b", "rigveda", t)
    return t or "?"


def chunk_work(chunk: dict[str, Any]) -> str:
    return work_key(str(chunk.get("work") or chunk.get("title") or ""))


def mention_count(chunk: dict[str, Any], forms: set[str]) -> int:
    """Quantas vezes o nome (em qualquer grafia dada) aparece no corpo do chunk.

    Só o corpo conta: na Nārada Smṛti ou no Nārada Bhakti Sūtra o nome está
    no título de todo trecho, inclusive nos que tratam de dívidas ou ordálios.
    """
    if not forms:
        return 0
    from vedic_pipeline.common.sanskrit import devanagari_to_iast, has_devanagari
    from vedic_pipeline.search.hybrid import tokenize

    count = 0
    for t in tokenize(str(chunk.get("text") or "")):
        if t in forms or (has_devanagari(t) and fold_ascii(devanagari_to_iast(t)) in forms):
            count += 1
    return count


def chunk_mentions(chunk: dict[str, Any], forms: set[str]) -> bool:
    return mention_count(chunk, forms) > 0


def apply_entity_mention_boost(entity: EntityQuery, pool: list[dict[str, Any]]) -> None:
    """Trecho que não cita o nome raramente serve a uma pergunta sobre ele.

    Quem cita o nome sobe; sobe um pouco mais quando o nome se repete (o
    episódio é sobre ele, não uma lista de sábios). Quem não cita desce,
    menos na obra que leva o nome no título (Nārada Bhakti Sūtra): ali o
    trecho é dele mesmo sem repetir o nome, e conta como uma menção.
    """
    forms = entity.all_forms()
    titled = entity_titled_works(entity, pool)
    for chunk in pool:
        n = mention_count(chunk, forms)
        if not n and titled and chunk_work(chunk) in titled:
            n = 1
        if n:
            delta = ENTITY_MENTION_BOOST + ENTITY_MENTION_STEP * (min(n, ENTITY_MENTION_CAP) - 1)
        else:
            delta = -ENTITY_NO_MENTION_PENALTY
        chunk["_entity_mentions"] = n
        chunk["score"] = round(float(chunk.get("score") or 0.0) + delta, 4)


def entity_work_injections(
    entity: EntityQuery,
    all_chunks: list[dict[str, Any]],
    scores: list[float],
    *,
    exclude_ids: set[str] | None = None,
    per_work: int = ENTITY_INJECT_PER_WORK,
    max_works: int = ENTITY_INJECT_MAX_WORKS,
) -> list[dict[str, Any]]:
    """Os melhores trechos (BM25 do nome) de cada obra que cita a entidade.

    Recall por obra: sem isso, Chāndogya VII.1 (diálogo Nārada–Sanatkumāra)
    ou o canto I do Rāmāyaṇa nunca entram no pool, dominado pelo épico.
    """
    skip = set(exclude_ids or ())
    by_work: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for i, score in enumerate(scores):
        if score <= 0:
            continue
        by_work[chunk_work(all_chunks[i])].append((score, i))
    ranked_works = sorted(by_work.items(), key=lambda kv: max(s for s, _ in kv[1]), reverse=True)
    out: list[dict[str, Any]] = []
    for _work, rows in ranked_works[: max(0, max_works)]:
        rows.sort(key=lambda r: r[0], reverse=True)
        taken = 0
        for _score, i in rows:
            if taken >= per_work:
                break
            ch = all_chunks[i]
            cid = str(ch.get("chunk_id") or id(ch))
            if cid in skip:
                continue
            skip.add(cid)
            copy = dict(ch)
            copy["_entity_injected"] = entity.lexical_query()
            out.append(copy)
            taken += 1
    return out


def work_cap(top_k: int, max_per_work: int = ENTITY_MAX_PER_WORK) -> int:
    """Teto por obra: 3, ou um quarto do top-k quando ele é grande."""
    return max(1, max_per_work, top_k // 4)


def entity_titled_works(entity: EntityQuery, hits: list[dict[str, Any]]) -> frozenset[str]:
    """Obras do pool cujo título traz o nome (Nārada Smṛti, Nārada Bhakti Sūtra)."""
    names = set(entity.names)
    return frozenset(w for w in {chunk_work(h) for h in hits} if names & set(w.split()))


def diversify_by_work(
    hits: list[dict[str, Any]],
    top_k: int,
    *,
    max_per_doc: int = 2,
    max_per_work: int = ENTITY_MAX_PER_WORK,
    titled: frozenset[str] = frozenset(),
    seed_min_mentions: int = 2,
) -> list[dict[str, Any]]:
    """Top-k em ordem de score com teto por obra (e por documento).

    Os volumes e capítulos de uma obra (Mahābhārata, Bhāgavata, Harivaṃśa)
    contam como uma obra. Obra com o nome no título (``titled``: a Nārada
    Smṛti numa pergunta sobre Nārada) entra com um trecho só na primeira
    passada: ela é atribuída a ele, não conta a história dele, e suas páginas
    repetem "Nārada diz" o bastante para lotar o top-k. Se faltar candidato
    de outras obras, completa com o que sobrou (ainda em ordem de score), em
    vez de deixar o top-k curto. O top-1 é sempre o melhor score.

    Antes disso, uma rodada dá uma vaga a cada obra cujo melhor trecho cita
    o nome ao menos ``seed_min_mentions`` vezes (``_entity_mentions``, posto
    por ``apply_entity_mention_boost``); sem essa marca, ninguém é semeado.
    """
    cap = work_cap(top_k, max_per_work)
    chosen: list[dict[str, Any]] = []
    chosen_pos: set[int] = set()
    per_doc: Counter[str] = Counter()
    per_work: Counter[str] = Counter()
    # rodada 0: o melhor trecho de cada obra que fala dele de fato (nome 2+
    # vezes no trecho, ou obra que leva o nome) — o Rāmāyaṇa I (Nārada conta
    # a história a Vālmīki) não pode perder a vaga para o 4º capítulo do
    # Harivaṃśa; lista de sábios com o nome uma vez não ganha vaga aqui.
    for pos, h in enumerate(hits):
        if len(chosen) >= top_k:
            break
        work = chunk_work(h)
        if per_work[work]:
            continue
        if int(h.get("_entity_mentions") or 0) < seed_min_mentions and work not in titled:
            continue
        did = str(h.get("doc_id") or h.get("title") or h.get("chunk_id") or "")
        per_work[work] += 1
        per_doc[did] += 1
        chosen_pos.add(pos)
        chosen.append(h)
    for enforce_work in (True, False):
        for pos, h in enumerate(hits):
            if len(chosen) >= top_k:
                break
            if pos in chosen_pos:
                continue
            did = str(h.get("doc_id") or h.get("title") or h.get("chunk_id") or "")
            work = chunk_work(h)
            if per_doc[did] >= max_per_doc:
                continue
            if enforce_work and per_work[work] >= (1 if work in titled else cap):
                continue
            per_work[work] += 1
            per_doc[did] += 1
            chosen_pos.add(pos)
            chosen.append(h)
    chosen.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    return chosen


# ------------------------------------------------------------- cobertura

# Obras de referência (por regex no título). Usadas só para dizer o que o
# acervo NÃO tem; não é lista de personagens.
REFERENCE_WORKS: tuple[tuple[str, str], ...] = (
    ("Ṛgveda", r"rigveda|rig veda|rgveda"),
    ("Atharvaveda", r"atharva"),
    ("Chāndogya Upaniṣad", r"chandogya"),
    ("Bhagavad-gītā", r"bhagavad|gita"),
    ("Mahābhārata", r"mahabharata"),
    ("Rāmāyaṇa", r"ramayan"),
    ("Harivaṃśa", r"harivam"),
    ("Bhāgavata Purāṇa", r"bhagavata"),
    ("Viṣṇu Purāṇa", r"vishnu purana|visnu purana"),
    ("Nārada Purāṇa", r"narad(?:a|iya) purana"),
    ("Padma Purāṇa", r"padma purana"),
    ("Śiva Purāṇa", r"(?:shiva|siva) purana"),
    ("Brahmavaivarta Purāṇa", r"brahma ?vaivart"),
    ("Skanda Purāṇa", r"skanda purana"),
    ("Mārkaṇḍeya Purāṇa", r"markandeya"),
    ("Nārada Bhakti Sūtra", r"bhakti sutra"),
    ("Nārada Smṛti", r"narada smrti|narada smriti|naradasmrti"),
)

_TITLE_SET_CACHE: dict[int, frozenset[str]] = {}


def _corpus_works(all_chunks: list[dict[str, Any]]) -> frozenset[str]:
    key = id(all_chunks)
    cached = _TITLE_SET_CACHE.get(key)
    if cached is not None and len(_TITLE_SET_CACHE) < 16:
        return cached
    titles = frozenset(fold_ascii(str(c.get("title") or "")) for c in all_chunks)
    if len(_TITLE_SET_CACHE) >= 16:
        _TITLE_SET_CACHE.clear()
    _TITLE_SET_CACHE[key] = titles
    return titles


def missing_reference_works(all_chunks: list[dict[str, Any]]) -> list[str]:
    titles = _corpus_works(all_chunks)
    blob = "\n".join(titles)
    return [label for label, pattern in REFERENCE_WORKS if not re.search(pattern, blob)]


def titled_works(entity: EntityQuery, all_chunks: list[dict[str, Any]]) -> list[str]:
    """Obras com o nome no título ("Nārada Smṛti", "Nārada Bhakti Sūtra")."""
    names = set(entity.names)
    works = {chunk_work(c) for c in all_chunks}
    return sorted(w for w in works if names & set(w.split()))


def corpus_coverage(
    entity: EntityQuery,
    all_chunks: list[dict[str, Any]],
    scores: list[float] | None = None,
) -> list[tuple[str, int]]:
    """[(obra, nº de trechos que citam o nome)] em ordem decrescente.

    Obra com o nome no título fica de fora: ali todo trecho "cita" o nome
    pelo título, e a contagem não diria nada (ver ``titled_works``).
    """
    if scores is None:
        from vedic_pipeline.search.hybrid import lexical_scores

        scores = lexical_scores(entity.lexical_query(), all_chunks, alternates=entity.alternates())
    titled = set(titled_works(entity, all_chunks))
    counts: Counter[str] = Counter()
    for i, score in enumerate(scores):
        if score > 0:
            work = chunk_work(all_chunks[i])
            if work not in titled:
                counts[work] += 1
    return counts.most_common()


def coverage_note(entity: EntityQuery, all_chunks: list[dict[str, Any]], *, limit: int = 12) -> str:
    """Nota curta para o prompt: onde o nome aparece e o que falta no acervo."""
    cov = corpus_coverage(entity, all_chunks)
    titled = titled_works(entity, all_chunks)
    missing = missing_reference_works(all_chunks)
    parts: list[str] = []
    if titled:
        parts.append("Obras do acervo com o nome no título: " + "; ".join(w.title() for w in titled) + ".")
    if cov:
        listed = "; ".join(f"{work.title()} ({n})" for work, n in cov[:limit])
        parts.append(f"Trechos do acervo que citam “{entity.display}”, por obra: {listed}.")
    elif not titled:
        parts.append(f"Nenhum trecho do acervo cita “{entity.display}” pelo nome.")
    if missing:
        parts.append("Obras de referência ausentes do acervo: " + ", ".join(missing) + ".")
    return " ".join(parts)
