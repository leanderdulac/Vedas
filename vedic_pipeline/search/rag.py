"""Camada RAG: recuperação + montagem de contexto citável + inferência."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vedic_pipeline.common.constants import DEFAULT_EMBED_DIR
from vedic_pipeline.common.style import STYLE_GUIDE_PT
from vedic_pipeline.search.embeddings import load_embedding_index, search_index

# cache em processo sensível ao mtime do índice
_INDEX_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def invalidate_index_cache(index_dir: Path | None = None) -> None:
    """Invalida o cache de índice em memória."""
    if index_dir is None:
        _INDEX_CACHE.clear()
    else:
        _INDEX_CACHE.pop(str(Path(index_dir).resolve()), None)


VEDIC_SYSTEM_PROMPT = (
    """Você é Veda Knowledge, professor de estudos védicos e vaiṣṇavas. Responde a perguntas
de estudantes com base nos textos do acervo licenciado que acompanham cada pergunta.

Conteúdo:
- Responda à intenção da pergunta já na primeira frase; depois desenvolva com as fontes.
- Explique, compare e tire conclusões a partir dos textos, citando cada fonte que usar.
- Não invente versos, números de mantra ou citações. Se a fonte só indica o capítulo (BG 2), cite o capítulo e nunca fabrique o śloka.
- Não use material com direitos autorais que não esteja nos textos fornecidos (por exemplo, edições BBT não autorizadas).
- Ignore textos fornecidos que não tenham relação com a pergunta, sem comentar sobre eles.
- Responda no idioma da pergunta. Se ela vier em inglês, siga as mesmas regras de redação em inglês.
- Extensão: de 3 a 6 parágrafos curtos (em geral 200 a 450 palavras). Perguntas simples pedem respostas curtas.

"""
    + STYLE_GUIDE_PT
)


def get_index(index_dir: Path = DEFAULT_EMBED_DIR, reload: bool = False) -> dict[str, Any]:
    dir_path = Path(index_dir)
    key = str(dir_path.resolve())
    emb_file = dir_path / "embeddings.npy"
    current_mtime = emb_file.stat().st_mtime if emb_file.exists() else 0.0

    cached = _INDEX_CACHE.get(key)
    if not reload and cached is not None:
        cached_mtime, index = cached
        if cached_mtime == current_mtime:
            return index

    index = load_embedding_index(dir_path)
    _INDEX_CACHE[key] = (current_mtime, index)
    return index


def retrieve(
    query: str,
    index_dir: Path = DEFAULT_EMBED_DIR,
    top_k: int = 5,
    tradition: str | None = None,
    language: str | None = None,
    reload: bool = False,
) -> list[dict[str, Any]]:
    index = get_index(index_dir, reload=reload)
    return search_index(
        query,
        index,
        top_k=top_k,
        tradition=tradition,
        language=language,
    )


def format_rag_context(
    hits: list[dict[str, Any]],
    max_chars: int = 9000,
) -> str:
    """Monta contexto com citações para prompt de LLM."""
    parts: list[str] = []
    used = 0
    for i, h in enumerate(hits, 1):
        locator = (h.get("locator") or "").strip()
        title = h.get("title") or "sem título"
        label = f"{locator} — {title}" if locator else title
        header = (
            f"[{i}] {label} "
            f"(tradição={h.get('tradition')}, idioma={h.get('language')}, "
            f"licença={h.get('license')}, score={h.get('score')})"
        )
        body = (h.get("text") or "").strip()
        block = f"{header}\n{body}\nFonte: {h.get('source_url') or 'n/a'}"
        if used + len(block) > max_chars and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n\n---\n\n".join(parts)


ENTITY_ANSWER_GUIDE = (
    "Esta pergunta nomeia um personagem ou tema que aparece em várias obras. {note}\n"
    "Comece dizendo quem ele é. Depois percorra as tradições presentes nos textos acima, "
    "um parágrafo para cada uma que tiver fonte (Veda e Upaniṣad; Mahābhārata e Bhagavad-gītā; "
    "Rāmāyaṇa; Purāṇas), com os episódios e o papel dele em cada obra. Nesta pergunta a extensão "
    "é maior: de 5 a 9 parágrafos (cerca de 450 a 800 palavras). Feche com uma frase curta dizendo "
    "quais obras importantes para o tema não estão no acervo, escolhendo entre as ausentes listadas "
    "(só as que a tradição de fato associa ao tema) e sem resumir o que elas dizem."
)


def build_rag_prompt(
    query: str,
    hits: list[dict[str, Any]],
    system_preamble: str | None = None,
    *,
    entity_note: str | None = None,
    max_context_chars: int | None = None,
) -> dict[str, str]:
    """Prompt orientado a resposta + inferência fundamentada.

    ``entity_note`` (consulta de entidade) traz a cobertura do nome no acervo
    e as obras ausentes; a resposta passa por cada tradição e diz o que falta.
    """
    preamble = system_preamble or VEDIC_SYSTEM_PROMPT
    if max_context_chars:
        context = format_rag_context(hits, max_chars=max_context_chars)
    else:
        context = format_rag_context(hits)
    if not hits:
        user = (
            f"Pergunta: {query}\n\n"
            "Nenhum texto do acervo foi encontrado para esta pergunta. Diga isso em uma ou duas "
            "frases simples e sugira reformular a pergunta ou ampliar o acervo licenciado."
        )
    else:
        user = (
            f"Textos do acervo licenciado:\n{context}\n\n"
            f"Pergunta do estudante: {query}\n\n"
            "Responda à pergunta em prosa corrida, seguindo as regras de redação do sistema. "
            "Cite [n] e o localizador canônico quando houver (RV 10.129.1, BG 2.47); não invente "
            "maṇḍala, hino ou número de śloka. Traduza para o português as passagens em inglês que "
            "citar. Se as fontes divergirem (por exemplo, jñāna e bhakti), mostre a diferença."
        )
        if entity_note:
            user = f"{user}\n\n{ENTITY_ANSWER_GUIDE.format(note=entity_note.strip())}"
    return {"system": preamble, "user": user, "context": context}
