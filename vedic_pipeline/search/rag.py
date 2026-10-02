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


def build_rag_prompt(
    query: str,
    hits: list[dict[str, Any]],
    system_preamble: str | None = None,
) -> dict[str, str]:
    """Prompt orientado a resposta + inferência fundamentada."""
    preamble = system_preamble or VEDIC_SYSTEM_PROMPT
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
    return {"system": preamble, "user": user, "context": context}
