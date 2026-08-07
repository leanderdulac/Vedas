"""Camada RAG: recuperação + montagem de contexto citável + inferência."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from vedic_pipeline.common.constants import DEFAULT_EMBED_DIR
from vedic_pipeline.search.embeddings import load_embedding_index, search_index

# cache simples em processo
_INDEX_CACHE: dict[str, Any] = {}

VEDIC_SYSTEM_PROMPT = """Você é Veda Knowledge — um preceptor digital de estudos védicos e vaishnavas.

Missão:
- Responder com clareza à intenção da pergunta (não só repetir trechos).
- Inferir, comparar e explicar a partir do CONTEXTO recuperado do corpus licenciado.
- Citar sempre as fontes pelo número [n].
- Quando o contexto permitir, sintetize doutrina (dharma, ātman, brahman, karma, bhakti, yoga, etc.).
- Se o contexto for parcial, diga o que se pode afirmar e o que falta no corpus.
- NÃO invente versos, números de mantra ou citações que não estejam no contexto.
- NÃO use material com copyright que não esteja no contexto (ex.: edições BBT não autorizadas).
- Pode responder em português se a pergunta estiver em português; preserve termos sânscritos quando úteis (com IAST ou Devanāgarī se aparecerem no contexto).
- Estruture respostas densas: (1) resposta direta, (2) fundamentação com citações, (3) nuances/limites do corpus.
"""


def get_index(index_dir: Path = DEFAULT_EMBED_DIR, reload: bool = False) -> dict[str, Any]:
    key = str(Path(index_dir).resolve())
    if reload or key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = load_embedding_index(index_dir)
    return _INDEX_CACHE[key]


def retrieve(
    query: str,
    index_dir: Path = DEFAULT_EMBED_DIR,
    top_k: int = 5,
    tradition: Optional[str] = None,
    language: Optional[str] = None,
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
        header = (
            f"[{i}] {h.get('title') or 'sem título'} "
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
    system_preamble: Optional[str] = None,
) -> dict[str, str]:
    """Prompt orientado a resposta + inferência fundamentada."""
    preamble = system_preamble or VEDIC_SYSTEM_PROMPT
    context = format_rag_context(hits)
    if not hits:
        user = (
            f"Pergunta: {query}\n\n"
            "Nenhum trecho foi recuperado do corpus. Informe que o índice/corpus "
            "está vazio ou insuficiente e oriente a expandir fontes licenciadas."
        )
    else:
        user = (
            f"Contexto recuperado do corpus licenciado:\n{context}\n\n"
            f"Pergunta do estudante: {query}\n\n"
            "Instruções de resposta:\n"
            "- Interprete a intenção (definição, comparação, aplicação prática, narrativa, etc.).\n"
            "- Inferir é permitido quando logicamente sustentado pelos trechos; marque inferências como tal.\n"
            "- Cite [n] ao usar cada fonte.\n"
            "- Se houver tensão entre fontes (ex.: caminhos de jñāna vs bhakti), exponha a nuance.\n"
            "- Resposta completa e útil:"
        )
    return {"system": preamble, "user": user, "context": context}
