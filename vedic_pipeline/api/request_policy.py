"""Server-owned choices for public retrieval and generation endpoints."""
import os
import secrets
from pathlib import Path

from fastapi import HTTPException

from vedic_pipeline.common.constants import DEFAULT_EMBED_DIR
from vedic_pipeline.llm.generate import DEFAULT_XAI_MODEL


def api_index_dir() -> str:
    return str(Path(os.environ.get('VEDIC_API_INDEX_DIR') or DEFAULT_EMBED_DIR).resolve())


def validate_index_dir(value: str) -> str:
    resolved = str(Path(value).resolve())
    if resolved != api_index_dir():
        raise ValueError('A API aceita somente o índice configurado no servidor')
    return resolved


def authorize_generation(provider: str, model: str | None, authorization: str | None) -> tuple[str, str | None]:
    # Resolve auto before checking authorization so a configured key cannot
    # silently turn an anonymous request into paid generation.
    selected = ('xai' if os.environ.get('XAI_API_KEY') else 'extractive') if provider == 'auto' else provider
    if selected == 'extractive':
        if model is not None:
            raise HTTPException(422, 'Modo extrativo não aceita seleção de modelo')
        return selected, None

    expected = os.environ.get('VEDIC_GENERATION_API_TOKEN', '')
    if not expected:
        # Instância local/privada: a chave xAI no servidor já autoriza a geração.
        # Em API pública, defina VEDIC_GENERATION_API_TOKEN.
        configured = (os.environ.get('XAI_MODEL') or DEFAULT_XAI_MODEL) if selected == 'xai' else (os.environ.get('VEDIC_LOCAL_LM') or 'gpt2')
        if model is not None and model != configured:
            raise HTTPException(422, 'Modelo não autorizado pelo servidor')
        return selected, configured
    supplied = authorization.removeprefix('Bearer ') if authorization and authorization.startswith('Bearer ') else ''
    if not secrets.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(401, 'Token de geração inválido', headers={'WWW-Authenticate': 'Bearer'})
    configured = (os.environ.get('XAI_MODEL') or DEFAULT_XAI_MODEL) if selected == 'xai' else (os.environ.get('VEDIC_LOCAL_LM') or 'gpt2')
    if model is not None and model != configured:
        raise HTTPException(422, 'Modelo não autorizado pelo servidor')
    return selected, configured


def authorize_media(authorization: str | None) -> None:
    """Gate de mídia paga (TTS/Imagine: /audio, /image, /video).

    Espelha a política de geração do /ask: em instância local/privada (sem
    VEDIC_GENERATION_API_TOKEN) a chave xAI no servidor autoriza; em API pública,
    o token obrigatório impede que anônimos gastem créditos de voz/imagem/vídeo
    — comportamento que antes ficava aberto em /audio, /image e /video.
    """
    expected = os.environ.get('VEDIC_GENERATION_API_TOKEN', '')
    if not expected:
        return
    supplied = authorization.removeprefix('Bearer ') if authorization and authorization.startswith('Bearer ') else ''
    if not secrets.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(401, 'Token de geração inválido', headers={'WWW-Authenticate': 'Bearer'})
