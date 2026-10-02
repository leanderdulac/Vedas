"""Server-owned choices for public retrieval and generation endpoints."""
import os
import re
import secrets
from pathlib import Path

from fastapi import HTTPException

from vedic_pipeline.common.constants import DEFAULT_EMBED_DIR, PROJECT_ROOT
from vedic_pipeline.common.env import env_flag
from vedic_pipeline.llm.generate import DEFAULT_XAI_MODEL

GENERATION_TOKEN_UNSET = (
    "Geração paga exige VEDIC_GENERATION_API_TOKEN nesta instância "
    "(VEDIC_REQUIRE_GENERATION_TOKEN=true ou VEDIC_PUBLIC_API=1)"
)

_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-/]{0,127}(?::[A-Za-z0-9._\-]{1,64})?$")


def validate_model_name(value: str, *, field: str = "model") -> str:
    """Allowlist sintática para nomes de modelo HF/xAI (anti-SSRF/path traversal)."""
    v = (value or "").strip()
    if not v or len(v) > 160:
        raise ValueError(f"{field}: nome de modelo inválido")
    if ".." in v or v.startswith(("/", ".", "-")) or "://" in v:
        raise ValueError(f"{field}: nome de modelo inválido")
    if any(c in v for c in (" ", ";", "&", "|", "$", "`", '"', "'", "\\", "<", ">", "\n", "\r", "\x00")):
        raise ValueError(f"{field}: nome de modelo inválido")
    if not _MODEL_RE.match(v):
        raise ValueError(f"{field}: nome de modelo inválido")
    return v


def validate_project_path(value: str, *, field: str = "path") -> str:
    """Confina caminhos de pipeline a diretórios permitidos (anti-LFI/traversal).

    Permite: PROJECT_ROOT, diretório temporário do SO e extras em
    VEDIC_ALLOWED_DATA_DIRS (separado por ':'). Bloqueia /etc, /root, home, etc.
    """
    import tempfile

    if not value or not value.strip():
        raise ValueError(f"{field}: caminho vazio")
    v = value.strip()
    if "\x00" in v:
        raise ValueError(f"{field}: caminho inválido")
    # Bloqueia URLs/esquemas (file://, http://) — apenas paths locais.
    if "://" in v:
        raise ValueError(f"{field}: apenas caminhos locais dentro do projeto")
    base = PROJECT_ROOT.resolve()
    tmp = Path(tempfile.gettempdir()).resolve()
    candidate = (base / v).resolve() if not Path(v).is_absolute() else Path(v).resolve()
    allowed: list[Path] = [base, tmp]
    extra = os.environ.get("VEDIC_ALLOWED_DATA_DIRS", "")
    for part in extra.split(":"):
        part = part.strip()
        if part:
            try:
                allowed.append(Path(part).resolve())
            except OSError:
                continue
    # data/ e artifacts/ do projeto são os casos comuns — já cobertos por base.
    for root in allowed:
        try:
            candidate.relative_to(root)
            return str(candidate)
        except ValueError:
            continue
    raise ValueError(f"{field}: fora dos diretórios permitidos")


def api_index_dir() -> str:
    return str(Path(os.environ.get('VEDIC_API_INDEX_DIR') or DEFAULT_EMBED_DIR).resolve())


def validate_index_dir(value: str) -> str:
    resolved = str(Path(value).resolve())
    if resolved != api_index_dir():
        raise ValueError('A API aceita somente o índice configurado no servidor')
    return resolved


def public_api() -> bool:
    """Instância publicada (`VEDIC_PUBLIC_API=1`, ligada em docker-compose.prod.yml).

    O default é privado: token vazio deixa a chave do servidor autorizar,
    para a UI local funcionar.
    """
    return env_flag('VEDIC_PUBLIC_API')


def generation_token_required() -> bool:
    """Fail-closed se `VEDIC_REQUIRE_GENERATION_TOKEN` ou `VEDIC_PUBLIC_API` estiver ligado.

    Prod/compose.prod liga as duas. Dev local continua aberto se o token
    estiver vazio e nenhuma das flags estiver ligada.
    """
    return env_flag("VEDIC_REQUIRE_GENERATION_TOKEN") or public_api()


def _generation_token() -> str:
    return (os.environ.get('VEDIC_GENERATION_API_TOKEN') or '').strip()


def _require_generation_token(authorization: str | None) -> None:
    """503 se o token é obrigatório e não foi configurado; 401 se não confere."""
    expected = _generation_token()
    if not expected:
        if generation_token_required():
            raise HTTPException(503, GENERATION_TOKEN_UNSET)
        # Instância local/privada: a chave do servidor já autoriza a geração.
        return
    supplied = authorization.removeprefix('Bearer ') if authorization and authorization.startswith('Bearer ') else ''
    if not secrets.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(401, 'Token de geração inválido', headers={'WWW-Authenticate': 'Bearer'})


def public_generation_policy() -> dict[str, bool]:
    xai = bool((os.environ.get("XAI_API_KEY") or "").strip())
    token = bool(_generation_token())
    require = generation_token_required()
    return {
        "require_token": require,
        "public_api": public_api(),
        "token_configured": token,
        "xai_configured": xai,
        "open_paid": bool(xai and not token and not require),
    }


def authorize_generation(provider: str, model: str | None, authorization: str | None) -> tuple[str, str | None]:
    # Resolve auto before checking authorization so a configured key cannot
    # silently turn an anonymous request into paid generation.
    selected = ('xai' if os.environ.get('XAI_API_KEY') else 'extractive') if provider == 'auto' else provider
    if selected == 'extractive':
        if model is not None:
            raise HTTPException(422, 'Modo extrativo não aceita seleção de modelo')
        return selected, None

    _require_generation_token(authorization)
    configured = (os.environ.get('XAI_MODEL') or DEFAULT_XAI_MODEL) if selected == 'xai' else (os.environ.get('VEDIC_LOCAL_LM') or 'gpt2')
    if model is not None and model != configured:
        raise HTTPException(422, 'Modelo não autorizado pelo servidor')
    return selected, configured


def authorize_media(authorization: str | None) -> None:
    """Gate de mídia (TTS, Imagine ou Stable Diffusion local).

    Instância privada, token vazio: a chave do servidor autoriza.
    Instância pública (`VEDIC_PUBLIC_API=1`): sem token configurado responde
    503; com token, a requisição precisa enviá-lo. Leitura de arquivo em
    cache continua fora desta função.
    """
    _require_generation_token(authorization)
