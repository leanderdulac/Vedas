"""Geração de respostas: SpaceXAI (xAI) prioritário + fallbacks."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger("vedic_pipeline.llm")

DEFAULT_XAI_MODEL = "grok-4.5"
DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"


def list_providers() -> dict[str, Any]:
    xai_key = bool(os.environ.get("XAI_API_KEY"))
    return {
        "xai": {
            "available": xai_key,
            "model": os.environ.get("XAI_MODEL", DEFAULT_XAI_MODEL),
            "base_url": os.environ.get("XAI_BASE_URL", DEFAULT_XAI_BASE_URL),
            "env": "XAI_API_KEY",
        },
        "local": {
            "available": True,
            "model": os.environ.get("VEDIC_LOCAL_LM", "gpt2"),
            "note": "Transformers causal LM — lento/CPU, só para smoke tests",
        },
        "extractive": {
            "available": True,
            "note": "Sem LLM: monta resposta a partir dos trechos recuperados",
        },
    }


def _generate_xai(
    system: str,
    user: str,
    model: Optional[str] = None,
    max_tokens: int = 1800,
) -> dict[str, Any]:
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("XAI_API_KEY não definida")

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("XAI_BASE_URL", DEFAULT_XAI_BASE_URL),
    )
    model_name = model or os.environ.get("XAI_MODEL", DEFAULT_XAI_MODEL)

    # Responses API (preferida na docs xAI)
    try:
        resp = client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            store=False,
            max_output_tokens=max_tokens,
        )
        text = getattr(resp, "output_text", None)
        if not text:
            # fallback parse
            parts = []
            for item in getattr(resp, "output", []) or []:
                for c in getattr(item, "content", []) or []:
                    t = getattr(c, "text", None)
                    if t:
                        parts.append(t)
            text = "\n".join(parts)
        return {
            "provider": "xai",
            "model": model_name,
            "answer": (text or "").strip(),
        }
    except Exception as exc:
        logger.warning("Responses API falhou (%s); tentando chat.completions", exc)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.35,
        )
        text = resp.choices[0].message.content or ""
        return {
            "provider": "xai",
            "model": model_name,
            "answer": text.strip(),
        }


def _generate_local(
    system: str,
    user: str,
    model: Optional[str] = None,
    max_new_tokens: int = 128,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = model or os.environ.get("VEDIC_LOCAL_LM", "gpt2")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(model_name)
    lm.eval()

    prompt = f"{system}\n\n{user}\n"
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        out = lm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    full = tok.decode(out[0], skip_special_tokens=True)
    # tenta devolver só a continuação
    answer = full[len(prompt) :].strip() if full.startswith(prompt[:200]) else full
    if not answer:
        answer = full.strip()
    return {
        "provider": "local",
        "model": model_name,
        "answer": answer,
    }


def _generate_extractive(user_context_block: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
    if not hits:
        return {
            "provider": "extractive",
            "model": None,
            "answer": (
                "Não há trechos recuperados no índice para fundamentar uma resposta. "
                "Execute build-index e tente novamente."
            ),
        }
    lines = [
        "Resposta extrativa (sem LLM externo) com base nos trechos recuperados:",
        "",
    ]
    for i, h in enumerate(hits[:3], 1):
        title = h.get("title") or "fonte"
        snippet = (h.get("text") or "").strip().replace("\n", " ")
        if len(snippet) > 400:
            snippet = snippet[:400] + "…"
        lines.append(f"[{i}] ({title}, score={h.get('score')}) {snippet}")
    lines.append("")
    lines.append(
        "Para resposta discursiva com modelo generativo, defina XAI_API_KEY "
        "(SpaceXAI/xAI) ou use --provider local."
    )
    return {
        "provider": "extractive",
        "model": None,
        "answer": "\n".join(lines),
    }


def generate_answer(
    system: str,
    user: str,
    *,
    provider: str = "auto",
    model: Optional[str] = None,
    hits: Optional[list[dict[str, Any]]] = None,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """
    provider:
      - auto: xai se XAI_API_KEY, senão extractive
      - xai | local | extractive
    """
    hits = hits or []
    provider = (provider or "auto").lower()

    if provider == "auto":
        provider = "xai" if os.environ.get("XAI_API_KEY") else "extractive"

    if provider == "xai":
        return _generate_xai(system, user, model=model, max_tokens=max_tokens)
    if provider == "local":
        return _generate_local(system, user, model=model)
    if provider == "extractive":
        return _generate_extractive(user, hits)

    raise ValueError(f"Provider desconhecido: {provider}")


def _stream_xai(
    system: str,
    user: str,
    model: Optional[str] = None,
    max_tokens: int = 1800,
):
    """Yields dicts: {type: meta|token}."""
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("XAI_API_KEY não definida")

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("XAI_BASE_URL", DEFAULT_XAI_BASE_URL),
    )
    model_name = model or os.environ.get("XAI_MODEL", DEFAULT_XAI_MODEL)
    yield {"type": "meta", "provider": "xai", "model": model_name}

    # chat.completions stream é o path mais portável
    stream = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0.35,
        stream=True,
    )
    for event in stream:
        try:
            delta = event.choices[0].delta
            text = getattr(delta, "content", None) or ""
        except (AttributeError, IndexError, KeyError):
            text = ""
        if text:
            yield {"type": "token", "text": text}


def stream_answer(
    system: str,
    user: str,
    *,
    provider: str = "auto",
    model: Optional[str] = None,
    hits: Optional[list[dict[str, Any]]] = None,
    max_tokens: int = 1024,
):
    """
    Generator de chunks de texto (streaming).
    Yields: {"type": "meta"|"token", ...}
    Fallback: gera resposta completa e emite tokens em pedaços.
    """
    hits = hits or []
    provider = (provider or "auto").lower()
    if provider == "auto":
        provider = "xai" if os.environ.get("XAI_API_KEY") else "extractive"

    if provider == "xai":
        try:
            yield from _stream_xai(system, user, model=model, max_tokens=max_tokens)
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("stream xAI falhou (%s); fallback one-shot", exc)

    # extractive / local / fallback: resposta completa fatiada
    result = generate_answer(
        system,
        user,
        provider=provider,
        model=model,
        hits=hits,
        max_tokens=max_tokens,
    )
    yield {
        "type": "meta",
        "provider": result.get("provider"),
        "model": result.get("model"),
    }
    text = result.get("answer") or ""
    # fatias ~48 chars para simular stream na UI
    step = 48
    for i in range(0, len(text), step):
        yield {"type": "token", "text": text[i : i + step]}
