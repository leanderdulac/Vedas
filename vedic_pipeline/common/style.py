"""Estilo editorial dos textos gerados e pós-processamento determinístico.

Três peças:

- ``STYLE_GUIDE_PT`` / ``STYLE_GUIDE_EN``: guia de redação anexado aos prompts
  de sistema (pergunta, explicação e tradução). Pede prosa corrida de
  professor, sem markdown decorativo nem os clichês típicos de chatbot.
- ``clean_prose``: limpeza determinística aplicada ao texto final (e ao
  cache já gravado). Tira negrito, títulos, réguas, marcadores de lista,
  rótulos de seção ("Resposta direta:"), aberturas e fechos de chatbot
  ("Claro!", "Em resumo,"), emoji e travessões; normaliza espaços.
- ``speech_text``: versão para voz (TTS/voz do navegador), sem nada que a
  síntese leria literalmente (asteriscos, colchetes de citação, cerquilhas).

O passo opcional de revisão por LLM (``edit_prose``) vale só para textos
longos não transmitidos em streaming e fica atrás de ``VEDIC_TEXT_EDITOR``.
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger("vedic_pipeline.common.style")

STYLE_GUIDE_PT = """Redação (obrigatório):
- Escreva em português do Brasil culto e natural, como um professor experiente que explica a um aluno atento. Nada de tom publicitário, entusiasmo artificial ou frases de efeito.
- Prosa corrida em parágrafos curtos (2 a 4 frases cada). Não use títulos, negrito, itálico, listas com marcadores ou numeradas, tabelas, réguas (---), emoji nem travessão (—); use vírgula, dois-pontos, ponto ou parênteses.
- Comece direto pelo conteúdo. Não repita a pergunta, não use rótulos como "Resposta direta:", "Fundamentação:", "Nuances:" ou "Conclusão:".
- Não feche com resumo nem com oferta de ajuda. Termine na última ideia relevante.
- Evite estas expressões: "Claro!", "Ótima pergunta", "Em resumo", "Em suma", "Em síntese", "É importante notar/ressaltar/destacar", "Vale ressaltar/destacar", "Cabe ressaltar", "mergulhar", "tapeçaria", "rica tapeçaria", "jornada", "fascinante", "profundo e multifacetado", "não apenas X, mas também Y", "desvendar", "no cerne de", "em última análise", "como mencionado". Evite também tríades retóricas (três adjetivos ou três frases paralelas seguidas).
- Sem anglicismos nem espanholismos: escreva "Si" ou "eu interior" (não "Self"), "hino" (não "himno"), "solene" (não "solemne"). Quando o trecho-fonte estiver em inglês, traduza a citação para o português entre aspas curvas “ ” e não deixe palavras inglesas soltas no texto (títulos de obras podem ficar como estão).
- Termos sânscritos em IAST com diacríticos, sempre da mesma forma (ātman, brahman, yajña, dharma, Ṛgveda, Upaniṣad, Bhagavad-gītā), em letra normal, sem itálico nem negrito. Nomes de deuses com inicial maiúscula (Agni, Indra, Varuṇa). Use a forma portuguesa no plural quando natural (as Upaniṣads).
- Fontes: indique o número entre colchetes logo após a afirmação, [3], e o localizador canônico quando existir no contexto, no formato RV 1.1.1, BG 2.47, AV 1.1.1.
- Inferências entram com naturalidade ("disso se deduz que", "é razoável entender que"), sem anunciar "inferência" entre parênteses.
- Limites das fontes: se faltar algo importante, diga numa frase simples ao final ("Os textos disponíveis aqui não tratam de..."). Não fale de "corpus recuperado", "trechos", "score", "contexto fornecido" nem de trechos corrompidos.
"""

STYLE_GUIDE_EN = """Writing (mandatory):
- Plain, natural, educated English, like an experienced teacher. No marketing tone or forced enthusiasm.
- Continuous prose in short paragraphs (2 to 4 sentences). No headings, bold, italics, bullet or numbered lists, tables, rules (---), emoji or em dashes; use commas, colons, full stops or parentheses.
- Start with the content itself. Do not restate the question; no labels like "Direct answer:", "Grounding:", "Nuances:" or "Conclusion:".
- No summary closing and no offer of further help.
- Avoid: "Certainly!", "Great question", "In summary", "In conclusion", "It is important to note", "delve", "tapestry", "journey", "fascinating", "not only X but also Y", "at its core", "ultimately". Avoid rhetorical triads.
- Sanskrit terms in IAST with diacritics, always spelled the same way (ātman, brahman, yajña, Ṛgveda, Upaniṣad), in roman type. Capitalize deity names (Agni, Indra).
- Sources: the bracketed number right after the claim, [3], plus the canonical locator when present in the context (RV 1.1.1, BG 2.47).
- If something important is missing from the sources, say so in one plain sentence at the end. Do not talk about "retrieved corpus", "chunks" or "scores".
"""


def style_guide(lang: str = "pt") -> str:
    return STYLE_GUIDE_EN if (lang or "pt").lower().startswith("en") else STYLE_GUIDE_PT


# ---------------------------------------------------------------- limpeza

# rótulos de seção que o modelo usa como título ou abre parágrafo com eles
_SECTION_LABELS = (
    r"resposta(?: direta| curta| breve)?|fundamenta[çc][ãa]o(?: com (?:o corpus|as fontes|cita[çc][õo]es))?"
    r"|nuances?(?: e limites(?: do corpus)?(?: recuperado)?)?|limites(?: do corpus)?(?: recuperado)?"
    r"|conclus[ãa]o|s[íi]ntese|resumo|contexto|an[áa]lise|observa[çc][õo]es?|notas?"
    r"|sentido do verso|termos?[- ]chave|infer[êe]ncia[^:\n]{0,50}|direct answer|answer|grounding|nuances? and limits"
    r"|corpus limits|limits|conclusion|summary|sense of the verse|key terms"
)
_LABEL_LINE_RE = re.compile(
    rf"^\s*(?:\(?\d+\)?[.)]?\s*)?(?:{_SECTION_LABELS})\s*:?\s*$", re.IGNORECASE
)
_LABEL_PREFIX_RE = re.compile(
    rf"^(\s*)(?:\(?\d+\)?[.)]?\s*)?(?:{_SECTION_LABELS})\s*:\s+(\S)", re.IGNORECASE
)

# aberturas/fechos de chatbot: removidos e a frase seguinte recapitalizada
_OPENERS_PT = (
    r"claro[!,.]\s*|com certeza[!,.]\s*|ótima pergunta[!.]?\s*|excelente pergunta[!.]?\s*"
    r"|em resumo,\s*|em suma,\s*|em síntese,\s*|resumindo,\s*|em última análise,\s*"
    r"|é importante (?:notar|ressaltar|destacar|lembrar|observar) que\s+"
    r"|vale (?:a pena )?(?:ressaltar|destacar|notar|lembrar|mencionar) que\s+"
    r"|cabe (?:ressaltar|destacar|notar|lembrar) que\s+"
    r"|convém (?:ressaltar|destacar|notar|lembrar) que\s+"
    r"|como mencionado(?: anteriormente)?,\s*"
)
_OPENERS_EN = (
    r"certainly[!,.]\s*|of course[!,.]\s*|great question[!.]?\s*|in summary,\s*|in conclusion,\s*"
    r"|to summarize,\s*|ultimately,\s*|it is (?:important|worth) (?:to note|noting) that\s+"
)
_OPENER_RE = re.compile(
    rf"(^|(?<=[.!?…])\s+|(?<=\n))(?:{_OPENERS_PT}|{_OPENERS_EN})(\S)", re.IGNORECASE
)

# fechos que oferecem ajuda (frase inteira)
_OFFER_RE = re.compile(
    r"(?:^|\s)(?:Se (?:quiser|desejar|precisar)[^.!?\n]*(?:posso|é só)[^.!?\n]*[.!?]"
    r"|Espero (?:ter ajudado|que isso ajude)[^.!?\n]*[.!?]"
    r"|(?:Fico|Estou) à disposição[^.!?\n]*[.!?]"
    r"|Let me know if[^.!?\n]*[.!?]|I hope this helps[^.!?\n]*[.!?])\s*$",
    re.IGNORECASE,
)

_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U0001F000-\U0001F2FF\u2600-\u26FF\uFE0F\u200d]"
)
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*(.*?)\s*#*\s*$")
_RULE_RE = re.compile(r"^\s*(?:[-*_]\s*){3,}$")
_BULLET_RE = re.compile(r"^(\s*)(?:[-*+•▪◦]|\d{1,2}[.)])\s+(?=\S)")
_BOLD_RE = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1", re.DOTALL)
# itálico com * ou _ (não pega asterisco isolado nem snake_case)
_ITALIC_STAR_RE = re.compile(r"(?<![*\w])\*(?=[^\s*])([^*\n]+?)(?<=[^\s*])\*(?![*\w])")
_ITALIC_UNDER_RE = re.compile(r"(?<![_\w])_(?=[^\s_])([^_\n]+?)(?<=[^\s_])_(?![_\w])")
_CODE_RE = re.compile(r"`([^`\n]+)`")
_BLOCKQUOTE_RE = re.compile(r"^\s*>\s?")
# travessão (— ou – com espaços) entre palavras; preserva intervalos "1.1.1–2"
_DASH_RE = re.compile(r"\s*(?:—|\s–\s|\s--\s)\s*")

# deslizes de idioma recorrentes do modelo
_LEXICAL_FIXES_PT = {
    r"\bhimnos\b": "hinos",
    r"\bhimno\b": "hino",
    r"\bsolemnes\b": "solenes",
    r"\bsolemne\b": "solene",
    r"\bṛgvedico\b": "ṛgvédico",
    r"\bSi-mesmo \(Self\)": "Si",
    r"\b(o|do|no|ao|pelo|um) Self\b": r"\1 Si",
}


def _strip_inline_markdown(line: str) -> str:
    line = _BOLD_RE.sub(r"\2", line)
    line = _ITALIC_STAR_RE.sub(r"\1", line)
    line = _ITALIC_UNDER_RE.sub(r"\1", line)
    line = _CODE_RE.sub(r"\1", line)
    return line.replace("**", "")


def _fix_dashes(line: str) -> str:
    stripped = line.lstrip()
    if stripped.startswith(("—", "–")):  # travessão de abertura (fala, verbete)
        line = stripped.lstrip("—– ").strip()

    n_dashes = len(_DASH_RE.findall(line))

    def repl(m: re.Match[str]) -> str:
        rest = line[m.end() :]
        # "termo — glosa" no começo da linha (glossário): dois-pontos
        before = line[: m.start()]
        if not rest or rest[0] in ".,;:!?)":
            return ""
        if (
            n_dashes == 1
            and before
            and len(before.split()) <= 3
            and not re.search(r"[.,;:!?]$", before)
        ):
            return ": "
        return ", "

    out = _DASH_RE.sub(repl, line)
    return re.sub(r",\s*,", ",", out)


def clean_prose(text: str | None, *, lang: str = "pt") -> str:
    """Limpeza determinística dos marcadores típicos de texto de chatbot.

    Idempotente: aplicar duas vezes dá o mesmo resultado. Não toca em
    citações [n], localizadores (RV 1.1.1–2), devanāgarī nem diacríticos.
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _EMOJI_RE.sub("", text)

    out_lines: list[str] = []
    for raw in text.split("\n"):
        line = raw.rstrip()
        if _RULE_RE.match(line):
            out_lines.append("")
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            line = heading.group(1)
            if _LABEL_LINE_RE.match(_strip_inline_markdown(line)):
                out_lines.append("")
                continue
            # título restante vira linha própria, sem cerquilha
        line = _BLOCKQUOTE_RE.sub("", line)
        line = _BULLET_RE.sub(r"\1", line).strip()
        line = _strip_inline_markdown(line)
        if _LABEL_LINE_RE.match(line):
            out_lines.append("")
            continue
        line = _LABEL_PREFIX_RE.sub(lambda m: m.group(1) + m.group(2).upper(), line)
        line = _fix_dashes(line)
        out_lines.append(line)

    text = "\n".join(out_lines)
    text = _OPENER_RE.sub(lambda m: m.group(1) + m.group(2).upper(), text)
    text = _OFFER_RE.sub("", text.rstrip())
    if not (lang or "pt").lower().startswith("en"):
        for pat, rep in _LEXICAL_FIXES_PT.items():
            text = re.sub(pat, rep, text)

    # citações: "[5][8]" -> "[5, 8]" e "frase. [7]" -> "frase [7]."
    text = re.sub(
        r"\[(\d+)\](?:\s*\[\d+\])+",
        lambda m: "[" + ", ".join(re.findall(r"\d+", m.group(0))) + "]",
        text,
    )
    text = re.sub(r"([.;:!?])[ \t]*(\[\d+(?:,\s*\d+)*\])(?=\s|$)", r" \2\1", text)

    # espaços e pontuação
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +([,.;:!?)])", r"\1", text)
    text = re.sub(r"\( +", "(", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


# ------------------------------------------------------------------- voz

_LOCATOR_LINE_RE = re.compile(r"^\s*(?:RV|AV|SV|YV|BG|YS)\s[\d.–-]+\s*$", re.MULTILINE)
_CITATION_RE = re.compile(r"\s*\[\d+(?:\s*[,–-]\s*\d+)*\]")


def speech_text(text: str | None, *, lang: str = "pt") -> str:
    """Texto para síntese de voz: prosa limpa, sem [n] nem símbolos lidos literalmente."""
    text = clean_prose(text, lang=lang)
    text = _CITATION_RE.sub("", text)
    # linhas que são só localizador ("RV 1.1.1") seriam soletradas pela voz
    text = _LOCATOR_LINE_RE.sub("", text)
    text = re.sub(r"[*#_`~|<>«»“”\"]", "", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return re.sub(r"[ \t]+", " ", text).strip()


# ------------------------------------------------- revisão opcional por LLM

DEFAULT_EDITOR_MODEL = "grok-4.20-0309-non-reasoning"

EDITOR_SYSTEM_PT = """Você é revisor de texto de uma editora brasileira de obras sobre a tradição védica.

Revise o texto abaixo para que soe escrito por um professor humano, em português do Brasil culto e natural.
- Corrija gramática, concordância, regência e pontuação; corte repetições e frases ocas.
- Troque palavras em inglês ou espanhol pelo equivalente português; traduza citações em inglês.
- Remova markdown (negrito, títulos, listas), travessões, rótulos de seção, aberturas e fechos de chatbot ("Claro!", "Em resumo").
- Não acrescente nem retire informação. Preserve exatamente as citações entre colchetes [n], os localizadores (RV 1.1.1), o devanāgarī e a grafia IAST dos termos sânscritos.
- Devolva só o texto revisado, sem comentários.
"""

EDITOR_SYSTEM_EN = """You are a copy editor for a publisher of books on the Vedic tradition.

Edit the text below so it reads as written by a human teacher, in plain educated English.
- Fix grammar and punctuation; cut repetition and empty phrases.
- Remove markdown (bold, headings, lists), em dashes, section labels and chatbot openers/closers.
- Do not add or remove information. Keep bracketed citations [n], locators (RV 1.1.1), Devanāgarī and IAST spellings exactly.
- Return only the edited text, with no comments.
"""


def editor_enabled() -> bool:
    return os.environ.get("VEDIC_TEXT_EDITOR", "1").strip().lower() in {"1", "true", "on", "yes"}


def _citations(text: str) -> set[str]:
    nums = {
        f"[{n}]"
        for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text)
        for n in re.findall(r"\d+", group)
    }
    return nums | set(
        re.findall(r"\b(?:RV|AV|SV|YV|BG|YS)\s\d+(?:\.\d+)*", text)
    )


def edit_prose(
    text: str,
    *,
    lang: str = "pt",
    provider: str = "xai",
    min_chars: int = 400,
) -> str:
    """Segunda passada de revisão (LLM barato). Em qualquer falha ou desvio
    (perda de citação, tamanho muito diferente) devolve o texto de entrada."""
    if not text or provider != "xai" or not editor_enabled() or len(text) < min_chars:
        return text
    from vedic_pipeline.llm.generate import generate_answer

    model = os.environ.get("VEDIC_TEXT_EDITOR_MODEL", DEFAULT_EDITOR_MODEL)
    system = EDITOR_SYSTEM_EN if (lang or "pt").lower().startswith("en") else EDITOR_SYSTEM_PT
    try:
        out = generate_answer(
            system,
            text,
            provider="xai",
            model=model,
            max_tokens=max(600, int(len(text) / 2.5)),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Revisão por LLM falhou (%s); mantendo texto original", exc)
        return text
    edited = clean_prose(out.get("answer") or "", lang=lang)
    ratio = len(edited) / max(1, len(text))
    if not edited or not (0.6 <= ratio <= 1.35):
        logger.warning("Revisão descartada (proporção %.2f)", ratio)
        return text
    if not _citations(text) <= _citations(edited):
        logger.warning("Revisão descartada (perdeu citações)")
        return text
    return edited
