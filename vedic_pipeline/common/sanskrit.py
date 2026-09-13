"""Processamento e transliteração de termos sânscritos (Devanāgarī, IAST e ASCII).

Suporta:
- Transliteração Devanāgarī -> IAST (ex.: आत्मन् -> ātman, ॐ -> om)
- Normalização/dobra de diacríticos IAST -> ASCII (ex.: ātman -> atman, kṛṣṇa -> krishna / krsna)
- Geração de variantes fonéticas e ortográficas para recall ampliado em buscas híbridas.
"""

from __future__ import annotations

import re
import unicodedata

# Mapeamento de vogais independentes Devanāgarī -> IAST
DEVA_VOWELS: dict[str, str] = {
    "अ": "a",
    "आ": "ā",
    "इ": "i",
    "ई": "ī",
    "उ": "u",
    "ऊ": "ū",
    "ऋ": "ṛ",
    "ॠ": "ṝ",
    "ऌ": "ḷ",
    "ॡ": "ḹ",
    "ए": "e",
    "ऐ": "ai",
    "ओ": "o",
    "औ": "au",
}

# Mapeamento de matras (vogais dependentes) Devanāgarī -> IAST
DEVA_MATRAS: dict[str, str] = {
    "ा": "ā",
    "ि": "i",
    "ी": "ī",
    "ु": "u",
    "ू": "ū",
    "ृ": "ṛ",
    "ॄ": "ṝ",
    "ॢ": "ḷ",
    "ॣ": "ḹ",
    "े": "e",
    "ै": "ai",
    "ो": "o",
    "ौ": "au",
}

# Consoantes Devanāgarī (carregam 'a' inerente a menos que haja matra ou virama)
DEVA_CONSONANTS: dict[str, str] = {
    "क": "k",
    "ख": "kh",
    "ग": "g",
    "घ": "gh",
    "ङ": "ṅ",
    "च": "c",
    "छ": "ch",
    "ज": "j",
    "झ": "jh",
    "ञ": "ñ",
    "ट": "ṭ",
    "ठ": "ṭh",
    "ड": "ḍ",
    "ढ": "ḍh",
    "ण": "ṇ",
    "त": "t",
    "थ": "th",
    "द": "d",
    "ध": "dh",
    "न": "n",
    "प": "p",
    "फ": "ph",
    "ब": "b",
    "भ": "bh",
    "म": "m",
    "य": "y",
    "र": "r",
    "ल": "l",
    "व": "v",
    "श": "ś",
    "ष": "ṣ",
    "स": "s",
    "ह": "h",
}

# Símbolos especiais
DEVA_SPECIAL: dict[str, str] = {
    "ं": "ṃ",  # Anusvara
    "ः": "ḥ",  # Visarga
    "ँ": "m̐",  # Candrabindu
    "ऽ": "'",   # Avagraha
    "ॐ": "om",  # Pranava
    "।": ".",   # Danda
    "॥": "..",  # Double Danda
}

VIRAMA = "्"

# Mapeamento direto de diacríticos IAST -> ASCII simples
IAST_TO_ASCII_MAP: dict[str, str] = {
    "ā": "a",
    "ī": "i",
    "ū": "u",
    "ṛ": "r",
    "ṝ": "r",
    "ḷ": "l",
    "ḹ": "l",
    "ṃ": "m",
    "ḥ": "h",
    "ś": "s",
    "ṣ": "s",
    "ñ": "n",
    "ṅ": "n",
    "ṭ": "t",
    "ḍ": "d",
    "ṇ": "n",
    "Ā": "A",
    "Ī": "I",
    "Ū": "U",
    "Ṛ": "R",
    "Ṝ": "R",
    "Ḷ": "L",
    "Ṹ": "L",
    "Ṃ": "M",
    "Ḥ": "H",
    "Ś": "S",
    "Ṣ": "S",
    "Ñ": "N",
    "Ṅ": "N",
    "Ṭ": "T",
    "Ḍ": "D",
    "Ṇ": "N",
}

# Substituições fonéticas frequentes em inglês/português
IAST_PHONETIC_VARIANTS: list[tuple[str, str]] = [
    ("ś", "sh"),
    ("ṣ", "sh"),
    ("ṛ", "ri"),
    ("c", "ch"),
    ("v", "w"),
]

DEVANAGARI_RANGE = re.compile(r"[\u0900-\u097F]")


def has_devanagari(text: str) -> bool:
    """Retorna True se o texto contiver caracteres Devanāgarī."""
    return bool(DEVANAGARI_RANGE.search(text))


def devanagari_to_iast(text: str) -> str:
    """Converte texto em Devanāgarī para IAST com tratamento de 'a' inerente e virama."""
    if not text:
        return ""

    out: list[str] = []
    chars = list(text)
    n = len(chars)
    i = 0

    while i < n:
        ch = chars[i]

        if ch == "ॐ":
            out.append("om")
            i += 1
            continue

        if ch in DEVA_VOWELS:
            out.append(DEVA_VOWELS[ch])
            i += 1
            continue

        if ch in DEVA_CONSONANTS:
            base_cons = DEVA_CONSONANTS[ch]
            # Verifica o próximo caractere para definir a vogal
            if i + 1 < n:
                next_ch = chars[i + 1]
                if next_ch == VIRAMA:
                    out.append(base_cons)
                    i += 2
                    continue
                if next_ch in DEVA_MATRAS:
                    out.append(base_cons + DEVA_MATRAS[next_ch])
                    i += 2
                    continue
                if next_ch in DEVA_SPECIAL:
                    # Consoante com 'a' inerente seguida de anusvara/visarga
                    out.append(base_cons + "a" + DEVA_SPECIAL[next_ch])
                    i += 2
                    continue
                if next_ch in DEVA_CONSONANTS or next_ch in DEVA_VOWELS or next_ch.isspace() or next_ch in ".,;:-!?'\"()[]{}":
                    # Consoante seguida de outra consoante ou espaço mantém 'a' inerente
                    out.append(base_cons + "a")
                    i += 1
                    continue
            # Fim da string: mantém 'a' inerente
            out.append(base_cons + "a")
            i += 1
            continue

        if ch in DEVA_MATRAS:
            # Matra isolada (raro/malformado)
            out.append(DEVA_MATRAS[ch])
            i += 1
            continue

        if ch in DEVA_SPECIAL:
            out.append(DEVA_SPECIAL[ch])
            i += 1
            continue

        if ch == VIRAMA:
            # Virama isolado
            i += 1
            continue

        # Caracteres comuns (pontuação, espaço, números)
        out.append(ch)
        i += 1

    return "".join(out)


def fold_for_search(text: str) -> str:
    """Remove acentos de recitação e diacríticos combinantes para match lexical.

    `इ॒षे त्वो॒र्जे` e `इषे त्वोर्जे` passam a ser iguais; `agním` ≈ `agnim`.
    """
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    chars: list[str] = []
    for ch in decomposed:
        # Mn: svarita/anudātta, acentos latinos. Matras Devanāgarī (Mc) ficam.
        if unicodedata.category(ch) == "Mn":
            continue
        if ch in "॒॑":
            continue
        chars.append(ch)
    return "".join(chars).casefold()


def iast_to_ascii(text: str) -> str:
    """Converte texto IAST para ASCII simples dobrando os diacríticos."""
    if not text:
        return ""
    res = "".join(IAST_TO_ASCII_MAP.get(c, c) for c in text)
    # Remove marcas diacríticas residuais caso unicode decomposto
    nfkd = unicodedata.normalize("NFKD", res)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def get_sanskrit_variants(text: str) -> list[str]:
    """
    Gera variantes de consulta para termos sânscritos.
    Exemplo: 'आत्मन्' -> ['आत्मन्', 'ātman', 'atman']
    Exemplo: 'kṛṣṇa' -> ['kṛṣṇa', 'krsna', 'krishna']
    """
    variants: list[str] = [text]

    if has_devanagari(text):
        iast = devanagari_to_iast(text)
        if iast and iast != text:
            variants.append(iast)
            ascii_text = iast_to_ascii(iast)
            if ascii_text and ascii_text not in variants:
                variants.append(ascii_text)
    else:
        # Texto já em alfabeto latino
        ascii_text = iast_to_ascii(text)
        if ascii_text and ascii_text != text:
            variants.append(ascii_text)

        # Variantes fonéticas comuns (ex: ś -> sh, ṛ -> ri)
        phonetic = text
        for orig, sub in IAST_PHONETIC_VARIANTS:
            if orig in phonetic:
                phonetic = phonetic.replace(orig, sub)
        if phonetic != text and phonetic not in variants:
            variants.append(phonetic)
        ascii_phonetic = iast_to_ascii(phonetic)
        if ascii_phonetic and ascii_phonetic not in variants:
            variants.append(ascii_phonetic)

    return list(dict.fromkeys(v.strip() for v in variants if v.strip()))
