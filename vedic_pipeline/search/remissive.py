"""Índice remissivo da busca e o personagem que merece uma figura.

O índice não é uma nuvem de palavras. Cada entrada é um nome canônico
(divindade, pessoa ou conceito), com as formas que o texto realmente usa,
os localizadores dos trechos recuperados e os verbetes "ver também".
A figura sai só para um personagem — o nome da consulta, ou, se ela não
nomear ninguém, o personagem mais presente nos trechos.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from vedic_pipeline.common.sanskrit import fold_for_search, iast_to_ascii

_KIND_PERSON = "personagem"
_KIND_CONCEPT = "conceito"


@dataclass(frozen=True)
class Gloss:
    id: str
    name: str
    kind: str
    epithet: str
    gloss: str
    aliases: tuple[str, ...]
    see_also: tuple[str, ...]
    visual: str
    avoid: str = ""


# aliases são nomes, não palavras genéricas ("fire", "self", "lord").
_GLOSSES: tuple[Gloss, ...] = (
    Gloss("agni", "Agni", _KIND_PERSON, "o fogo do sacrifício", "Sacerdote escolhido, que leva a oblação aos deuses.", ("agni", "agnim", "agne", "अग्नि", "अग्निम्"), ("soma", "indra"), "Agni, Vedic fire god, fiery red-bronze skin, exactly two bearded heads, crowns of living flame with seven tongues of fire rising above them, a white ram at his side, holding a flaming ladle, sacred fire all around.", "blue skin, grey skin, single head, peaceful blue god"),
    Gloss("indra", "Indra", _KIND_PERSON, "o que empunha o vajra", "Rei dos deuses no R̥gveda, matador de Vṛtra.", ("indra", "indram", "इन्द्र"), ("agni", "soma", "vritra"), "Indra, king of the gods, golden-skinned warrior holding the vajra thunderbolt, chariot of two tawny horses, storm clouds and lightning."),
    Gloss("soma", "Soma", _KIND_PERSON, "a bebida e o deus", "O suco prensado e a divindade que ele é.", ("soma", "somam", "सोम"), ("indra", "agni"), "Soma, moon god with pale silver skin, crescent moon crown, holding a cup of golden soma juice beside pressing stones, night sky, white flowers."),
    Gloss("varuna", "Varuṇa", _KIND_PERSON, "o guardião do ṛta", "Soberano das águas e do juramento.", ("varuna", "varuṇa", "वरुण"), ("mitra", "rita"), "Varuna, god of the cosmic waters, dark blue skin, holding a pasha, a coiled lasso of glowing light, a makara sea creature behind him, ocean waves and night stars."),
    Gloss("mitra", "Mitra", _KIND_PERSON, "o aliado", "Divindade do pacto, quase sempre junto de Varuṇa.", ("mitra", "मित्र"), ("varuna",), "Mitra, god of friendship and oaths, dawn-colored golden skin, open hands in blessing, seated beside a darker companion, morning light."),
    Gloss("ushas", "Uṣas", _KIND_PERSON, "a aurora", "A deusa que abre o caminho do dia.", ("ushas", "usas", "uṣas", "उषस्"), ("surya", "savitr"), "Ushas, goddess of dawn, young woman in rose and saffron veils, chariot drawn by red cows, opening the gates of the sky, pink sunrise."),
    Gloss("surya", "Sūrya", _KIND_PERSON, "o sol", "O olho dos deuses, que percorre o céu.", ("surya", "sūrya", "सूर्य"), ("savitr", "ushas"), "Surya, sun god with golden skin, holding two lotuses, chariot drawn by seven horses, radiant sun disc halo."),
    Gloss("savitr", "Savitṛ", _KIND_PERSON, "o impulsionador", "O sol que inspira, nomeado na Gāyatrī.", ("savitr", "savitri", "savitar", "सवितृ"), ("surya", "gayatri"), "Savitr, golden sun god with golden arms raised, sending forth light before sunrise, golden chariot, quiet dawn sky."),
    Gloss("vayu", "Vāyu", _KIND_PERSON, "o vento", "O sopro que corre à frente dos deuses.", ("vayu", "vāyu", "वायु"), ("indra",), "Vayu, god of wind, pale skin, streaming hair and billowing scarf, riding an antelope, swirling wind and golden dust."),
    Gloss("rudra", "Rudra", _KIND_PERSON, "o que ruge", "Arqueiro da montanha, pai dos Maruts.", ("rudra", "रुद्र"), ("maruts",), "Rudra, fierce mountain archer, matted hair, bow and arrows at rest, tiger skin, storm clouds over the Himalaya, compassionate and severe."),
    Gloss("maruts", "Maruts", _KIND_PERSON, "a tropa da tempestade", "Os jovens que acompanham Indra e Rudra.", ("maruts", "marut", "मरुत्"), ("rudra", "indra"), "The Maruts, band of young storm gods with lightning spears and golden ornaments, riding through storm clouds, wind and rain."),
    Gloss("vishnu", "Viṣṇu", _KIND_PERSON, "o dos passos largos", "Aquele cujos três passos medem o mundo.", ("vishnu", "viṣṇu", "visnu", "विष्णु"), ("indra",), "Vishnu as Trivikrama, giant blue-skinned god striding across earth, sky and heaven in three steps, yellow dhoti, cosmic background."),
    Gloss("brihaspati", "Bṛhaspati", _KIND_PERSON, "o senhor da prece", "Sacerdote dos deuses.", ("brihaspati", "brhaspati", "बृहस्पति"), ("indra",), "Brihaspati, priest of the gods, golden skin, white beard, holding a golden axe, seated before a blazing fire altar."),
    Gloss("ashvins", "Aśvins", _KIND_PERSON, "os gêmeos médicos", "Chegam com a aurora e curam.", ("ashvins", "asvins", "aśvins", "अश्विन्"), ("ushas",), "The Ashvins, divine twin horsemen, two identical young gods on one golden chariot at dawn, honey-colored light, healing herbs."),
    Gloss("sarasvati", "Sarasvatī", _KIND_PERSON, "o rio e a fala", "A corrente que inspira o hino.", ("sarasvati", "sarasvatī", "सरस्वती"), ("vac",), "Sarasvati, goddess of speech, woman in a white sari playing the veena, seated on a white lotus by a flowing river, swan beside her."),
    Gloss("aditi", "Aditi", _KIND_PERSON, "a sem amarras", "Mãe dos Ādityas, a liberdade.", ("aditi", "अदिति"), ("surya", "varuna"), "Aditi, mother of the gods, serene woman with a sky-blue cloak full of stars, radiant children of light around her, boundless sky."),
    Gloss("yama", "Yama", _KIND_PERSON, "o primeiro que morreu", "Rei dos pais, no caminho dos que partiram.", ("yama", "यम"), ("varuna",), "Yama, king of the departed, dark green skin, crown, holding a staff and a coiled pasha lasso, black buffalo behind him, two dogs, dusk.", "skeleton, skull, gore, horror"),
    Gloss("hiranyagarbha", "Hiraṇyagarbha", _KIND_PERSON, "o germe de ouro", "Aquele que surgiu no princípio.", ("hiranyagarbha", "hiraṇyagarbha", "हिरण्यगर्भ"), ("prajapati", "purusha"), "Hiranyagarbha, a glowing golden cosmic egg floating on dark primordial waters, light radiating outward, the beginning of creation."),
    Gloss("purusha", "Puruṣa", _KIND_PERSON, "a pessoa cósmica", "O ser cujo sacrifício é o mundo.", ("purusha", "puruṣa", "purusa", "पुरुष"), ("prajapati", "hiranyagarbha"), "Purusha, the cosmic man, vast serene giant with many heads and eyes, stars and worlds within his body, cosmic sky."),
    Gloss("prajapati", "Prajāpati", _KIND_PERSON, "o senhor das criaturas", "Quem gera e se esvazia na criação.", ("prajapati", "prajāpati", "प्रजापति"), ("purusha", "hiranyagarbha"), "Prajapati, lord of creatures, bearded progenitor seated on a lotus, animals and beings emerging around him, dawn light."),
    Gloss("vritra", "Vṛtra", _KIND_PERSON, "o que cobre as águas", "A serpente que Indra enfrenta.", ("vritra", "vṛtra", "vrtra", "वृत्र"), ("indra",), "Vritra, giant serpent dragon coiled over a mountain, holding back the waters, dark scales, storm clouds and lightning."),
    Gloss("krishna", "Kṛṣṇa", _KIND_PERSON, "o de pele escura", "O auriga e mestre da Bhagavad-gītā.", ("krishna", "kṛṣṇa", "krsna", "कृष्ण"), ("arjuna",), "Krishna, blue-skinned charioteer in yellow silk, peacock feather crown, driving a war chariot with white horses, Kurukshetra battlefield."),
    Gloss("arjuna", "Arjuna", _KIND_PERSON, "o arqueiro", "O guerreiro que pergunta na Gītā.", ("arjuna", "अर्जुन"), ("krishna",), "Arjuna, archer prince holding the Gandiva bow lowered, standing in a war chariot, questioning, dawn over the Kurukshetra battlefield."),
    Gloss("rama", "Rāma", _KIND_PERSON, "o da linhagem de Raghu", "O príncipe do Rāmāyaṇa.", ("rama", "rāma", "raghava", "राम"), ("sita",), "Rama, blue-skinned prince with a great bow and quiver, bark-cloth garments, forest hermitage, calm face."),
    Gloss("sita", "Sītā", _KIND_PERSON, "a nascida do sulco", "A filha da terra, companheira de Rāma.", ("sita", "sītā", "सीता"), ("rama",), "Sita, princess in a red sari with a lotus in hand, standing by a plowed furrow, forest and distant palace."),
    Gloss("brahman", "Brahman", _KIND_CONCEPT, "o absoluto", "O que os Upaniṣads dizem ser um, sem segundo.", ("brahman", "ब्रह्मन्"), ("atman",), ""),
    Gloss("atman", "Ātman", _KIND_CONCEPT, "o si", "O si que os textos aproximam de Brahman.", ("atman", "ātman", "atma", "आत्मन्"), ("brahman", "purusha"), ""),
    Gloss("dharma", "Dharma", _KIND_CONCEPT, "a ordem de cada um", "Dever, lei, o que sustenta.", ("dharma", "धर्म"), ("rita", "karma"), ""),
    Gloss("karma", "Karma", _KIND_CONCEPT, "a ação e seu fruto", "O ato que permanece.", ("karma", "कर्म"), ("dharma",), ""),
    Gloss("rita", "Ṛta", _KIND_CONCEPT, "a ordem do cosmos", "O curso certo que Varuṇa guarda.", ("rita", "ṛta", "rta", "ऋत"), ("varuna", "dharma"), ""),
    Gloss("om", "Oṃ", _KIND_CONCEPT, "a sílaba", "Praṇava, som de abertura.", ("om", "aum", "praṇava", "pranava", "ॐ"), ("brahman",), ""),
    Gloss("moksha", "Mokṣa", _KIND_CONCEPT, "a liberação", "O sair do ciclo.", ("moksha", "mokṣa", "moksa", "मोक्ष"), ("atman",), ""),
    Gloss("yoga", "Yoga", _KIND_CONCEPT, "a união", "Disciplina e junção.", ("yoga", "योग"), ("atman",), ""),
    Gloss("bhakti", "Bhakti", _KIND_CONCEPT, "a devoção", "O amor que se oferece.", ("bhakti", "भक्ति"), ("krishna",), ""),
    Gloss("gayatri", "Gāyatrī", _KIND_CONCEPT, "o metro e a prece", "O hino a Savitṛ, e o metro de vinte e quatro sílabas.", ("gayatri", "gāyatrī", "gayatris", "गायत्री"), ("savitr",), ""),
    Gloss("vac", "Vāc", _KIND_CONCEPT, "a fala", "A palavra como potência.", ("vac", "vāc", "वाच्"), ("sarasvati",), ""),
)


def _norm(text: str) -> str:
    return fold_for_search(iast_to_ascii(text or ""))


def _compile(alias: str) -> re.Pattern[str]:
    token = _norm(alias)
    return re.compile(rf"(?<!\w){re.escape(token)}(?!\w)")


_BY_ID: dict[str, Gloss] = {g.id: g for g in _GLOSSES}
_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    g.id: tuple(_compile(a) for a in g.aliases if _norm(a)) for g in _GLOSSES
}


def _blob(*parts: Any) -> str:
    return _norm("\n".join(str(p) for p in parts if p))


def _mentioned(normed: str, gloss_id: str) -> bool:
    return any(p.search(normed) for p in _PATTERNS[gloss_id])


def _hit_blob(hit: dict[str, Any]) -> str:
    text = str(hit.get("text") or "")[:2000]
    return _blob(hit.get("title"), hit.get("heading"), hit.get("locator"), hit.get("work"), text)


def _locator(hit: dict[str, Any], index: int) -> dict[str, Any]:
    label = str(hit.get("locator") or hit.get("heading") or hit.get("title") or "").strip()
    if not label:
        label = f"trecho {index + 1}"
    return {"index": index, "locator": label[:80]}


def _ref(gloss_id: str, present_ids: set[str]) -> dict[str, Any] | None:
    other = _BY_ID.get(gloss_id)
    if other is None:
        return None
    return {"id": other.id, "name": other.name, "present": gloss_id in present_ids}


def get_figure(figure_id: str) -> Gloss | None:
    gloss = _BY_ID.get((figure_id or "").strip().lower())
    if gloss is None or gloss.kind != _KIND_PERSON:
        return None
    return gloss


# Estilo dos retratos (ver docs/IMAGE_STYLE.md). `VEDIC_FIGURE_STYLE`:
# - sculpted (padrão): busto escuro de bronze e obsidiana com filigrana gravada,
#   chiaroscuro e fundo quase preto, como a série de devas que o Leandro escolheu;
# - cinematic: pintura devocional cinematográfica, halo dourado e raios de luz;
# - miniature: a miniatura Rajput antiga.
# A iconografia vem primeiro; o que não deve aparecer vai para o negativo
# (o "no X" no positivo atrai X).
FIGURE_STYLES: dict[str, str] = {
    "sculpted": (
        "Dark cinematic fantasy portrait, ultra-detailed digital art: the deity rendered like a living "
        "bronze-and-obsidian sculpture, skin and garments covered in intricate engraved filigree, paisley and "
        "tribal ornament, heavy ornate crown, earrings and layered necklaces, intense piercing gaze, dramatic "
        "chiaroscuro rim light, near-black smoky background, muted palette of bronze, charcoal and one glowing "
        "accent color, head-and-shoulders bust, centered, razor-sharp detail."
    ),
    "cinematic": (
        "Epic cinematic Hindu devotional digital painting, hyper-detailed semi-realistic rendering, volumetric "
        "god rays and a glowing golden halo, dramatic sky, rich gold, saffron and deep blue palette, ornate gold "
        "jewelry and crown, serene majestic expression, centered heroic composition, film still."
    ),
    "miniature": "Indian devotional painting, Rajput miniature, vivid colors, gold detail, ornate border.",
}
# CLIP (fallback local) lê só 77 tokens: versão curta do mesmo estilo.
FIGURE_COMPACT_STYLES: dict[str, str] = {
    "sculpted": "Dark ornate bronze bust, engraved filigree, chiaroscuro, black background.",
    "cinematic": "Epic devotional painting, golden halo, god rays.",
    "miniature": FIGURE_STYLES["miniature"],
}
DEFAULT_FIGURE_STYLE = "sculpted"
FIGURE_STYLE = FIGURE_STYLES[DEFAULT_FIGURE_STYLE]

FIGURE_NEGATIVE = (
    "text, letters, caption, watermark, logo, signature, modern clothing, deformed face, "
    "extra heads, extra fingers, blurry, lowres, cropped"
)
# Só a miniatura foge do render; os estilos novos têm cara de render de propósito.
_MINIATURE_NEGATIVE = "photo, photorealistic, 3d render"


def figure_style() -> str:
    choice = (os.environ.get("VEDIC_FIGURE_STYLE") or "").strip().lower()
    return choice if choice in FIGURE_STYLES else DEFAULT_FIGURE_STYLE


def figure_prompt(gloss: Gloss, *, compact: bool = False) -> str:
    visual = gloss.visual.strip() or f"{gloss.name}, {gloss.epithet}."
    styles = FIGURE_COMPACT_STYLES if compact else FIGURE_STYLES
    return f"{visual} {styles[figure_style()]}"


def figure_negative(gloss: Gloss) -> str:
    base = FIGURE_NEGATIVE
    if figure_style() == "miniature":
        base = f"{base}, {_MINIATURE_NEGATIVE}"
    return f"{gloss.avoid}, {base}" if gloss.avoid else base


def annotate_search(query: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Devolve `figure` (ou null) e `entries` do índice desta recuperação."""
    query_norm = _blob(query)
    hit_norms = [_hit_blob(h) for h in hits]
    found: list[tuple[Gloss, bool, list[dict[str, Any]]]] = []
    for gloss in _GLOSSES:
        in_query = bool(query_norm) and _mentioned(query_norm, gloss.id)
        locators = [
            _locator(hit, i)
            for i, (hit, normed) in enumerate(zip(hits, hit_norms, strict=True))
            if _mentioned(normed, gloss.id)
        ]
        if in_query or locators:
            found.append((gloss, in_query, locators))

    present = {g.id for g, _, _ in found}

    def score(item: tuple[Gloss, bool, list[dict[str, Any]]]) -> tuple[int, int, str]:
        gloss, in_query, locators = item
        # Nome dito na consulta vence a frequência nos trechos.
        return (1 if in_query else 0, len(locators), gloss.name)

    people = [item for item in found if item[0].kind == _KIND_PERSON]
    figure: dict[str, Any] | None = None
    if people:
        gloss, in_query, locators = max(people, key=score)
        if in_query or locators:
            figure = _public(gloss, in_query, locators, present)

    entries = [
        _public(gloss, in_query, locators, present)
        for gloss, in_query, locators in sorted(found, key=lambda item: (item[0].kind != _KIND_PERSON, item[0].name))
    ]
    return {"figure": figure, "entries": entries}


def _public(
    gloss: Gloss,
    in_query: bool,
    locators: list[dict[str, Any]],
    present: set[str],
) -> dict[str, Any]:
    see = [ref for ref in (_ref(i, present) for i in gloss.see_also) if ref]
    return {
        "id": gloss.id,
        "name": gloss.name,
        "kind": gloss.kind,
        "epithet": gloss.epithet,
        "gloss": gloss.gloss,
        "in_query": in_query,
        "count": len(locators),
        "locators": locators[:8],
        "see_also": see,
    }
