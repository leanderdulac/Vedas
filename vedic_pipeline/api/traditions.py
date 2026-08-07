"""Catálogo de tradições e eixos do conhecimento védico (UI + API)."""

from __future__ import annotations

from typing import Any

TRADITIONS: list[dict[str, Any]] = [
    {
        "id": "vedic",
        "name_sa": "वेद",
        "name_en": "Veda",
        "name_pt": "Veda",
        "description_pt": (
            "Os quatro Vedas — Ṛg, Yajur, Sāma e Atharva — e as camadas "
            "associadas (Saṃhitā, Brāhmaṇa, Āraṇyaka)."
        ),
        "description_en": (
            "The four Vedas — Rig, Yajur, Sama and Atharva — and associated layers."
        ),
        "icon": "ॐ",
        "color": "#c45c26",
        "order": 1,
    },
    {
        "id": "upanishad",
        "name_sa": "उपनिषद्",
        "name_en": "Upanishad",
        "name_pt": "Upanishad",
        "description_pt": (
            "Textos de jñāna que investigam Ātman, Brahman e a natureza do real."
        ),
        "description_en": "Wisdom texts on Ātman, Brahman and the nature of reality.",
        "icon": "✦",
        "color": "#d4a84b",
        "order": 2,
    },
    {
        "id": "itihasa",
        "name_sa": "इतिहास",
        "name_en": "Itihasa",
        "name_pt": "Itihāsa",
        "description_pt": "Mahābhārata e Rāmāyaṇa — épicos e dharma em narrativa.",
        "description_en": "Mahabharata and Ramayana — epic narratives of dharma.",
        "icon": "⚔",
        "color": "#8b3a2a",
        "order": 3,
    },
    {
        "id": "purana",
        "name_sa": "पुराण",
        "name_en": "Purana",
        "name_pt": "Purāṇa",
        "description_pt": "Cosmologia, genealogias e ciclos de tempo (yugas).",
        "description_en": "Cosmology, genealogies and cycles of time (yugas).",
        "icon": "☽",
        "color": "#5c4a7a",
        "order": 4,
    },
    {
        "id": "vaishnava",
        "name_sa": "वैष्णव",
        "name_en": "Vaishnava",
        "name_pt": "Vaishnava",
        "description_pt": (
            "Literatura de bhakti e teologias vaishnavas — Gītā, purāṇas, ācāryas."
        ),
        "description_en": "Bhakti literature and Vaishnava theologies.",
        "icon": "lotus",
        "color": "#2d6a4f",
        "order": 5,
    },
    {
        "id": "jyotisha",
        "name_sa": "ज्योतिष",
        "name_en": "Jyotisha",
        "name_pt": "Jyotiṣa",
        "description_pt": "Astronomia e astrologia védicas como aṅga do Veda.",
        "description_en": "Vedic astronomy and astrology as a limb of the Veda.",
        "icon": "☉",
        "color": "#b8860b",
        "order": 6,
    },
    {
        "id": "tantra",
        "name_sa": "तन्त्र",
        "name_en": "Tantra / Agama",
        "name_pt": "Tantra / Āgama",
        "description_pt": "Tradições śaiva, śākta e āgamas de ritual e yoga interior.",
        "description_en": "Shaiva, Shakta and Agama traditions of ritual and inner yoga.",
        "icon": "△",
        "color": "#9b2226",
        "order": 7,
    },
    {
        "id": "yoga",
        "name_sa": "योग",
        "name_en": "Yoga / Darshana",
        "name_pt": "Yoga / Darśana",
        "description_pt": "Yoga-sūtra, sāṃkhya e sistemas filosóficos clássicos.",
        "description_en": "Yoga-sutra, Samkhya and classical philosophical systems.",
        "icon": "◎",
        "color": "#1d3557",
        "order": 8,
    },
    {
        "id": "grammar",
        "name_sa": "व्याकरण",
        "name_en": "Sanskrit Grammar",
        "name_pt": "Gramática Sânscrita",
        "description_pt": "Pāṇini e a tradição de vyākaraṇa — porta de entrada do śāstra.",
        "description_en": "Panini and the vyakarana tradition — gateway to shastra.",
        "icon": "अ",
        "color": "#457b9d",
        "order": 9,
    },
]


def list_traditions() -> list[dict[str, Any]]:
    return sorted(TRADITIONS, key=lambda t: t["order"])


def get_tradition(tradition_id: str) -> dict[str, Any] | None:
    tid = (tradition_id or "").lower()
    for t in TRADITIONS:
        if t["id"] == tid:
            return t
    return None
