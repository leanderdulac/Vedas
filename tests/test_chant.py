"""Recitação cantada (VEDIC_RECITATION_STYLE=chant): configuração, letra e fallback."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from vedic_pipeline.llm import chant

BG_12_13 = ["अद्वेष्टा सर्वभूतानां मैत्रः करुण एव च", "निर्ममो निरहङ्कारः समदुःखसुखः क्षमी", "12.13"]


def _bundle() -> dict:
    return {
        "verse_id": "BG.12.13",
        "locator": "BG 12.13",
        "has_sanskrit": True,
        "witnesses": [{"role": "sa", "text": " । ".join(BG_12_13[:2]) + " ॥"}],
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "spoken"),
        ("", "spoken"),
        ("spoken", "spoken"),
        ("chant", "chant"),
        (" CHANT ", "chant"),
        ("sung", "spoken"),
    ],
)
def test_recitation_style(raw, expected):
    env = {k: v for k, v in os.environ.items() if k != "VEDIC_RECITATION_STYLE"}
    if raw is not None:
        env["VEDIC_RECITATION_STYLE"] = raw
    with patch.dict(os.environ, env, clear=True):
        assert chant.recitation_style() == expected


def test_unavailable_without_acestep_dir():
    with patch.dict(os.environ, {"VEDIC_ACESTEP_DIR": ""}):
        assert not chant.chant_available()
        assert chant.missing_requirements() == ["VEDIC_ACESTEP_DIR não definido"]


def test_unavailable_reports_missing_pieces(tmp_path):
    with patch.dict(
        os.environ,
        {"VEDIC_ACESTEP_DIR": str(tmp_path), "VEDIC_CHANT_TANPURA": str(tmp_path / "x.ogg")},
    ):
        missing = " | ".join(chant.missing_requirements())
    assert (
        "checkout do ACE-Step" in missing
        and "acestep-v15-turbo" in missing
        and "tanpura" in missing
    )


def test_lyrics_iast_one_pada_per_line():
    lines = chant.lyric_lines(padas=BG_12_13)
    assert lines == [
        "adveṣṭā sarvabhūtānāṁ",
        "maitraḥ karuṇa eva ca",
        "nirmamo nirahaṅkāraḥ",
        "samaduḥkhasukhaḥ kṣamī",
    ]
    assert chant.lyrics_for(lines).startswith("[Verse]\nadveṣṭā")
    assert chant.chant_duration(lines) == 34  # receita vencedora: 34 s


def test_lyrics_strip_vedic_accents_and_map_la():
    lines = chant.lyric_lines(padas=["अ॒ग्निमी॑ळे पु॒रोहि॑तं य॒ज्ञस्य॑ दे॒वमृ॒त्विज॑म्", "होता॑रं रत्न॒धात॑मम्"])
    assert lines[0] == "agnimīḍe purohitaṁ"
    assert all("॒" not in ln and "॑" not in ln for ln in lines)


def test_lyrics_from_free_text_drops_references():
    text = "janmādy asya yato 'nvayād itarataś cārtheṣv abhijñaḥ svarāṭ / tene brahma hṛdā ya ādikavaye // 1.1.1 //"
    lines = chant.lyric_lines(text=text)
    assert lines and not any(ch.isdigit() for ln in lines for ch in ln)


def test_duration_is_clamped():
    assert chant.chant_duration(["a"]) == 20
    assert chant.chant_duration(["ka" * 400]) == 150


def test_cache_key_depends_on_text_and_recipe(tmp_path, monkeypatch):
    monkeypatch.setattr(chant, "AUDIO_DIR", tmp_path)
    a = chant.cache_path("BG.12.13", ["adveṣṭā"])
    assert a != chant.cache_path("BG.12.13", ["maitraḥ"])
    monkeypatch.setitem(chant.RECIPE, "seed", 34)
    assert a != chant.cache_path("BG.12.13", ["adveṣṭā"])


def test_cached_chant_raises_when_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(chant, "AUDIO_DIR", tmp_path)
    with patch.dict(os.environ, {"VEDIC_ACESTEP_DIR": ""}), pytest.raises(chant.ChantUnavailable):
        chant.cached_chant("BG.12.13", ["adveṣṭā"])


# --- endpoint: canto com fallback para a voz falada ---------------------------------


def _mp3(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.write_bytes(b"ID3" + b"\0" * 2000)
    return p


@pytest.fixture
def client():
    from vedic_pipeline.api.app import create_app

    env = {"VEDIC_PIPELINE_API_TOKEN": "", "VEDIC_GENERATION_API_TOKEN": "", "VEDIC_PUBLIC_API": ""}
    with (
        patch.dict(os.environ, env),
        patch("vedic_pipeline.api.verse_service.get_verse", return_value=_bundle()),
    ):
        yield TestClient(create_app())


def test_spoken_mode_never_touches_chant(client, tmp_path):
    spoken = _mp3(tmp_path, "spoken.mp3")
    with (
        patch.dict(os.environ, {"VEDIC_RECITATION_STYLE": "spoken"}),
        patch("vedic_pipeline.llm.tts.get_cached_verse_audio", return_value=spoken),
        patch("vedic_pipeline.llm.chant.cached_chant") as gen,
    ):
        r = client.get("/api/v1/verses/BG.12.13/audio")
    assert r.status_code == 200 and r.headers["x-recitation-style"] == "spoken"
    gen.assert_not_called()


def test_chant_mode_serves_chant(client, tmp_path):
    sung = _mp3(tmp_path, "sung.mp3")
    with (
        patch.dict(os.environ, {"VEDIC_RECITATION_STYLE": "chant"}),
        patch("vedic_pipeline.llm.chant.get_cached_chant", return_value=None),
        patch("vedic_pipeline.llm.chant.missing_requirements", return_value=[]),
        patch("vedic_pipeline.llm.chant.cached_chant", return_value=sung) as gen,
    ):
        r = client.get("/api/v1/verses/BG.12.13/audio")
    assert r.status_code == 200
    assert r.headers["x-recitation-style"] == "chant"
    assert "CC BY 4.0" in r.headers["x-recitation-credit"]
    lines = gen.call_args.args[1]
    assert lines[0] == "adveṣṭā sarvabhūtānāṁ" and len(lines) == 4


def test_chant_failure_falls_back_to_spoken(client, tmp_path):
    spoken = _mp3(tmp_path, "spoken.mp3")
    with (
        patch.dict(os.environ, {"VEDIC_RECITATION_STYLE": "chant"}),
        patch("vedic_pipeline.llm.chant.get_cached_chant", return_value=None),
        patch("vedic_pipeline.llm.chant.missing_requirements", return_value=[]),
        patch("vedic_pipeline.llm.chant.cached_chant", side_effect=RuntimeError("MPS caiu")),
        patch("vedic_pipeline.llm.tts.get_cached_verse_audio", return_value=spoken),
    ):
        r = client.get("/api/v1/verses/BG.12.13/audio")
    assert r.status_code == 200 and r.headers["x-recitation-style"] == "spoken"


def test_chant_unavailable_falls_back_to_spoken(client, tmp_path):
    spoken = _mp3(tmp_path, "spoken.mp3")
    with (
        patch.dict(os.environ, {"VEDIC_RECITATION_STYLE": "chant", "VEDIC_ACESTEP_DIR": ""}),
        patch("vedic_pipeline.llm.chant.get_cached_chant", return_value=None),
        patch("vedic_pipeline.llm.chant.cached_chant") as gen,
        patch("vedic_pipeline.llm.tts.get_cached_verse_audio", return_value=spoken),
    ):
        r = client.get("/api/v1/verses/BG.12.13/audio")
    assert r.status_code == 200 and r.headers["x-recitation-style"] == "spoken"
    gen.assert_not_called()


def test_recitation_info_endpoint(client):
    with patch.dict(os.environ, {"VEDIC_RECITATION_STYLE": "chant", "VEDIC_ACESTEP_DIR": ""}):
        info = client.get("/api/v1/recitation").json()
    assert info["style"] == "chant" and info["chant_available"] is False
    assert any("sankalp" in c and "CC BY 4.0" in c for c in info["credits"])
    assert "missing" not in info
    with patch.dict(os.environ, {"VEDIC_RECITATION_STYLE": "spoken"}):
        assert client.get("/api/v1/recitation").json()["credits"] == []


def test_recite_validates_text(client):
    assert client.post("/api/v1/recite", json={"text": ""}).status_code == 422
    assert client.post("/api/v1/recite", json={"text": "a" * 801}).status_code == 422


def test_recite_chant_then_fallback(client, tmp_path):
    sung, spoken = _mp3(tmp_path, "sung.mp3"), _mp3(tmp_path, "spoken.mp3")
    body = {
        "text": "janmādy asya yato 'nvayād itarataś cārtheṣv abhijñaḥ svarāṭ",
        "label": "BhP.1.1.1",
    }
    with (
        patch.dict(os.environ, {"VEDIC_RECITATION_STYLE": "chant"}),
        patch("vedic_pipeline.llm.chant.get_cached_chant", return_value=sung),
    ):
        assert client.post("/api/v1/recite", json=body).headers["x-recitation-style"] == "chant"
    with (
        patch.dict(os.environ, {"VEDIC_RECITATION_STYLE": "chant"}),
        patch("vedic_pipeline.llm.chant.get_cached_chant", return_value=None),
        patch("vedic_pipeline.llm.chant.missing_requirements", return_value=["x"]),
        patch("vedic_pipeline.llm.tts.get_cached_verse_audio", return_value=spoken),
    ):
        assert client.post("/api/v1/recite", json=body).headers["x-recitation-style"] == "spoken"


def test_chant_without_token_plays_cached_spoken(tmp_path):
    from vedic_pipeline.api.app import create_app

    spoken = _mp3(tmp_path, "spoken.mp3")
    env = {
        "VEDIC_RECITATION_STYLE": "chant",
        "VEDIC_GENERATION_API_TOKEN": "secret",
        "XAI_API_KEY": "k",
    }
    with (
        patch.dict(os.environ, env),
        patch("vedic_pipeline.api.verse_service.get_verse", return_value=_bundle()),
        patch("vedic_pipeline.llm.chant.get_cached_chant", return_value=None),
        patch("vedic_pipeline.llm.chant.missing_requirements", return_value=[]),
        patch("vedic_pipeline.llm.chant.cached_chant") as gen,
        TestClient(create_app()) as c,
    ):
        with patch("vedic_pipeline.llm.tts.get_cached_verse_audio", return_value=spoken):
            r = c.get("/api/v1/verses/BG.12.13/audio")
            assert r.status_code == 200 and r.headers["x-recitation-style"] == "spoken"
        with patch("vedic_pipeline.llm.tts.get_cached_verse_audio", return_value=None):
            assert c.get("/api/v1/verses/BG.12.13/audio").status_code == 401
    gen.assert_not_called()
