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


# --- melhor de N tomadas (VEDIC_CHANT_TAKES) ---------------------------------------


@pytest.mark.parametrize(
    ("raw", "n"), [(None, 3), ("1", 1), ("3", 3), ("0", 1), ("9", 6), ("x", 3)]
)
def test_chant_takes(raw, n, monkeypatch):
    if raw is None:
        monkeypatch.delenv("VEDIC_CHANT_TAKES", raising=False)
    else:
        monkeypatch.setenv("VEDIC_CHANT_TAKES", raw)
    assert chant.chant_takes() == n


def test_take_seeds():
    assert chant.take_seeds(1) == [33]
    assert chant.take_seeds(3) == [33, 44, 77]
    assert len(set(chant.take_seeds(6))) == 6


def test_cer_devanagari_and_latin():
    ref = "अद्वेष्टा सर्वभूतानां मैत्रः करुण एव च ।"
    assert chant.cer(ref, "अद्वेष्टा सर्वभूतानाम्, मैत्र करुण एवच।") == 0.0
    assert 0 < chant.cer(ref, "अंदरष्टा सर्वभूताना मत्र करुण एवच") < 0.3
    assert chant.cer(ref, "") == 1.0
    lines = ["sa vai puṁsāṁ paro dharmo"]
    assert chant.cer("", "स वै पुंसां परो धर्मो", lines) < 0.15
    assert chant.cer("", "हरि ओम", lines) > 0.5


@pytest.fixture
def fake_takes(tmp_path, monkeypatch):
    """Worker falso: cada tomada grava um MP3 com a seed no conteúdo."""
    monkeypatch.setattr(chant, "AUDIO_DIR", tmp_path)
    monkeypatch.setattr(chant, "missing_requirements", lambda: [])
    monkeypatch.setattr(chant, "chant_available", lambda: True)
    calls: list[int] = []

    def run_take(key, lines, seed, dest, title):
        calls.append(seed)
        d = chant._takes_dir(dest)
        d.mkdir(parents=True, exist_ok=True)
        mp3, voc = d / f"s{seed}.mp3", d / f"s{seed}_vocal.mp3"
        mp3.write_bytes(f"seed{seed}".encode() + b"\0" * 2000)
        voc.write_bytes(f"voc{seed}".encode())
        return {"seed": seed, "mp3": str(mp3), "vocal": str(voc), "secs": 1.0}

    monkeypatch.setattr(chant, "_run_take", run_take)
    scheduled: list[str] = []
    monkeypatch.setattr(
        chant, "_schedule_upgrade", lambda key, lines, ref, title: scheduled.append(key) or True
    )
    return calls, scheduled


LINES = ["adveṣṭā sarvabhūtānāṁ", "maitraḥ karuṇa eva ca"]
REF = "अद्वेष्टा सर्वभूतानां मैत्रः करुण एव च"


def test_single_take_is_old_behaviour(fake_takes, monkeypatch):
    calls, scheduled = fake_takes
    monkeypatch.setenv("VEDIC_CHANT_TAKES", "1")
    dest = chant.cached_chant("BG.12.13", LINES, reference=REF)
    assert calls == [33] and scheduled == []
    assert dest.read_bytes().startswith(b"seed33")
    meta = chant.chant_meta(dest)
    assert meta["status"] == "final" and meta["chosen_seed"] == 33


def test_first_take_served_then_best_installed(fake_takes, monkeypatch):
    calls, scheduled = fake_takes
    monkeypatch.setenv("VEDIC_CHANT_TAKES", "3")
    monkeypatch.setenv("XAI_API_KEY", "k")
    dest = chant.cached_chant("BG.12.13", LINES, reference=REF)
    assert calls == [33] and scheduled == ["BG.12.13"]
    assert (
        dest.read_bytes().startswith(b"seed33")
        and chant.chant_meta(dest)["status"] == "provisional"
    )
    said = {"voc33": "अंदरष्टा सर्वभूताना मत्र करुण", "voc44": REF, "voc77": "हरि ओम"}
    monkeypatch.setattr(chant, "transcribe", lambda p: said[Path(p).read_bytes().decode()])
    meta = chant.upgrade("BG.12.13", LINES, reference=REF)
    assert calls == [33, 44, 77]
    assert meta["status"] == "final" and meta["chosen_seed"] == 44 and meta["chosen_cer"] == 0.0
    assert [t["seed"] for t in meta["takes"]] == [33, 44, 77]
    assert dest.read_bytes().startswith(b"seed44")
    assert not chant._takes_dir(dest).exists()
    assert chant.upgrade("BG.12.13", LINES, reference=REF)["status"] == "final"  # idempotente


def test_stt_unavailable_keeps_first_take(fake_takes, monkeypatch):
    monkeypatch.setenv("VEDIC_CHANT_TAKES", "3")
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    dest = chant.cached_chant("BG.12.13", LINES, reference=REF)
    meta = chant.upgrade("BG.12.13", LINES, reference=REF)
    assert meta["chosen_seed"] == 33 and "XAI_API_KEY" in meta["stt_error"]
    assert dest.read_bytes().startswith(b"seed33")


def test_stt_error_keeps_first_take(fake_takes, monkeypatch):
    monkeypatch.setenv("VEDIC_CHANT_TAKES", "3")
    monkeypatch.setenv("XAI_API_KEY", "k")
    dest = chant.cached_chant("BG.12.13", LINES, reference=REF)

    def boom(path):
        raise RuntimeError("503")

    monkeypatch.setattr(chant, "transcribe", boom)
    meta = chant.upgrade("BG.12.13", LINES, reference=REF)
    assert meta["chosen_seed"] == 33 and meta["stt_error"]
    assert dest.read_bytes().startswith(b"seed33")


def test_failed_background_take_is_skipped(fake_takes, monkeypatch):
    calls, _ = fake_takes
    monkeypatch.setenv("VEDIC_CHANT_TAKES", "3")
    monkeypatch.setenv("XAI_API_KEY", "k")
    chant.cached_chant("BG.12.13", LINES, reference=REF)
    ok = chant._run_take

    def flaky(key, lines, seed, dest, title):
        if seed == 44:
            raise RuntimeError("MPS caiu")
        return ok(key, lines, seed, dest, title)

    monkeypatch.setattr(chant, "_run_take", flaky)
    said = {"voc33": "हरि ओम", "voc77": REF}
    monkeypatch.setattr(chant, "transcribe", lambda p: said[Path(p).read_bytes().decode()])
    meta = chant.upgrade("BG.12.13", LINES, reference=REF)
    assert [t["seed"] for t in meta["takes"]] == [33, 77] and meta["chosen_seed"] == 77


def test_resume_upgrade_only_for_provisional(fake_takes, monkeypatch):
    _, scheduled = fake_takes
    monkeypatch.setenv("VEDIC_CHANT_TAKES", "3")
    chant.cached_chant("BG.12.13", LINES, reference=REF)
    scheduled.clear()
    assert chant.resume_upgrade("BG.12.13", LINES, reference=REF) is True
    monkeypatch.setenv("XAI_API_KEY", "k")
    monkeypatch.setattr(chant, "transcribe", lambda p: REF)
    chant.upgrade("BG.12.13", LINES, reference=REF)
    assert chant.resume_upgrade("BG.12.13", LINES, reference=REF) is False


def test_endpoint_exposes_seed_and_status(client, tmp_path, monkeypatch):
    monkeypatch.setattr(chant, "AUDIO_DIR", tmp_path)
    lines = chant.lyric_lines(padas=BG_12_13)
    dest = chant.cache_path("BG.12.13", lines)
    dest.write_bytes(b"ID3" + b"\0" * 2000)
    chant._write_meta(dest, {"status": "final", "chosen_seed": 77})
    with patch.dict(os.environ, {"VEDIC_RECITATION_STYLE": "chant"}):
        r = client.get("/api/v1/verses/BG.12.13/audio")
    assert r.headers["x-recitation-style"] == "chant"
    assert r.headers["x-chant-seed"] == "77" and r.headers["x-chant-status"] == "final"
