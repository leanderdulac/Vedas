"""Worker de canto do Vedas: roda no venv do ACE-Step 1.5 (MIT), não no do app.

Protocolo: uma requisição JSON por linha no stdin, uma resposta JSON por linha
no stdout ({"ok": true, "timings": {...}} ou {"ok": false, "error": "..."}).
Todo o resto que as bibliotecas imprimem vai para o stderr (log do app).

Etapas: ACE-Step text2music (receita fixa vinda do app) -> Demucs htdemucs
(MIT) isola a voz -> mix sobre a tanpura real (sankalp, CC BY 4.0, afinada
na tônica por reamostragem) com sala curta -> ffmpeg loudnorm -16 LUFS para
MP3 com o crédito nos metadados. Modelos carregam na primeira requisição e
ficam na memória; stdin fechado encerra o processo.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from fractions import Fraction

# canal do protocolo = stdout original; stdout das bibliotecas -> stderr
_PROTO = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
os.dup2(2, 1)
sys.stdout = sys.stderr

import numpy as np  # noqa: E402

TANPURA_SA_HZ = 235.5  # Sa médio medido na gravação (corda grave em 117,8 Hz)
SR = 44100
_STATE: dict = {}


def reply(obj: dict) -> None:
    _PROTO.write(json.dumps(obj, ensure_ascii=False) + "\n")
    _PROTO.flush()


def load_ace(root: str, recipe: dict):
    if "dit" in _STATE:
        return _STATE["dit"], _STATE["llm"]
    sys.path.insert(0, root)
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler

    dit = AceStepHandler()
    msg = dit.initialize_service(project_root=root, config_path=recipe["dit"], device="mps")
    print("[chant_worker] DiT:", msg, file=sys.stderr)
    llm = LLMHandler()
    backend = "mlx" if sys.platform == "darwin" else "pt"
    msg = llm.initialize(
        checkpoint_dir=os.path.join(root, "checkpoints"),
        lm_model_path=recipe["lm"],
        backend=backend,
        device="auto",
    )
    print("[chant_worker] LM:", msg, file=sys.stderr)
    _STATE["dit"], _STATE["llm"] = dit, llm
    return dit, llm


def load_demucs():
    if "demucs" not in _STATE:
        from demucs.pretrained import get_model

        model = get_model("htdemucs")
        model.eval()
        _STATE["demucs"] = model
    return _STATE["demucs"]


def resample(y: np.ndarray, sr_in: float, sr_out: float) -> np.ndarray:
    from scipy.signal import resample_poly

    frac = Fraction(sr_out / sr_in).limit_denominator(2000)
    return resample_poly(y, frac.numerator, frac.denominator, axis=-1)


def generate(root: str, req: dict, tmp: str) -> tuple[np.ndarray, int]:
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    recipe = req["recipe"]
    dit, llm = load_ace(root, recipe)
    params = GenerationParams(
        task_type="text2music",
        caption=recipe["caption"],
        lyrics=req["lyrics"],
        vocal_language=recipe["vocal_language"],
        bpm=recipe["bpm"],
        keyscale=recipe["keyscale"],
        timesignature=recipe["timesignature"],
        duration=float(req["duration"]),
        seed=recipe["seed"],
        inference_steps=recipe["inference_steps"],
        shift=recipe["shift"],
        thinking=True,
        use_cot_caption=False,
        use_cot_language=False,
    )
    config = GenerationConfig(
        batch_size=1, use_random_seed=False, seeds=[recipe["seed"]], audio_format="flac"
    )
    result = generate_music(dit, llm, params, config, save_dir=tmp)
    if not result.success:
        raise RuntimeError(f"ACE-Step falhou: {result.error}")
    audio = result.audios[0]
    wav = audio["tensor"]
    wav = (
        wav.detach().cpu().float().numpy()
        if hasattr(wav, "detach")
        else np.asarray(wav, dtype=np.float32)
    )
    if wav.ndim == 1:
        wav = np.stack([wav, wav])
    return wav, int(audio.get("sample_rate", 48000))


def isolate_vocals(wav: np.ndarray, sr: int) -> np.ndarray:
    import torch
    from demucs.apply import apply_model

    model = load_demucs()
    x = resample(wav, sr, model.samplerate).astype(np.float32)
    ref = x.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    t = torch.from_numpy((x - mean) / std)[None]
    with torch.no_grad():
        out = apply_model(model, t, device="cpu", split=True, overlap=0.25, progress=False)[0]
    vocals = out[model.sources.index("vocals")].numpy() * std + mean
    return resample(vocals.mean(0), model.samplerate, SR)


def trim(x: np.ndarray, top_db: float = 35.0) -> np.ndarray:
    hop = 512
    n = len(x) // hop
    if n < 2:
        return x
    rms = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms / rms.max())
    idx = np.where(db > -top_db)[0]
    if not len(idx):
        return x
    return x[idx[0] * hop : (idx[-1] + 1) * hop]


def tanpura_bed(
    path: str, dur: float, tonic_hz: float, xfade: float = 2.5, width: float = 0.55
) -> np.ndarray:
    import soundfile as sf

    y, s0 = sf.read(path, always_2d=True)
    y = y.T.astype(np.float64)
    if y.shape[0] == 1:
        y = np.vstack([y, y])
    y = y[:, int(1.0 * s0) : y.shape[1] - int(1.0 * s0)]
    ratio = (tonic_hz * 2) / TANPURA_SA_HZ  # corda grave -> tônica do canto
    y = resample(y, s0, SR / ratio)  # reinterpretado a SR => altura * ratio
    n, xf = y.shape[1], int(xfade * SR)
    total = int(dur * SR)
    out = np.zeros((2, total + n))
    t = np.linspace(0, np.pi / 2, xf)
    fin, fout = np.sin(t), np.cos(t)
    pos, i = 0, 0
    while pos < total:
        seg = y[::-1].copy() if i % 2 else y.copy()
        if pos > 0:
            seg[:, :xf] *= fin
        seg[:, -xf:] *= fout
        out[:, pos : pos + n] += seg
        pos += n - xf
        i += 1
    out = out[:, :total]
    mid, side = (out[0] + out[1]) / 2, (out[0] - out[1]) / 2
    out = np.stack([mid + width * side, mid - width * side])
    f = int(1.5 * SR)
    out[:, :f] *= np.linspace(0, 1, f)
    out[:, -f:] *= np.linspace(1, 0, f)
    return out / (np.abs(out).max() + 1e-9)


def room(v: np.ndarray, rt: float = 1.9, seed: int = 3) -> np.ndarray:
    from scipy.signal import fftconvolve, lfilter

    rng = np.random.default_rng(seed)
    n = int(rt * SR)
    env = np.exp(-6.9 * np.arange(n) / n)
    wet = []
    for _ in range(2):
        ir = rng.standard_normal(n) * env
        ir = lfilter([0.25], [1, -0.75], ir)  # passa-baixa suave
        ir[: int(0.012 * SR)] = 0
        wet.append(fftconvolve(v, ir)[: len(v)])
    w = np.stack(wet)
    return w / (np.abs(w).max() + 1e-9)


def mix_and_encode(vocal: np.ndarray, req: dict, tmp: str) -> float:
    import soundfile as sf

    recipe = req["recipe"]
    x = trim(vocal)
    x = x / (np.abs(x).max() + 1e-9)
    fo = int(0.8 * SR)
    x[-fo:] *= np.linspace(1, 0, fo)
    v = np.concatenate([np.zeros(int(2.5 * SR)), x, np.zeros(int(3.5 * SR))])
    voice = 0.85 * np.stack([v, v]) + 0.15 * room(v)
    bed = tanpura_bed(req["tanpura"], len(v) / SR, float(recipe["tonic_hz"]))[:, : len(v)]
    mix = voice / (np.abs(voice).max() + 1e-9) * 0.9 + bed * 10 ** (
        float(recipe["tanpura_db"]) / 20
    )
    mix /= np.abs(mix).max() + 1e-9
    wav = os.path.join(tmp, "mix.wav")
    sf.write(wav, (0.9 * mix).T.astype(np.float32), SR)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        wav,
        "-af",
        "loudnorm=I=-16:TP=-1.5",
        "-ar",
        str(SR),
        "-b:a",
        "192k",
        "-metadata",
        f"title={req.get('title', '')}",
        "-metadata",
        "artist=Vedas (ACE-Step 1.5)",
        "-metadata",
        f"comment={req.get('comment', '')}",
        req["out"],
    ]
    subprocess.run(cmd, check=True)
    return len(v) / SR


def handle(root: str, req: dict) -> dict:
    timings: dict = {}
    with tempfile.TemporaryDirectory(prefix="vedas_chant_") as tmp:
        t = time.monotonic()
        wav, sr = generate(root, req, tmp)
        timings["acestep_s"] = round(time.monotonic() - t, 1)
        t = time.monotonic()
        vocal = isolate_vocals(wav, sr)
        timings["demucs_s"] = round(time.monotonic() - t, 1)
        t = time.monotonic()
        dur = mix_and_encode(vocal, req, tmp)
        timings["mix_s"] = round(time.monotonic() - t, 1)
    return {"ok": True, "timings": timings, "duration_s": round(dur, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acestep-dir", required=True)
    args = ap.parse_args()
    os.chdir(args.acestep_dir)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            reply(handle(args.acestep_dir, json.loads(line)))
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]})


if __name__ == "__main__":
    main()
