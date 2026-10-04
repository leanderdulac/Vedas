"""Recitação cantada dos versos em sânscrito (`VEDIC_RECITATION_STYLE=chant`).

O canto sai do ACE-Step 1.5 (MIT) rodando localmente (MPS/MLX no Mac) com a
receita aprovada na rodada v3: letra em IAST, um pāda por linha,
`vocal_language=sa`, 66 bpm, C# maior, seed fixa 33 e legenda de canto
devocional. A voz é isolada com Demucs htdemucs (MIT) e mixada sobre a
tanpura real "Acoustic Tanpura 10" de sankalp (Freesound #155272, CC BY 4.0,
crédito obrigatório: vai no MP3, no cabeçalho HTTP e na UI).

O ACE-Step e o Demucs ficam num venv próprio (`VEDIC_ACESTEP_DIR`, ver
`scripts/setup_chant.sh`): o app só fala com `scripts/chant_worker.py` por
stdin/stdout (JSON por linha). O worker sobe na primeira recitação cantada
(carga preguiçosa) e mantém o modelo na memória. Cada verso gerado vai para o
cache em `data/audio/`. Qualquer falha (venv ausente, modelo sem checkpoint,
erro ou timeout) devolve `ChantUnavailable`/exceção e o endpoint cai na voz
falada de sempre.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from vedic_pipeline.common.sanskrit import devanagari_to_iast, has_devanagari

logger = logging.getLogger("vedic_pipeline.llm.chant")

STYLES = ("spoken", "chant")
AUDIO_DIR = Path("data/audio")
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_SCRIPT = REPO_ROOT / "scripts" / "chant_worker.py"
DEFAULT_TANPURA = Path("data/assets/tanpura/sankalp_155272_acoustic_tanpura_10.ogg")

CAPTION = (
    "Devotional Sanskrit shloka chant, Bhagavad Gita verse recitation, solo male vocal, warm calm baritone, "
    "slow meditative tempo, sustained vowels, tanpura drone, soft harmonium, Indian classical, bhajan, temple ambience"
)
# Receita vencedora da rodada v3 (BG 12.13: FA 0,75, CER 0,13, platô 0,51).
RECIPE: dict[str, Any] = {
    "version": 1,
    "dit": "acestep-v15-turbo",
    "lm": "acestep-5Hz-lm-1.7B",
    "caption": CAPTION,
    "vocal_language": "sa",
    "bpm": 66,
    "keyscale": "C# major",
    "timesignature": "4",
    "seed": 33,
    "inference_steps": 8,
    "shift": 3.0,
    "tonic_hz": 138.6,  # C#3: a corda grave da tanpura vai para a tônica
    "tanpura_db": -24.0,
}
TANPURA_CREDIT = '"Acoustic Tanpura 10" by sankalp (freesound.org/s/155272), CC BY 4.0'
MODEL_CREDIT = "Canto gerado por ACE-Step 1.5 (MIT); voz isolada com Demucs (MIT)"

_IAST_VOWEL_RE = re.compile(r"ai|au|[aāiīuūṛṝḷḹeo]", re.IGNORECASE)
_ACCENT_RE = re.compile(r"[\u0951\u0952\u0300\u0301]")  # ॑ ॒ e acentos agudo/grave combinantes
_DANDA_RE = re.compile(r"[।॥|/]+")
_DIGITS_RE = re.compile(r"[0-9०-९]+")
_LETTER_RE = re.compile(r"[A-Za-zĀ-ſḀ-ỿ\u0900-\u097F]")


class ChantUnavailable(RuntimeError):
    """O canto local não está configurado (venv, checkpoints, tanpura ou ffmpeg)."""


def recitation_style() -> str:
    """`VEDIC_RECITATION_STYLE` = `spoken` (padrão) | `chant`; valor desconhecido vira `spoken`."""
    raw = (os.environ.get("VEDIC_RECITATION_STYLE") or "spoken").strip().lower()
    if raw not in STYLES:
        logger.warning("VEDIC_RECITATION_STYLE=%r desconhecido; usando spoken", raw)
        return "spoken"
    return raw


def acestep_dir() -> Path | None:
    raw = (os.environ.get("VEDIC_ACESTEP_DIR") or "").strip()
    return Path(raw).expanduser() if raw else None


def acestep_python() -> Path | None:
    raw = (os.environ.get("VEDIC_ACESTEP_PYTHON") or "").strip()
    if raw:
        return Path(raw).expanduser()
    root = acestep_dir()
    return root / ".venv" / "bin" / "python" if root else None


def tanpura_path() -> Path:
    raw = (os.environ.get("VEDIC_CHANT_TANPURA") or "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_TANPURA


def chant_timeout() -> float:
    try:
        return max(30.0, float(os.environ.get("VEDIC_CHANT_TIMEOUT", "600")))
    except ValueError:
        return 600.0


def missing_requirements() -> list[str]:
    """O que falta para o canto local; lista vazia = disponível."""
    missing: list[str] = []
    root = acestep_dir()
    if root is None:
        return ["VEDIC_ACESTEP_DIR não definido"]
    if not (root / "acestep").is_dir():
        missing.append(f"checkout do ACE-Step ausente em {root}")
    py = acestep_python()
    if py is None or not py.exists():
        missing.append(f"python do ACE-Step ausente ({py})")
    ckpt = root / "checkpoints"
    for name in (RECIPE["dit"], RECIPE["lm"], "vae"):
        if not (ckpt / name).is_dir():
            missing.append(f"checkpoint {name} ausente")
    if not tanpura_path().exists():
        missing.append(f"tanpura ausente ({tanpura_path()}); rode scripts/fetch_tanpura.py")
    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg fora do PATH")
    if not WORKER_SCRIPT.exists():
        missing.append("scripts/chant_worker.py ausente")
    return missing


def chant_available() -> bool:
    return not missing_requirements()


def _clean_line(text: str) -> str:
    text = _ACCENT_RE.sub("", text)
    if has_devanagari(text):
        # o transliterador comum não cobre o ḷa védico: ळ/ळ्ह = ड/ढ intervocálicos
        # (grafia clássica īḍe); ajuste só aqui, sem mexer na busca
        text = devanagari_to_iast(text.replace("ळ्ह", "ढ").replace("ळ", "ड"))
    text = _DIGITS_RE.sub(" ", text)
    text = re.sub(r"[.\[\]()“”\"'‘’,;:!?\-–—]+", " ", text)
    # grafia da receita vencedora: anusvāra como ṁ
    text = text.replace("ṃ", "ṁ").replace("Ṃ", "Ṁ")
    return re.sub(r"\s+", " ", text).strip()


def lyric_lines(padas: list[str] | None = None, text: str | None = None) -> list[str]:
    """Um pāda por linha, em IAST; descarta numeração e restos sem letras."""
    raw: list[str] = []
    if padas:
        for p in padas:
            raw.extend(_DANDA_RE.split(p))
    elif text:
        for ln in text.split("\n"):
            raw.extend(_DANDA_RE.split(ln))
    lines = [_clean_line(p) for p in raw]
    out: list[str] = []
    for ln in lines:
        if _LETTER_RE.search(ln):
            out.extend(_split_long(ln))
    return out


def _split_long(line: str, limit: int = 14) -> list[str]:
    """Meia-estrofe (ex.: 16 sílabas do anuṣṭubh) vira dois pādas na fronteira
    de palavra mais próxima do meio — a receita vencedora cantava um pāda por linha."""
    words = line.split()
    total = len(_IAST_VOWEL_RE.findall(line))
    if total < limit or len(words) < 2:
        return [line]
    best, acc, best_gap = 1, 0, total
    for i, w in enumerate(words[:-1], 1):
        acc += len(_IAST_VOWEL_RE.findall(w))
        if abs(total / 2 - acc) < best_gap:
            best, best_gap = i, abs(total / 2 - acc)
    return [" ".join(words[:best]), " ".join(words[best:])]


def lyrics_for(lines: list[str]) -> str:
    return "[Verse]\n" + "\n".join(lines)


def syllable_count(lines: list[str]) -> int:
    return sum(len(_IAST_VOWEL_RE.findall(ln)) for ln in lines)


def chant_duration(lines: list[str]) -> int:
    """~0,875 s por sílaba a 66 bpm mais 6 s de entrada (BG 12.13 -> 34 s); 20–150 s."""
    return int(min(150, max(20, round(6 + 0.875 * syllable_count(lines)))))


def _digest(lines: list[str]) -> str:
    blob = json.dumps({"lines": lines, "recipe": RECIPE}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def cache_path(key: str, lines: list[str]) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", key)[:80]
    return AUDIO_DIR / f"{safe}_chant_{_digest(lines)}.mp3"


def get_cached_chant(key: str, lines: list[str]) -> Path | None:
    dest = cache_path(key, lines)
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    return None


class _Worker:
    """Processo do worker do ACE-Step (venv próprio), criado sob demanda."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen[str] | None = None
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.lock = threading.Lock()

    def _spawn(self) -> None:
        py, root = acestep_python(), acestep_dir()
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        log = open(AUDIO_DIR / "chant_worker.log", "a", encoding="utf-8")  # noqa: SIM115
        env = {**os.environ, "TOKENIZERS_PARALLELISM": "false", "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
        env.pop("XAI_API_KEY", None)  # o worker não precisa de chave nenhuma
        self.proc = subprocess.Popen(
            [str(py), str(WORKER_SCRIPT), "--acestep-dir", str(root)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            encoding="utf-8",
            env=env,
            cwd=str(root),
        )
        self.lines = queue.Queue()
        out = self.proc.stdout

        def pump() -> None:
            assert out is not None
            for line in out:
                self.lines.put(line)
            self.lines.put(None)

        threading.Thread(target=pump, name="chant-worker-stdout", daemon=True).start()
        logger.info("Worker de canto iniciado (pid %s)", self.proc.pid)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
        self.proc = None

    def request(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        with self.lock:
            if not self.alive():
                self._spawn()
            assert self.proc is not None and self.proc.stdin is not None
            self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            deadline = time.monotonic() + timeout
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    self.stop()
                    raise TimeoutError(f"worker de canto passou de {timeout:.0f}s")
                try:
                    line = self.lines.get(timeout=left)
                except queue.Empty:
                    continue
                if line is None:
                    self.stop()
                    raise RuntimeError("worker de canto encerrou (ver data/audio/chant_worker.log)")
                line = line.strip()
                if line.startswith("{"):
                    reply = json.loads(line)
                    if not reply.get("ok"):
                        raise RuntimeError(f"worker de canto: {reply.get('error')}")
                    return reply


_WORKER = _Worker()
_GEN_LOCK = threading.Lock()


def shutdown_worker() -> None:
    _WORKER.stop()


def cached_chant(key: str, lines: list[str], *, title: str = "") -> Path:
    """Devolve o MP3 cantado do verso (cache ou geração local). Levanta em falha."""
    if not lines:
        raise ValueError("verso sem linhas para cantar")
    cached = get_cached_chant(key, lines)
    if cached is not None:
        return cached
    missing = missing_requirements()
    if missing:
        raise ChantUnavailable("; ".join(missing))
    with _GEN_LOCK:
        cached = get_cached_chant(key, lines)
        if cached is not None:
            return cached
        dest = cache_path(key, lines)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part.mp3")
        payload = {
            "lyrics": lyrics_for(lines),
            "duration": chant_duration(lines),
            "out": str(tmp.resolve()),
            "tanpura": str(tanpura_path().resolve()),
            "recipe": RECIPE,
            "title": title or key,
            "comment": f"{MODEL_CREDIT}. Tanpura: {TANPURA_CREDIT}.",
        }
        t0 = time.monotonic()
        reply = _WORKER.request(payload, chant_timeout())
        if not tmp.exists() or tmp.stat().st_size < 1000:
            raise RuntimeError("worker de canto não gravou o MP3")
        tmp.replace(dest)
        logger.info(
            "Canto de %s gerado em %.1fs (%s)",
            key,
            time.monotonic() - t0,
            json.dumps(reply.get("timings", {})),
        )
        return dest


def credits() -> list[str]:
    return [MODEL_CREDIT, f"Tanpura: {TANPURA_CREDIT}"]


def describe() -> dict[str, Any]:
    style = recitation_style()
    missing = missing_requirements() if style == "chant" else []
    return {
        "style": style,
        "chant_available": style == "chant" and not missing,
        "missing": missing,
        "credits": credits() if style == "chant" else [],
    }
