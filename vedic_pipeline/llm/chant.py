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
    "version": 2,
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


BASE_SEEDS = (33, 44, 77)  # 33 = receita aprovada; 44 e 77 também foram bons na rodada v3
STT_URL_PATH = "/stt"
STT_MODEL = "grok-voice-transcribe-2.0"


def chant_takes() -> int:
    """`VEDIC_CHANT_TAKES` (padrão 3, 1–6): quantas tomadas por verso; 1 = sem seleção."""
    raw = (os.environ.get("VEDIC_CHANT_TAKES") or "3").strip()
    try:
        n = int(raw)
    except ValueError:
        logger.warning("VEDIC_CHANT_TAKES=%r inválido; usando 3", raw)
        return 3
    return min(6, max(1, n))


def take_seeds(n: int) -> list[int]:
    seeds = list(BASE_SEEDS[:n])
    while len(seeds) < n:
        seeds.append(101 * len(seeds))
    return seeds


def _digest(lines: list[str]) -> str:
    blob = json.dumps(
        {"lines": lines, "recipe": RECIPE, "seeds": take_seeds(chant_takes())},
        ensure_ascii=False,
        sort_keys=True,
    )
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


def meta_path(dest: Path) -> Path:
    return dest.with_suffix(".json")


def chant_meta(dest: Path) -> dict[str, Any]:
    try:
        return json.loads(meta_path(dest).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_meta(dest: Path, meta: dict[str, Any]) -> None:
    tmp = meta_path(dest).with_suffix(".json.part")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(meta_path(dest))


def _takes_dir(dest: Path) -> Path:
    return dest.parent / "chant_takes" / dest.stem


def _run_take(key: str, lines: list[str], seed: int, dest: Path, title: str) -> dict[str, Any]:
    """Uma tomada no worker: MP3 mixado + voz isolada (antes da tanpura) para o STT."""
    tdir = _takes_dir(dest)
    tdir.mkdir(parents=True, exist_ok=True)
    mp3, vocal = tdir / f"s{seed}.mp3", tdir / f"s{seed}_vocal.mp3"
    payload = {
        "lyrics": lyrics_for(lines),
        "duration": chant_duration(lines),
        "seed": seed,
        "out": str(mp3.resolve()),
        "vocal_out": str(vocal.resolve()),
        "tanpura": str(tanpura_path().resolve()),
        "recipe": RECIPE,
        "title": title or key,
        "comment": f"{MODEL_CREDIT} (seed {seed}). Tanpura: {TANPURA_CREDIT}.",
    }
    t0 = time.monotonic()
    reply = _WORKER.request(payload, chant_timeout())
    if not mp3.exists() or mp3.stat().st_size < 1000:
        raise RuntimeError("worker de canto não gravou o MP3")
    secs = round(time.monotonic() - t0, 1)
    logger.info(
        "Tomada seed=%s de %s em %.1fs (%s)", seed, key, secs, json.dumps(reply.get("timings", {}))
    )
    return {"seed": seed, "mp3": str(mp3), "vocal": str(vocal), "secs": secs}


def _install(src: str | Path, dest: Path) -> None:
    tmp = dest.with_suffix(".part.mp3")
    shutil.copyfile(src, tmp)
    tmp.replace(dest)


def stt_available() -> bool:
    return bool(os.environ.get("XAI_API_KEY"))


def transcribe(path: str | Path) -> str:
    """xAI speech-to-text (REST, US$0,10/h) da voz isolada; hindi = devanāgarī na saída."""
    import httpx

    key = os.environ.get("XAI_API_KEY")
    if not key:
        raise RuntimeError("XAI_API_KEY não definida")
    base = os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1").rstrip("/")
    with open(path, "rb") as fh, httpx.Client(timeout=60.0) as client:
        resp = client.post(
            f"{base}{STT_URL_PATH}",
            headers={"Authorization": f"Bearer {key}"},
            files={"file": ("take.mp3", fh, "audio/mpeg")},
            data={"model": STT_MODEL, "language": "hi"},
        )
    resp.raise_for_status()
    return str(resp.json().get("text") or "")


def _norm_deva(text: str) -> str:
    text = _ACCENT_RE.sub("", text)
    text = re.sub(r"[\s।॥|/,.\-?!…\"'0-9०-९]", "", text)
    for a, b in (("ङ्", "ं"), ("म्", "ं"), ("ँ", "ं"), ("ः", ""), ("्", "")):
        text = text.replace(a, b)
    return text


def _norm_latin(text: str) -> str:
    from vedic_pipeline.common.sanskrit import iast_to_ascii

    if has_devanagari(text):
        text = devanagari_to_iast(text)
    text = iast_to_ascii(text.lower())
    return re.sub(r"[^a-z]", "", text.replace("h", ""))


def _levenshtein(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def cer(reference: str, transcript: str, lines: list[str] | None = None) -> float:
    """CER da transcrição contra o devanāgarī do verso (normalizado: sem pontuação,
    anusvāra/visarga/virāma unificados). Sem devanāgarī de referência, compara em
    latim (transcrição transliterada contra o IAST, sem diacríticos)."""
    if reference and has_devanagari(reference):
        ref, hyp = _norm_deva(reference), _norm_deva(transcript)
    else:
        ref, hyp = _norm_latin(reference or " ".join(lines or [])), _norm_latin(transcript)
    if not ref:
        return 1.0
    return round(min(1.0, _levenshtein(ref, hyp) / len(ref)), 3)


_UPGRADES: set[str] = set()
_UPGRADES_LOCK = threading.Lock()


def _schedule_upgrade(key: str, lines: list[str], reference: str, title: str) -> bool:
    dest = cache_path(key, lines)
    with _UPGRADES_LOCK:
        if str(dest) in _UPGRADES:
            return False
        _UPGRADES.add(str(dest))

    def run() -> None:
        try:
            upgrade(key, lines, reference=reference, title=title)
        except Exception:  # noqa: BLE001
            logger.exception("Seleção de tomadas falhou para %s; fica a primeira", key)
        finally:
            with _UPGRADES_LOCK:
                _UPGRADES.discard(str(dest))

    threading.Thread(target=run, name=f"chant-upgrade-{key}", daemon=True).start()
    return True


def upgrade(key: str, lines: list[str], *, reference: str = "", title: str = "") -> dict[str, Any]:
    """Gera as tomadas que faltam, pontua todas pelo CER do STT e instala a melhor.

    STT ausente ou com erro: fica a primeira tomada bem-sucedida (a já servida)."""
    dest = cache_path(key, lines)
    meta = chant_meta(dest)
    if meta.get("status") != "provisional" or not dest.exists():
        return meta
    takes = [t for t in meta.get("takes", []) if Path(t.get("mp3", "")).exists()]
    done = {t["seed"] for t in takes}
    for seed in take_seeds(int(meta.get("n_takes") or chant_takes())):
        if seed in done:
            continue
        try:
            takes.append(_run_take(key, lines, seed, dest, title))
        except Exception:  # noqa: BLE001
            logger.exception("Tomada seed=%s de %s falhou; seguindo sem ela", seed, key)
    chosen = takes[0] if takes else None
    stt_error = None
    if chosen and len(takes) > 1:
        if not stt_available():
            stt_error = "XAI_API_KEY ausente"
        else:
            try:
                for t in takes:
                    t["transcript"] = transcribe(t["vocal"])
                    t["cer"] = cer(reference, t["transcript"], lines)
                chosen = min(takes, key=lambda t: (t["cer"], takes.index(t)))
            except Exception as exc:  # noqa: BLE001
                logger.warning("STT falhou para %s (%s); fica a primeira tomada", key, exc)
                stt_error = f"{type(exc).__name__}: {exc}"[:200]
                chosen = takes[0]
    if chosen and Path(chosen["mp3"]).resolve() != dest.resolve():
        _install(chosen["mp3"], dest)
    meta.update(
        {
            "status": "final",
            "chosen_seed": chosen["seed"] if chosen else meta.get("chosen_seed"),
            "chosen_cer": chosen.get("cer") if chosen else None,
            "stt_error": stt_error,
            "takes": [{k: t.get(k) for k in ("seed", "cer", "transcript", "secs")} for t in takes],
        }
    )
    _write_meta(dest, meta)
    shutil.rmtree(_takes_dir(dest), ignore_errors=True)
    logger.info("Canto de %s: seed %s escolhida (%s)", key, meta["chosen_seed"], meta["takes"])
    return meta


def resume_upgrade(key: str, lines: list[str], *, reference: str = "", title: str = "") -> bool:
    """Retoma a seleção de um cache provisório (ex.: o app recarregou no meio)."""
    dest = cache_path(key, lines)
    if (
        chant_takes() < 2
        or chant_meta(dest).get("status") != "provisional"
        or not chant_available()
    ):
        return False
    return _schedule_upgrade(key, lines, reference, title)


def cached_chant(key: str, lines: list[str], *, title: str = "", reference: str = "") -> Path:
    """MP3 cantado do verso (cache ou geração local). Levanta em falha.

    Com `VEDIC_CHANT_TAKES` > 1 a primeira tomada (seed 33) é servida já, como
    cache provisório, e as outras são geradas e pontuadas em segundo plano; a
    melhor substitui o arquivo (o primeiro pedido fica abaixo dos ~100 s do túnel)."""
    if not lines:
        raise ValueError("verso sem linhas para cantar")
    cached = get_cached_chant(key, lines)
    if cached is not None:
        return cached
    missing = missing_requirements()
    if missing:
        raise ChantUnavailable("; ".join(missing))
    n = chant_takes()
    with _GEN_LOCK:
        cached = get_cached_chant(key, lines)
        if cached is not None:
            return cached
        dest = cache_path(key, lines)
        dest.parent.mkdir(parents=True, exist_ok=True)
        first = _run_take(key, lines, take_seeds(n)[0], dest, title)
        _install(first["mp3"], dest)
        meta = {
            "key": key,
            "status": "provisional" if n > 1 else "final",
            "n_takes": n,
            "chosen_seed": first["seed"],
            "chosen_cer": None,
            "takes": [first],
            "recipe_version": RECIPE["version"],
        }
        _write_meta(dest, meta)
        if n == 1:
            shutil.rmtree(_takes_dir(dest), ignore_errors=True)
    if n > 1:
        _schedule_upgrade(key, lines, reference, title)
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
