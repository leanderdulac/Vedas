"""Coletor privado do Śrīmad-Bhāgavatam (Vedabase, pt-br).

Conteúdo © Bhaktivedanta Book Trust (BBT). Uso privado, com permissão pendente:
o resultado fica só em ``data/raw/vedabase_sb_ptbr/`` (gitignored) e só entra no
índice quando ``VEDIC_ENABLE_LICENSED_SOURCES`` estiver ligado.

Recolhe APENAS, por verso: Devanāgarī, transliteração e tradução em português.
Sinônimos palavra-a-palavra, significados (purports), prefácios e introduções são
descartados ANTES de gravar em disco: o cache guarda só os fragmentos dos três
campos, não a página inteira.

Rastreamento educado: respeita robots.txt (incluindo Crawl-delay), 1 requisição
por vez, User-Agent descritivo, retentativas com backoff, retomada pelo cache.

Uso:
    python -m vedic_pipeline.etl.vedabase_sb crawl --out data/raw/vedabase_sb_ptbr
    python -m vedic_pipeline.etl.vedabase_sb build --out data/raw/vedabase_sb_ptbr  # só remonta o JSONL
"""
from __future__ import annotations

import argparse
import contextlib
import html as htmllib
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
import urllib.robotparser
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://vedabase.io"
ROOT_PATH = "/pt-br/library/sb/"
USER_AGENT = (
    "VedasPrivateResearch/1.0 (private study corpus; 1 request at a time; "
    "honours robots.txt Crawl-delay; owner: Leandro Franca de Mello)"
)
DEFAULT_DELAY = 10.0  # robots.txt de vedabase.io: Crawl-delay: 10
MAX_CANTO = 12
LICENCE_NOTE = "BBT — permission pending, private use"

_NUM_SEG = re.compile(r"^\d+(?:-\d+)?$")


# --------------------------------------------------------------------------- parse
def _balanced_div(doc: str, class_name: str) -> str | None:
    """HTML interno do primeiro <div class="class_name"> (com divs aninhados)."""
    m = re.search(rf'<div[^>]*class="{re.escape(class_name)}"[^>]*>', doc)
    if not m:
        return None
    depth, i = 1, m.end()
    tag = re.compile(r"<(/?)div\b[^>]*>", re.I)
    while depth:
        t = tag.search(doc, i)
        if not t:
            return None
        depth += -1 if t.group(1) else 1
        i = t.end()
        if depth == 0:
            return doc[m.end():t.start()]
    return None


def _to_text(fragment: str | None) -> str:
    if not fragment:
        return ""
    s = re.sub(r"<h2\b.*?</h2>", "", fragment, flags=re.S | re.I)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"</(?:p|div)>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = htmllib.unescape(s).replace("\u00a0", " ")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in s.split("\n")]
    out: list[str] = []
    for ln in lines:
        if ln or (out and out[-1]):
            out.append(ln)
    return "\n".join(out).strip()


def extract_fragments(page_html: str) -> dict[str, str | None]:
    """Só os fragmentos permitidos: título, devanāgarī, transliteração, tradução."""
    h1 = re.search(r"<h1\b[^>]*>(.*?)</h1>", page_html, re.S)
    return {
        "title": h1.group(1) if h1 else None,
        "devanagari": _balanced_div(page_html, "av-devanagari"),
        "verse_text": _balanced_div(page_html, "av-verse_text"),
        "translation": _balanced_div(page_html, "av-translation"),
    }


def fragments_to_cache_html(frags: dict[str, str | None]) -> str:
    parts = []
    for key in ("title", "devanagari", "verse_text", "translation"):
        val = frags.get(key)
        if val is not None:
            parts.append(f'<section data-field="{key}">{val}</section>')
    return "\n".join(parts) + "\n"


def cache_html_to_fragments(cached: str) -> dict[str, str | None]:
    out: dict[str, str | None] = {k: None for k in ("title", "devanagari", "verse_text", "translation")}
    for m in re.finditer(r'<section data-field="(\w+)">(.*?)</section>', cached, re.S):
        out[m.group(1)] = m.group(2)
    return out


def parse_locator(path: str) -> tuple[int, int, str, int, int]:
    segs = [s for s in path.strip("/").split("/") if s]
    canto, chapter, verse = int(segs[-3]), int(segs[-2]), segs[-1]
    a, _, b = verse.partition("-")
    return canto, chapter, verse, int(a), int(b or a)


def record_from_fragments(path: str, frags: dict[str, str | None]) -> dict:
    canto, chapter, verse, v0, v1 = parse_locator(path)
    rec = {
        "canto": canto,
        "chapter": chapter,
        "verse": verse,
        "verse_start": v0,
        "verse_end": v1,
        "locator": f"SB {canto}.{chapter}.{verse}",
        "devanagari": _to_text(frags.get("devanagari")),
        "transliteration": _to_text(frags.get("verse_text")),
        "translation_pt": _to_text(frags.get("translation")),
        "url": BASE + path,
        "licence": LICENCE_NOTE,
    }
    missing = [k for k in ("devanagari", "transliteration", "translation_pt") if not rec[k]]
    if missing:
        rec["missing"] = missing
    return rec


def child_links(page_html: str, parent_path: str) -> list[str]:
    """Links diretos de 1 nível abaixo de parent_path cujo segmento é numérico."""
    out = set()
    for href in re.findall(r'href="([^"#?]+)"', page_html):
        if not href.startswith(parent_path) or href == parent_path:
            continue
        rest = href[len(parent_path):].strip("/")
        if "/" in rest or not _NUM_SEG.match(rest):
            continue
        out.add(parent_path + rest + "/")

    def key(p: str):
        seg = p.strip("/").split("/")[-1]
        return int(seg.split("-")[0])

    return sorted(out, key=key)


# --------------------------------------------------------------------------- crawl
class Fetcher:
    def __init__(self, delay: float, log):
        self.delay = delay
        self.last = 0.0
        self.log = log
        self.requests = 0

    def get(self, path: str) -> tuple[int, str]:
        backoff = 30.0
        for attempt in range(6):
            wait = self.last + self.delay + random.uniform(0, 1.0) - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()
            self.requests += 1
            req = urllib.request.Request(BASE + path, headers={
                "User-Agent": USER_AGENT, "Accept-Language": "pt-BR,pt;q=0.9"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    return r.status, r.read().decode("utf-8", "replace")
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return 404, ""
                retry_after = e.headers.get("Retry-After") if e.headers else None
                if e.code in (429, 500, 502, 503, 504):
                    pause = float(retry_after) if retry_after and retry_after.isdigit() else backoff
                    self.log(f"HTTP {e.code} em {path}; tentativa {attempt + 1}, pausa {pause:.0f}s")
                    time.sleep(pause)
                    backoff = min(backoff * 2, 900)
                    continue
                return e.code, ""
            except Exception as e:  # rede
                self.log(f"erro de rede em {path}: {e!r}; tentativa {attempt + 1}, pausa {backoff:.0f}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 900)
        return -1, ""


def check_robots(paths: list[str]) -> tuple[bool, float]:
    rp = urllib.robotparser.RobotFileParser(BASE + "/robots.txt")
    req = urllib.request.Request(BASE + "/robots.txt", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        rp.parse(r.read().decode("utf-8", "replace").splitlines())
    ok = all(rp.can_fetch(USER_AGENT, BASE + p) for p in paths)
    delay = rp.crawl_delay(USER_AGENT) or DEFAULT_DELAY
    return ok, float(delay)


def build_jsonl(out: Path) -> dict:
    cache = out / "cache"
    records = []
    for f in cache.glob("sb/*/*/*.html"):
        rel = "/pt-br/library/" + str(f.relative_to(cache).with_suffix("")) + "/"
        frags = cache_html_to_fragments(f.read_text("utf-8"))
        records.append(record_from_fragments(rel, frags))
    records.sort(key=lambda r: (r["canto"], r["chapter"], r["verse_start"]))
    tmp = out / "sb_ptbr.jsonl.tmp"
    with tmp.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(out / "sb_ptbr.jsonl")
    per_canto: dict[str, int] = {}
    for r in records:
        per_canto[str(r["canto"])] = per_canto.get(str(r["canto"]), 0) + 1
    return {"records": len(records), "per_canto": per_canto,
            "incomplete": sum(1 for r in records if r.get("missing"))}


def crawl(out: Path, delay_override: float | None, cantos: list[int] | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "cache").mkdir(exist_ok=True)
    logf = (out / "crawl.log").open("a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        logf.write(line + "\n")
        logf.flush()
        print(line, flush=True)

    ok, robots_delay = check_robots([ROOT_PATH, ROOT_PATH + "1/1/1/"])
    if not ok:
        log("robots.txt proíbe o caminho; abortando")
        sys.exit(2)
    delay = max(robots_delay, delay_override or 0.0)
    log(f"robots.txt ok; Crawl-delay={robots_delay}s; usando {delay}s entre requisições")
    f = Fetcher(delay, log)

    gaps_path = out / "gaps.jsonl"
    known_gaps = set()
    if gaps_path.exists():
        for ln in gaps_path.read_text("utf-8").splitlines():
            with contextlib.suppress(Exception):
                known_gaps.add(json.loads(ln)["path"])

    def gap(path: str, reason: str) -> None:
        if path in known_gaps:
            return
        known_gaps.add(path)
        with gaps_path.open("a", encoding="utf-8") as g:
            g.write(json.dumps({"path": path, "url": BASE + path, "reason": reason,
                                "at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + "\n")
        log(f"lacuna: {path} ({reason})")

    def links(path: str, cache_name: str) -> list[str] | None:
        cf = out / "cache" / "index" / f"{cache_name}.json"
        if cf.exists():
            return json.loads(cf.read_text("utf-8"))
        status, body = f.get(path)
        if status != 200:
            gap(path, f"HTTP {status}")
            return None
        found = child_links(body, path)
        cf.parent.mkdir(parents=True, exist_ok=True)
        cf.write_text(json.dumps(found), "utf-8")
        return found

    started = time.time()
    state = {"started_at": datetime.now().isoformat(timespec="seconds"), "delay_s": delay}
    canto_list = cantos or list(range(1, MAX_CANTO + 1))
    # descobre todos os capítulos primeiro (para ETA)
    chapters: list[tuple[int, str]] = []
    for c in canto_list:
        ch = links(f"{ROOT_PATH}{c}/", f"canto_{c}")
        if ch is None:
            continue
        chapters += [(c, p) for p in ch]
    log(f"{len(chapters)} capítulos nos cantos {canto_list}")

    fetched_verses = 0
    for c, chpath in chapters:
        cname = "_".join(chpath.strip("/").split("/")[-2:])
        verses = links(chpath, f"chapter_{cname}")
        if verses is None:
            continue
        new_in_chapter = 0
        for vpath in verses:
            rel = vpath[len("/pt-br/library/"):].strip("/")
            cf = out / "cache" / (rel + ".html")
            if cf.exists() or vpath in known_gaps:
                continue
            status, body = f.get(vpath)
            if status != 200:
                gap(vpath, f"HTTP {status}")
                continue
            frags = extract_fragments(body)
            cf.parent.mkdir(parents=True, exist_ok=True)
            cf.write_text(fragments_to_cache_html(frags), "utf-8")
            rec = record_from_fragments(vpath, frags)
            if rec.get("missing"):
                gap(vpath, "sem " + ", ".join(rec["missing"]))
            fetched_verses += 1
            new_in_chapter += 1
        if new_in_chapter:
            summary = build_jsonl(out)
            elapsed = time.time() - started
            # ETA pelos capítulos que faltam (média de versos/capítulo até aqui)
            idx = chapters.index((c, chpath)) + 1
            remaining_ch = len(chapters) - idx
            avg_v = max(1.0, summary["records"] / max(1, idx))
            eta_h = remaining_ch * avg_v * (delay + 0.7) / 3600
            state.update({"last_chapter": chpath, "chapters_done": idx, "chapters_total": len(chapters),
                          "fetched_this_run": fetched_verses, "requests_this_run": f.requests,
                          "elapsed_h": round(elapsed / 3600, 2), "eta_h": round(eta_h, 1),
                          "updated_at": datetime.now().isoformat(timespec="seconds"), **summary})
            (out / "progress.json").write_text(json.dumps(state, ensure_ascii=False, indent=1), "utf-8")
            log(f"capítulo {chpath} ok (+{new_in_chapter}); total {summary['records']} versos; ETA ~{eta_h:.1f} h")
    summary = build_jsonl(out)
    state.update({"finished_at": datetime.now().isoformat(timespec="seconds"),
                  "elapsed_h": round((time.time() - started) / 3600, 2), "done": True, **summary})
    (out / "progress.json").write_text(json.dumps(state, ensure_ascii=False, indent=1), "utf-8")
    log(f"FIM: {summary}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["crawl", "build"])
    ap.add_argument("--out", default="data/raw/vedabase_sb_ptbr")
    ap.add_argument("--delay", type=float, default=None, help="mínimo entre requisições (nunca abaixo do Crawl-delay)")
    ap.add_argument("--cantos", default=None, help="ex.: 1,2,3")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.cmd == "build":
        print(json.dumps(build_jsonl(out), ensure_ascii=False))
        return
    cantos = [int(x) for x in a.cantos.split(",")] if a.cantos else None
    crawl(out, a.delay, cantos)


if __name__ == "__main__":
    main()
