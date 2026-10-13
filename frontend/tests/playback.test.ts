import assert from "node:assert/strict";
import { beforeEach, test } from "node:test";
import type { AudioLike } from "../src/data/playback.ts";
import {

  hasActivePlayback,
  playVerseAudio,
  speakExclusive,
  stopAllPlayback,
} from "../src/data/playback.ts";

type Listener = ((ev?: unknown) => unknown) | null;

class FakeAudio implements AudioLike {
  onplaying: Listener = null;
  onended: Listener = null;
  onerror: Listener = null;
  src: string;
  paused = true;
  private resolvePlay!: () => void;
  private rejectPlay!: (e: Error) => void;
  constructor(url: string) {
    this.src = url;
  }
  play(): Promise<void> {
    this.paused = false;
    return new Promise((res, rej) => {
      this.resolvePlay = res;
      this.rejectPlay = rej;
    });
  }
  pause() {
    this.paused = true;
  }
  removeAttribute(name: string) {
    if (name === "src") this.src = "";
  }
  load() {
    // Navegador real: limpar o src de um elemento ainda com handler dispara `error`.
    if (!this.src) this.onerror?.();
  }
  // helpers do teste
  startPlaying() {
    this.resolvePlay();
    this.onplaying?.();
  }
  fail() {
    this.onerror?.();
    this.rejectPlay(new Error("NotSupportedError"));
  }
  end() {
    this.paused = true;
    this.onended?.();
  }
}

function harness() {
  const audios: FakeAudio[] = [];
  const speech = { active: 0, started: 0, stops: 0, pending: [] as Array<() => void> };
  const deps = {
    createAudio: (url: string) => {
      const a = new FakeAudio(url);
      audios.push(a);
      return a;
    },
    speakFallback: (onEnd: () => void) => {
      speech.active += 1;
      speech.started += 1;
      speech.pending.push(onEnd);
    },
    stopSpeech: () => {
      speech.stops += 1;
      speech.active = 0;
    },
  };
  const playingAudios = () => audios.filter((a) => !a.paused).length;
  return { audios, speech, deps, playingAudios };
}

const tick = () => new Promise((r) => setTimeout(r, 0));

beforeEach(() => stopAllPlayback());

test("replaying a verse never starts the browser voice over the MP3", async () => {
  const h = harness();
  const notices: string[] = [];
  playVerseAudio("/a/RV.1.1.5", h.deps, { onFallback: () => notices.push("fallback") });
  h.audios[0].startPlaying();
  await tick();
  h.audios[0].end();
  // Segundo clique em ▶ Ouvir: o elemento antigo é limpo e não pode cair na voz sintetizada.
  playVerseAudio("/a/RV.1.1.5", h.deps, { onFallback: () => notices.push("fallback") });
  h.audios[1].startPlaying();
  await tick();
  assert.equal(h.speech.started, 0);
  assert.deepEqual(notices, []);
  assert.equal(h.playingAudios(), 1);
});

test("starting another verse stops the previous one (one voice at a time)", async () => {
  const h = harness();
  let idleA = 0;
  playVerseAudio("/a/RV.1.1.4", h.deps, { onIdle: () => (idleA += 1) });
  h.audios[0].startPlaying();
  await tick();
  playVerseAudio("/a/RV.1.1.5", h.deps);
  h.audios[1].startPlaying();
  await tick();
  assert.equal(h.audios[0].paused, true);
  assert.equal(h.audios[0].src, "");
  assert.equal(h.playingAudios(), 1);
  assert.equal(idleA, 1);
  assert.equal(h.speech.started, 0);
});

test("stop while the MP3 is still loading does not trigger the fallback voice", async () => {
  const h = harness();
  const handle = playVerseAudio("/a/RV.1.1.5", h.deps);
  handle.stop();
  h.audios[0].fail(); // play() rejeita com AbortError depois do stop
  await tick();
  assert.equal(h.speech.started, 0);
  assert.equal(hasActivePlayback(), false);
});

test("browser voice is only a fallback when the backend audio fails", async () => {
  const h = harness();
  let fallbacks = 0;
  let idle = 0;
  playVerseAudio("/a/RV.1.1.5", h.deps, { onFallback: () => (fallbacks += 1), onIdle: () => (idle += 1) });
  h.audios[0].fail();
  await tick();
  assert.equal(fallbacks, 1);
  assert.equal(h.speech.started, 1);
  assert.equal(h.playingAudios(), 0);
  h.speech.pending[0]();
  assert.equal(idle, 1);
  assert.equal(hasActivePlayback(), false);
});

test("error after playback started ends cleanly without switching voices", async () => {
  const h = harness();
  playVerseAudio("/a/RV.1.1.5", h.deps);
  h.audios[0].startPlaying();
  await tick();
  h.audios[0].onerror?.();
  assert.equal(h.speech.started, 0);
  assert.equal(hasActivePlayback(), false);
});

test("MP3 playback silences a browser voice already speaking", async () => {
  const h = harness();
  let idle = 0;
  speakExclusive(() => {}, h.deps.stopSpeech, { onIdle: () => (idle += 1) });
  const stopsBefore = h.speech.stops;
  playVerseAudio("/a/RV.1.1.5", h.deps);
  assert.equal(idle, 1);
  assert.ok(h.speech.stops > stopsBefore);
});

test("stopping a stale handle does not stop the current playback", async () => {
  const h = harness();
  const first = playVerseAudio("/a/RV.1.1.4", h.deps);
  playVerseAudio("/a/RV.1.1.5", h.deps);
  h.audios[1].startPlaying();
  await tick();
  first.stop();
  assert.equal(h.audios[1].paused, false);
  assert.equal(hasActivePlayback(), true);
});
