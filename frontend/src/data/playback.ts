// Uma voz por vez no app inteiro.
//
// O bug da "voz dupla": parar um <audio> com `src = ""` dispara o evento
// `error` no elemento antigo. O handler antigo achava que o MP3 tinha falhado
// e ligava a voz do navegador (speechSynthesis) por cima do MP3 novo, com o
// aviso "Usando voz sintetizada". Aqui cada reprodução é uma sessão: quem
// começa encerra a anterior, os handlers são soltos antes de limpar o
// elemento e eventos de uma sessão encerrada são ignorados.

export interface AudioLike {
  onplaying: ((this: any, ev: any) => any) | null;
  onended: ((this: any, ev: any) => any) | null;
  onerror: ((this: any, ev: any) => any) | null;
  play(): Promise<void>;
  pause(): void;
  removeAttribute(name: string): void;
  load(): void;
}

export interface PlaybackDeps {
  createAudio: (url: string) => AudioLike;
  /** Voz do navegador, só quando o MP3 do backend falha antes de tocar. */
  speakFallback: (onEnd: () => void) => void;
  stopSpeech: () => void;
}

export interface PlaybackCallbacks {
  onPlaying?: () => void;
  onFallback?: () => void;
  /** Chamado uma única vez quando a sessão termina, por qualquer motivo. */
  onIdle?: () => void;
}

export interface PlaybackHandle {
  stop: () => void;
  isActive: () => boolean;
}

type Session = { id: number; stop: () => void };

let current: Session | null = null;
let seq = 0;

/** Encerra a reprodução ativa (MP3 ou voz do navegador), se houver. */
export function stopAllPlayback(): void {
  const prev = current;
  current = null;
  prev?.stop();
}

export function hasActivePlayback(): boolean {
  return current !== null;
}

function claim(stop: () => void): Session {
  stopAllPlayback();
  seq += 1;
  const session = { id: seq, stop };
  current = session;
  return session;
}

function release(session: Session): void {
  if (current?.id === session.id) current = null;
}

function isCurrent(session: Session): boolean {
  return current?.id === session.id;
}

function silence(audio: AudioLike): void {
  audio.onplaying = null;
  audio.onended = null;
  audio.onerror = null;
  try {
    audio.pause();
    // `removeAttribute` + `load()` descarrega sem disparar `error`;
    // `src = ""` dispararia (e foi a origem da voz dupla).
    audio.removeAttribute("src");
    audio.load();
  } catch {
    /* ignore */
  }
}

/**
 * Toca o MP3 do verso. A voz do navegador só entra se o MP3 falhar antes de
 * começar a tocar, e nunca junto com ele.
 */
export function playVerseAudio(url: string, deps: PlaybackDeps, cb: PlaybackCallbacks = {}): PlaybackHandle {
  let audio: AudioLike | null = null;
  let started = false;
  let fellBack = false;
  let finished = false;

  const teardown = () => {
    if (audio) {
      const a = audio;
      audio = null;
      silence(a);
    }
  };

  const finish = () => {
    if (finished) return;
    finished = true;
    teardown();
    if (fellBack) deps.stopSpeech();
    release(session);
    cb.onIdle?.();
  };

  const session = claim(finish);
  // Qualquer fala pendente (eco, palavra da análise) para antes do MP3.
  deps.stopSpeech();

  const fallback = () => {
    if (!isCurrent(session) || finished || started || fellBack) return;
    fellBack = true;
    teardown();
    cb.onFallback?.();
    deps.speakFallback(() => {
      if (isCurrent(session)) finish();
    });
  };

  const a = deps.createAudio(url);
  audio = a;
  a.onplaying = () => {
    if (!isCurrent(session) || fellBack) return;
    started = true;
    cb.onPlaying?.();
  };
  a.onended = () => {
    if (isCurrent(session)) finish();
  };
  a.onerror = () => {
    if (!isCurrent(session)) return;
    // Erro depois de começar a tocar: encerra, sem trocar de voz no meio.
    if (started) finish();
    else fallback();
  };

  let playPromise: Promise<void> | undefined;
  try {
    playPromise = a.play();
  } catch {
    fallback();
  }
  if (playPromise && typeof playPromise.then === "function") {
    playPromise.then(
      () => {
        if (!isCurrent(session) || fellBack || started) return;
        started = true;
        cb.onPlaying?.();
      },
      () => fallback(),
    );
  }

  return {
    stop: () => {
      if (isCurrent(session)) stopAllPlayback();
    },
    isActive: () => isCurrent(session),
  };
}

/** Voz do navegador como reprodução exclusiva (tradução, palavra, eco). */
export function speakExclusive(
  speak: (onEnd: () => void) => void,
  stopSpeech: () => void,
  cb: { onIdle?: () => void } = {},
): PlaybackHandle {
  let finished = false;
  const finish = () => {
    if (finished) return;
    finished = true;
    stopSpeech();
    release(session);
    cb.onIdle?.();
  };
  const session = claim(finish);
  speak(() => {
    if (isCurrent(session)) finish();
  });
  return {
    stop: () => {
      if (isCurrent(session)) stopAllPlayback();
    },
    isActive: () => isCurrent(session),
  };
}
