import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, VerseBundle, VerseTranslation, VerseUnit } from "../api/client";

function speakBrowser(text: string, lang: string) {
  if (typeof window === "undefined" || !window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(text);
  utter.lang = lang;
  utter.rate = 0.85;
  window.speechSynthesis.speak(utter);
}

function revoke(url: string | null) {
  if (url && url.startsWith("blob:")) {
    try {
      URL.revokeObjectURL(url);
    } catch {
      /* ignore */
    }
  }
}

export default function VerseCard({
  unit,
  isTarget,
  highlightRef,
  hasImage,
  hasVideo,
}: {
  unit: VerseUnit;
  isTarget: boolean;
  highlightRef?: (el: HTMLElement | null) => void;
  hasImage?: boolean;
  hasVideo?: boolean;
}) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [busy, setBusy] = useState<"audio" | "pt" | "en" | "trad" | "image" | "video" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [bundle, setBundle] = useState<VerseBundle | null>(null);
  const [explanation, setExplanation] = useState<{ lang: "pt" | "en"; text: string } | null>(null);
  const [translation, setTranslation] = useState<VerseTranslation | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoNote, setVideoNote] = useState<string | null>(null);
  const cancelled = useRef(false);

  useEffect(() => {
    cancelled.current = false;
    return () => {
      cancelled.current = true;
    };
  }, []);

  useEffect(() => {
    if (hasImage) setImageUrl(api.verseImageUrl(unit.verse_id));
    if (hasVideo) setVideoUrl(api.verseVideoFileUrl(unit.verse_id));
  }, [hasImage, hasVideo, unit.verse_id]);

  useEffect(() => {
    return () => {
      revoke(imageUrl);
      stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function ensureBundle() {
    if (bundle) return bundle;
    const next = await api.verse(unit.verse_id);
    setBundle(next);
    return next;
  }

  async function play() {
    setError(null);
    setBusy("audio");
    try {
      await ensureBundle().catch(() => undefined);
      const url = api.verseAudioUrl(unit.verse_id);
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 30000);
      const res = await fetch(url, { signal: ctrl.signal });
      clearTimeout(timer);
      if (res.ok) {
        const blob = await res.blob();
        const src = URL.createObjectURL(blob);
        if (audioRef.current) {
          audioRef.current.pause();
        }
        const prev = (audioRef.current as HTMLAudioElement & { __src?: string } | null)?.__src;
        revoke(prev ?? null);
        const audio = new Audio(src);
        (audio as HTMLAudioElement & { __src?: string }).__src = src;
        audioRef.current = audio;
        audio.onended = () => setPlaying(false);
        await audio.play();
        setPlaying(true);
        return;
      }
      speakBrowser(unit.text, unit.verse_id.startsWith("BG") || /[\u0900-\u097F]/.test(unit.text) ? "hi-IN" : "en-US");
      setPlaying(true);
    } catch (e) {
      speakBrowser(unit.text, "hi-IN");
      setPlaying(true);
      setError(e instanceof Error ? e.message : "Áudio do navegador");
    } finally {
      setBusy(null);
    }
  }

  function stop() {
    audioRef.current?.pause();
    if (typeof window !== "undefined") window.speechSynthesis?.cancel();
    setPlaying(false);
  }

  function speakTranslation() {
    if (!translation?.translation) return;
    setError(null);
    stop();
    speakBrowser(translation.translation, "pt-BR");
    setPlaying(true);
    const done = () => setPlaying(false);
    if (typeof window !== "undefined" && window.speechSynthesis) {
      // utterance.onend dispara só se o texto ainda existir; fallback por tempo.
      setTimeout(done, Math.max(4000, translation.translation.length * 90));
    }
  }

  async function illustrate() {
    setError(null);
    setBusy("image");
    try {
      await ensureBundle();
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 60000);
      const res = await fetch(api.verseImageUrl(unit.verse_id), { signal: ctrl.signal });
      clearTimeout(timer);
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(typeof body.detail === "string" ? body.detail : res.statusText);
      }
      const blob = await res.blob();
      setImageUrl((prev) => {
        revoke(prev);
        return URL.createObjectURL(blob);
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Falha ao ilustrar");
    } finally {
      setBusy(null);
    }
  }

  async function animate() {
    setError(null);
    setBusy("video");
    setVideoNote("Gerando still e pedindo o vídeo…");
    try {
      if (!imageUrl) {
        const res = await fetch(api.verseImageUrl(unit.verse_id));
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(typeof body.detail === "string" ? body.detail : "Não foi possível gerar o still");
        }
        const blob = await res.blob();
        if (cancelled.current) return;
        setImageUrl((prev) => {
          revoke(prev);
          return URL.createObjectURL(blob);
        });
      }
      await api.startVerseVideo(unit.verse_id);
      for (let i = 0; i < 40; i += 1) {
        if (cancelled.current) return;
        const st = await api.verseVideoStatus(unit.verse_id);
        if (st.ready || st.status === "done") {
          if (!cancelled.current) {
            setVideoUrl(api.verseVideoFileUrl(unit.verse_id));
            setVideoNote(null);
          }
          return;
        }
        if (st.status === "failed" || st.status === "expired" || st.status === "error") {
          throw new Error(`Vídeo ${st.status}`);
        }
        if (!cancelled.current) setVideoNote(`Vídeo ${st.status || "pendente"}…`);
        await new Promise((r) => setTimeout(r, 4000));
      }
      throw new Error("Tempo esgotado à espera do vídeo");
    } catch (e) {
      if (!cancelled.current) {
        setError(e instanceof Error ? e.message : "Falha no vídeo");
        setVideoNote(null);
      }
    } finally {
      if (!cancelled.current) setBusy(null);
    }
  }

  async function explain(lang: "pt" | "en") {
    setError(null);
    setBusy(lang);
    try {
      await ensureBundle();
      const res = await api.explainVerse(unit.verse_id, lang, "auto");
      setExplanation({ lang, text: res.explanation });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Falha ao explicar");
    } finally {
      setBusy(null);
    }
  }

  async function translate() {
    setError(null);
    setBusy("trad");
    try {
      await ensureBundle();
      const res = await api.translateVerse(unit.verse_id, "pt", "auto");
      setTranslation(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Falha ao traduzir");
    } finally {
      setBusy(null);
    }
  }

  const aligned = (bundle?.witnesses || []).filter((w) => w.role !== "sa" && w.text && w.text !== unit.text);

  return (
    <section
      id={unit.verse_id}
      className={`verse-block${isTarget ? " is-target" : ""}`}
      ref={highlightRef}
    >
      <div className="verse-meta">
        <a className="verse-locator" href={`#${encodeURIComponent(unit.verse_id)}`}>
          {unit.locator}
        </a>
        <div className="verse-actions">
          <button type="button" className="btn btn-ghost verse-btn" onClick={playing ? stop : play} disabled={busy === "audio"}>
            {playing ? "■ Parar" : busy === "audio" ? "…" : "▶ Ouvir"}
          </button>
          <button type="button" className="btn btn-ghost verse-btn" onClick={() => explain("pt")} disabled={busy === "pt"}>
            {busy === "pt" ? "…" : "PT"}
          </button>
          <button type="button" className="btn btn-ghost verse-btn" onClick={() => explain("en")} disabled={busy === "en"}>
            {busy === "en" ? "…" : "EN"}
          </button>
          <button type="button" className="btn btn-ghost verse-btn" onClick={() => translate()} disabled={busy === "trad"} title="Traduzir o verso para português">
            {busy === "trad" ? "…" : "Traduzir"}
          </button>
          <button type="button" className="btn btn-ghost verse-btn" onClick={() => illustrate()} disabled={busy === "image"}>
            {busy === "image" ? "…" : "Imagem"}
          </button>
          <button type="button" className="btn btn-ghost verse-btn" onClick={() => animate()} disabled={busy === "video"}>
            {busy === "video" ? "…" : "Vídeo"}
          </button>
          <Link
            className="btn btn-ghost verse-btn"
            to={`/perguntar?verse=${encodeURIComponent(unit.verse_id)}&q=${encodeURIComponent(`Explique o verso ${unit.locator} e converse comigo sobre o que ele narra.`)}`}
          >
            Conversar
          </Link>
        </div>
      </div>
      <p className="verse-text">{unit.text}</p>
      {aligned.length > 0 && (
        <div className="verse-witnesses">
          {aligned.map((w) => (
            <p key={w.role} className={`verse-witness verse-witness-${w.role}`}>
              <span className="verse-witness-label">{w.role}</span>
              {w.text}
            </p>
          ))}
        </div>
      )}
      {explanation && (
        <div className="verse-explain">
          <div className="verse-explain-lang">{explanation.lang === "pt" ? "Explicação" : "Explanation"}</div>
          <p>{explanation.text}</p>
        </div>
      )}
      {translation && (
        <div className="verse-explain verse-translation">
          <div className="verse-explain-lang">
            Tradução ({translation.source_role === "sa" ? "do sânscrito" : translation.source_role})
            {translation.cached ? " · em cache" : ""}
          </div>
          {translation.translation && <p>{translation.translation}</p>}
          {translation.translation && (
            <button
              type="button"
              className="btn btn-ghost verse-btn"
              onClick={speakTranslation}
              disabled={playing}
              title="Ouvir a tradução em português"
            >
              {playing ? "■ Parar" : "▶ Ouvir tradução"}
            </button>
          )}
          {!translation.translation && translation.note && <p className="muted">{translation.note}</p>}
          {translation.references && translation.references.length > 0 && (
            <div className="verse-witnesses">
              {translation.references.map((w, i) => (
                <p key={`${w.role}-${i}`} className={`verse-witness verse-witness-${w.role}`}>
                  <span className="verse-witness-label">{w.role}</span>
                  {w.text}
                </p>
              ))}
            </div>
          )}
        </div>
      )}
      {imageUrl && (
        <figure className="verse-media">
          <img src={imageUrl} alt={`Ilustração de ${unit.locator}`} />
        </figure>
      )}
      {videoUrl && (
        <video className="verse-media-video" src={videoUrl} controls playsInline />
      )}
      {videoNote && <p className="verse-audio-note">{videoNote}</p>}
      {error && <p className="verse-error">{error}</p>}
    </section>
  );
}
