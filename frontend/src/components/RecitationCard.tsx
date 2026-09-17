import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, Pada } from "../api/client";
import { RECITATIONS, echoGapMs } from "../data/recitation";

type Phase = "idle" | "loading" | "reciting" | "echo" | "done";

function speak(text: string, rate: number): Promise<void> {
  return new Promise((resolve) => {
    if (typeof window === "undefined" || !window.speechSynthesis) return resolve();
    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = "hi-IN";
    utter.rate = rate;
    utter.onend = () => resolve();
    utter.onerror = () => resolve();
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utter);
  });
}

function stopSpeaking() {
  if (typeof window !== "undefined") window.speechSynthesis?.cancel();
}

export default function RecitationCard() {
  const [entry, setEntry] = useState(RECITATIONS[0]);
  const [customId, setCustomId] = useState("");
  const [padas, setPadas] = useState<Pada[] | null>(null);
  const [locator, setLocator] = useState<string>("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [step, setStep] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const runId = useRef(0);
  const echoTimer = useRef<number | null>(null);

  const current: Pada | null = padas ? padas[step] ?? null : null;
  const total = padas?.length ?? 0;

  const reset = useCallback(() => {
    runId.current += 1;
    if (echoTimer.current) window.clearTimeout(echoTimer.current);
    echoTimer.current = null;
    stopSpeaking();
    setPhase("idle");
    setStep(0);
  }, []);

  useEffect(() => reset, [reset]);

  async function loadVerse(verseId: string) {
    reset();
    setError(null);
    setPadas(null);
    setPhase("loading");
    const myRun = ++runId.current;
    try {
      const res = await api.versePadas(verseId);
      if (runId.current !== myRun) return;
      setPadas(res.padas);
      setLocator(res.locator || verseId);
      setPhase("idle");
    } catch (e) {
      if (runId.current !== myRun) return;
      setPhase("idle");
      setError(e instanceof Error ? e.message : "Falha ao carregar o verso");
    }
  }

  const advance = useCallback(
    async (from: number) => {
      if (!padas) return;
      const myRun = runId.current;
      const next = from + 1;
      if (next >= padas.length) {
        setPhase("done");
        return;
      }
      setStep(next);
      setPhase("reciting");
      await speak(padas[next].sa || padas[next].iast, 0.7);
      if (runId.current !== myRun) return;
      setPhase("echo");
      if (echoTimer.current) window.clearTimeout(echoTimer.current);
      echoTimer.current = window.setTimeout(() => {
        void advance(next);
      }, echoGapMs(padas[next]));
    },
    [padas],
  );

  async function start() {
    if (!padas || padas.length === 0) return;
    reset();
    const myRun = runId.current;
    setPhase("reciting");
    await speak(padas[0].sa || padas[0].iast, 0.7);
    if (runId.current !== myRun) return;
    setPhase("echo");
    if (echoTimer.current) window.clearTimeout(echoTimer.current);
    echoTimer.current = window.setTimeout(() => {
      void advance(0);
    }, echoGapMs(padas[0]));
  }

  /** Eco imediato: o usuário repetiu e chama o próximo pāda. */
  function nextPada() {
    if (echoTimer.current) window.clearTimeout(echoTimer.current);
    stopSpeaking();
    void advance(step);
  }

  function replayPada() {
    const p = current;
    if (!p) return;
    if (echoTimer.current) window.clearTimeout(echoTimer.current);
    stopSpeaking();
    setPhase("reciting");
    void speak(p.sa || p.iast, 0.6).then(() => {
      setPhase("echo");
      echoTimer.current = window.setTimeout(() => {
        void advance(step);
      }, echoGapMs(p));
    });
  }

  function pickCustom(e: React.FormEvent) {
    e.preventDefault();
    const id = customId.trim();
    if (!id) return;
    void loadVerse(id);
  }

  const statusLabel = useMemo(() => {
    switch (phase) {
      case "loading":
        return "Carregando o verso…";
      case "reciting":
        return "O app recita — ouça";
      case "echo":
        return "Sua vez — repita o pāda";
      case "done":
        return "Verso completo ॥";
      default:
        return total > 0 ? `${total} pāda${total > 1 ? "s" : ""} prontos` : "Escolha um verso";
    }
  }, [phase, total]);

  return (
    <article className="card recite-card">
      <div className="recite-head">
        <div>
          <h2>Modo eco</h2>
          <p className="muted">
            O app recita um pāda, pausa — você repete. Assim os vaidikas ensinam: a voz
            responde à voz, até o verso morar na memória.
          </p>
        </div>
      </div>

      <form className="recite-picker" onSubmit={pickCustom}>
        <div className="ops-actions">
          {RECITATIONS.map((r) => (
            <button
              key={r.verseId}
              type="button"
              className={`btn ${entry.verseId === r.verseId ? "btn-primary" : "btn-ghost"}`}
              onClick={() => {
                setEntry(r);
                void loadVerse(r.verseId);
              }}
              title={r.note}
            >
              {r.label}
            </button>
          ))}
        </div>
        <div className="recite-custom">
          <input
            value={customId}
            onChange={(e) => setCustomId(e.target.value)}
            placeholder="Outro verso: RV.1.1.2, BG.2.63, AV.1.2.1…"
            aria-label="verse id"
          />
          <button type="submit" className="btn btn-ghost">
            Carregar
          </button>
        </div>
      </form>

      {entry.note && <p className="recite-note muted">{entry.note}</p>}

      {error && <p className="verse-error">{error}</p>}

      {padas && padas.length > 0 && (
        <>
          <div className="recite-padas" role="list">
            {padas.map((p, i) => (
              <div
                key={p.index}
                role="listitem"
                className={`recite-pada${i === step && (phase === "reciting" || phase === "echo") ? " is-active" : ""}${
                  i < step || phase === "done" ? " is-done" : ""
                }`}
                onClick={() => {
                  if (i !== step) return;
                  stopSpeaking();
                  void speak(p.sa || p.iast, 0.6);
                }}
                title="Ouvir novamente"
              >
                <span className="recite-pada-num">{p.index}</span>
                <span className="recite-pada-sa deva">{p.sa || p.iast}</span>
                {p.iast && p.sa && <span className="recite-pada-iast">{p.iast}</span>}
              </div>
            ))}
          </div>

          <div className="recite-controls ops-actions">
            {phase === "idle" && (
              <button type="button" className="btn btn-primary" onClick={start}>
                ▶ Começar o eco
              </button>
            )}
            {phase === "reciting" && (
              <button type="button" className="btn btn-ghost" onClick={stopSpeaking}>
                ■ Pausar
              </button>
            )}
            {phase === "echo" && (
              <>
                <button type="button" className="btn btn-primary" onClick={nextPada}>
                  ✔ Repeti — próximo
                </button>
                <button type="button" className="btn btn-ghost" onClick={replayPada}>
                  ↻ Repetir o pāda
                </button>
              </>
            )}
            {(phase === "reciting" || phase === "echo" || phase === "done") && (
              <button type="button" className="btn btn-ghost" onClick={reset}>
                Recomeçar
              </button>
            )}
          </div>

          <div className="recite-status">
            <span className="recite-phase">{statusLabel}</span>
            <span className="muted">
              {locator} · pāda {Math.min(step + 1, total || 1)} de {total}
            </span>
          </div>
        </>
      )}

      {padas && padas.length === 0 && !error && (
        <p className="muted">Este verso não tem pāda recitável (sem sânscrito/IAST).</p>
      )}
    </article>
  );
}