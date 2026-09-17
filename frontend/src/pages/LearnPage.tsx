import { useState } from "react";
import { Link } from "react-router-dom";
import {
  VARNA_GROUPS,
  VOCAB,
  VOCAB_CATEGORIES,
  LESSONS,
  speakSanskrit,
} from "../data/sanskrit";

type Tab = "varna" | "vocab" | "licoes";

const TABS: { id: Tab; label: string; hint: string }[] = [
  { id: "varna", label: "Varṇamālā", hint: "O alfabeto: sons e pontos de articulação" },
  { id: "vocab", label: "Vocabulário", hint: "Palavras vivas do corpus védico" },
  { id: "licoes", label: "Lições", hint: "Sandhi, casos e verbos pelo próprio verso" },
];

function VarnaCard({ v }: { v: (typeof VARNA_GROUPS)[0]["items"][0] }) {
  return (
    <button
      type="button"
      className="learn-letter"
      onClick={() => speakSanskrit(v.example?.word ?? v.deva)}
      title="Ouvir pronúncia"
    >
      <span className="learn-letter-deva">{v.deva}</span>
      <span className="learn-letter-iast">{v.iast}</span>
      <span className="learn-letter-art">{v.articulation}</span>
      {v.example && (
        <span className="learn-letter-ex">
          {v.example.word} <span className="dim">· {v.example.iast}</span>
        </span>
      )}
    </button>
  );
}

function VocabCard({ entry }: { entry: (typeof VOCAB)[0] }) {
  return (
    <article className="card learn-word">
      <div className="learn-word-head">
        <button type="button" className="learn-word-deva" onClick={() => speakSanskrit(entry.deva)} title="Ouvir">
          {entry.deva}
        </button>
        <div>
          <div className="learn-word-iast">{entry.iast}</div>
          <div className="learn-word-gloss">{entry.gloss}</div>
        </div>
      </div>
      <p className="learn-word-meaning">{entry.meaning}</p>
      <div className="ops-actions">
        <Link className="btn btn-ghost verse-btn" to={`/busca?q=${encodeURIComponent(entry.query)}&lang=sa`}>
          Ver no corpus
        </Link>
        <button type="button" className="btn btn-ghost verse-btn" onClick={() => speakSanskrit(entry.deva)}>
          ▶ Ouvir
        </button>
      </div>
    </article>
  );
}

function LessonCard({ lesson }: { lesson: (typeof LESSONS)[0] }) {
  return (
    <article className="card learn-lesson">
      <h2>{lesson.title}</h2>
      <p className="muted">{lesson.subtitle}</p>
      {lesson.sections.map((s, i) => (
        <div key={i} className="stack" style={{ marginTop: "0.75rem" }}>
          <h3 style={{ margin: 0 }}>{s.heading}</h3>
          {s.body.map((p, j) => (
            <p key={j} style={{ margin: "0.25rem 0" }}>
              {p}
            </p>
          ))}
          {s.examples.length > 0 && (
            <table className="learn-table">
              <tbody>
                {s.examples.map((ex, k) => (
                  <tr key={k}>
                    <td className="deva learn-table-deva" onClick={() => speakSanskrit(ex.deva)} title="Ouvir">
                      {ex.deva}
                    </td>
                    <td className="learn-table-iast">{ex.iast}</td>
                    <td className="dim">{ex.gloss}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      ))}
    </article>
  );
}

export default function LearnPage() {
  const [tab, setTab] = useState<Tab>("varna");
  const [category, setCategory] = useState<string>("");

  const visibleVocab = category ? VOCAB.filter((v) => v.category === category) : VOCAB;
  const active = TABS.find((t) => t.id === tab)!;

  return (
    <div className="container" style={{ maxWidth: 1080 }}>
      <div className="section-head">
        <div>
          <h1>
            Aprenda Saṃskṛtam <span className="deva">संस्कृतम्</span>
          </h1>
          <p className="muted">
            Do som à palavra, da palavra ao verso: o alfabeto como o Śikṣā o ensina, o vocabulário
            que habita o corpus e as regras que dão forma ao hino. Toque nas letras para ouvir.
          </p>
        </div>
      </div>

      <div className="ops-actions" style={{ marginBottom: "1rem" }}>
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            className={`btn ${tab === t.id ? "btn-primary" : "btn-ghost"}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>
      <p className="muted" style={{ marginTop: "-0.5rem" }}>
        {active.hint}
      </p>

      {tab === "varna" && (
        <div className="stack">
          {VARNA_GROUPS.map((g) => (
            <section key={g.id} className="learn-group">
              <h2 className="learn-group-title">{g.label}</h2>
              <div className="learn-letters">
                {g.items.map((v) => (
                  <VarnaCard key={v.deva} v={v} />
                ))}
              </div>
            </section>
          ))}
          <div className="card dim" style={{ fontSize: "0.9rem" }}>
            A ordem é a da varṇamālā tradicional: svaras, depois as cinco vargas (gutural → labial),
            semivogais e ūṣman. Cada varga tem cinco sons: surda, surda aspirada, sonora, sonora
            aspirada e nasal — o mesmo ponto de articulação, cinco maneiras de soltar o som.
          </div>
        </div>
      )}

      {tab === "vocab" && (
        <div className="stack">
          <div className="ops-actions">
            <button
              type="button"
              className={`btn ${category === "" ? "btn-primary" : "btn-ghost"}`}
              onClick={() => setCategory("")}
            >
              Todas
            </button>
            {VOCAB_CATEGORIES.map((c) => (
              <button
                key={c.id}
                type="button"
                className={`btn ${category === c.id ? "btn-primary" : "btn-ghost"}`}
                onClick={() => setCategory(c.id)}
              >
                {c.label}
              </button>
            ))}
          </div>
          <div className="learn-words">
            {visibleVocab.map((v) => (
              <VocabCard key={v.deva} entry={v} />
            ))}
          </div>
        </div>
      )}

      {tab === "licoes" && (
        <div className="stack">
          {LESSONS.map((l) => (
            <LessonCard key={l.id} lesson={l} />
          ))}
        </div>
      )}
    </div>
  );
}