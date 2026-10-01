import { useEffect, useState } from "react";
import type { RemissiveEntry, SearchFigure } from "../api/client";
import { api } from "../api/client";

type Props = {
  figure: SearchFigure | null;
  entries: RemissiveEntry[];
  onSeek: (name: string) => void;
  onLocate: (index: number) => void;
};

export default function SearchAside({ figure, entries, onSeek, onLocate }: Props) {
  const [imageState, setImageState] = useState<"loading" | "ready" | "error">("loading");
  useEffect(() => {
    setImageState("loading");
  }, [figure?.id]);
  const people = entries.filter((e) => e.kind === "personagem");
  const concepts = entries.filter((e) => e.kind === "conceito");
  if (!figure && entries.length === 0) return null;

  return (
    <div className="search-aside">
      {figure && (
        <article className="card figure-card">
          <h2 className="panel-title">Figura</h2>
          <div className="figure-frame">
            {imageState !== "error" && (
              <img
                key={figure.id}
                src={api.figureImageUrl(figure.id)}
                alt={`Figura de ${figure.name}`}
                onLoad={() => setImageState("ready")}
                onError={() => setImageState("error")}
              />
            )}
            {imageState === "loading" && <p className="dim figure-status">Gerando a figura de {figure.name}…</p>}
            {imageState === "error" && (
              <p className="dim figure-status">A figura de {figure.name} não ficou pronta. A busca segue sem ela.</p>
            )}
          </div>
          <h3 className="figure-name">{figure.name}</h3>
          <p className="figure-epithet">{figure.epithet}</p>
          <p className="muted figure-gloss">{figure.gloss}</p>
          <p className="dim figure-note">Retrato gerado a partir do nome. Não é uma testemunha do texto.</p>
          {figure.see_also.length > 0 && (
            <p className="figure-see">
              Ver também{" "}
              {figure.see_also.map((ref) => (
                <button key={ref.id} type="button" className="chip chip-gold figure-chip" onClick={() => onSeek(ref.name)}>
                  {ref.name}
                </button>
              ))}
            </p>
          )}
        </article>
      )}

      {entries.length > 0 && (
        <article className="card remissive">
          <h2 className="panel-title">Índice remissivo</h2>
          <p className="muted remissive-lead">
            Nomes canônicos destes trechos, com o localizador e o ver também. Não é a lista crua de palavras.
          </p>
          <IndexGroup title="Personagens" entries={people} onSeek={onSeek} onLocate={onLocate} />
          <IndexGroup title="Conceitos" entries={concepts} onSeek={onSeek} onLocate={onLocate} />
        </article>
      )}
    </div>
  );
}

function IndexGroup({
  title,
  entries,
  onSeek,
  onLocate,
}: {
  title: string;
  entries: RemissiveEntry[];
  onSeek: (name: string) => void;
  onLocate: (index: number) => void;
}) {
  if (entries.length === 0) return null;
  return (
    <section className="remissive-group">
      <h3>{title}</h3>
      <ul>
        {entries.map((entry) => (
          <li key={entry.id}>
            <button type="button" className="remissive-term" onClick={() => onSeek(entry.name)}>
              {entry.name}
            </button>
            <span className="remissive-epithet">, {entry.epithet}</span>
            {entry.locators.length > 0 && (
              <span className="remissive-locs">
                {entry.locators.map((loc) => (
                  <button key={`${entry.id}-${loc.index}`} type="button" className="remissive-loc" onClick={() => onLocate(loc.index)}>
                    {loc.locator}
                  </button>
                ))}
              </span>
            )}
            {entry.see_also.length > 0 && (
              <span className="remissive-see">
                {" "}
                ver também{" "}
                {entry.see_also.map((ref, i) => (
                  <span key={ref.id}>
                    {i > 0 ? ", " : ""}
                    <button type="button" className="remissive-link" onClick={() => onSeek(ref.name)}>
                      {ref.name}
                    </button>
                  </span>
                ))}
              </span>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}
