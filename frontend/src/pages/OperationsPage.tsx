import { FormEvent, useEffect, useMemo, useState } from "react";
import { PipelineJob, pipelineApi } from "../api/client";

type Status = PipelineJob["status"];

const STATUS_CHIP: Record<Status, string> = {
  queued: "chip",
  running: "chip chip-blue",
  done: "chip chip-green",
  error: "chip chip-red",
};

function fmtTime(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString();
}

function JobCard({ job }: { job: PipelineJob }) {
  const detail = job.status === "error" ? job.error : job.result ? JSON.stringify(job.result, null, 2) : "";
  const [open, setOpen] = useState(false);
  const params = Object.entries(job.params || {});
  return (
    <article className="card ops-job">
      <div className="ops-job-head">
        <div>
          <strong>{job.kind}</strong>{" "}
          <span className="dim" title={job.job_id}>
            {job.job_id}
          </span>
        </div>
        <span className={STATUS_CHIP[job.status] || "chip"}>{job.status}</span>
      </div>
      <div className="dim" style={{ fontSize: "0.82rem" }}>
        criado {fmtTime(job.created_at)}
        {job.finished_at ? ` · concluído ${fmtTime(job.finished_at)}` : ""}
      </div>
      {params.length > 0 && (
        <div className="dim" style={{ fontSize: "0.78rem", wordBreak: "break-word" }}>
          {params.map(([k, v]) => (
            <span key={k} className="chip" style={{ marginRight: 6 }}>
              {k}: {String(v)}
            </span>
          ))}
        </div>
      )}
      {detail && (
        <button type="button" className="btn btn-ghost" onClick={() => setOpen((v) => !v)}>
          {open ? "Ocultar detalhe" : "Ver detalhe"}
        </button>
      )}
      {open && detail && <pre className="ops-job-result">{detail}</pre>}
    </article>
  );
}

export default function OperationsPage() {
  const [token, setToken] = useState(pipelineApi.getStoredToken());
  const [tokenInput, setTokenInput] = useState("");
  const [probe, setProbe] = useState<"idle" | "ok" | "bad">("idle");
  const [jobs, setJobs] = useState<PipelineJob[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  // forms
  const [manifest, setManifest] = useState("fixtures/sources_local.json");
  const [syncDb, setSyncDb] = useState(false);
  const [backend, setBackend] = useState("numpy");
  const [dim, setDim] = useState("");
  const [corpus, setCorpus] = useState("");
  const [vocabSize, setVocabSize] = useState("32000");
  const [baseModel, setBaseModel] = useState("gpt2");
  const [maxSteps, setMaxSteps] = useState("10");
  const [kindFilter, setKindFilter] = useState("");

  useEffect(() => {
    if (!token) {
      setJobs([]);
      setProbe("idle");
      return;
    }
    let active = true;
    const load = async () => {
      try {
        const res = await pipelineApi.listJobs(token, 40);
        if (!active) return;
        setJobs(res.items ?? []);
        setProbe("ok");
      } catch {
        if (!active) return;
        setProbe("bad");
      }
    };
    load();
    const timer = window.setInterval(load, 3000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [token]);

  const saveToken = () => {
    const next = tokenInput.trim();
    pipelineApi.storeToken(next);
    setToken(next);
    setTokenInput("");
    setActionError(null);
  };

  const clearToken = () => {
    pipelineApi.storeToken("");
    setToken("");
    setTokenInput("");
    setJobs([]);
    setActionError(null);
  };

  const run = async (kind: string, fn: () => Promise<PipelineJob>) => {
    if (!token) {
      setActionError("Informe o token do pipeline primeiro.");
      return;
    }
    setBusy(kind);
    setActionError(null);
    try {
      const job = await fn();
      setJobs((prev) => [job, ...prev.filter((j) => j.job_id !== job.job_id)]);
      setProbe("ok");
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const onSubmit = (e: FormEvent) => e.preventDefault();

  const activeJobs = useMemo(() => jobs.filter((j) => j.status === "queued" || j.status === "running").length, [jobs]);
  const kinds = useMemo(() => Array.from(new Set(jobs.map((j) => j.kind))).sort(), [jobs]);
  const visibleJobs = useMemo(
    () => (kindFilter ? jobs.filter((j) => j.kind === kindFilter) : jobs),
    [jobs, kindFilter]
  );

  return (
    <div className="container" style={{ maxWidth: 980 }}>
      <div className="section-head">
        <div>
          <h1>Operações</h1>
          <p className="muted">
            Fila de jobs do pipeline (ingestão, tokenização, treino, índice e banco). Requer{" "}
            <code>VEDIC_PIPELINE_API_TOKEN</code> configurado no servidor — o token digitado
            fica apenas no <code>sessionStorage</code> desta aba.
          </p>
        </div>
      </div>

      <form className="card stack" onSubmit={onSubmit}>
        <div className="field">
          <label htmlFor="ops-token">Token do pipeline (Authorization: Bearer)</label>
          <div className="ops-actions">
            <input
              id="ops-token"
              className="input"
              type="password"
              style={{ flex: 1, minWidth: 240 }}
              value={tokenInput}
              onChange={(e) => setTokenInput(e.target.value)}
              placeholder={token ? "•••••••• (salvo nesta aba)" : "Cole o token…"}
              autoComplete="off"
            />
            <button type="button" className="btn btn-primary" onClick={saveToken} disabled={!tokenInput.trim()}>
              Salvar token
            </button>
            {token && (
              <button type="button" className="btn btn-ghost" onClick={clearToken}>
                Remover
              </button>
            )}
          </div>
          {probe === "ok" && (
            <div className="dim" style={{ marginTop: 6 }}>
              ✓ Token aceito pelo servidor — {jobs.length} jobs na fila{activeJobs ? ` (${activeJobs} ativos)` : ""}.
            </div>
          )}
          {probe === "bad" && (
            <div className="error" style={{ marginTop: 6 }}>
              Token rejeitado (401) ou operações desabilitadas (503). Verifique o valor e o servidor.
            </div>
          )}
        </div>
      </form>

      {actionError && (
        <div className="error" style={{ marginTop: "1rem" }}>
          {actionError}
        </div>
      )}

      <div className="ops-grid" style={{ marginTop: "1rem" }}>
        <section className="card stack">
          <h2>Ingestão</h2>
          <div className="field">
            <label htmlFor="ops-manifest">Manifesto</label>
            <input
              id="ops-manifest"
              className="input"
              value={manifest}
              onChange={(e) => setManifest(e.target.value)}
            />
          </div>
          <label style={{ display: "flex", gap: 8, alignItems: "center", fontSize: "0.9rem" }}>
            <input type="checkbox" checked={syncDb} onChange={(e) => setSyncDb(e.target.checked)} />
            Sincronizar com PostgreSQL após ingestão
          </label>
          <button
            type="button"
            className="btn btn-primary"
            disabled={busy !== null || !token}
            onClick={() => run("ingest", () => pipelineApi.ingest(token, { manifest, sync_db: syncDb }))}
          >
            {busy === "ingest" ? "Enfileirando…" : "Ingerir manifesto"}
          </button>
        </section>

        <section className="card stack">
          <h2>Índice de embeddings</h2>
          <div className="form-row">
            <div className="field">
              <label htmlFor="ops-backend">Backend</label>
              <select id="ops-backend" className="select" value={backend} onChange={(e) => setBackend(e.target.value)}>
                <option value="numpy">numpy</option>
                <option value="pgvector">pgvector</option>
                <option value="both">both</option>
              </select>
            </div>
            <div className="field">
              <label htmlFor="ops-dim">Dimensão (opcional)</label>
              <input
                id="ops-dim"
                className="input"
                type="number"
                min={64}
                max={3072}
                value={dim}
                onChange={(e) => setDim(e.target.value)}
                placeholder="384"
              />
            </div>
          </div>
          <button
            type="button"
            className="btn btn-primary"
            disabled={busy !== null || !token}
            onClick={() =>
              run("build-index", () =>
                pipelineApi.buildIndex(token, {
                  backend,
                  embedding_dim: dim ? Number(dim) : null,
                })
              )
            }
          >
            {busy === "build-index" ? "Enfileirando…" : "Construir índice"}
          </button>
        </section>

        <section className="card stack">
          <h2>Banco (PostgreSQL/pgvector)</h2>
          <div className="form-row">
            <div className="field">
              <label htmlFor="ops-dbinit-dim">db-init dim</label>
              <input
                id="ops-dbinit-dim"
                className="input"
                type="number"
                min={64}
                max={3072}
                value={dim}
                onChange={(e) => setDim(e.target.value)}
                placeholder="384 (vazio = default)"
              />
            </div>
            <div className="field">
              <label htmlFor="ops-corpus">db-sync corpus</label>
              <input
                id="ops-corpus"
                className="input"
                value={corpus}
                onChange={(e) => setCorpus(e.target.value)}
                placeholder="data/corpus.jsonl (vazio = default)"
              />
            </div>
          </div>
          <div className="ops-actions">
            <button
              type="button"
              className="btn btn-primary"
              disabled={busy !== null || !token}
              onClick={() => run("db-init", () => pipelineApi.dbInit(token, dim ? Number(dim) : undefined))}
            >
              {busy === "db-init" ? "…" : "Inicializar schema"}
            </button>
            <button
              type="button"
              className="btn btn-secondary"
              disabled={busy !== null || !token}
              onClick={() => run("db-sync", () => pipelineApi.dbSync(token, corpus.trim() || undefined))}
            >
              {busy === "db-sync" ? "…" : "Sincronizar corpus"}
            </button>
          </div>
        </section>

        <section className="card stack">
          <h2>Treino (avançado)</h2>
          <div className="form-row">
            <div className="field">
              <label htmlFor="ops-base-model">Base model</label>
              <input
                id="ops-base-model"
                className="input"
                value={baseModel}
                onChange={(e) => setBaseModel(e.target.value)}
              />
            </div>
            <div className="field">
              <label htmlFor="ops-max-steps">max_steps</label>
              <input
                id="ops-max-steps"
                className="input"
                type="number"
                min={1}
                value={maxSteps}
                onChange={(e) => setMaxSteps(e.target.value)}
              />
            </div>
          </div>
          <div className="ops-actions">
            <button
              type="button"
              className="btn btn-primary"
              disabled={busy !== null || !token}
              onClick={() =>
                run("train", () =>
                  pipelineApi.train(token, {
                    base_model: baseModel.trim() || undefined,
                    max_steps: maxSteps ? Number(maxSteps) : null,
                  })
                )
              }
            >
              {busy === "train" ? "…" : "Treinar modelo"}
            </button>
            <button
              type="button"
              className="btn btn-secondary"
              disabled={busy !== null || !token}
              onClick={() =>
                run("tokenize", () =>
                  pipelineApi.tokenize(token, { vocab_size: vocabSize ? Number(vocabSize) : undefined })
                )
              }
            >
              {busy === "tokenize" ? "…" : "Treinar tokenizer"}
            </button>
          </div>
          <div className="field">
            <label htmlFor="ops-vocab">Vocab size (tokenizer)</label>
            <input
              id="ops-vocab"
              className="input"
              type="number"
              min={1000}
              max={256000}
              value={vocabSize}
              onChange={(e) => setVocabSize(e.target.value)}
            />
          </div>
        </section>
      </div>

      <section className="stack" style={{ marginTop: "1.5rem" }}>
        <div className="section-head">
          <div>
            <h2>Jobs</h2>
            <p className="muted">
              {token
                ? `${jobs.length} job(s)${activeJobs ? ` · ${activeJobs} em execução/fila` : ""} — atualiza a cada 3s.`
                : "Informe o token para listar os jobs."}
            </p>
          </div>
        </div>
        {kinds.length > 0 && (
          <div className="ops-actions">
            <button
              type="button"
              className={`btn ${kindFilter === "" ? "btn-primary" : "btn-ghost"}`}
              onClick={() => setKindFilter("")}
            >
              Todos
            </button>
            {kinds.map((k) => (
              <button
                key={k}
                type="button"
                className={`btn ${kindFilter === k ? "btn-primary" : "btn-ghost"}`}
                onClick={() => setKindFilter(kindFilter === k ? "" : k)}
              >
                {k}
              </button>
            ))}
          </div>
        )}
        {visibleJobs.length === 0 && (
          <div className="card dim">
            {jobs.length === 0 ? "Nenhum job no servidor." : "Nenhum job deste tipo."}
          </div>
        )}
        {visibleJobs.map((j) => (
          <JobCard key={j.job_id} job={j} />
        ))}
      </section>
    </div>
  );
}
