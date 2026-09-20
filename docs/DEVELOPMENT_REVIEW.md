# Vedas — estado de desenvolvimento (2026-09-20)

Revisão do **Vedas** apenas. O documento anterior misturava outro projeto e um
corpus de 11 documentos na porta 8100 — isso está obsoleto.

Para operação: [`docs/DEPLOYMENT.md`](DEPLOYMENT.md).  
Para o CE: [`docs/reranker_decision.md`](reranker_decision.md).

## O que está estável

- Retrieval híbrido + locator (hino/obra) no default. Gold expandido
  (`fixtures/smoke_queries.json`, **37** queries) **37/37** no Mac (CE off).
- API + UI (biblioteca, busca, RAG extractive/SSE, versos, jobs).
- CI: `pytest` + ruff + smoke CI (4 queries) + frontend lint/test/build.
- Hardening: tokens de pipeline/geração, rate limit, índice fixo na API,
  SSRF/LFI no crawler, imagem API slim vs perfil `train`, `/metrics` não
  publicado no Caddy.
- Fila `POST .../async` + `GET /jobs`.
- Cross-Encoder **v5 PROMOTE** (e v4) no gold-37 e **default off**. Opt-in
  recomendado: `artifacts/reranker_domain_v5` quando existir localmente.

## Corpus: o que é reproduzível

O número **~1054 docs** do README é um **snapshot de um ambiente**, não o
default de um clone. Perfis em `fixtures/corpus_profiles.json`:

| Perfil | Manifesto | Reproduzível |
|--------|-----------|----------------|
| `local` | `fixtures/sources_local.json` | sim |
| `bootstrap` (default) | `fixtures/sources_vedic_corpus.json` | sim (CI) |
| `open-web` | `fixtures/sources_open_web.json` | sim (bulk) |
| `canonical-snapshot` | — | **não** (~1054; não está no git) |

```bash
vedic-pipeline corpus-status --profile bootstrap
vedic-pipeline corpus-status --write-lock data/corpus.lock.json
vedic-pipeline corpus-status --verify-lock data/corpus.lock.json
```

Yajur/Sāma/Atharva completos exigem ingestão (ex. Vedic Heritage), não um
tarball no repositório.

## Deploy: checklist fail-closed

```bash
vedic-pipeline deploy-check --mode local
vedic-pipeline deploy-check --mode prod
```

Produção (`docker-compose.prod.yml`):

- `VEDIC_REQUIRE_GENERATION_TOKEN=true` — sem token, `/ask` xAI e mídia
  paga devolvem **503** (extractive continua público).
- `VEDIC_ENABLE_RERANKER=false` salvo opt-in com **dir local**.
- Prometheus interno: `--profile metrics` (rede `vedas_internal`, sem porta).
- O DNS check do crawler **não** tem kill-switch.

## CE depois do PROMOTE

PROMOTE ≠ ligar. `fixtures/reranker_promote.json` + `/api/v1/health.reranker`
registram `eligible_opt_in=true` e `default_enabled=false`.

```bash
vedic-pipeline reranker-status
# só então, local:
export VEDIC_ENABLE_RERANKER=true
export VEDIC_RERANKER_MODEL=artifacts/reranker_domain_v5
```

`deploy-check` falha se o CE estiver on sem diretório local (MiniLM genérico).

## Ainda aberto (não bloqueia o núcleo)

- Auth de usuário / multi-tenant / HA — fora do compose single-instance.
- `npm audit` pontual no frontend (overrides já cobrem o lote anterior).
- Ampliar o cânone via manifestos; gravar lock depois do ingest.
- Treino causal (`gpt2`) continua smoke — não usar para ranking.
