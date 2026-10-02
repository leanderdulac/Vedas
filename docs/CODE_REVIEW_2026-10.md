# Vedas — auditoria de código (2026-10-02)

Revisão investigativa do repositório **Vedas** (`veda-knowledge` 2.0.0): pipeline RAG full-stack (crawler → índice → retrieval híbrido + locators → CE opcional → API → frontend). **Não houve refatoração ampla.** A única correção de código nesta revisão é o bloqueio público de `/metrics` no Caddy (controle que a documentação já afirmava existir). `VEDIC_ENABLE_RERANKER` permanece **false** por default. Nenhum peso de modelo nem corpus foi commitado.

---

## Veredito

O projeto está **maduro para um corpus licenciado single-instance**, com hardening consciente (tokens, SSRF parcial, confinamentos de path, CE off, gold de locators, `deploy-check`). O núcleo de retrieval híbrido + locator é o pedaço mais trabalhado e melhor testado.

Não é “pronto para multi-tenant / internet aberta sem operação cuidadosa”. Os riscos reais são: **SSRF por rebinding DNS** no crawler, **geração paga aberta** quando há `XAI_API_KEY` sem token (dev/default), **híbrido/locator silencioso em deploy só-pgvector**, e **heurísticas de hino RV que capturam qualquer `X.Y`**. A suíte local está verde (ruff, 250 testes, frontend). O estado operacional (corpus canônico ~1054 docs) **não** é o default de um clone — é um snapshot.

**Saúde geral: B+ (sólido, com dívidas de segurança e de retrieval bem delimitadas).**

---

## 1. Arquitetura e fluxos

```text
manifesto JSON
    → licenses.validate_source
    → download.download_source (HTTP(S) + SSRF / file:// em PROJECT_ROOT)
    → extractors | vedicheritage | structured_json
    → data/corpus.jsonl (+ opcional db-sync → PostgreSQL documents)

chunking (unidades canônicas ou janela 800/120)
    → numpy: artifacts/embeddings/{embeddings.npy, chunks.jsonl, index_meta.json}
    → pgvector: chunks + chunk_embeddings (vector(VEDIC_EMBEDDING_DIM))

query
    → expand_query (até 3 variantes)
    → search_index (numpy) OU search_pgvector
    → hybrid_rerank:
         pool = hits semânticos + top léxico do corpus
         + locator_hymn/work/deity injections
         fusão sem*0.65 + lex
         title/size + hymn/work/deity boosts
         [CE rerank se VEDIC_ENABLE_RERANKER] + locator de novo
         phrase boost + anthology demotion
         diversify_by_doc(max_per_doc=2)
    → build_rag_prompt → generate_answer (xai | local | extractive)
    → FastAPI /api/v1/{search,ask,ask/stream} → SPA React
```

| Camada | Módulos-chave |
|--------|----------------|
| Ingestão | `crawler/ingest.py`, `crawler/download.py`, `crawler/licenses.py`, `etl/*` |
| Índice | `search/embeddings.py`, `storage/vectors.py`, `etl/chunking.py`, `etl/structure.py` |
| Retrieval | `llm/ask.py:retrieve_hits`, `search/hybrid.py`, `search/rag.py` |
| CE | `search/reranker.py` (default off), `search/reranker_status.py` |
| Geração | `llm/generate.py`, `llm/ask.py`, `api/request_policy.py` |
| API | `api/app.py` (`create_app`), schemas, jobs, rate_limit, metrics |
| UI | `frontend/src/pages/*`, `frontend/src/api/client.ts` |
| Deploy | `Dockerfile`, `docker-compose.yml` / `.prod.yml`, `Caddyfile`, `ops/deploy_check.py` |

**Entradas:** CLI `vedic-pipeline` (`cli.py`), ASGI `vedic_knowledge_pipeline:app`, scripts (`bulk_ingest_open.py`, `smoke_rag.py`, treino CE).

**Detalhe importante:** o léxico de corpus inteiro e a injeção de locator leem `all_chunks` do índice **numpy** (`llm/ask.py` 82–93). `backend=pgvector` sozinho, sem `artifacts/embeddings/`, ainda chama `hybrid_rerank` mas **sem** injeção/expansão léxica.

---

## 2. Resultados dos checks (exatos)

Ambiente: Python 3.12.3, Node 22.14.0, deps via `pip install -r requirements_vedic_pipeline.txt` + `pip install -e ".[dev]"` e `frontend/npm ci`. Data: 2026-10-02.

| Check | Comando | Resultado |
|-------|---------|-----------|
| Ruff | `python3 -m ruff check .` | **All checks passed** |
| Pytest + coverage | `python3 -m pytest -v --cov=vedic_pipeline --cov-report=term-missing --cov-fail-under=60` | **250 passed**, 1 warning, 32 subtests, **68.21%** (gate 60%) em 16.93s (inclui o teste novo do Caddy). |
| Frontend lint | `npm run lint` (`tsc --noEmit`) | **OK** |
| Frontend tests | `npm test` | **18/18 passed** (0 fail) |
| Frontend audit | `npm audit --audit-level=moderate` | **0 vulnerabilities** |
| Frontend build | `npm run build` | **OK** (vite 6.4.3, 59 módulos, ~1.2s) |
| `pip-audit -r requirements-*.txt` | isolamento cria venv | **Falhou neste ambiente** (`ensurepip` / `python3-venv` ausente). No CI o passo é `continue-on-error`. |
| `pip-audit --local` | pacotes já instalados | Várias CVEs **transitivas** (jinja2 3.1.2, pyjwt 2.7.0, urllib3 2.6.3, setuptools 78.1.0, wheel 0.42.0, oauthlib). O app **não** usa JWT; jinja2 entra via stack web/docs. Não bloqueante para o produto, mas o passo informacional do CI precisa de venv. |
| Smoke CI (`ingest` + `build-index` + `smoke_rag.py --strict`) | igual a `.github/workflows/smoke.yml` | **Não executado aqui** — exige baixar o MiniLM (~centenas de MB) e gravar índice. A lógica de gold/locator está coberta em `tests/test_locator_boost.py` + `tests/test_smoke_gold.py`. |

Coverage por caminho crítico (do relatório pytest):

| Módulo | Cover | Nota |
|--------|-------|------|
| `search/hybrid.py` | 91% | Melhor coberto; `rrf_fuse` e deity inject quase mortos |
| `search/reranker.py` | 91% | Default off testado |
| `ops/deploy_check.py` | 99% | |
| `llm/ask.py` | 40% | `ask()` / auto-pgvector / stream pouco exercitados de verdade |
| `llm/generate.py` | 20% | xAI/local só mockados |
| `crawler/download.py` | 46% | HTTP real (L138–195) sem teste |
| `storage/vectors.py` | 37% | busca pgvector sem teste de integração |
| `etl/extractors.py` | 15% | |
| `etl/vedaweb.py` | 0% | Morto para a suíte |
| `api/traditions.py` | 0% | Dados estáticos |
| **TOTAL** | **68.21%** | Acima do gate |

`VEDIC_ENABLE_RERANKER` default `"false"` verificado em `search/reranker.py:29–36`, `.env.example:95`, `docker-compose.prod.yml:51`, `tests/test_rate_limit.py::ConsistencyTests` (`health["reranker"]["enabled"]` / `default_enabled` false).

---

## 3. Achados

### Critical

Nenhum. Sem RCE não autenticado, sem segredos vivos no git, ops de pipeline 503 sem `VEDIC_PIPELINE_API_TOKEN`.

---

### High

#### H1. SSRF: validação DNS não está pinada na conexão TCP (TOCTOU / rebinding)

- **Onde:** `vedic_pipeline/crawler/download.py:22–51`, `149–154`; o mesmo padrão em `vedic_pipeline/llm/imagine.py:104–132`.
- **Problema:** `is_safe_url` resolve o host com `socket.getaddrinfo` e bloqueia RFC1918/loopback/link-local. Em seguida `httpx.Client.get(current)` resolve **de novo**. Um host que responde IP público no check e `127.0.0.1` / metadata no connect passa.
- **Impacto:** ingest HTTP (`/ingest` se o token de pipeline vazar, ou CLI/scripts com manifesto atacante) pode alcançar serviços internos.
- **Correção:** resolver uma vez, conectar ao IP validado (transport custom / `http://<ip>/` + `Host:`) e recusar mismatch; revalidar em cada redirect (já há loop manual). Testar com host de rebinding.

*Não corrigido aqui — mudança de transporte, não é “pequena/baixa risco”.*

#### H2. Geração paga aberta no default/dev se `XAI_API_KEY` existir

- **Onde:** `vedic_pipeline/api/request_policy.py:110–120`; `docker-compose.yml` (sem `VEDIC_REQUIRE_GENERATION_TOKEN`); frontend `AskPage` usa `provider=auto` e **não** envia Bearer de geração (`frontend/src/api/client.ts:290–330`).
- **Problema:** sem `VEDIC_GENERATION_API_TOKEN` e sem `VEDIC_REQUIRE_GENERATION_TOKEN`, qualquer cliente em `:8000` dispara `/ask`, stream, TTS, Imagine. Compose de prod falha-fechado (`true`); o de dev não.
- **Impacto:** billing xAI + superfície de abuso se a API for publicada “como no compose de dev”.
- **Correção:** em qualquer host compartilhado usar `docker-compose.prod.yml` ou setar `VEDIC_REQUIRE_GENERATION_TOKEN=true` + token. README L395–399 diz que “sem token a geração HTTP retorna 503” — isso só é verdade com a flag de prod; **alinhar o README**. UI de prod precisa de um caminho para Bearer de geração (hoje só o token de pipeline existe no `sessionStorage`).

#### H3. `/metrics` público no ingress Caddy (documentado como interno) — **corrigido nesta PR**

- **Onde (antes):** `Caddyfile` catch-all `handle { reverse_proxy api:8000 }` proxava `/metrics`. Comentário + `docs/DEPLOYMENT.md:131–133` + `docs/DEVELOPMENT_REVIEW.md:16–17` diziam o contrário.
- **Impacto:** no hostname público (perfil `ingress`) qualquer um lia uptime, contagens de corpus/chunks, jobs e séries HTTP.
- **Correção aplicada:** `handle /metrics { respond 404 }` antes do catch-all. Teste: `tests/test_deploy_check.py::test_caddyfile_denies_public_metrics`. Prometheus interno (`vedas_internal`) continua raspando `api:8000` direto.

---

### Medium

#### M1. Identidade de rate-limit spoofável via `X-Forwarded-For`

- **Onde:** `vedic_pipeline/api/rate_limit.py:68–75` (default `VEDIC_TRUST_PROXY_HEADERS=true`); `Dockerfile:48` `--forwarded-allow-ips '*'`.
- **Problema:** o app usa o **primeiro** hop do XFF. Cliente que alcance a API sem um proxy que reescreva o header (ou uvicorn confiando em qualquer peer) rotaciona IPs e esvazia o teto de `/ask`/`/search`/mídia.
- **Correção:** default `false`; em prod, usar só o hop imediato do Caddy (`$remote_addr`) ou o rightmost untrusted. Restringir `--forwarded-allow-ips` à rede do Caddy. Teste de spoof XFF.

#### M2. Híbrido/locator desliga em silêncio no modo só-pgvector

- **Onde:** `vedic_pipeline/llm/ask.py:82–93`.
- **Problema:** `all_chunks` só carrega se `backend == "numpy"` ou `Path(index_dir).exists()`. Sem sidecar numpy, `hybrid_rerank(..., all_chunks=None)` **não** injeta hino/obra/deidade nem expande léxico. Ranking cai sem erro; `retrieval_backend` ainda reporta `pgvector+hybrid`.
- **Correção:** carregar metadados de `chunks` no Postgres para o híbrido, **ou** falhar alto se hybrid=true e o índice numpy não existir. Teste de integração pgvector-only.

#### M3. `\d{1,2}\.\d{1,3}` solto é tratado como id de Ṛgveda

- **Onde:** `vedic_pipeline/search/hybrid.py:253–257` (`_EXPLICIT_HYMN_RE`).
- **Reproduzido nesta auditoria:**

  | Query | `extract_query_hymn_ids` | `extract_query_work_keys` |
  |-------|--------------------------|---------------------------|
  | `BG 2.47` | `['2.47']` | `[]` |
  | `Yoga Sutra 1.2` | `['1.2']` | `['yoga-sutra']` |
  | `version 1.2 of the text` | `['1.2']` | `[]` |
  | `see 3.14` | `['3.14']` | `[]` |

- **Impacto:** injeção/boost de hinos RV errados (RV 2.47 no lugar da Gītā; RV 1.2 no Yoga-sūtra). Nomes canônicos (Nasadiya, etc.) vencem conflito, mas números “nus” não.
- **Correção:** remover o ramo `\b(\d{1,2}\.\d{1,3})\b` **ou** exigir contexto RV/hymn/sūkta. Não fazer isso sem re-rodar o gold-37 (`10.129` solto ainda precisa funcionar via nome ou `RV`).

#### M4. `pull_dir` S3 usa `startswith` (escape de prefixo)

- **Onde:** `vedic_pipeline/storage/objects.py:254–256`.
- **Problema:** `str(target).startswith(str(dest.resolve()))` aceita `/app/data_evil` se `dest` for `/app/data`. `..` após `resolve()` é pego; irmão com prefixo compartilhado não. Teste atual (`test_pull_rejects_escape_keys`) só cobre `../`.
- **Correção:** `target.is_relative_to(dest.resolve())` (já usado no SPA em `app.py:936`). Teste com dest=`.../artifacts` e key que resolva para `.../artifacts_backup`.

#### M5. `VEDIC_DISABLE_SSRF_DNS_CHECK` no deploy-check, inexistente no crawler

- **Onde:** `vedic_pipeline/ops/deploy_check.py:132–140`; **zero** referências em `download.py`. `docs/DEVELOPMENT_REVIEW.md:56` diz que não há kill-switch.
- **Impacto:** operador pode achar que desligou o DNS check; o comportamento não muda. O check falha se a env estiver setada (conservador), mas o nome é um footgun.
- **Correção:** remover a env do checklist **ou** implementar a flag, logada e proibida em `--mode prod`.

#### M6. Download “streaming” não faz stream

- **Onde:** `vedic_pipeline/crawler/download.py:149–185`.
- **Problema:** `client.get()` (httpx, sem `stream=True`) carrega o body inteiro. O loop `iter_bytes` e o teto de 25 MB atuam **depois**. `Content-Length` mentiroso ou ausente → pico de memória. `imagine._download_media_bytes` lê `resp.content` de uma vez (L132).
- **Correção:** `client.stream("GET", ...)` + teto no iterador; recusar se `Content-Length` for inválido em vez de engolir o `ValueError`.

#### M7. Docs / OpenAPI públicos no ingress

- **Onde:** FastAPI default (`/docs`, `/redoc`, `/openapi.json`). README L31 anuncia `/docs`. Caddy catch-all proxia.
- **Impacto:** superfície de API (incl. ops de pipeline) visível no hostname público. Ops continuam 401/503 sem token.
- **Correção:** `docs_url=None` quando `VEDIC_REQUIRE_GENERATION_TOKEN` ou um `VEDIC_DISABLE_DOCS`; ou `handle /docs` / `/redoc` / `/openapi.json` 404 no Caddy.

#### M8. README vs código (vários)

| Item | Docs | Código |
|------|------|--------|
| Porta PG | README L190 `localhost:5432` | compose / `.env.example` host **5433** |
| Smoke | tabela L166 **10/10** | gold **37** (`fixtures/smoke_queries.json`); CI 4 queries |
| Geração sem token | README L399 “retorna 503” | 503 só com `VEDIC_REQUIRE_GENERATION_TOKEN`; senão aberto se houver xAI |
| Chunk default | exemplos 1000/150 | `constants.py` **800/120** |
| Rate-limit /metrics | `rate_limit.py:11–12` “metrics não entram” | `_RULES` inclui `/metrics` (L30); teste `test_read_routes_are_not_limited` só faz 1 request e passa |
| Catálogo da UI | narrativa “full-stack PG” | `catalog_service.py` lê **JSONL** em memória |
| Hybrid = RRF | docstring `hybrid.py:1` | `rrf_fuse` (L162–171) **nunca é chamado**; fusão é soma ponderada |

#### M9. Dependências sem pin + audit frágil

- **Onde:** `requirements-api.txt`, `requirements_vedic_pipeline.txt`, `pyproject.toml` (`>=`).
- **Impacto:** drift de supply-chain; `pip-audit` no CI é informacional e aqui nem rodou isolado.
- **Correção:** pins / `uv.lock` no runtime da API; tornar o audit do CI bloqueante para o arquivo da API; ignorar só o extra `train`.

---

### Low

#### L1. Código morto / duplicado

- `rrf_fuse` (`hybrid.py:162–171`) sem callers.
- `etl/vedaweb.py` (0% coverage, caminho paralelo ao ingest atual).
- Pipeline sync vs async em `app.py` (L600–884) duplica ingest/tokenize/train/build-index/db — risco de divergir (já quase iguais).
- `vedic_knowledge_pipeline.py:28–45` reexporta símbolos só para compat.

#### L2. Health ainda vaza contagens de DB

- **Onde:** `api/app.py:171–189` comenta “não expõe contagens cruas” e em seguida copia `database.counts` / `embedding_dim` se existirem.

#### L3. Phrase boost + scores sem bound

- **Onde:** `apply_phrase_boost` +1.25 (`hybrid.py:1026–1036`); boosts de locator +1.45 / +1.05. Scores deixam de ser comparáveis entre queries. Phrase em corpo de antologia ainda compete com o demote (−1.35). Intencional para o gold; frágil fora dele.

#### L4. `search_index` filtra tradição/idioma **depois** do matmul em todos os vetores

- **Onde:** `search/embeddings.py:184–190`. Em ~24k chunks é aceitável; não escala sem índice invertido de metadados.

#### L5. Cache de tokens do híbrido

- **Onde:** `_CHUNK_TOKEN_CACHE` (`hybrid.py:90–121`), limpa aos 50k. Churn adversarial de `chunk_id` = flush + CPU. DoS local, não remoto sem índice.

#### L6. Frontend: sem token de geração; Ask `auto` quebra em prod fail-closed

- Documentado em `.env.example:27–31`. UI não oferece Bearer de geração. Em prod, `/perguntar` com `auto`+xAI → 401/503. Extractive continua público.

#### L7. Prompt injection (RAG)

- Contexto recuperado + `history` (12×6000) entram no prompt (`rag.py`, `generate.py:36–48`). Mitigação só instrucional. Esperado para RAG.

#### L8. Jobs em processo

- `api/jobs.py`: ThreadPool, máx. 4 workers, persistência JSON. Sem isolamento multi-réplica (o próprio docstring avisa). `job_id` é uuid; `get_job` só lê o dict em memória (sem path traversal no GET).

#### L9. Compose de dev

- API em `0.0.0.0:8000`, Postgres `vedas:vedas` só em `127.0.0.1:5433`, user 10001 — ok para local. MinIO `vedas-minio-dev-only`.

#### L10. `compare_digest` com tamanhos diferentes

- `app.py:129`, `request_policy.py:122,144`. Em 3.12 **não** crasham (retornam False). Testes de 401 sem header passam. Sem ação.

---

## 4. Heurísticas do híbrido (resumo de risco)

O stack em `hybrid_rerank` (`hybrid.py:919–1023`) é uma política de produto, não um ranker teórico:

1. **Injeção** (até 4 chunks/hino, 4/obra, 4/deidade) salva o gold (Gāyatrī, Īśā, Nasadiya) quando o denso/léxico falha.
2. **Boost empilhado** (título 1.45, texto 1.05, obra 1.45, phrase 1.25, deidade 1.05) + **demote** de antologia 1.35 — afinado contra “Principal Upanishads” / “selected hymns”.
3. **CE** só se flag; depois o locator **reaplica** (o MiniLM genérico inverte Nasadiya). Default off — correto.
4. **Riscos:** M3 (X.Y genérico), M2 (pgvector-only), phrase vs antologia, scores não calibrados, lexical O(n chunks) por request (hotspot).
5. **Cobertura:** `tests/test_locator_boost.py` é excelente para o gold; **não** cobre “BG 2.47”, “YS 1.2”, nem pgvector-only.

---

## 5. Segurança (o que está bem)

- Gate de licença (`licenses.py`) + allowlist.
- LFI de `file://` com `is_relative_to(PROJECT_ROOT)` (`download.py:63–71`).
- Redirects HTTP revalidados (sem `follow_redirects=True`).
- `index_dir` da API pinado em `VEDIC_API_INDEX_DIR` (`request_policy.py:73–81`); testes de symlink.
- Paths de pipeline via `validate_project_path`.
- Tokens com `secrets.compare_digest`; pipeline 503 se unset.
- SQL parametrizado (`vectors.py:189–227`).
- `np.load(..., allow_pickle=False)`.
- CORS: `*` removido se `allow_credentials=True` (`app.py:61–63`).
- Headers de segurança no middleware; Caddy HSTS/CSP em prod.
- SPA fallback com `is_relative_to`.
- Métricas com path cardinality colapsada (`metrics.py:37–72`).
- `.gitignore` cobre `artifacts/`, `data/corpus.jsonl`, `.env`, pesos.
- User não-root 10001; DB de prod sem porta pública; `POSTGRES_PASSWORD` obrigatório no compose.prod.

---

## 6. Performance

| Hotspot | Onde | Nota |
|---------|------|------|
| Léxico full-corpus | `hybrid_rerank` + `lexical_scores` | O(chunks × tokens) ~24k; cache ajuda |
| Até 3 encodes + buscas | `retrieve_hits` variantes[:3] × fetch_k=36 | |
| Filtro metadado pós-matmul | `search_index` | |
| Catálogo JSONL em RAM | `catalog_service.get_cached_corpus` | ok até dezenas de MB; não é PG |
| Cold start ST + npy | `embeddings._load_st_model`, `get_index` | |
| CE top ~36 | só se opt-in | |

Não há budget test de latência no CI.

---

## 7. Docker / deploy

- **Dev compose:** API publicada em 8000, trust proxy **false**, geração aberta se houver xAI — coerente com “máquina local”.
- **Prod compose:** token de geração fail-closed, CE off, redes internal/public, Caddy perfil `ingress`, Prometheus perfil `metrics`. **H3** era o furo do Caddy (agora fechado).
- **Dockerfile:** imagem slim (`requirements-api.txt`); `/tokenize` e `/train` → 501. `--forwarded-allow-ips '*'` é largo (M1).
- **Dockerfile.train:** deps completas, `sleep infinity`, user 10001.
- **Caddyfile:** `/api/*` + catch-all para a SPA; `/metrics` agora 404 no público.
- **`deploy-check`:** senha fraca, token de pipeline, geração aberta, CE genérico, flag SSRF fantasma.

---

## 8. Lacunas de teste (caminhos críticos)

| Área | Há teste? | Falta |
|------|-----------|--------|
| SSRF básico | loopback, 169.254, file:// | rebinding / IP pinado |
| Auth geração/mídia | `test_request_policy`, `test_rate_limit` | E2E UI prod + Bearer de geração |
| Rate limit | exaustão, disable | spoof XFF; `/metrics` de verdade limitado |
| Hybrid locators | extenso | `BG 2.47` / `YS 1.2`; só-pgvector |
| Download HTTP | rejeições | stream + teto + Content-Length mentiroso |
| `pull_dir` | `../` | prefixo `artifacts` vs `artifacts_backup` |
| `ask()` real | mockado na API | generate.py 20% |
| CORS | não | |
| `/docs` em prod | não | |
| Smoke gold-37 | script + CI recorte 4 | MiniLM no CI (já existe no workflow; não rodado nesta VM) |

---

## 9. Correção feita nesta PR

1. **Caddyfile:** `handle /metrics { respond 404 }` para cumprir `docs/DEPLOYMENT.md` §4.1.
2. **Teste:** `tests/test_deploy_check.py::test_caddyfile_denies_public_metrics`.

Nada mais foi alterado no pipeline, no default do reranker, nem em dados/pesos.

---

## 10. Top 5 próximos passos

1. **Pin de conexão no crawler (H1)** + teste de rebinding; mesmo helper para Imagine.
2. **Fechar geração paga por default em hosts não-privados (H2)** e alinhar README L395–399; decidir se a UI ganha Bearer de geração ou se prod força extractive.
3. **Híbrido honesto no pgvector (M2):** chunks do Postgres **ou** erro se o sidecar numpy faltar.
4. **Apertar `_EXPLICIT_HYMN_RE` (M3)** e adicionar casos `BG 2.47` / `Yoga Sutra 1.2` / `version 1.2` no `test_locator_boost`, revalidando o gold-37.
5. **Rate-limit + proxy (M1):** default `VEDIC_TRUST_PROXY_HEADERS=false`; `--forwarded-allow-ips` só o Caddy; teste de XFF.

Depois: `is_relative_to` no S3 pull (M4), stream real no download (M6), pins + audit bloqueante (M9), docs OpenAPI em prod (M7).

---

## 11. Como ler este documento

- Relatório de **investigação** (2026-10-02), não um roadmap de features.
- Gold de retrieval e decisão do CE: `docs/reranker_decision.md`, `docs/ce_promote_gate.md`, `docs/DEVELOPMENT_REVIEW.md`.
- Operação: `docs/DEPLOYMENT.md`.
- Default do CE continua **off**. Pesos em `artifacts/reranker_domain_v*` não entram no git.
