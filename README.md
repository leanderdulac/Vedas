# Veda Knowledge

Aplicação full-stack de conhecimento védico: **biblioteca**, **busca semântica**, **Q&A RAG** e pipeline de ingestão/treino com fontes licenciadas.

## Aplicação web (frontend + backend)

```bash
# Backend deps (pip install -e . ou requirements)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,db]"   # ou: pip install -r requirements_vedic_pipeline.txt

# Corpus + índice (offline)
vedic-pipeline ingest --manifest fixtures/sources_local.json
vedic-pipeline build-index --backend numpy

# Frontend
cd frontend && npm install && cd ..

# Dev (API :8000 + Vite :5173)
chmod +x scripts/run_dev.sh scripts/run_prod.sh
./scripts/run_dev.sh

# Ou produção (UI embutida na API :8000)
./scripts/run_prod.sh
```

| URL | Função |
|-----|--------|
| http://localhost:5173 | UI React (dev, proxy `/api`) |
| http://localhost:8000 | API + UI buildada |
| http://localhost:8000/docs | OpenAPI |

### Páginas

- **Início** — śloka, estatísticas, tradições
- **Biblioteca** — catálogo com filtros (tradição/idioma/texto)
- **Busca** — recuperação semântica com scores
- **Aprenda** — saṃskṛtam: varṇamālā pronunciável, vocabulário védico ligado ao corpus e lições de sandhi/subantas/tiṅantas
- **Perguntar** — chat RAG + painel de fontes
- **Documento** — leitor com Devanāgarī
- **Operações** — fila de jobs do pipeline (ingest/index/db/treino) com token no `sessionStorage`
- **Sobre** — licenças e health do sistema

### API da UI (`/api/v1`)

| Método | Endpoint | Função |
|--------|----------|--------|
| GET | `/api/v1/health` | status |
| GET | `/api/v1/stats` | contagens do corpus |
| GET | `/api/v1/traditions` | catálogo de tradições |
| GET | `/api/v1/documents` | lista filtrada |
| GET | `/api/v1/documents/{id}` | documento completo |
| POST | `/api/v1/search` | busca semântica |
| POST | `/api/v1/ask` | Q&A RAG |
| GET | `/api/v1/verses/{id}` | bundle do verso (testemunhas sa/IAST/EN) |
| GET | `/api/v1/verses/{id}/padas` | **pāda do verso** (segmentação determinística; alimenta o modo eco) |
| GET | `/api/v1/verses/{id}/audio` | recitação TTS (cache em disco) |
| POST | `/api/v1/verses/{id}/explain` | explicação PT/EN (LLM ou extrativa) |
| POST | `/api/v1/verses/{id}/translate` | **tradução do sânscrito** para PT/EN (LLM; cache aberto) |
| POST | `/api/v1/verses/{id}/analyze` | **vyākaraṇa interlinear**: pāda + análise palavra-por-palavra (LLM; cache aberto) |

A tradução (`/translate`) usa o texto sânscrito como fonte (fallback IAST→EN),
apoia-se nas testemunhas IAST/EN e cacheia o resultado em `data/translations/`.
Leitura do cache é aberta (sem custo); geração nova passa pela política de
geração (`VEDIC_GENERATION_API_TOKEN`). Sem LLM configurado, devolve as
testemunhas alinhadas como referência — nunca uma tradução inventada.

---

## Pipeline de dados / ML

Abaixo, a arquitetura de módulos e os comandos de ingestão/treino/busca.

## Arquitetura

```text
vedic_pipeline/
├── crawler/    # manifesto, gate de licença, download
├── etl/        # extração + chunking
├── train/      # BPE + causal LM
├── search/     # embeddings numpy + montagem RAG
├── storage/    # PostgreSQL catálogo + pgvector
├── llm/        # /ask — xAI | local | extractive
├── api/        # FastAPI
└── common/
```

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_vedic_pipeline.txt
cp .env.example .env   # opcional: DATABASE_URL, XAI_API_KEY
```

## Corpus védico ampliado + RAG com inferência

```bash
# Bootstrap: 11 obras PD (Ṛgveda, Upaniṣads, Gītā, Yoga-sūtra, Rāmāyaṇa, Mahābhārata…)
python scripts/bootstrap_vedic_corpus.py --reset --rebuild-index --backend both --sync-db

# Q&A com recuperação híbrida + Grok (XAI_API_KEY no .env)
python -m vedic_pipeline ask "Compare ātman nas Upaniṣads e na Gītā" --provider xai --top-k 12
```

### Ingestão em massa (fontes abertas verificadas)

```bash
# Dry-run do manifesto Gutenberg + Sacred Texts + fixtures
python scripts/bulk_ingest_open.py --dry-run

# Download + corpus + PostgreSQL (resume em data/bulk_state.json)
python scripts/bulk_ingest_open.py \
  --manifest fixtures/sources_open_web.json \
  --include-local-fixtures \
  --sync-db

# Ṛgveda mandalas 1 e 10 completas (Griffith / sacred-texts)
python scripts/generate_rigveda_manifest.py --books 1,10 --out fixtures/sources_rigveda_m1_m10.json
python scripts/bulk_ingest_open.py --manifest fixtures/sources_rigveda_m1_m10.json --sync-db

# Dedupe (fp + fixtures superadas) e alinhar DB (purge de órfãos)
python -m vedic_pipeline dedupe
python -m vedic_pipeline db-sync          # purge default
# python -m vedic_pipeline db-sync --no-purge

# Depois: reindexar (demora se o Mahābhārata completo estiver no corpus)
python -m vedic_pipeline build-index --backend both
# ou:
python scripts/bulk_ingest_open.py --rebuild-index --backend both --limit 0

# SBE/Purāṇas completos (Chāndogya, Bṛhadāraṇyaka, Viṣṇu, Garuḍa…)
python scripts/build_sbe_manifest.py
python scripts/bulk_ingest_open.py --manifest fixtures/sources_sbe_complete.json --sync-db

# Mārkaṇḍeya Purāṇa (inglês Pargiter 1904 + sânscrito Devanāgarī, scans OCR)
VEDIC_MAX_DOWNLOAD_BYTES=134217728 \
python scripts/bulk_ingest_open.py --manifest fixtures/sources_markandeya.json --sync-db

# Smoke + 10 queries gold de regressão de retrieval
python scripts/smoke_rag.py --strict --json-out data/smoke_report.json
```

### Contagens canônicas (snapshot 2026-08-07/08)

> Snapshot de um ambiente específico (bulk ingest completo). Deploys novos
> partem de `fixtures/` e crescem conforme o manifesto usado — os números
> abaixo não são automáticos nem garantidos.

| Métrica | Valor |
|---------|-------|
| Documentos no corpus / PG | **1054** (alinhados, purge on) |
| Caracteres | **~19,8M** |
| Chunks + embeddings | **23.702** (numpy e pgvector) |
| Hinos Ṛgveda (Griffith) | **1028** (mandalas **1–10** completas) |
| Upaniṣads PD adicionais | Kena, Kaṭha, Muṇḍaka, Māṇḍūkya, Taittirīya, Aitareya, Praśna, Chāndogya, Bṛhadāraṇyaka, Śvetāśvatara (+ Müller) |
| Ranking | híbrido + boost título/hino + **max 2 chunks/doc** |
| Ask | JSON + **SSE** `/api/v1/ask/stream` |
| Smoke retrieval | **10/10** (`python scripts/smoke_rag.py --strict`) |

**Obras de base:** Mahābhārata Ganguli vols. 1–4, Rāmāyaṇa Valmiki, Upaniṣads (Müller + páginas SBE/sacred-texts), Gītā Arnold, Yoga-sūtra Johnston, Manu, Viṣṇu Purāṇa integral (Wilson), Garuḍa Purāṇa (Wood), Mārkaṇḍeya Purāṇa (Pargiter en + Devanāgarī, scans OCR), Ṛgveda Griffith completo, Chāndogya e Bṛhadāraṇyaka integrais (SBE01/SBE15), saṃhitās Vedic Heritage em sânscrito.

```bash
# Ṛgveda mandalas 2–9
python scripts/generate_rigveda_manifest.py --books 2-9 --out fixtures/sources_rigveda_m2_m9.json
python scripts/bulk_ingest_open.py --manifest fixtures/sources_rigveda_m2_m9.json --sync-db

# Upaniṣads PD adicionais
python scripts/bulk_ingest_open.py --manifest fixtures/sources_upanishads_open.json --sync-db

# Reindex (encode 1×) + pgvector
python -m vedic_pipeline build-index --backend numpy --chunk-size 1000 --overlap 150 --batch-size 64
python scripts/numpy_to_pgvector.py
```
**Não é o cânone inteiro** (mandalas 2–9, Yajur/Sāma/Atharva, Purāṇas completos, edições com copyright). Amplie o manifesto e rode o bulk de novo (resumível via `data/bulk_state.json`).

RAG: expansão de consulta, híbrido semântico+lexical, inferência citável com Grok.

## PostgreSQL + pgvector

```bash
docker compose up -d
export DATABASE_URL=postgresql://vedas:vedas@localhost:5432/vedas

python -m vedic_pipeline db-init
python -m vedic_pipeline db-sync
python -m vedic_pipeline build-index --backend pgvector   # ou: both
python -m vedic_pipeline db-check
python -m vedic_pipeline search " पूर्णमदः " --backend pgvector
```

## Q&A com SpaceXAI (xAI)

```bash
export XAI_API_KEY=...          # https://console.x.ai
# opcional: XAI_MODEL=grok-4.5

python -m vedic_pipeline ask "Explain verse 6 of the Isha Upanishad from the sources" \
  --provider auto --top-k 5
```

Providers:

| Provider | Quando usar |
|----------|-------------|
| `auto` | `xai` se `XAI_API_KEY`, senão `extractive` |
| `xai` | SpaceXAI / xAI (`https://api.x.ai/v1`) |
| `local` | Transformers causal (ex. gpt2) — smoke test |
| `extractive` | Só trechos recuperados, sem geração |

## API

```bash
python -m vedic_pipeline serve --port 8000
```

| Método | Endpoint | Função |
|--------|----------|--------|
| GET | `/health` | status, DB, providers LLM |
| POST | `/ingest` | manifesto → corpus (+ `sync_db`) |
| POST | `/tokenize` | treina BPE |
| POST | `/train` | fine-tune causal |
| POST | `/build-index` | `numpy` \| `pgvector` \| `both` (+ `embedding_dim`) |
| POST | `/search` | busca semântica |
| POST | `/ask` | RAG + geração |
| POST | `/db/init` | schema PG (`?dim=` configura `vector(N)`) |
| POST | `/db/sync` | JSONL → PG |
| POST | `/*/async` | mesma op acima, assíncrona (202 + `job_id`) |
| GET | `/jobs` | lista jobs |
| GET | `/jobs/{id}` | status/resultado do job |

Ops pesadas (`ingest`, `tokenize`, `train`, `build-index`, `db/init`, `db/sync`)
têm variante `POST .../async` que retorna `202 {"job_id", "status":"queued"}`;
acompanhe com `GET /jobs/{id}` até `done`/`error`. Mesma autenticação
(`Authorization: Bearer <VEDIC_PIPELINE_API_TOKEN>`).

```bash
# exemplo async
curl -X POST http://localhost:8000/build-index/async \
  -H "Authorization: Bearer $VEDIC_PIPELINE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"backend":"pgvector","embedding_dim":384}'
curl -H "Authorization: Bearer $VEDIC_PIPELINE_API_TOKEN" \
  http://localhost:8000/jobs/<job_id>
```

Dimensão pgvector configurável via `VEDIC_EMBEDDING_DIM` (default 384),
`--dim` na CLI (`db-init`, `build-index`) ou `embedding_dim`/`?dim=` na API.
Trocou o modelo de embeddings → `db-init` com a nova dim + `build-index --backend pgvector`.

Runtime slim: a imagem Docker instala `requirements-api.txt` (sem
`datasets`/`accelerate`/`sentencepiece` de treino). Para treino local use
`pip install -r requirements_vedic_pipeline.txt` ou `pip install -e ".[train]"`;
`/tokenize` e `/train` na imagem de API respondem 501.

## Artefatos remotos (S3/MinIO) + treino isolado

Operação é local-first; o S3 serve de backup/restauração de `data/raw` e `artifacts/`:

```bash
# status do backend (local por padrão)
vedic-pipeline artifacts status

# com MinIO local (console em http://localhost:9001)
docker compose --profile s3 up -d minio
export VEDIC_S3_BUCKET=vedas VEDIC_S3_ENDPOINT=http://localhost:9000
export AWS_ACCESS_KEY_ID=vedas AWS_SECRET_ACCESS_KEY=vedas-minio-dev-only

vedic-pipeline artifacts push --dir artifacts --dry-run
vedic-pipeline artifacts push --dir artifacts
vedic-pipeline artifacts push --dir data/raw --prefix raw
vedic-pipeline artifacts pull --prefix artifacts --dir /tmp/restore
```

Sem `VEDIC_S3_BUCKET`, os comandos explicam que o backend está desabilitado
(em vez de traceback). `boto3` é opcional: `pip install -e ".[s3]"`.

Treino e pipeline pesado rodam isolados do perfil `train` (imagem completa):

```bash
docker compose --profile train run --rm train vedic-pipeline train-model --max-steps 10
docker compose --profile train run --rm train vedic-pipeline build-index --backend pgvector
```

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the Self according to the Isha Upanishad?",
    "provider": "extractive",
    "top_k": 3
  }'
```

## Treino

Default causal: **`gpt2`**.

```bash
python -m vedic_pipeline train-tokenizer --vocab-size 8000 --eval
python -m vedic_pipeline train-model --base-model gpt2 --block-size 128 --max-steps 10
```

## Jurídico

Fontes sem licença na lista permitida são bloqueadas. Material BBT/Vedabase somente com autorização explícita.

## Smoke / regressão

```bash
# health corpus + índice + 10 gold queries + ask extractive
python scripts/smoke_rag.py
python scripts/smoke_rag.py --backend numpy --strict
python scripts/smoke_rag.py --backend pgvector --strict --json-out data/smoke_report.json
```

Gold set: `fixtures/smoke_queries.json` (Isha, Gītā, Nasadiya, Puruṣa, Agni 1.1, Yoga-sūtra, Rāmāyaṇa, Mahābhārata, Manu…).  
CI: `.github/workflows/smoke.yml` + `fixtures/smoke_queries_ci.json`.

## Docker

```bash
docker compose up -d db
# API full-stack (build multi-stage: frontend + FastAPI)
docker compose up -d --build api
# UI: http://localhost:8000  ·  docs: http://localhost:8000/docs
```

## Streaming Q&A

```bash
curl -N -X POST http://localhost:8000/api/v1/ask/stream \
  -H 'Content-Type: application/json' \
  -d '{"query":"Explain the Self in the Isha Upanishad","provider":"extractive","top_k":5}'
```

Eventos SSE: `meta` → `token*` → `done` (ou `error`). UI: página **Perguntar** com toggle Streaming.

## Próximos passos

- ~~MinIO/S3 para raw e artefatos~~ ✅ (`vedic-pipeline artifacts` + perfil `s3`)
- ~~Deploy multi-container (crawler / etl / train / api)~~ ✅ parcial (API slim + `train` isolado; fila de jobs em-processo)
- ~~Dimensão de embedding configurável no schema~~ ✅ (`VEDIC_EMBEDDING_DIM` / `--dim`)  
- ~~Rerank cross-encoder opcional~~ ✅ (`VEDIC_ENABLE_RERANKER` / `VEDIC_RERANKER_MODEL`)  
- ~~Fila de jobs para ops pesadas~~ ✅ (`POST .../async` + `GET /jobs`)  

## Desenvolvimento local revisado

Veja `docs/DEVELOPMENT_REVIEW.md` para correções, prioridades, execução local e validação.

As operações HTTP `/ingest`, `/tokenize`, `/train`, `/build-index`, `/db/init` e `/db/sync` (e suas variantes `/async`, mais `GET /jobs`) exigem `VEDIC_PIPELINE_API_TOKEN` e o cabeçalho `Authorization: Bearer <token>`. Sem token configurado, ficam desabilitadas (503). Os comandos da CLI continuam disponíveis sem esse token.

Testes & Qualidade: `pytest -v` (ou `python -m unittest discover -s tests -v`) e linter via `ruff check .`; frontend: `npm run lint`, `npm test` e `npm audit` dentro de `frontend/` (Node.js 22+). O score mostrado nas fontes é uma pontuação de ordenação, não uma probabilidade.

### Controle de geração e índice na API

Busca e chat aceitam somente o diretório definido em `VEDIC_API_INDEX_DIR` (padrão `artifacts/embeddings`). A CLI mantém a possibilidade de escolher outros diretórios.

Para usar `provider=xai` ou `provider=local` por HTTP, configure `VEDIC_GENERATION_API_TOKEN` e envie `Authorization: Bearer <token>`. O modelo permitido vem de `XAI_MODEL` ou `VEDIC_LOCAL_LM`; o cliente não pode escolher outro. `auto` também exige autorização se selecionar xAI. Sem configuração do token, geração HTTP retorna 503; com token ausente ou incorreto na requisição, retorna 401. Essas respostas acontecem antes de iniciar o streaming ou recuperar documentos.

`provider=extractive` continua público e não aceita `model`. A interface web atual usa esse modo automaticamente quando não há chave xAI; para usar geração protegida, clientes HTTP precisam enviar o cabeçalho. Não exponha o token em `VITE_*` nem no bundle público. O token do pipeline não concede acesso à geração.
