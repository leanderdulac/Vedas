# Veda Knowledge

Aplicação full-stack de conhecimento védico: **biblioteca**, **busca semântica**, **Q&A RAG** e pipeline de ingestão/treino com fontes licenciadas.

## Aplicação web (frontend + backend)

```bash
# Backend deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_vedic_pipeline.txt

# Corpus + índice (offline)
python -m vedic_pipeline ingest --manifest fixtures/sources_local.json
python -m vedic_pipeline build-index --backend numpy

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
- **Perguntar** — chat RAG + painel de fontes
- **Documento** — leitor com Devanāgarī
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

---

## Pipeline de dados / ML

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

# Smoke + 10 queries gold de regressão de retrieval
python scripts/smoke_rag.py --strict --json-out data/smoke_report.json
```

### Contagens canônicas (snapshot 2026-08-07/08)

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

**Obras de base:** Mahābhārata Ganguli vols. 1–4, Rāmāyaṇa Valmiki, Upaniṣads (Müller + páginas SBE/sacred-texts), Gītā Arnold, Yoga-sūtra Johnston, Manu, Viṣṇu Purāṇa (abertura), Ṛgveda Griffith completo.

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
| POST | `/build-index` | `numpy` \| `pgvector` \| `both` |
| POST | `/search` | busca semântica |
| POST | `/ask` | RAG + geração |
| POST | `/db/init` | schema PG |
| POST | `/db/sync` | JSONL → PG |

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

- MinIO/S3 para raw e artefatos  
- Deploy multi-container (crawler / etl / train / api)  
- Dimensão de embedding configurável no schema  
- Rerank cross-encoder opcional  
