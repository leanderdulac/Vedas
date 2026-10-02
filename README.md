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
| GET | `/api/v1/verses/{id}/audio` | recitação sânscrito (TTS, cache em disco) |
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

# Smoke + gold expandido (37 queries) de regressão de retrieval
python scripts/smoke_rag.py --strict --json-out data/smoke_report.json
```

### Contagens canônicas (índice local, 2026-09-17)

> Snapshot de **um** ambiente: esta máquina, bulk ingest completo
> (`artifacts/embeddings/index_meta.json`). **Não** é o default de um clone:
> isso começa em `bootstrap` (`fixtures/sources_vedic_corpus.json`) e cresce
> conforme o manifesto — os números abaixo não são automáticos nem garantidos.
> O perfil `canonical-snapshot` em `fixtures/corpus_profiles.json` ainda
> registra o snapshot de agosto (1.054 documentos, 23.702 chunks); o índice
> numpy atual é o da tabela. Depois do ingest, grave um lock e verifique:
>
> ```bash
> vedic-pipeline corpus-status --profile bootstrap
> vedic-pipeline corpus-status --write-lock data/corpus.lock.json
> ```

| Métrica | Valor |
|---------|-------|
| Documentos no corpus | **6.973** (5.061 em sânscrito, 1.912 em inglês) |
| Caracteres | **~31,1M** |
| Chunks + embeddings | **67.170**, dim **384**, modelo `paraphrase-multilingual-MiniLM-L12-v2` |
| Hinos Ṛgveda (Griffith) | **1028** (mandalas **1–10** completas) |
| Upaniṣads PD adicionais | Kena, Kaṭha, Muṇḍaka, Māṇḍūkya, Taittirīya, Aitareya, Praśna, Chāndogya, Bṛhadāraṇyaka, Śvetāśvatara (+ Müller) |
| Ranking | híbrido + boost título/hino + **max 2 chunks/doc** |
| Ask | JSON + **SSE** `/api/v1/ask/stream` |
| Smoke retrieval | **37/37** no gold (`fixtures/smoke_queries.json`); o CI usa 4 queries |

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

Redação: os prompts de /ask, explicação e tradução levam o guia de estilo de
`vedic_pipeline/common/style.py` (prosa corrida de professor, sem markdown, travessões nem
clichês de chatbot; IAST consistente; citações `[n]` + localizador). Todo texto final passa
por `clean_prose` (determinístico); a voz recebe `speech_text`. Explicações de verso ganham
ainda uma revisão por um modelo barato (`VEDIC_TEXT_EDITOR=1`, padrão; `0` desliga).

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

## Ilustração, vídeo e áudio (Stable Diffusion)

`VEDIC_MEDIA_BACKEND=auto` usa a família Stable Diffusion quando o extra
está instalado e, sem ele, o xAI Imagine.

```bash
pip install -e ".[media]"
# Apple Silicon. Na primeira geração, HF_HUB_OFFLINE=0 para baixar os pesos.
export VEDIC_MEDIA_BACKEND=diffusion VEDIC_DEVICE=mps HF_HUB_OFFLINE=0
```

| Mídia | Modelo | O que faz |
| --- | --- | --- |
| Imagem | `stabilityai/sdxl-turbo` (fallback `sd-turbo`) | Still do verso e retrato das figuras. No verso, um refresh (`?refresh=1`) refina o JPEG que já existe; na figura, gera de novo do zero. |
| Vídeo | `stable-video-diffusion-img2vid-xt` | Alguns segundos a partir do still, com movimento lento. |
| Áudio | TTS + `stable-audio-open-1.0` | A fala continua no TTS. O Stable Audio só acrescenta um drone de tanpura por baixo — ele não pronuncia sânscrito. |

O retrato das figuras cabe nos 77 tokens do CLIP: a iconografia vem primeiro
(no Agni: pele vermelha, duas cabeças, sete línguas de fogo, carneiro, concha),
depois o estilo de pintura devocional. O que não deve aparecer (letras, pele
azul) vai no prompt negativo, que só pesa com CFG: por isso o turbo roda com
`VEDIC_SD_TURBO_GUIDANCE=1.5`. Sem `VEDIC_SD_MODEL`, o SDXL-Turbo é usado se
estiver no cache; offline e sem os pesos, cai para o sd-turbo. Num M5 Max, uma
figura de 512 px leva ~0,8 s; o vídeo SVD (14 quadros, 1024x576) ~1,5 min.

### Modo qualidade das imagens

O turbo a 512 px sai tosco. Comparação em outubro de 2026 (Agni e RV 1.1.1,
M5 Max, mesmo prompt):

| Candidato | Tempo/imagem | Resultado |
| --- | --- | --- |
| `grok-imagine-image-quality` (xAI) | ~6 s, US$ 0,05 | Duas cabeças, carneiro, conchas; cena do verso fiel. Escolhido. |
| `grok-imagine-image-2.0` (xAI) | ~18 s, US$ 0,04 | Mesma qualidade, mas escreveu "अग्नि" na moldura. |
| SDXL base 1.0 (+ refiner), 1024 px, 30 passos | 14–16 s | Estilo de miniatura autêntico, mas o Agni vira uma deusa de muitos braços. |
| Playground v2.5, 1024 px | ~13 s | Pintura rica, uma cabeça, sem carneiro; verso confuso. |
| Turbo 512 → img2img 1024 | ~6 s | Mais nítido, mesma composição tosca. |

Por isso `VEDIC_IMAGE_BACKEND=xai` manda figuras e stills para o Grok Imagine
e o vídeo e o drone continuam locais. O Imagine não tem prompt negativo: as
proibições entram como uma frase "Avoid: …" e o prompt pede para não escrever
letras. Se o xAI falhar e o extra media existir, a imagem sai do turbo local
(`render_image(fast=True)`). Para ficar 100% local em qualidade, use
`VEDIC_IMAGE_BACKEND=diffusion` com `VEDIC_SD_MODEL` completo: 1024 px, 30
passos e CFG 7 por padrão, com `VEDIC_SD_REFINER` e `VEDIC_SD_UPSCALE`
opcionais.

O estilo segue o acervo de referência do Leandro (ver `docs/IMAGE_STYLE.md`): os retratos usam o
busto escultural escuro (`VEDIC_FIGURE_STYLE=sculpted`) e os versos usam a pintura devocional
cinematográfica (`VEDIC_SCENE_STYLE=cinematic`); `miniature` volta ao estilo antigo. As
referências opcionais (`VEDIC_IMAGE_STYLE_REF`, `VEDIC_SCENE_STYLE_REF`) apontam para imagens
locais que nunca entram no git.

Retratos com verbete (devas, Nārada, Brahmā, Śiva, Vyāsa, Vālmīki, Hanumān…) usam a iconografia
curada; sábios não herdam a coroa do estilo. Uma consulta que só nomeia alguém sem verbete
("Vasishtha") ganha uma figura `nome-<slug>` se o nome é próprio e atestado no acervo
(maiúscula nos trechos e ≥3 chunks no índice lexical).

Sem `ffmpeg` no `PATH`, a recitação é gravada sem o drone. `VEDIC_AUDIO_BED=0`
desliga a cama e mantém a voz sozinha.

O drone fica 22 dB abaixo da voz em RMS (`VEDIC_AUDIO_BED_DB`, de -40 a -12)
e passa por um passa-baixa em ~1,2 kHz. Antes ele era normalizado e somado com
ganho fixo, ficava só ~3 dB abaixo da voz, e os harmônicos de 1 a 4 kHz soavam
como um segundo narrador. No front, `src/data/playback.ts` garante uma voz por
vez: o MP3 do backend nunca toca junto com a voz do navegador, que só entra
quando o MP3 falha antes de começar.

### Vídeo dos versos (~5 s)

O SVD img2vid-xt gera 25 quadros a 768×432 (~90 s e ~21 GB no MPS do M5 Max;
1024×576 com 25 quadros não cabe nos 36 GB). Os quadros tocam a 5 fps (5 s),
o `ffmpeg minterpolate` interpola até 24 fps e um zoom lento (Ken Burns) com
upscale lanczos fecha em 1024×576, H.264 com faststart. O still quadrado entra
inteiro, com as laterais preenchidas pelo próprio still desfocado, então o
alto das chamas não é mais cortado.

| Variável | Padrão | Efeito |
| --- | --- | --- |
| `VEDIC_SVD_FRAMES` | 25 | Quadros do SVD (máx. 25 no xt) |
| `VEDIC_SVD_FPS` | 5 | Ritmo dos quadros; duração = quadros / fps |
| `VEDIC_VIDEO_INTERP_FPS` | 24 | Interpolação por movimento; 0 desliga |
| `VEDIC_VIDEO_ZOOM` | 1.06 | Zoom do Ken Burns; 1 desliga |
| `VEDIC_SVD_WIDTH` | 768 | Largura em que o SVD roda (512–1024) |
| `VEDIC_VIDEO_OUT_WIDTH` | 1024 | Largura do MP4 final |
| `VEDIC_SVD_FRAMING` | `pad` | `pad` (still inteiro) ou `cover` (recorte 16:9) |

#### Motor de vídeo configurável (`VEDIC_VIDEO_BACKEND`)

O `POST /api/v1/verses/{id}/video` grava um job pendente em
`data/media/jobs/` e renderiza numa thread de fundo; o `GET` só lê esse job.
O motor sai de `VEDIC_VIDEO_BACKEND`, no mesmo padrão do `VEDIC_IMAGE_BACKEND`:

| Valor | Motor | Observações |
| --- | --- | --- |
| `svd` | Stable Video Diffusion local | Padrão com o extra media (vazio segue `VEDIC_MEDIA_BACKEND`) |
| `xai` | Grok Imagine image-to-video (`XAI_VIDEO_MODEL`, padrão `grok-imagine-video-1.5`) | Usa `XAI_API_KEY`; 720p 16:9 |
| `runway` | Runway API image-to-video (`RUNWAY_VIDEO_MODEL`, padrão `gen4.5`; `gen4_turbo` é mais barato) | Usa `RUNWAYML_API_SECRET`; 1280:720 |

Nas APIs, o still quadrado vai inteiro num quadro 16:9 (laterais com o próprio
still desfocado), como no SVD, e o MP4 é baixado com a validação SSRF das
imagens (teto de 50 MB). Se a API falhar (sem chave, sem crédito, recusa,
tempo esgotado) e o extra media existir, o mesmo job cai no SVD e registra
`fallback_from` e `fallback_reason`; o job de sucesso guarda modelo e custo
(`cost_usd`, e `credits` na Runway). `VEDIC_VIDEO_SECONDS` (5, de 2 a 10)
define a duração pedida e `VEDIC_VIDEO_TIMEOUT` (600 s) a espera máxima.

Comparação no RV 1.1.5 (mesmo still do Grok Imagine, outubro de 2026):

| Motor | Render | Saída | Custo | Resultado |
| --- | --- | --- | --- | --- |
| SVD img2vid-xt (local, MPS) | ~90–160 s | 5 s, 1024×576, 24 fps | grátis (~21 GB de memória) | Fiel ao still, mas movimento quase nulo e rostos borrados |
| `grok-imagine-video-1.5` (xAI) | ~36 s | 5 s, 1280×720, 24 fps, com áudio ambiente | US$ 0,71 (cobrado no `usage`) | Nítido, chamas e fumaça sobem, push-in lento; o Agni ganha uma segunda cabeça no meio do vídeo |
| Runway `gen4.5` / `gen4_turbo` | — | — | 12 / 5 créditos por s (US$ 0,60 / 0,25 em 5 s) | Não testado: a conta estava sem créditos |

## SLM local (continued-pretraining LoRA) + migração de embeddings

**SLM** — adaptar um modelo pequeno ao corpus védico (base recomendada
`Qwen/Qwen2.5-0.5B`; vocab cobre Devanāgarī; no Apple Silicon use
`VEDIC_DEVICE=mps`). Docs com OCR ruidoso (scans) são excluídos do treino:

```bash
# stats do dataset sem baixar nada
python scripts/train_slm_lora.py --dry-run

# treino (HF_HUB_OFFLINE=0 na primeira vez para baixar o modelo-base)
HF_HUB_OFFLINE=0 VEDIC_DEVICE=mps python scripts/train_slm_lora.py \
  --base-model Qwen/Qwen2.5-0.5B --max-steps 500 --out artifacts/slm-qwen05

# usar como fallback de geração local
export VEDIC_LOCAL_LM=artifacts/slm-qwen05
```

Device de treino unificado (`VEDIC_DEVICE=cuda|mps|cpu`, default conservador
cpu; MPS exige flag explícita por causa do histórico de SIGSEGV): vale para
`train-model`, `train_slm_lora.py`, o fine-tune do reranker, a difusão e o
encode dos embeddings. Sem a variável, a busca semântica continua em CPU.

**Embeddings** — trocar o modelo de embeddings (ex.: `intfloat/multilingual-e5-small`,
mesma dim 384, prefixos `query:`/`passage:` aplicados automaticamente):

```bash
# backup do índice atual + reindex (numpy e/ou pgvector) + smoke de validação
VEDIC_EMBEDDING_MODEL=intfloat/multilingual-e5-small \
  python scripts/migrate_embeddings.py --backend both --smoke gold

# ou fixar no env para todos os comandos
export VEDIC_EMBEDDING_MODEL=intfloat/multilingual-e5-small
```

O modelo gravado no `index_meta.json` é a fonte da verdade na busca (o
pgvector passa a consultar com o mesmo modelo do índice, mesmo que o env
mude depois).

`build-index` também grava o BM25 invertido (`lexical.npz` ao lado dos
embeddings). A busca usa esse arquivo em vez de retokenizar os chunks.
Se o arquivo não existir ou o corpus tiver mudado, ele é refeito na
primeira carga do índice.

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

`train-model` monta `TrainingArguments` filtrando kwargs que a versão instalada do `transformers` não aceita (no 5.x, `overwrite_output_dir` foi removido). `--max-steps`, `--batch-size`, `--block-size` e `--out` continuam iguais.

## Reranker (Cross-Encoder)

**Desligado por padrão.** Domínio **v5** (e v4) passaram o gate no gold de **37** (híbrido 37/37, CE 37/37, Nasadiya incl. variantes ok) mas continua **opt-in**. Prefira v5 quando `artifacts/reranker_domain_v5` existir localmente. Genérico e v1–v3 não promover (9/10 no gold antigo). Ver [`docs/reranker_decision.md`](docs/reranker_decision.md).

```bash
# Opt-in explícito (só depois de um CE de domínio validado)
export VEDIC_ENABLE_RERANKER=true
export VEDIC_RERANKER_MODEL=artifacts/reranker_domain_v5

# Pares fracos a partir do gold (fixture minúscula, sem embeddings)
python scripts/build_rerank_pairs.py --dry-run --out data/rerank/pairs.jsonl

# Fine-tune ( --dry-run não baixa o modelo nem treina )
python scripts/train_reranker.py --help
python scripts/train_reranker.py --dry-run --pairs data/rerank/pairs.jsonl --out artifacts/reranker

# Gate A/B (híbrido vs CE de domínio). Exit 0 = apto a opt-in.
python scripts/eval_reranker_smoke.py --model artifacts/reranker_domain_v5 --json-out data/rerank_eval.json
```

## Jurídico

Fontes sem licença na lista permitida são bloqueadas. Material BBT/Vedabase somente com autorização explícita.

## Smoke / regressão

```bash
# health corpus + índice + gold expandido + ask extractive
python scripts/smoke_rag.py
python scripts/smoke_rag.py --backend numpy --strict
python scripts/smoke_rag.py --backend pgvector --strict --json-out data/smoke_report.json
```

Consultas de entidade ("Narada Muni", "Nārada", "Quem foi Vyāsa?") passam por
`vedic_pipeline/search/entity.py`: tira títulos e palavras de pergunta, soma grafias do corpus
(Griffith escreve "Nárad"), injeta os melhores trechos de cada obra que cita o nome, limita o
top-k por obra (os quatro volumes do Mahābhārata contam como uma) e, no `/ask`, usa 14 trechos,
contexto maior e um prompt que percorre cada tradição e diz quais obras de referência faltam.
Pergunta com hino ou obra nomeados segue o caminho do localizador. `VEDIC_ENTITY_MODE=false` desliga.

Gold set: `fixtures/smoke_queries.json` (37 queries: 19 anteriores + 18 beyond-stress após locator PRs #5–#9).  
CI: `.github/workflows/smoke.yml` + `fixtures/smoke_queries_ci.json` (recorte rápido de 4 queries). O gate de promote do CE usa o gold expandido, não o CI.

O smoke A/B (v1–v3 HOLD no gold de 10; **v5 PROMOTE** no gold de 37 — v4 também PROMOTE, default ainda off) está em [`docs/reranker_decision.md`](docs/reranker_decision.md); critérios do gate em [`docs/ce_promote_gate.md`](docs/ce_promote_gate.md).

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
- ~~Rerank cross-encoder opcional~~ ✅ (`VEDIC_ENABLE_RERANKER` default **off**; `VEDIC_RERANKER_MODEL`) — [decisão A/B](docs/reranker_decision.md)
- ~~Fila de jobs para ops pesadas~~ ✅ (`POST .../async` + `GET /jobs`)
- ~~Corpus reproduzível vs snapshot~~ ✅ (`vedic-pipeline corpus-status` + lock)
- ~~Checklist de deploy / geração fail-closed~~ ✅ (`deploy-check`, `VEDIC_REQUIRE_GENERATION_TOKEN`)
- ~~CE pós-PROMOTE sem default on~~ ✅ (`reranker-status`, `fixtures/reranker_promote.json`)  

## Desenvolvimento local revisado

Veja `docs/DEVELOPMENT_REVIEW.md` para o estado atual (corpus, deploy, CE). Comandos:

```bash
vedic-pipeline corpus-status --profile bootstrap
vedic-pipeline deploy-check --mode local   # ou --mode prod
vedic-pipeline reranker-status
```

As operações HTTP `/ingest`, `/tokenize`, `/train`, `/build-index`, `/db/init` e `/db/sync` (e suas variantes `/async`, mais `GET /jobs`) exigem `VEDIC_PIPELINE_API_TOKEN` e o cabeçalho `Authorization: Bearer <token>`. Sem token configurado, ficam desabilitadas (503). Os comandos da CLI continuam disponíveis sem esse token.

Testes & Qualidade: `pytest -v` (ou `python -m unittest discover -s tests -v`) e linter via `ruff check .`; frontend: `npm run lint`, `npm test` e `npm audit` dentro de `frontend/` (Node.js 22+). O score mostrado nas fontes é uma pontuação de ordenação, não uma probabilidade.

### Controle de geração e índice na API

Busca e chat aceitam somente o diretório definido em `VEDIC_API_INDEX_DIR` (padrão `artifacts/embeddings`). A CLI mantém a possibilidade de escolher outros diretórios.

Há dois modos de geração (`/ask` com xAI ou modelo local, e a mídia nova em `/audio`, `/image`, `/video`):

| Modo | Quando | Sem `VEDIC_GENERATION_API_TOKEN` |
|------|--------|----------------------------------|
| Privado (default) | `VEDIC_PUBLIC_API` desligado | A chave do servidor autoriza. A UI local funciona. |
| Público | `VEDIC_PUBLIC_API=1` (`docker-compose.prod.yml`) | **503.** Com o token configurado, a requisição envia `Authorization: Bearer <token>` ou recebe **401**. |

O modelo permitido vem de `XAI_MODEL` ou `VEDIC_LOCAL_LM`; o cliente não escolhe outro. `auto` com `XAI_API_KEY` cai em xAI e segue a mesma regra. A checagem acontece antes do streaming e antes de recuperar documentos.

`provider=extractive` continua aberto nos dois modos e não aceita `model`. Não exponha o token em `VITE_*` nem no bundle público. O token do pipeline não concede acesso à geração. Leitura de mídia já em cache continua aberta.
