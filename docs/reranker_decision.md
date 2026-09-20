# Decisão: Cross-Encoder reranker desligado por padrão

**Data da medição:** 2026-09-20  
**Máquina:** Mac `leandrofrancademello` · corpus local (~6973 docs / ~67170 chunks, MiniLM multilíngue, backend numpy; Postgres `:5433` offline)

## Resultado A/B no gold de smoke

Gold **do snapshot** (2026-09-20): `fixtures/smoke_queries.json` tinha **10 queries**. Relatórios locais: `data/smoke_report_rerank_off.json` / `data/smoke_report_rerank_on.json`.

O arquivo atual é o **gold expandido** (37 queries: 19 anteriores + 18 beyond-stress após locator PRs #5–#9). O gate de promote (`eval_reranker_smoke.py`) e o smoke de retrieval devem usar esse gold — não o recorte histórico de 10 nem `fixtures/smoke_queries_ci.json` (recorte rápido de CI).

| Modo | Pass | Notas |
|------|------|-------|
| Híbrido **sem** CrossEncoder (`VEDIC_ENABLE_RERANKER=false`) | **10/10** | ranking numpy + léxico + boosts |
| Reranker ON, CE **sem** cache | 10/10 | o modelo não carregou — ranking = OFF |
| Reranker ON, `cross-encoder/ms-marco-MiniLM-L-6-v2` em cache | **9/10** | CE genérico ativo |
| CE de **domínio** v1 (117 pares gold) | **9/10** | Nasadiya falha |
| CE de domínio v2 | **9/10** | Nasadiya falha |
| CE de domínio v3 (hard neg. explícitos 10.125 / 10.5) | **9/10** | tops errados: RV **10.125** depois **10.5** |

### Delta com o CE genérico ligado

- O top-3 mudou em **7/10** queries.
- **Regressão:** `nasadiya` (title_mismatch). O CE empurrou RV 10.125 / 10.5 / 2.38 no lugar de **RV 10.129**.
- Latência: 1ª query ~24s (cold start do CE); demais ~iguais ao OFF (+100–200ms).

### Fine-tune de domínio **não** corrigiu Nasadiya

Treinar o MiniLM ms-marco nos pares gold (v1→v3, inclusive hard negatives de 10.125 / 10.5) **não** tirou RV 10.129 do buraco: o smoke com CE de domínio continua **9/10**, mesmos decoys. O híbrido sem CE segue **10/10**.

Conclusão: CE fine-tune **sozinho** não é suficiente para opt-in. Antes de ligar:

1. **Boost de locator/hino** no híbrido (e de novo após o CE) — PoC local restaurou Nasadiya sob pressão do CE; agora está em `apply_locator_hymn_boost`.
2. **Pares maiores/melhores** (mais decoys, mais âncoras de hino, holdout real) — o snapshot de 10 queries era estreito demais para o CE generalizar 10.129 vs 10.125. O gold expandido (Kaṭha, Gītā 2.47, Yoga 1.2, épicos, Īśā em Devanāgarī, Śvetāśvatara, Praśna, Muṇḍaka) é o conjunto que o gate deve medir.

## Gate v4 (gold de 19; histórico)

Medição Mac 2026-09-20 no gold de **19** (antes da expansão beyond-stress). Critérios inalterados (CE ≥ híbrido, incluindo Nasadiya). Default continua off.

| Item | Valor |
|------|--------|
| Gold | `fixtures/smoke_queries.json` (**19** queries) |
| Pares locais (não commitados) | `data/rerank/pairs_gold19.jsonl` (**381** pares) |
| Modelo | `artifacts/reranker_domain_v4` (pesos locais; **não** commitados) |
| Híbrido (CE off) | **19/19** |
| Domínio CE v4 | **19/19** |
| Nasadiya | OK — RV **10.129** no topo (locator boost + CE) |
| `eval_reranker_smoke.py` | **PROMOTE** (CE >= híbrido, Nasadiya ok) |

v4 foi **apto a opt-in** já no gold de 19. Isso **não** ligou o CE no default.

## Gate v4 no gold-37 (Mac, 2026-09-20)

Medição Mac no gold atual (`fixtures/smoke_queries.json`, **37** queries):

| Item | Valor |
|------|--------|
| Híbrido (CE off) | **37/37** |
| Domínio CE v4 | **37/37** |
| Nasadiya (canônico + wrong-number / Devanāgarī / PT) | híbrido ✓ e CE ✓ |
| `eval_reranker_smoke.py` | **PROMOTE** (CE >= híbrido, Nasadiya ok) |

Apto a opt-in (caminho anterior). Default **continua OFF**.

## Gate v5 no gold-37 (Mac, 2026-09-20)

Medição Mac 2026-09-20 no gold atual (`fixtures/smoke_queries.json`, **37** queries). Treino local no gold-37; pesos **não** commitados. Default continua off.

| Item | Valor |
|------|--------|
| Gold | `fixtures/smoke_queries.json` (**37** queries) |
| Pares locais (não commitados) | `data/rerank/pairs_gold37.jsonl` (**487** pares: 210 pos / 277 neg; train 407 / eval 80) |
| Modelo | `artifacts/reranker_domain_v5` (pesos locais; **não** commitados) |
| Base | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Treino | epochs 3, batch 8, max_length 256, lr 2e-5, device mps (Mac); `train_meta.json` |
| Híbrido (CE off) | **37/37** |
| Domínio CE v5 | **37/37** |
| Nasadiya (canônico + wrong-number / Devanāgarī / PT) | híbrido ✓ e CE ✓ |
| `eval_reranker_smoke.py` | **PROMOTE** (CE >= híbrido, Nasadiya ok) |

v5 é o candidato mais recente treinado no gold-37 e o **opt-in recomendado** quando os pesos existem localmente. v4 continua um caminho de opt-in válido. Isso **não** liga o CE no default.

```bash
# Continua off sem estas variáveis (ou com ENABLE=false)
export VEDIC_ENABLE_RERANKER=true
export VEDIC_RERANKER_MODEL=artifacts/reranker_domain_v5
```

Critérios e o que o exit code mede: [`docs/ce_promote_gate.md`](ce_promote_gate.md).

## Política atual

1. `VEDIC_ENABLE_RERANKER` **default off** (`false`). Genérico e domínio v1–v3: **não promover**. Domínio **v5** (e **v4**) passaram o gate no gold de **37** (seções acima) mas **não** viram default — só opt-in via env apontando para um dir local. Prefira **v5** quando os pesos existirem localmente.
2. `VEDIC_RERANKER_MODEL` pode ser um id HF **ou** um diretório local (`artifacts/reranker_domain_v5`; v4 permanece válido); o loader resolve o path absoluto quando o dir existe.
3. Fine-tune só faz sentido com pares que **protejam Nasadiya** — positivos com marcadores de hino (`10.129` / Nasadiya) em vez do rótulo amplo "Rig Veda", e hard negatives 10.125 / 10.5 / 2.38 quando aparecerem nos candidatos — **e** com o boost de locator abaixo.
4. **Não** treinar GPT-2 / LM causal para ranking.

## Boost de locator / hino (híbrido ± CE)

Quando a query cita **Nasadiya** (→ 10.129), **Purusha Sukta** (→ 10.90), **Gāyatrī** (→ 3.62, também `tat savitur…` / `तत्सवितुर्…`), **Hiraṇyagarbha** (→ 10.121), **Vāk/Vāc Sūkta** (→ 10.125) ou um **RV X.Y** explícito (`hymn 1.1`, `10.129`, `RV 10.90`), os candidatos cujo `title` / `locator` / texto contêm esse **id** sobem. Casa `10.5` sem `10.50`; também `HYMN X.129` (romano). Gāyatrī não sobe Chandogya III.12 só porque o texto fala do metro. Se o nome canônico e um id explícito discordam (ex. “Nasadiya … 10.125”), o nome vence — o id conflitante não entra no boost.

**PR #5** = mapa nomeado + boost (reordena o que já está no pool). **PR #6** = injeção de recall: se `extract_query_hymn_ids` devolve um id RV, chunks cujo título/locator casam esse id entram no conjunto híbrido *antes* do score/boost (teto por id; mesmas regras de match — sem Chandogya pelo nome). Sem isso, “Gayatri mantra…” nunca vê RV 3.62 no top-40 e o boost não tem o que subir.

**Série seguinte (obra/Veda):** o mesmo padrão em nível de obra e coleção — extrair o rótulo da query, injetar chunks cujo *título* casa, boost (teto como no PR #6). Īśā (`īśāvāsyam…` / `ईशावास्यम्…` / `isavasya`) → Isha Upanishad, não RV que só partilha idaṃ/jagat. Nachiketas → Kaṭha mesmo se a query nomear Kauṣītaki. `Sama Veda` / `सामवेद` → títulos `Sāmaveda SV…`; `Shukla Yajur` / `Vajasaneyi` / `Yajurveda VS` → títulos `Yajurveda VS…`, não Chandogya nem antologia Müller.

**Passo 4 (near-miss / antologia):** se o hino ou a obra nomeada já tem um chunk de *título* específico no pool, rebaixa coleções genéricas (`Rig Veda selected hymns`, `Principal Upanishads (English core)`) — elas citam o texto no corpo e ganhavam o top-1. Bṛhadāraṇyaka (`neti neti`), Māṇḍūkya (pelo nome, não Om sozinho), Rāmāyaṇa (Sītā+abdução/Rāvaṇa vence rótulo Mahābhārata), Yoga-sūtra (definição de yoga PT/EN e Patañjali). Query curta `Agni` prefere hinos RV Agni; `dharma` sozinho não é pinado (sinal inseguro).

Aplicado **no híbrido** (antes do CE, para o hino certo entrar no slice) e **de novo depois do CE** quando o reranker está ligado — senão o CE volta a enterrar 10.129.

Isso **não** liga o CE. Só reduz a regressão se alguém optar pelo modelo local.

## Gate train → eval → promote

Não ligar o CE em produção por feeling. O loop é:

```text
build_rerank_pairs.py  →  train_reranker.py  →  eval_reranker_smoke.py  →  opt-in
         pares JSONL           artifacts/reranker      híbrido vs CE
```

```bash
# 1. Pares (fixture minúscula, sem embeddings) ou híbrido local (CE forçado off)
python scripts/build_rerank_pairs.py --dry-run --out data/rerank/pairs.jsonl
python scripts/build_rerank_pairs.py \
  --gold fixtures/smoke_queries.json \
  --candidates caminho/candidates.json \
  --out data/rerank/pairs.jsonl

# 2. Fine-tune (dry-run não baixa o modelo; grava train_meta.json com counts/device)
python scripts/train_reranker.py --dry-run --pairs data/rerank/pairs.jsonl --out artifacts/reranker
python scripts/train_reranker.py --pairs data/rerank/pairs.jsonl --out artifacts/reranker

# 3. Gate A/B no gold expandido — exit ≠ 0 se o CE de domínio for pior OU se Nasadiya falhar
python scripts/eval_reranker_smoke.py \
  --model artifacts/reranker \
  --queries fixtures/smoke_queries.json \
  --backend numpy \
  --json-out data/rerank_eval.json

# 4. Só então, opt-in local (default continua false)
export VEDIC_ENABLE_RERANKER=true
export VEDIC_RERANKER_MODEL=artifacts/reranker_domain_v5
```

### Critérios do gate (`eval_reranker_smoke.py`)

Promote (**exit 0**) somente se **ambos** valerem:

| Critério | Falha (exit 1) |
|----------|----------------|
| Pass rate do CE de domínio **≥** pass rate do híbrido (CE off) | `ce_worse_overall` |
| O CE **passa Nasadiya** (`nasadiya` / RV 10.129) quando a query está no gold | `nasadiya_failed` (+ `nasadiya_regressed` se o híbrido passava) |

Medição Mac 2026-09-20 (domínio v1–v3, gold de 10): **não promove** — 9/10 vs híbrido 10/10, Nasadiya falha com tops 10.125 / 10.5.

Medição Mac 2026-09-20 (domínio **v4**, gold de 19): **PROMOTE** — híbrido 19/19, CE 19/19, Nasadiya com RV 10.129 no topo. Default **continua OFF**.

Medição Mac 2026-09-20 (domínio **v4**, gold de **37**): **PROMOTE** — híbrido 37/37, CE 37/37, Nasadiya (incl. wrong-number / Devanāgarī / PT) ok. Default **continua OFF**.

Medição Mac 2026-09-20 (domínio **v5**, gold de **37**): **PROMOTE** — híbrido 37/37, CE 37/37, Nasadiya (incl. wrong-number / Devanāgarī / PT) ok. Default **continua OFF**. Opt-in recomendado quando `artifacts/reranker_domain_v5` existir localmente.

O JSON traz `per_query` (ok híbrido vs CE + `top_titles`) e um bloco dedicado `nasadiya`.

**Gold do gate:** sempre `fixtures/smoke_queries.json` expandido (**37** queries: 19 anteriores + 18 beyond-stress pós locator PRs #5–#9 — decoys de hino, Devanāgarī/IAST, PT, Veda/obra nomeada, épico errado). Só entram queries com `expect_title_any` estrito que passaram no híbrido (CE off; short-dharma excluído). O CI reduzido (`smoke_queries_ci.json`, 4 queries) não substitui o gate. CE continua **off** por default.

O default permanece **OFF**. `eval_reranker_smoke.py` troca `VEDIC_ENABLE_RERANKER` / `VEDIC_RERANKER_MODEL` no processo e recarrega o singleton; não deixa o CE ligado ao sair.

Latência p95 CPU não deve ficar >2× o híbrido (critério operacional, fora do exit code).

## O que não fazer agora

- Não substituir `/ask` / xAI por LM causal fine-tuned para “melhorar ranking”.
- Não treinar embedding sânscrito-específico antes de medir falhas reais do MiniLM atual.
- Não commitar pesos grandes, `data/rerank/*.jsonl` nem corpus em git.
- Não ligar o CE genérico `cross-encoder/ms-marco-MiniLM-L-6-v2` **nem** o CE de domínio v1–v3 em produção.
- Não tratar o fine-tune gold-only (v1–v3) como correção de Nasadiya. v5 (e v4) passaram o gate; mesmo assim o default fica **off**.
- Não setar `VEDIC_ENABLE_RERANKER=true` no `.env` de deploy sem um dir local validado.
