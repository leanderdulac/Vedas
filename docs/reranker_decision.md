# Decisão: Cross-Encoder reranker desligado por padrão

**Data da medição:** 2026-09-20  
**Máquina:** Mac `leandrofrancademello` · corpus local (~6973 docs / ~67170 chunks, MiniLM multilíngue, backend numpy; Postgres `:5433` offline)

## Resultado A/B no gold de smoke

Gold **do snapshot** (2026-09-20): `fixtures/smoke_queries.json` tinha **10 queries**. Relatórios locais: `data/smoke_report_rerank_off.json` / `data/smoke_report_rerank_on.json`.

O arquivo atual é o **gold expandido** (19 queries: 10 originais + 9 append-only verificadas no híbrido). O gate de promote (`eval_reranker_smoke.py`) e o smoke de retrieval devem usar esse gold — não o recorte histórico de 10 nem `fixtures/smoke_queries_ci.json`.

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

## Política atual

1. `VEDIC_ENABLE_RERANKER` **default off** (`false`). **Não promover** o CE (genérico nem domínio v1–v3).
2. `VEDIC_RERANKER_MODEL` pode ser um id HF **ou** um diretório local (`artifacts/reranker`); o loader resolve o path absoluto quando o dir existe.
3. Fine-tune só faz sentido com pares que **protejam Nasadiya** — positivos com marcadores de hino (`10.129` / Nasadiya) em vez do rótulo amplo "Rig Veda", e hard negatives 10.125 / 10.5 / 2.38 quando aparecerem nos candidatos — **e** com o boost de locator abaixo.
4. **Não** treinar GPT-2 / LM causal para ranking.

## Boost de locator / hino (híbrido ± CE)

Quando a query cita **Nasadiya** (→ 10.129), **Purusha Sukta** (→ 10.90) ou um **RV X.Y** explícito (`hymn 1.1`, `10.129`, `RV 10.90`), os candidatos cujo `title` / `locator` / texto contêm esse id sobem. Casa `10.5` sem `10.50`; também `HYMN X.129` (romano).

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

# 4. Só então, opt-in local/prod (default continua false)
export VEDIC_ENABLE_RERANKER=true
export VEDIC_RERANKER_MODEL=artifacts/reranker
```

### Critérios do gate (`eval_reranker_smoke.py`)

Promote (**exit 0**) somente se **ambos** valerem:

| Critério | Falha (exit 1) |
|----------|----------------|
| Pass rate do CE de domínio **≥** pass rate do híbrido (CE off) | `ce_worse_overall` |
| O CE **passa Nasadiya** (`nasadiya` / RV 10.129) quando a query está no gold | `nasadiya_failed` (+ `nasadiya_regressed` se o híbrido passava) |

Medição Mac 2026-09-20 (domínio v1–v3, gold de 10): **não promove** — 9/10 vs híbrido 10/10, Nasadiya falha com tops 10.125 / 10.5.

O JSON traz `per_query` (ok híbrido vs CE + `top_titles`) e um bloco dedicado `nasadiya`.

**Gold do gate:** sempre `fixtures/smoke_queries.json` expandido (19 queries: ids originais estáveis + Kaṭha Nachiketas, Gītā 2.47, Yoga 1.2, rapto de Sītā, leito de flechas de Bhīṣma, Īśā em Devanāgarī, Śvetāśvatara, Praśna, Muṇḍaka). Só entram queries com `expect_title_any` estrito que passaram no híbrido (CE off). O CI reduzido (`smoke_queries_ci.json`) não substitui o gate. CE continua **off** por default.

O default permanece **OFF**. `eval_reranker_smoke.py` troca `VEDIC_ENABLE_RERANKER` / `VEDIC_RERANKER_MODEL` no processo e recarrega o singleton; não deixa o CE ligado ao sair.

Latência p95 CPU não deve ficar >2× o híbrido (critério operacional, fora do exit code).

## O que não fazer agora

- Não substituir `/ask` / xAI por LM causal fine-tuned para “melhorar ranking”.
- Não treinar embedding sânscrito-específico antes de medir falhas reais do MiniLM atual.
- Não commitar pesos grandes nem corpus em git.
- Não ligar o CE genérico `cross-encoder/ms-marco-MiniLM-L-6-v2` **nem** o CE de domínio v1–v3 em produção.
- Não tratar o fine-tune gold-only como correção de Nasadiya.
