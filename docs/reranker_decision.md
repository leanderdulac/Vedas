# Decisão: Cross-Encoder reranker desligado por padrão

**Data da medição:** 2026-09-20  
**Máquina:** Mac `leandrofrancademello` · corpus local (~6973 docs / ~67170 chunks, MiniLM multilíngue, backend numpy; Postgres `:5433` offline)

## Resultado A/B no gold de smoke

Gold: `fixtures/smoke_queries.json` (10 queries). Relatórios locais: `data/smoke_report_rerank_off.json` / `data/smoke_report_rerank_on.json`.

| Modo | Pass | Notas |
|------|------|-------|
| Híbrido **sem** CrossEncoder (`VEDIC_ENABLE_RERANKER=false`) | **10/10** | ranking numpy + léxico + boosts |
| Reranker ON, CE **sem** cache | 10/10 | o modelo não carregou — ranking = OFF |
| Reranker ON, `cross-encoder/ms-marco-MiniLM-L-6-v2` em cache | **9/10** | CE genérico ativo |

### Delta com o CE genérico ligado

- O top-3 mudou em **7/10** queries.
- **Regressão:** `nasadiya` (title_mismatch). O CE empurrou RV 10.125 / 10.5 / 2.38 no lugar de **RV 10.129**.
- Latência: 1ª query ~24s (cold start do CE); demais ~iguais ao OFF (+100–200ms).

Conclusão: o MiniLM ms-marco **não** é ganho gratuito neste gold. O híbrido sozinho já passa 10/10.

## Política atual

1. `VEDIC_ENABLE_RERANKER` **default off** (`false`). O caminho de produção não carrega o CE a menos que se opte explicitamente (`true` / `1` / `on` / `yes`).
2. `VEDIC_RERANKER_MODEL` pode ser um id HF **ou** um diretório local (`artifacts/reranker`); o loader resolve o path absoluto quando o dir existe.
3. Fine-tune só faz sentido com pares que **protejam Nasadiya** — positivos com marcadores de hino (`10.129` / Nasadiya) em vez do rótulo amplo "Rig Veda", e hard negatives 10.125 / 10.5 / 2.38 quando aparecerem nos candidatos.
4. **Não** treinar GPT-2 / LM causal para ranking.

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

# 3. Gate A/B no gold — exit ≠ 0 se o CE de domínio for pior OU se Nasadiya regride
python scripts/eval_reranker_smoke.py \
  --model artifacts/reranker \
  --queries fixtures/smoke_queries.json \
  --backend numpy \
  --json-out data/rerank_eval.json

# 4. Só então, opt-in local/prod
export VEDIC_ENABLE_RERANKER=true
export VEDIC_RERANKER_MODEL=artifacts/reranker
```

### Critérios do gate (`eval_reranker_smoke.py`)

Promote (**exit 0**) somente se **ambos** valerem:

| Critério | Falha (exit 1) |
|----------|----------------|
| Pass rate do CE de domínio **≥** pass rate do híbrido (CE off) | `ce_worse_overall` |
| Se o híbrido **passa** Nasadiya (`nasadiya` / RV 10.129), o CE também passa | `nasadiya_regressed` |

O JSON traz `per_query` (ok híbrido vs CE + `top_titles`) e um bloco dedicado `nasadiya`.

O default permanece **OFF**. `eval_reranker_smoke.py` troca `VEDIC_ENABLE_RERANKER` / `VEDIC_RERANKER_MODEL` no processo e recarrega o singleton; não deixa o CE ligado ao sair.

Latência p95 CPU não deve ficar >2× o híbrido (critério operacional, fora do exit code).

## O que não fazer agora

- Não substituir `/ask` / xAI por LM causal fine-tuned para “melhorar ranking”.
- Não treinar embedding sânscrito-específico antes de medir falhas reais do MiniLM atual.
- Não commitar pesos grandes nem corpus em git.
- Não ligar o CE genérico `cross-encoder/ms-marco-MiniLM-L-6-v2` em produção.
