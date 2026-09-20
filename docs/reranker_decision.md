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
2. Manter o CE genérico desligado até existir um modelo **adaptado ao domínio** que bata o híbrido num gold **expandido** (não só as 10 queries de smoke).
3. Fine-tune só faz sentido com pares que **protejam Nasadiya** e as demais queries óbvias — labels fracos a partir de `expect_title_any`, sem inventar teologia.

## Como treinar um CE de domínio (scaffold)

Não promover automaticamente. Fluxo:

```bash
# 1. Pares a partir do gold (fixture minúscula, sem embeddings)
python scripts/build_rerank_pairs.py --dry-run --out data/rerank/pairs.jsonl

# 1b. Ou candidatos pré-computados / híbrido local (CE forçado off)
python scripts/build_rerank_pairs.py \
  --gold fixtures/smoke_queries.json \
  --candidates caminho/candidates.json \
  --out data/rerank/pairs.jsonl

# 2. Fine-tune (dry-run não baixa o modelo)
python scripts/train_reranker.py --dry-run --pairs data/rerank/pairs.jsonl
python scripts/train_reranker.py --pairs data/rerank/pairs.jsonl --out artifacts/reranker

# 3. Avaliar — só ligar o CE se Δ ≥ 0 no gold + hold-out, sem regressão Nasadiya
VEDIC_ENABLE_RERANKER=true VEDIC_RERANKER_MODEL=artifacts/reranker \
  python scripts/smoke_rag.py --backend numpy --strict
```

Critérios de promote (plano 2026-09-20): smoke ≥ baseline (ideal 10/10); hold-out ≥ baseline; latência p95 CPU não >2× o híbrido.

## O que não fazer agora

- Não substituir `/ask` / xAI por LM causal fine-tuned para “melhorar ranking”.
- Não treinar embedding sânscrito-específico antes de medir falhas reais do MiniLM atual.
- Não commitar pesos grandes nem corpus em git.
