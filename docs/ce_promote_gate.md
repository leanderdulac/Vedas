# Gate de promote do Cross-Encoder

O CE **não** entra no default só porque um fine-tune “parece melhor”. `scripts/eval_reranker_smoke.py` compara o híbrido (CE forçado off) com um modelo local e decide **PROMOTE** ou **HOLD**.

Resumo operacional da decisão: [`docs/reranker_decision.md`](reranker_decision.md).

## Gold

Sempre `fixtures/smoke_queries.json` — **19** queries (10 originais + 9 append-only verificadas no híbrido). Não usar o recorte histórico de 10 nem `fixtures/smoke_queries_ci.json`.

Pares usados no treino local do v4 (não commitados): `data/rerank/pairs_gold19.jsonl` (381 pares).

## Critérios (exit 0 = PROMOTE)

Promote somente se **ambos** valerem:

| Critério | Falha (exit 1) |
|----------|----------------|
| Pass rate do CE de domínio **≥** pass rate do híbrido (CE off) | `ce_worse_overall` |
| O CE **passa Nasadiya** (`nasadiya` / RV 10.129) quando a query está no gold | `nasadiya_failed` (+ `nasadiya_regressed` se o híbrido passava) |

Latência p95 CPU não deve ficar >2× o híbrido (critério operacional, **fora** do exit code).

```bash
python scripts/eval_reranker_smoke.py \
  --model artifacts/reranker_domain_v4 \
  --queries fixtures/smoke_queries.json \
  --backend numpy \
  --json-out data/rerank_eval.json
```

O script troca `VEDIC_ENABLE_RERANKER` / `VEDIC_RERANKER_MODEL` só no processo e restaura ao sair. Não deixa o CE ligado.

## Resultado v4 (Mac, 2026-09-20)

| Modo | Pass | Notas |
|------|------|-------|
| Híbrido (CE off) | **19/19** | locator/hymn boost |
| Domínio CE `artifacts/reranker_domain_v4` | **19/19** | Nasadiya OK — RV **10.129** no topo (locator boost + CE) |

Saída: **PROMOTE** (CE >= híbrido, Nasadiya ok).

v1–v3 (gold de 10): HOLD — 9/10 vs híbrido 10/10, Nasadiya com tops 10.125 / 10.5.

## Política depois do PROMOTE

PROMOTE = **apto a opt-in**, não “ligar em produção”.

- Default permanece `VEDIC_ENABLE_RERANKER=false`.
- Opt-in: `VEDIC_ENABLE_RERANKER=true` e `VEDIC_RERANKER_MODEL` apontando para um **dir local** (ex. `artifacts/reranker_domain_v4`).
- Não commitar pesos, `data/rerank/*.jsonl` nem o modelo.

O mapa nomeado de locator (PR #5, pós-v4, híbrido) inclui também Gāyatrī→3.62, Hiraṇyagarbha→10.121 e Vāk/Vāc Sūkta→10.125; nome canônico vence id explícito conflitante. A injeção de recall (PR seguinte) coloca no pool híbrido chunks cujo título/locator casam o id extraído — o boost do PR #5 sozinho não recupera Gāyatrī se RV 3.62 nunca entrou nos candidatos. Isso não liga o CE.
