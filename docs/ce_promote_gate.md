# Gate de promote do Cross-Encoder

O CE **não** entra no default só porque um fine-tune “parece melhor”. `scripts/eval_reranker_smoke.py` compara o híbrido (CE forçado off) com um modelo local e decide **PROMOTE** ou **HOLD**.

Resumo operacional da decisão: [`docs/reranker_decision.md`](reranker_decision.md).

## Gold

Sempre `fixtures/smoke_queries.json` — **37** queries (19 anteriores + 18 beyond-stress após locator PRs #5–#9, verificadas no híbrido Mac top-1; CE off). Não usar o recorte histórico de 10 nem `fixtures/smoke_queries_ci.json` (recorte rápido de CI, 4 queries — não explode o runtime; não substitui o gate).

O gold cresceu. Os critérios de promote **não** mudam: CE ≥ híbrido no pass rate, incluindo Nasadiya. Medição Mac no gold-37: **PROMOTE** (37/37 ambos os lados). Default do CE continua off. Short-dharma ficou de fora de propósito (24/25 no stress).

Pares usados no treino local do v4 (não commitados): `data/rerank/pairs_gold19.jsonl` (381 pares) — medição histórica no gold de 19.

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

## Resultado v4 no gold de 19 (Mac, 2026-09-20) — histórico

Medição histórica no gold de **19** (antes da expansão beyond-stress):

| Modo | Pass | Notas |
|------|------|-------|
| Híbrido (CE off) | **19/19** | locator/hymn boost |
| Domínio CE `artifacts/reranker_domain_v4` | **19/19** | Nasadiya OK — RV **10.129** no topo (locator boost + CE) |

Saída: **PROMOTE** (CE >= híbrido, Nasadiya ok). Isso **não** ligou o CE.

v1–v3 (gold de 10): HOLD — 9/10 vs híbrido 10/10, Nasadiya com tops 10.125 / 10.5.

## Resultado v4 no gold-37 (Mac, 2026-09-20)

Comando (parent, Mac, backend numpy):

```bash
python scripts/eval_reranker_smoke.py \
  --model artifacts/reranker_domain_v4 \
  --queries fixtures/smoke_queries.json \
  --backend numpy
```

| Modo | Pass | Notas |
|------|------|-------|
| Híbrido (CE off) | **37/37** | gold expandido (19 + 18 beyond-stress) |
| Domínio CE `artifacts/reranker_domain_v4` | **37/37** | CE ≥ híbrido |

Nasadiya no gold-37 — híbrido ✓ e CE ✓, inclusive variantes:

- `nasadiya` (canônico)
- wrong-number (`wrong-nasadiya-10-125`)
- Devanāgarī (`nasadiya-devanagari-only`)
- PT (`pt-nasadiya`)

Saída: **PROMOTE** (CE >= híbrido, Nasadiya ok). Apto a opt-in. Default **continua off**.

```bash
# Continua off sem estas variáveis (ou com ENABLE=false)
export VEDIC_ENABLE_RERANKER=true
export VEDIC_RERANKER_MODEL=artifacts/reranker_domain_v4
```

## Política depois do PROMOTE

PROMOTE = **apto a opt-in**, não “ligar em produção”.

- Default permanece `VEDIC_ENABLE_RERANKER=false`.
- Opt-in: `VEDIC_ENABLE_RERANKER=true` e `VEDIC_RERANKER_MODEL` apontando para um **dir local** (ex. `artifacts/reranker_domain_v4`).
- Não commitar pesos, `data/rerank/*.jsonl` nem o modelo.

Série de locator (híbrido, CE continua off): **PR #5** mapa nomeado de hino (Gāyatrī→3.62, Hiraṇyagarbha→10.121, Vāk→10.125; nome vence id conflitante) → **PR #6** injeção de recall pelo id RV → **PR #8** locator de obra/Veda (Īśā, Nachiketas→Kaṭha, Sāmaveda SV, Śukla Yajur / Vājasaneyi VS) → **este (passo 4)** near-miss / demote de antologia: quando o título específico da obra/hino está no pool, coleções genéricas (`Rig Veda selected hymns`, `Principal Upanishads`) descem; Sītā+abdução → Rāmāyaṇa (mesmo com rótulo Mahābhārata); neti neti → Bṛhadāraṇyaka; Māṇḍūkya pelo nome; definição de yoga PT/EN → Yoga-sūtra. Mesmo teto de injeção. Isso não liga o CE.
