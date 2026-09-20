"""Comparação gold-smoke: híbrido (CE off) vs Cross-Encoder de domínio."""

from __future__ import annotations

from typing import Any


def is_nasadiya_item(item: dict[str, Any]) -> bool:
    blob = f"{item.get('id') or ''} {item.get('query') or ''}".lower()
    return "nasadiya" in blob or "nāsadīya" in blob


def _pass_rate(suite: dict[str, Any]) -> tuple[int, int, float]:
    total = int(suite.get("total") or len(suite.get("results") or []) or 0)
    passed = int(suite.get("passed") or 0)
    rate = (passed / total) if total else 0.0
    return passed, total, rate


def find_nasadiya_row(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in results:
        if is_nasadiya_item(item):
            return item
    return None


def compare_smoke_suites(
    hybrid: dict[str, Any],
    ce: dict[str, Any],
    *,
    model: str = "",
) -> dict[str, Any]:
    """Compara duas suítes `run_retrieval_suite` (híbrido vs CE).

    Gate de promote (exit 0):
    - pass rate do CE >= pass rate do híbrido
    - se o híbrido passa Nasadiya, o CE também passa
    """
    h_pass, h_total, h_rate = _pass_rate(hybrid)
    c_pass, c_total, c_rate = _pass_rate(ce)
    hybrid_results = list(hybrid.get("results") or [])
    ce_results = list(ce.get("results") or [])
    ce_by_id = {str(item.get("id") or ""): item for item in ce_results}

    per_query: list[dict[str, Any]] = []
    for h_item in hybrid_results:
        qid = str(h_item.get("id") or "")
        c_item = ce_by_id.get(qid) or {}
        per_query.append(
            {
                "id": qid,
                "query": h_item.get("query"),
                "hybrid_ok": bool(h_item.get("ok")),
                "ce_ok": bool(c_item.get("ok")),
                "hybrid_top_titles": list(h_item.get("top_titles") or []),
                "ce_top_titles": list(c_item.get("top_titles") or []),
                "nasadiya": is_nasadiya_item(h_item),
            }
        )
    for c_item in ce_results:
        qid = str(c_item.get("id") or "")
        if qid and all(row["id"] != qid for row in per_query):
            per_query.append(
                {
                    "id": qid,
                    "query": c_item.get("query"),
                    "hybrid_ok": False,
                    "ce_ok": bool(c_item.get("ok")),
                    "hybrid_top_titles": [],
                    "ce_top_titles": list(c_item.get("top_titles") or []),
                    "nasadiya": is_nasadiya_item(c_item),
                }
            )

    nas_h = find_nasadiya_row(hybrid_results)
    nas_c = find_nasadiya_row(ce_results)
    nasadiya: dict[str, Any] | None = None
    reasons: list[str] = []
    if c_rate < h_rate:
        reasons.append("ce_worse_overall")
    if nas_h is not None or nas_c is not None:
        seed = nas_h or nas_c or {}
        nasadiya = {
            "id": seed.get("id"),
            "query": seed.get("query"),
            "hybrid_ok": bool(nas_h and nas_h.get("ok")),
            "ce_ok": bool(nas_c and nas_c.get("ok")),
            "hybrid_top_titles": list((nas_h or {}).get("top_titles") or []),
            "ce_top_titles": list((nas_c or {}).get("top_titles") or []),
        }
        if nasadiya["hybrid_ok"] and not nasadiya["ce_ok"]:
            reasons.append("nasadiya_regressed")

    promote = not reasons
    return {
        "model": model,
        "hybrid_passed": h_pass,
        "hybrid_total": h_total,
        "hybrid_pass_rate": round(h_rate, 4),
        "ce_passed": c_pass,
        "ce_total": c_total,
        "ce_pass_rate": round(c_rate, 4),
        "nasadiya": nasadiya,
        "per_query": per_query,
        "promote": promote,
        "fail_reasons": reasons,
    }


def gate_exit_code(comparison: dict[str, Any]) -> int:
    return 0 if comparison.get("promote") else 1
