"""Pre-registered hypothesis test для router vs static threshold на verified gold.

H1: router строго лучше static threshold хотя бы на одном из 3 pre-registered
бюджетов (25%, 40%, 50% LLM cost) после Bonferroni @α/3.

Этот модуль НЕ запускает эвалюацию — он только применяет statistical decision
rule к выходу router_eval_gold (router_stats_gold.parquet). Это обеспечивает,
что коррекция multiple comparison документирована и неизменна.
"""
from __future__ import annotations
import logging
from typing import Any

import pandas as pd

PRE_REGISTERED_BUDGETS: tuple[float, ...] = (0.25, 0.40, 0.50)
BONFERRONI_N: int = 3

logger = logging.getLogger(__name__)


def bonferroni_corrected_alpha(alpha: float = 0.05) -> float:
    return alpha / BONFERRONI_N


def evaluate_h1(stats: pd.DataFrame, alpha: float = 0.05) -> dict[str, Any]:
    """Применить H1 решающее правило.

    Parameters
    ----------
    stats : DataFrame с колонками budget_target, delta, p_mcnemar, ci_lo, ci_hi.
        Ожидается, что budget_target включает все 3 PRE_REGISTERED_BUDGETS.
    alpha : float

    Returns
    -------
    dict с ключами:
        h1_passed : bool
        significant_budgets : list of budgets where router strictly > static.
        alpha_corrected : float
        details : list of per-budget dicts.
    """
    alpha_c = bonferroni_corrected_alpha(alpha)
    significant: list[float] = []
    details = []
    for b in PRE_REGISTERED_BUDGETS:
        row = stats[stats["budget_target"] == b]
        if row.empty:
            logger.warning("Budget %.2f not found in stats — assuming non-significant.", b)
            details.append({"budget": b, "found": False})
            continue
        r = row.iloc[0]
        is_sig = (r["p_mcnemar"] < alpha_c) and (r["ci_lo"] > 0)
        if is_sig:
            significant.append(b)
        details.append({
            "budget": b,
            "found": True,
            "delta": r["delta"],
            "p_mcnemar": r["p_mcnemar"],
            "ci_lo": r["ci_lo"],
            "ci_hi": r["ci_hi"],
            "significant_after_bonferroni": is_sig,
        })
    return {
        "h1_passed": len(significant) > 0,
        "significant_budgets": significant,
        "alpha": alpha,
        "alpha_corrected": alpha_c,
        "bonferroni_n": BONFERRONI_N,
        "details": details,
    }


def main():
    """Прочитать router_stats_gold.parquet, применить H1, напечатать decision."""
    import json
    from src.common import PROCESSED_DIR
    stats = pd.read_parquet(f"{PROCESSED_DIR}/router_stats_gold.parquet")
    result = evaluate_h1(stats)
    print(json.dumps(result, indent=2, default=str))
    if result["h1_passed"]:
        print(f"\n[H1 PASS] significant budgets: {result['significant_budgets']}")
        print("→ Continue with 'router-centric' narrative (Phase F path A).")
    else:
        print("\n[H1 FAIL] no budget significantly favors router after Bonferroni.")
        print("→ Trigger Plan B4: production-ready cost-aware enrichment system (Phase F path B).")
    return result


if __name__ == "__main__":
    main()
