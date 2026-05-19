"""E4: rerun headline with table_gated policy instead of naive per_attr_table.

McNemar (one-sided binomtest on discordant pairs) p<0.05 AND Δ_acc > 3pp
gates the per-(cat, attr) decision to actually route to LLM. Otherwise
keep cascade. Compares against the naive per_attr_table row from
router_pareto_gold.parquet (already produced upstream).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.common import PROCESSED_DIR
from src.pipeline.router.train import _apply_gold_overrides, _enrich_with_product_meta


def build_per_attr_table_gated(train_df, p_threshold=0.05, min_delta_pp=3.0):
    """Build per-(cat, attr) policy gated by McNemar significance + min delta."""
    decisions = {}
    train_df = train_df[train_df["silver_gt"].notna()].copy()
    for (cat, attr), group in train_df.groupby(["category", "attr"]):
        cas_correct = (group["cascade_pred"].astype(str) == group["silver_gt"].astype(str))
        llm_correct = (group["llm_pred"].astype(str) == group["silver_gt"].astype(str))
        n_b = int((cas_correct & ~llm_correct).sum())
        n_c = int((~cas_correct & llm_correct).sum())
        cas_acc = float(cas_correct.mean())
        llm_acc = float(llm_correct.mean())
        delta_pp = (llm_acc - cas_acc) * 100

        if (n_b + n_c) > 0:
            p = float(binomtest(n_c, n_b + n_c, p=0.5, alternative="greater").pvalue)
        else:
            p = 1.0

        gate_llm = (delta_pp > min_delta_pp) and (p < p_threshold)
        decisions[(cat, attr)] = {
            "route_to_llm": bool(gate_llm),
            "cas_acc": cas_acc,
            "llm_acc": llm_acc,
            "delta_pp": delta_pp,
            "mcnemar_p": p,
            "n": int(len(group)),
        }
    return decisions


def apply_gated_policy(test_df, gated_decisions):
    """Apply gated policy → final_pred per row."""
    rows = []
    for _, r in test_df.iterrows():
        key = (r["category"], r["attr"])
        decision = gated_decisions.get(key, {"route_to_llm": False})
        if decision["route_to_llm"]:
            pred = r["llm_pred"]
            routed_llm = 1
        else:
            pred = r["cascade_pred"]
            routed_llm = 0
        rows.append({
            "code": r["code"], "category": r["category"], "attr": r["attr"],
            "final_pred": pred, "routed_to_llm": routed_llm,
            "silver_gt": r["silver_gt"],
        })
    return pd.DataFrame(rows)


def main():
    print("Loading router_train.parquet")
    df = pd.read_parquet(Path(PROCESSED_DIR) / "router_train.parquet")
    df = _enrich_with_product_meta(df, PROCESSED_DIR)
    df_with_gt, (train, val, test) = _apply_gold_overrides(df, PROCESSED_DIR)
    print(f"Train rows: {len(train)}; Test rows: {len(test)}")

    gated_decisions = build_per_attr_table_gated(train)
    final_df = apply_gated_policy(test, gated_decisions)
    final_df["correct"] = (
        final_df["final_pred"].astype(str) == final_df["silver_gt"].astype(str)
    )

    headline_acc = float(final_df["correct"].mean())
    llm_cost = float(final_df["routed_to_llm"].mean())

    # Compare with naive per_attr_table from cached pareto file.
    # Schema there: strategy / threshold / cost / accuracy (single row for per_attr_table).
    naive_pareto = pd.read_parquet(Path(PROCESSED_DIR) / "router_pareto_gold.parquet")
    naive_sub = naive_pareto[naive_pareto["strategy"] == "per_attr_table"]
    if not naive_sub.empty:
        # per_attr_table is a single static decision per (cat, attr), so usually
        # only one row. If multiple appear, pick the one closest to its own cost.
        if len(naive_sub) > 1:
            idx = (naive_sub["cost"] - llm_cost).abs().idxmin()
        else:
            idx = naive_sub.index[0]
        naive_acc = float(naive_sub.loc[idx, "accuracy"])
        naive_cost = float(naive_sub.loc[idx, "cost"])
    else:
        naive_acc = naive_cost = None

    delta_pp = (headline_acc - naive_acc) * 100 if naive_acc is not None else None
    if delta_pp is None:
        decision = "insufficient_data"
    elif delta_pp >= -0.3:
        decision = "switch_to_table_gated_production"
    else:
        decision = "keep_naive_with_sensitivity_note"

    n_routed_pairs = sum(1 for d in gated_decisions.values() if d["route_to_llm"])
    n_total_pairs = len(gated_decisions)

    # --- Fair-cost Pareto comparison ---
    # At cost=0.20 (gated's natural cost), what does static_threshold curve give?
    # This is a fairer comparison: both at same cost.
    static_curve = naive_pareto[naive_pareto["strategy"] == "static_threshold"].sort_values("cost")
    if not static_curve.empty:
        if 0.20 < static_curve["cost"].min() or 0.20 > static_curve["cost"].max():
            static_acc_at_020 = None
        else:
            static_acc_at_020 = float(np.interp(0.20,
                                                 static_curve["cost"].values,
                                                 static_curve["accuracy"].values))
    else:
        static_acc_at_020 = None

    # Also: what's the naive_pareto per_attr_table at ITS cost (not interpolated)?
    naive_at_natural = float(naive_pareto[naive_pareto["strategy"] == "per_attr_table"]["accuracy"].iloc[0]) if not naive_pareto[naive_pareto["strategy"] == "per_attr_table"].empty else None
    naive_natural_cost = float(naive_pareto[naive_pareto["strategy"] == "per_attr_table"]["cost"].iloc[0]) if not naive_pareto[naive_pareto["strategy"] == "per_attr_table"].empty else None

    fair_delta_pp_vs_static = (headline_acc - static_acc_at_020) * 100 if static_acc_at_020 is not None else None

    # Pareto-dominance check: gated dominates naive iff (gated acc >= naive acc AND gated cost <= naive cost)
    # OR (gated acc > naive acc AND gated cost <= naive cost)
    if naive_at_natural is not None and naive_natural_cost is not None:
        gated_dominates = (headline_acc >= naive_at_natural) and (llm_cost <= naive_natural_cost)
        naive_dominates = (naive_at_natural >= headline_acc) and (naive_natural_cost <= llm_cost)
        if gated_dominates:
            pareto_status = "gated_dominates_naive"
        elif naive_dominates:
            pareto_status = "naive_dominates_gated"
        else:
            pareto_status = "non_dominated_pareto_choice"
    else:
        pareto_status = "insufficient_data_for_pareto"

    summary = {
        "table_gated_headline_acc": headline_acc,
        "table_gated_llm_cost": llm_cost,
        "naive_headline_acc": naive_acc,
        "naive_llm_cost": naive_cost,
        "delta_pp": delta_pp,
        "n_routed_pairs": n_routed_pairs,
        "n_total_pairs": n_total_pairs,
        "decision": decision,
        "fair_cost_analysis": {
            "static_threshold_acc_at_gated_cost_020": static_acc_at_020,
            "fair_delta_pp_vs_static_at_020": fair_delta_pp_vs_static,
            "naive_table_natural_cost": naive_natural_cost,
            "naive_table_natural_acc": naive_at_natural,
            "gated_table_cost": llm_cost,
            "gated_table_acc": headline_acc,
            "pareto_status": pareto_status,
            "comment": (
                "naive per_attr_table is a single fixed config (no cost-tuning). "
                "Comparing 'naive at its natural cost' vs 'gated at its natural cost' "
                "is cost-confounded. The fair_delta vs static_threshold interpolated "
                "at gated cost isolates gating effect on Pareto position."
            ),
        },
    }

    pareto_row = pd.DataFrame([{
        "strategy": "per_attr_table_gated",
        "threshold": float("nan"),
        "cost": llm_cost,
        "accuracy": headline_acc,
    }])
    pareto_row.to_parquet(
        Path(PROCESSED_DIR) / "router_pareto_gold_table_gated.parquet"
    )

    with open(
        Path(PROCESSED_DIR) / "router_pareto_gold_table_gated_summary.json", "w"
    ) as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
