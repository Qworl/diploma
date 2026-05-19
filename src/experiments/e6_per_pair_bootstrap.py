"""E6: per-pair 95% bootstrap CI for Δ(hybrid - cascade) on consensus_gold test."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import PROCESSED_DIR
from src.pipeline.router.data import FOOD_CATS  # noqa: F401  (kept for parity with task spec)
from src.pipeline.router.train import _apply_gold_overrides, _enrich_with_product_meta
from src.pipeline.router.baselines import build_per_attr_table

N_BOOT = 5000
SEED = 42


def get_tier_for_pair(cat, attr):
    """Return tier of (cat, attr) per validation taxonomy."""
    try:
        from src.eval.validation_sources import get_tier
        t = get_tier(cat, attr)
        return t.value if t is not None else "unknown"
    except Exception:
        return "unknown"


def main():
    print("Loading router_train.parquet")
    df = pd.read_parquet(Path(PROCESSED_DIR) / "router_train.parquet")
    df = _enrich_with_product_meta(df, PROCESSED_DIR)
    df_with_gt, (train, val, test) = _apply_gold_overrides(df, PROCESSED_DIR)
    df = df_with_gt
    test = test[test["silver_gt"].notna()].copy()
    print(f"Test rows: {len(test)}")

    # build_per_attr_table needs cascade_correct + llm_correct columns
    train = train.copy()
    if "cascade_correct" not in train.columns:
        train["cascade_correct"] = (
            train["cascade_pred"].astype(str) == train["silver_gt"].astype(str)
        ).astype(int)
    train["llm_correct"] = (
        train["llm_pred"].astype(str) == train["silver_gt"].astype(str)
    ).astype(int)

    # Build naive per-attr policy from train
    policy_table = build_per_attr_table(train)

    # Apply policy to test → hybrid prediction
    # `build_per_attr_table` returns {(cat,attr): bool} where True means "use LLM"
    def _hybrid_pred(row):
        decision = policy_table.get((row["category"], row["attr"]))
        if decision is None:
            return row["cascade_pred"]
        if isinstance(decision, dict):
            if decision.get("route_to_llm") or decision.get("preferred") == "llm":
                return row["llm_pred"]
            return row["cascade_pred"]
        if isinstance(decision, (bool, np.bool_)):
            return row["llm_pred"] if bool(decision) else row["cascade_pred"]
        if str(decision).lower() == "llm":
            return row["llm_pred"]
        return row["cascade_pred"]
    test["hybrid_pred"] = test.apply(_hybrid_pred, axis=1)

    test["cas_correct"] = (test["cascade_pred"].astype(str) == test["silver_gt"].astype(str))
    test["hyb_correct"] = (test["hybrid_pred"].astype(str) == test["silver_gt"].astype(str))

    rng = np.random.default_rng(SEED)
    results = []
    pairs = sorted(test.groupby(["category", "attr"]).groups.keys())
    print(f"Pairs to bootstrap: {len(pairs)}")

    for cat, attr in pairs:
        sub = test[(test["category"] == cat) & (test["attr"] == attr)]
        if len(sub) < 5:
            continue
        codes = sub["code"].unique()
        cas_acc = float(sub["cas_correct"].mean())
        hyb_acc = float(sub["hyb_correct"].mean())
        delta_obs_pp = (hyb_acc - cas_acc) * 100

        # Cluster bootstrap by product code
        deltas = []
        for _ in range(N_BOOT):
            sampled_codes = rng.choice(codes, size=len(codes), replace=True)
            boot = pd.DataFrame({"code": sampled_codes}).merge(
                sub[["code", "cas_correct", "hyb_correct"]],
                on="code", how="left"
            )
            cas_b = boot["cas_correct"].mean()
            hyb_b = boot["hyb_correct"].mean()
            deltas.append(hyb_b - cas_b)
        deltas = np.array(deltas) * 100
        ci_low = float(np.percentile(deltas, 2.5))
        ci_high = float(np.percentile(deltas, 97.5))

        results.append({
            "category": cat,
            "attr": attr,
            "tier": get_tier_for_pair(cat, attr),
            "n_products": int(len(codes)),
            "n_cells": int(len(sub)),
            "cas_acc": cas_acc,
            "hyb_acc": hyb_acc,
            "delta_pp_observed": delta_obs_pp,
            "delta_pp_ci_low": ci_low,
            "delta_pp_ci_high": ci_high,
            "ci_crosses_zero": bool(ci_low <= 0 <= ci_high),
        })

    out_df = pd.DataFrame(results)
    out_path = Path(PROCESSED_DIR) / "per_pair_bootstrap_ci.parquet"
    out_df.to_parquet(out_path)
    print(out_df.to_string(index=False))

    # Summary by tier
    summary_per_tier = []
    for tier in ["gold", "close_to_gold", "silver_strong", "unknown"]:
        sub = out_df[out_df["tier"] == tier]
        if sub.empty:
            continue
        summary_per_tier.append({
            "tier": tier,
            "n_pairs": int(len(sub)),
            "n_pairs_ci_crosses_zero": int(sub["ci_crosses_zero"].sum()),
            "mean_delta_pp": float(sub["delta_pp_observed"].mean()),
            "max_delta_pp": float(sub["delta_pp_observed"].max()),
            "min_delta_pp": float(sub["delta_pp_observed"].min()),
        })

    summary_path = Path(PROCESSED_DIR) / "per_pair_bootstrap_ci_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"per_tier_summary": summary_per_tier, "n_pairs_total": int(len(out_df))}, f, indent=2)
    print(f"\nSaved: {out_path}, {summary_path}")


if __name__ == "__main__":
    main()
