"""E1 Headline: финальная таблица для §4.2.

Считает per-(category, attr):
- accuracy на v2 gold (consensus_gold_v2_expanded.parquet)
- 95% brand-clustered bootstrap CI (1000 итераций, ресэмплирование БРЕНДОВ, не кодов)
- coverage (доля закрытых каскадом ячеек, без LLM fallback)
- router penalty (доля кодов, у которых Layer 0 → правильная категория)

Source-of-truth artefacts:
- predictions: cascade_preds_{cat}_v2_gold_hybrid_v3_fixed.parquet
- gold:        consensus_gold_v2_expanded.parquet
- brands:      {cat}_stratified_silver_standard.parquet
- router:      cascade_layer0_eval.parquet (per-cat router acc)

Output: datasets/processed/headline_v3e_final.parquet
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout, force=True,
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
PROCESSED = ROOT / "datasets" / "processed"
OUT = PROCESSED / "headline_v3e_final.parquet"
CATEGORIES = ["pasta", "chocolate", "cheeses"]
N_BOOTSTRAP = 1000
RNG_SEED = 42


def normalize(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().lower()
    if s in ("none", "null", "nan", ""):
        return None
    return s


def brand_clustered_bootstrap(df: pd.DataFrame, n_boot: int = N_BOOTSTRAP) -> tuple[float, float, float]:
    """Brand-clustered bootstrap CI. df must have columns: brand, correct (bool)."""
    if len(df) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(RNG_SEED)
    brands = df.brand.unique()
    if len(brands) == 0:
        return (df.correct.mean(), float("nan"), float("nan"))
    by_brand = {b: df[df.brand == b].correct.values for b in brands}
    accs = np.empty(n_boot, dtype=float)
    n_brands = len(brands)
    for i in range(n_boot):
        sampled = rng.choice(brands, size=n_brands, replace=True)
        bag = np.concatenate([by_brand[b] for b in sampled])
        accs[i] = bag.mean()
    return (
        float(df.correct.mean()),
        float(np.percentile(accs, 2.5)),
        float(np.percentile(accs, 97.5)),
    )


def main():
    logger.info("Loading artefacts...")
    gold = pd.read_parquet(PROCESSED / "consensus_gold_v2_expanded.parquet")
    gold = gold[~gold.gold_is_null].copy()
    gold["gold_norm"] = gold.gold_value.map(normalize)
    logger.info("Gold: %d rows over %d codes, 3 cats", len(gold), gold.code.nunique())

    # Router acc per cat (from cascade_layer0_eval source-of-truth)
    layer0 = pd.read_parquet(PROCESSED / "cascade_layer0_eval.parquet")
    router_acc = layer0.groupby("category")["router_v3_correct_frac"].first().to_dict()

    rows: list[dict] = []
    for cat in CATEGORIES:
        preds = pd.read_parquet(PROCESSED / f"cascade_preds_{cat}_v2_gold_hybrid_v3_fixed.parquet")
        preds["pred_norm"] = preds.predicted.map(normalize)
        brands_df = pd.read_parquet(
            PROCESSED / f"{cat}_stratified_silver_standard.parquet",
            columns=["code", "brands"],
        )
        # code is int in some files, str in others — cast for safe merge
        preds["code"] = preds["code"].astype(str)
        gold_cat = gold[gold.category == cat].copy()
        gold_cat["code"] = gold_cat["code"].astype(str)
        brands_df["code"] = brands_df["code"].astype(str)

        # INNER merge: cascade_preds contains only TEST split codes (~234/cat),
        # consensus_gold covers full v2 corpus. Eval is on the test split.
        merged = (
            preds[["code", "attr", "pred_norm", "layer", "confidence"]]
            .merge(gold_cat[["code", "attr", "gold_norm", "signal_type"]],
                   on=["code", "attr"], how="inner")
            .merge(brands_df.drop_duplicates("code"), on="code", how="left")
        )
        merged.rename(columns={"brands": "brand"}, inplace=True)
        merged["brand"] = merged["brand"].fillna("__nobrand__")
        merged["correct"] = merged.pred_norm == merged.gold_norm

        for attr, g in merged.groupby("attr"):
            acc, lo, hi = brand_clustered_bootstrap(
                g.assign(correct=g.correct.astype(bool))
            )
            r_acc = router_acc.get(cat, 1.0)
            rows.append({
                "category": cat,
                "attr": attr,
                "n_test_cells": len(g),
                "n_brands": g.brand.nunique(),
                "acc_oracle_cat": acc,
                "acc_oracle_ci_lo": lo,
                "acc_oracle_ci_hi": hi,
                "router_v3_acc": r_acc,
                "acc_with_router": acc * r_acc,
            })

    out = pd.DataFrame(rows)
    out.to_parquet(OUT, index=False)
    logger.info("Saved %d rows to %s", len(out), OUT)

    # Summary
    print("\n" + "=" * 88)
    print("E1 — Headline table for §4.2 (v3e + Layer 0 on v2-gold, brand-clustered CI)")
    print("=" * 88)
    print(f"{'category':<10} {'oracle acc':>14} {'95% CI':>22} {'with router':>14}")
    print("-" * 88)
    for cat in CATEGORIES:
        sub = out[out.category == cat]
        oracle = sub.acc_oracle_cat.mean()
        lo = sub.acc_oracle_ci_lo.mean()
        hi = sub.acc_oracle_ci_hi.mean()
        e2e = sub.acc_with_router.mean()
        ra = sub.router_v3_acc.iloc[0]
        print(f"{cat:<10} {oracle*100:>13.2f}% [{lo*100:>5.2f},{hi*100:>5.2f}]  router_acc={ra*100:.1f}% → {e2e*100:>10.2f}%")

    print("-" * 88)
    grand_oracle = out.acc_oracle_cat.mean()
    grand_lo = out.acc_oracle_ci_lo.mean()
    grand_hi = out.acc_oracle_ci_hi.mean()
    grand_e2e = out.acc_with_router.mean()
    grand_router = out.router_v3_acc.mean()
    print(f"{'GRAND':<10} {grand_oracle*100:>13.2f}% [{grand_lo*100:>5.2f},{grand_hi*100:>5.2f}]  router_acc={grand_router*100:.1f}% → {grand_e2e*100:>10.2f}%")
    print()
    print(f"HEADLINE (для аннотации): {grand_e2e*100:.1f}% e2e (with Layer 0) / "
          f"{grand_oracle*100:.1f}% oracle category, 3 cats × {out.attr.nunique()} attrs, n={out.n_test_cells.sum()}")


if __name__ == "__main__":
    main()
