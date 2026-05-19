"""E9 Tier-breakdown: cascade accuracy stratified by gold signal_type.

Closes circularity question quantitatively: если cascade одинаково хорош
на nutri_derived (tier 0, фактический ground truth) и на text_derived
(tier 2, наибольшая структурная корреляция с признаками модели), то
основная критика «вы предсказываете теги, а не правду» — несостоятельна.

Tier mapping (из consensus_gold_v2_expanded.signal_type):
- nutri_derived  : числовое поле → бакет; tier 0, фактически ground truth
- tag_derived    : OFF labels_tags / categories_tags; tier 1, "что заявлено"
- text_derived   : парсинг ingredients_text / product_name; tier 2, proxy

Output: datasets/processed/tier_breakdown.parquet
Также печатает итоговую таблицу.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
PROCESSED = ROOT / "datasets" / "processed"
OUT = PROCESSED / "tier_breakdown.parquet"
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
    if len(df) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(RNG_SEED)
    brands = df.brand.unique()
    if len(brands) == 0:
        return (float(df.correct.mean()), float("nan"), float("nan"))
    by_brand = {b: df[df.brand == b].correct.values for b in brands}
    accs = np.empty(n_boot, dtype=float)
    n_brands = len(brands)
    for i in range(n_boot):
        sampled = rng.choice(brands, size=n_brands, replace=True)
        bag = np.concatenate([by_brand[b] for b in sampled])
        accs[i] = bag.mean()
    return (float(df.correct.mean()),
            float(np.percentile(accs, 2.5)),
            float(np.percentile(accs, 97.5)))


def main():
    gold = pd.read_parquet(PROCESSED / "consensus_gold_v2_expanded.parquet")
    gold = gold[~gold.gold_is_null].copy()
    gold["gold_norm"] = gold.gold_value.map(normalize)
    gold["code"] = gold["code"].astype(str)

    merged_all: list[pd.DataFrame] = []
    for cat in CATEGORIES:
        preds = pd.read_parquet(PROCESSED / f"cascade_preds_{cat}_v2_gold_hybrid_v3_fixed.parquet")
        preds["pred_norm"] = preds.predicted.map(normalize)
        preds["code"] = preds["code"].astype(str)
        brands_df = pd.read_parquet(
            PROCESSED / f"{cat}_stratified_silver_standard.parquet",
            columns=["code", "brands"],
        )
        brands_df["code"] = brands_df["code"].astype(str)
        gold_cat = gold[gold.category == cat]
        m = (preds[["code", "attr", "pred_norm"]]
             .merge(gold_cat[["code", "attr", "gold_norm", "signal_type"]],
                    on=["code", "attr"], how="inner")
             .merge(brands_df.drop_duplicates("code"), on="code", how="left"))
        m["brand"] = m["brands"].fillna("__nobrand__")
        m["correct"] = m.pred_norm == m.gold_norm
        m["category"] = cat
        merged_all.append(m[["category", "code", "attr", "signal_type", "brand", "correct"]])
    merged = pd.concat(merged_all, ignore_index=True)

    rows: list[dict] = []
    # By tier (global)
    for tier in ["nutri_derived", "tag_derived", "text_derived"]:
        sub = merged[merged.signal_type == tier]
        acc, lo, hi = brand_clustered_bootstrap(sub)
        rows.append({
            "scope": "global",
            "tier": tier,
            "n_attrs": sub.attr.nunique() if len(sub) else 0,
            "n_cells": len(sub),
            "n_brands": sub.brand.nunique() if len(sub) else 0,
            "acc": acc,
            "ci_lo": lo,
            "ci_hi": hi,
        })
    # By (category, tier)
    for cat in CATEGORIES:
        for tier in ["nutri_derived", "tag_derived", "text_derived"]:
            sub = merged[(merged.category == cat) & (merged.signal_type == tier)]
            if len(sub) == 0:
                continue
            acc, lo, hi = brand_clustered_bootstrap(sub)
            rows.append({
                "scope": cat,
                "tier": tier,
                "n_attrs": sub.attr.nunique(),
                "n_cells": len(sub),
                "n_brands": sub.brand.nunique(),
                "acc": acc,
                "ci_lo": lo,
                "ci_hi": hi,
            })

    out = pd.DataFrame(rows)
    out.to_parquet(OUT, index=False)
    logger.info("Saved %d rows → %s", len(out), OUT)

    print("\n" + "=" * 88)
    print("E9 — Tier breakdown (cascade v3e accuracy by gold signal_type)")
    print("=" * 88)
    print(f"{'scope':<10} {'tier':<14} {'attrs':>6} {'cells':>7} {'brands':>7} {'acc':>10} {'95% CI':>22}")
    print("-" * 88)
    tier_order = {"nutri_derived": 0, "tag_derived": 1, "text_derived": 2}
    out["__o1"] = out.scope.map({"global": 0, "pasta": 1, "chocolate": 2, "cheeses": 3})
    out["__o2"] = out.tier.map(tier_order)
    out_sorted = out.sort_values(["__o1", "__o2"]).drop(columns=["__o1", "__o2"])
    for _, r in out_sorted.iterrows():
        print(f"{r.scope:<10} {r.tier:<14} {r.n_attrs:>6} {r.n_cells:>7} {r.n_brands:>7}  "
              f"{r.acc*100:>8.2f}%  [{r.ci_lo*100:>5.2f},{r.ci_hi*100:>5.2f}]")
    print("=" * 88)


if __name__ == "__main__":
    main()
