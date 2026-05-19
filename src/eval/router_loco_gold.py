"""LOCO router evaluation на verified gold (Task E.4).

Для каждой категории c из FOOD_CATS:
  1. Загрузить router_train + apply gold overrides + brand-disjoint splits.
  2. Train router на (train + val) рядов где category != c
     (с внутренним 90/10 re-split на train/val для калибратора).
  3. Eval на test рядах где category == c.
  4. Compute Δ(router − static) на 3 pre-registered бюджетах (25%, 40%, 50%).

Output: datasets/processed/router_loco_gold.parquet
  (6 категорий × 3 бюджета = до 18 строк).

LOCO leakage check: brand_set / class_freq_table / brand_attr_acc_table
строятся внутри `train_router` ТОЛЬКО на train split (см. строки 123-125
src/pipeline/router/train.py), поэтому передача splits=(train_o, val_o, test_held)
гарантирует отсутствие утечки данных из held-out категории.

Usage:
    python -m src.eval.router_loco_gold
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.common import MODELS_DIR, PROCESSED_DIR, setup_logging
from src.eval.router_pre_registered import PRE_REGISTERED_BUDGETS
from src.eval.router_stats import compute_router_vs_static_at_budgets
from src.pipeline.router.data import FOOD_CATS
from src.pipeline.router.features import featurize
from src.pipeline.router.train import (
    _apply_gold_overrides,
    _calibrator_predict,
    _enrich_with_product_meta,
    save_artefacts,
    train_router,
)

logger = logging.getLogger(__name__)


def main():
    setup_logging()

    # Load + enrich + apply gold overrides once.
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "router_train.parquet"))
    logger.info("Loaded %d rows", len(df))
    df = _enrich_with_product_meta(df, PROCESSED_DIR)
    df, (train_all, val_all, test_all) = _apply_gold_overrides(df, PROCESSED_DIR)
    logger.info(
        "Gold overrides applied. Train=%d Val=%d Test=%d",
        len(train_all), len(val_all), len(test_all),
    )

    models_dir = Path(MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for holdout in FOOD_CATS:
        logger.info("=== LOCO holdout: %s ===", holdout)

        # Build "5 other categories" train pool (train+val combined),
        # and held-out test from current category.
        train_others = pd.concat(
            [
                train_all[train_all["category"] != holdout],
                val_all[val_all["category"] != holdout],
            ],
            ignore_index=True,
        )
        test_held = test_all[test_all["category"] == holdout].copy()

        if len(train_others) == 0 or len(test_held) == 0:
            logger.warning(
                "LOCO %s: empty train_others (%d) or test_held (%d), skip",
                holdout, len(train_others), len(test_held),
            )
            continue

        # Re-split train_others into train+val (90/10, seeded) so train_router
        # has a val slice for calibrator fitting + ECE selection.
        train_o, val_o = train_test_split(
            train_others, test_size=0.1, random_state=42,
        )
        logger.info(
            "  train_others=%d → train_o=%d val_o=%d ; test_held=%d",
            len(train_others), len(train_o), len(val_o), len(test_held),
        )

        # Train router. `train_router` builds brand_set / class_freq_table /
        # brand_attr_acc_table from `train_o` only — no LOCO leakage.
        artefacts = train_router(
            df=df,  # only used for column-presence check
            splits=(train_o, val_o, test_held),
        )

        # Save artefacts under loco_gold_{holdout} suffix.
        save_artefacts(artefacts, models_dir, suffix=f"_loco_gold_{holdout}")

        # Inference on held-out test split.
        X_test, _ = featurize(
            test_held,
            brand_set=artefacts.brand_set,
            class_freq_table=artefacts.class_freq_table,
            brand_attr_acc_table=artefacts.brand_attr_acc_table,
        )
        raw = artefacts.model.predict_proba(X_test)[:, 1]
        p_correct = _calibrator_predict(artefacts.calibrator, raw)

        # Stats at default budgets (which include the 3 pre-registered).
        stats = compute_router_vs_static_at_budgets(test_held, p_correct)

        for b in PRE_REGISTERED_BUDGETS:
            row = stats[stats["budget_target"] == b]
            if row.empty:
                logger.warning(
                    "Budget %.2f not found for holdout=%s", b, holdout,
                )
                continue
            r = row.iloc[0]
            rows.append({
                "holdout": holdout,
                "budget": float(b),
                "n_test": int(len(test_held)),
                "router_acc": float(r["router_accuracy"]),
                "static_acc": float(r["static_accuracy"]),
                "delta_pp": float(r["delta"]) * 100.0,
                "p_mcnemar": float(r["p_mcnemar"]),
                "ci_lo_pp": float(r["ci_lo"]) * 100.0,
                "ci_hi_pp": float(r["ci_hi"]) * 100.0,
                "calibrator_choice": artefacts.calibrator_choice,
                "calibrator_ece": artefacts.calibrator_ece,
                "val_auc_raw": artefacts.val_metrics.get("auc_raw"),
            })

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(PROCESSED_DIR, "router_loco_gold.parquet")
    out_df.to_parquet(out_path, index=False)
    logger.info("Saved %s (%d rows)", out_path, len(out_df))

    # Print summary table.
    if len(out_df) == 0:
        logger.warning("No rows produced — output table is empty.")
        return

    print()
    print("=== LOCO router gold: Δ(router − static) by holdout × budget [пп] ===")
    pivot = out_df.pivot_table(
        index="holdout", columns="budget", values="delta_pp", aggfunc="first",
    )
    print(pivot.round(2).to_string())
    print()
    for b in PRE_REGISTERED_BUDGETS:
        sub = out_df[out_df["budget"] == float(b)]
        if not sub.empty:
            print(
                f"Mean Δ at budget {int(b*100)}%: "
                f"{sub['delta_pp'].mean():.2f} пп "
                f"(min={sub['delta_pp'].min():.2f}, max={sub['delta_pp'].max():.2f})"
            )


if __name__ == "__main__":
    main()
