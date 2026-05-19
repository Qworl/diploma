"""
Pareto-curve router evaluation на consensus_gold (per-source эталон).

Отличие от `router_eval.py`: вместо `silver_gt` использует
`gt_consensus` из `consensus_gold_v1_emulated.parquet` (см. §6.12.1-2 notebook'а).
На gold + close_to_gold tier silver = consensus by definition, на
silver_strong tier — 3-LLM majority consensus (без Haiku).

Это закрывает A1 ревизии (router gold validation): router-Pareto
пересчитан на эталоне, где silver-noise устранён там, где это возможно.

Output:
    datasets/processed/router_pareto_gold.parquet
    datasets/processed/router_stats_gold.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle

import numpy as np
import pandas as pd

from src.common import MODELS_DIR, PROCESSED_DIR, setup_logging
from src.eval.router_eval import (
    pareto_curve_router, pareto_curve_static,
    pareto_curve_per_attr_table, pareto_curve_random,
)
from src.eval.router_stats import compute_router_vs_static_at_budgets
from src.pipeline.router.baselines import build_per_attr_table
from src.pipeline.router.data import (
    FOOD_CATS, build_training_dataset, by_product_split,
)
from src.pipeline.router.train import (
    _apply_gold_overrides, _calibrator_predict, _enrich_with_product_meta,
)
from src.pipeline.router.features import featurize

logger = logging.getLogger(__name__)


def _norm(v) -> str:
    """Case-insensitive normalize, чтобы 'True'/'true'/'wheat'/'Wheat' совпадали."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip().lower()


def _replace_gt_with_consensus(df: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Заменить silver_gt → gt_consensus где он есть. Нормализует case для
    всех предсказаний (cascade_pred, llm_pred, silver_gt) чтобы сравнения работали."""
    df = df.copy()
    df["code"] = df["code"].astype(str)
    gold = gold.copy()
    gold["code"] = gold["code"].astype(str)

    # consensus_gt уже в lowercase (нормализован в build_consensus_gold);
    # cascade_pred/llm_pred в исходном case → нормализуем здесь, чтобы
    # сравнение string-vs-string не ломалось.
    for col in ("cascade_pred", "llm_pred", "silver_gt"):
        if col in df.columns:
            df[col] = df[col].apply(_norm)

    joined = df.merge(
        gold[["category", "code", "attr", "gt_consensus", "tier"]],
        on=["category", "code", "attr"], how="left",
    )
    has_cons = joined["gt_consensus"].notna()
    joined.loc[has_cons, "silver_gt"] = joined.loc[has_cons, "gt_consensus"].apply(_norm)
    n_replaced = int(has_cons.sum())
    logger.info("Replaced gt: %d of %d rows (%.0f%%) via consensus_gold",
                n_replaced, len(joined), n_replaced / max(len(joined), 1) * 100)
    return joined.drop(columns=["gt_consensus"])


def main():
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tier",
        choices=["all", "gold_and_close", "silver_strong"],
        default="all",
        help="Filter test slice to a tier subset (default: all).",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffix appended to output parquet filenames, e.g. '_gold_tier_only'.",
    )
    parser.add_argument(
        "--llm-suffix",
        default="",
        help="Suffix для direct_llm_eval_{cat}_stratified_{suffix}.parquet "
             "(e.g. 'sonnet45', 'gemini25flash', 'llama3b'). "
             "Default (empty) = используется baseline LLM из router_train.parquet. "
             "Когда задан, output идёт в router_pareto_gold_{suffix}.parquet "
             "(если --output-suffix не задан явно).",
    )
    args = parser.parse_args()

    # Resolve final output suffix. If --output-suffix задан явно — используем его.
    # Иначе, если --llm-suffix задан — генерим '_<llm-suffix>'. Иначе пусто.
    if args.output_suffix:
        out_suffix = args.output_suffix
    elif args.llm_suffix:
        out_suffix = f"_{args.llm_suffix}"
    else:
        out_suffix = ""

    # 1. Загрузить router train data + enrich.
    # Когда задан --llm-suffix, пересобираем join с альтернативными LLM-результатами
    # (direct_llm_eval_{cat}_stratified_{suffix}.parquet) вместо чтения cached
    # router_train.parquet. Это нужно для E1 circularity check.
    if args.llm_suffix:
        logger.info("Rebuilding router data with llm_suffix=%r", args.llm_suffix)
        df = build_training_dataset(
            FOOD_CATS, PROCESSED_DIR, llm_suffix=args.llm_suffix
        )
    else:
        logger.info("Loading router training dataset")
        df = pd.read_parquet(os.path.join(PROCESSED_DIR, "router_train.parquet"))
    df = _enrich_with_product_meta(df, PROCESSED_DIR)

    # 2. Apply gold overrides + brand-disjoint splits (E.3, 2026-05-13).
    # `_apply_gold_overrides` does (a) silver_gt ← gt_consensus.combine_first(silver_gt)
    # plus recompute cascade_correct, AND (b) load per-category {cat}_gold_split.parquet
    # to produce brand-disjoint train/val/test slices identical to D.1 router training.
    df_with_gt, (train, val, test) = _apply_gold_overrides(df, PROCESSED_DIR)
    df = df_with_gt
    logger.info("Brand-disjoint splits: train=%d val=%d test=%d", len(train), len(val), len(test))
    logger.info("Test size: %d rows, %d products", len(test), test["code"].nunique())

    # 2b. Tier filter (E.5, 2026-05-13). Filters TEST slice only — router
    # was trained on all-tier data; we measure per-tier performance.
    if args.tier != "all":
        from src.eval.validation_sources import (
            VALIDATION_SOURCE, get_tier, SourceTier,
        )

        if args.tier == "gold_and_close":
            target_tier_values = {SourceTier.GOLD.value, SourceTier.CLOSE_TO_GOLD.value}
        elif args.tier == "silver_strong":
            target_tier_values = {SourceTier.SILVER_STRONG.value}
        else:
            target_tier_values = set()

        pair_in_tier = set()
        for (cat, attr), _src in VALIDATION_SOURCE.items():
            t = get_tier(cat, attr)
            if t is not None and t.value in target_tier_values:
                pair_in_tier.add((cat, attr))

        def _in_tier(row):
            return (row["category"], row["attr"]) in pair_in_tier

        n_before = len(test)
        test = test[test.apply(_in_tier, axis=1)].copy()
        logger.info(
            "Tier filter [%s]: test %d → %d rows over %d (cat, attr) pairs",
            args.tier, n_before, len(test), len(pair_in_tier),
        )
        if len(test) == 0:
            raise RuntimeError(
                f"Tier filter [{args.tier}] resulted in empty test set."
            )

    # 3. Load gold-trained router + standalone calibrator (D.2 / D.2.1).
    with open(os.path.join(MODELS_DIR, "router_gold_xgb.pkl"), "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]

    with open(os.path.join(MODELS_DIR, "router_gold_calibrator.pkl"), "rb") as f:
        calibrator = pickle.load(f)

    with open(os.path.join(MODELS_DIR, "router_gold_meta.json")) as f:
        meta = json.load(f)
    logger.info("Loaded router_gold (calibrator=%s, calibrator_ece=%s)",
                meta.get("calibrator", "?"), meta.get("calibrator_ece", "?"))
    brand_set = set(meta["brand_set"])
    class_freq_table = {tuple(e["key"]): e["value"]
                        for e in meta.get("class_freq_table", [])}
    brand_attr_acc_table = {tuple(e["key"]): e["value"]
                            for e in meta.get("brand_attr_acc_table", [])}

    X_test, _ = featurize(test, brand_set=brand_set,
                          class_freq_table=class_freq_table,
                          brand_attr_acc_table=brand_attr_acc_table)
    raw = model.predict_proba(X_test)[:, 1]
    p_correct = _calibrator_predict(calibrator, raw)

    # 4. Pareto curves (router + static + per_attr + random + anchors)
    train_with_llm = train.copy()
    train_with_llm["llm_correct"] = (
        train_with_llm["llm_pred"].astype(str) == train_with_llm["silver_gt"].astype(str)
    ).astype(int)
    table = build_per_attr_table(train_with_llm)

    parts = [
        pareto_curve_router(test, p_correct),
        pareto_curve_static(test),
        pareto_curve_per_attr_table(test, table),
        pareto_curve_random(test),
    ]
    pure_cascade = pd.DataFrame([{
        "strategy": "cascade_only", "threshold": 0.0, "cost": 0.0,
        "accuracy": float((test["cascade_pred"].astype(str) == test["silver_gt"].astype(str)).mean()),
    }])
    pure_llm = pd.DataFrame([{
        "strategy": "all_llm", "threshold": 1.0, "cost": 1.0,
        "accuracy": float((test["llm_pred"].astype(str) == test["silver_gt"].astype(str)).mean()),
    }])
    pareto = pd.concat(parts + [pure_cascade, pure_llm], ignore_index=True)

    out_pareto = os.path.join(
        PROCESSED_DIR, f"router_pareto_gold{out_suffix}.parquet"
    )
    pareto.to_parquet(out_pareto, index=False)
    logger.info("Saved %s (%d rows)", out_pareto, len(pareto))

    # 5. Router vs static stats (McNemar + paired bootstrap)
    stats = compute_router_vs_static_at_budgets(test, p_correct)
    out_stats = os.path.join(
        PROCESSED_DIR, f"router_stats_gold{out_suffix}.parquet"
    )
    stats.to_parquet(out_stats, index=False)
    logger.info("Saved %s (%d rows)", out_stats, len(stats))

    # 6. Anchor points report
    print()
    print("=== Anchor points (gold-эталон) ===")
    for _, r in pareto[pareto["strategy"].isin(["cascade_only", "all_llm",
                                                  "per_attr_table"])].iterrows():
        print(f'  {r["strategy"]:15} cost={r["cost"]*100:5.1f}%  acc={r["accuracy"]*100:.1f}%')

    print()
    print("=== Router stats vs static (gold-эталон) ===")
    show = stats.copy()
    show["budget"] = show["budget_target"].apply(lambda x: f"{x*100:.0f}%")
    show["router_acc"] = show["router_accuracy"].apply(lambda x: f"{x*100:.1f}%")
    show["static_acc"] = show["static_accuracy"].apply(lambda x: f"{x*100:.1f}%")
    show["delta"] = show["delta"].apply(lambda x: f"{x*100:+.1f} пп")
    show["p_mcnemar"] = show["p_mcnemar"].apply(lambda x: f"{x:.3f}" if x >= 0.001 else "<0.001")
    cols = ["budget", "router_acc", "static_acc", "delta", "p_mcnemar", "router_strictly_better"]
    print(show[cols].to_string(index=False))


if __name__ == "__main__":
    main()
