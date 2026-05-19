"""
SOTA-baseline: direct LLM zero-shot на всём тесте — без каскада.

Для каждой категории прогоняет LLM (по умолчанию Claude Haiku 4.5) на полном
test split (RANDOM_STATE=42, TEST_SIZE=0.2 — те же 250 продуктов на категорию,
что и для остальных метрик). Результаты позволяют сравнить:

- direct LLM accuracy vs cascade (regex_ml_bayes) accuracy на одном и том же тесте;
- стоимость direct LLM на товар vs стоимость каскада (≈0 + α*~5% LLM).

Записывает:
- direct_llm_eval_<cat>_stratified.parquet — per-product результаты
- direct_llm_summary.parquet — сводка category × attr × accuracy

Usage:
    python scripts/eval_direct_llm_baseline.py --all
    python scripts/eval_direct_llm_baseline.py --category pasta_stratified --max-products 50
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.common import PROCESSED_DIR, RANDOM_STATE, TEST_SIZE, setup_logging
from src.pipeline.schemas import (
    BEVERAGE_SCHEMA,
    CEREALS_SCHEMA,
    CHEESES_SCHEMA,
    CHOCOLATE_SCHEMA,
    COSMETICS_SCHEMA,
    PASTA_SCHEMA,
)
from src.pipeline.llm_fallback import enrich_product

logger = logging.getLogger(__name__)

SCHEMA_BY_CATEGORY = {
    "pasta": PASTA_SCHEMA,
    "chocolate": CHOCOLATE_SCHEMA,
    "beverages": BEVERAGE_SCHEMA,
    "cheeses": CHEESES_SCHEMA,
    "cereals": CEREALS_SCHEMA,
    "cosmetics": COSMETICS_SCHEMA,
}

INPUT_FIELDS = ["product_name", "brands", "ingredients_text", "quantity"]


def _normalize(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value)
    if s.lower() in ("none", "nan", ""):
        return None
    return s


def evaluate(category_stratified: str, model: str,
             max_products: int | None = None,
             enforce_json: bool = True,
             output_suffix: str = "",
             exclude_attrs: list[str] | None = None) -> pd.DataFrame:
    base_cat = category_stratified.replace("_stratified", "")
    schema = SCHEMA_BY_CATEGORY[base_cat]
    if exclude_attrs:
        excluded_present = [a for a in exclude_attrs if a in schema]
        logger.info("[%s] excluded attrs from LLM eval: %s",
                    category_stratified, excluded_present)
        schema = {a: v for a, v in schema.items() if a not in exclude_attrs}

    silver_path = os.path.join(PROCESSED_DIR, f"{category_stratified}_silver_standard.parquet")
    silver = pd.read_parquet(silver_path)
    silver["code"] = silver["code"].astype(str)

    # тот же split что и в run_experiments / классификаторах
    _, test_idx = train_test_split(
        np.arange(len(silver)), test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    test = silver.iloc[test_idx].copy().reset_index(drop=True)
    if max_products is not None:
        test = test.head(max_products)
    test_codes = test["code"].tolist()
    logger.info("[%s] direct LLM на %d тестовых продуктах",
                category_stratified, len(test_codes))

    # список целевых атрибутов из schema
    target_attrs = [a for a in schema.keys()]

    rows = []
    t0 = time.time()
    _suffix = f"_{output_suffix}" if output_suffix else ""
    checkpoint_path = os.path.join(PROCESSED_DIR,
                                    f"direct_llm_eval_{category_stratified}{_suffix}.parquet")
    for i, (_, row) in enumerate(test.iterrows()):
        product = {f: row.get(f) for f in INPUT_FIELDS if f in row.index}
        product["code"] = row["code"]
        try:
            llm_pred = enrich_product(
                product, schema, backend="openrouter",
                model=model, enforce_json=enforce_json,
            )
        except Exception as exc:
            logger.warning("  [%d/%d] %s — LLM error: %s",
                           i + 1, len(test), row["code"], exc)
            continue

        for attr in target_attrs:
            if attr not in row.index:
                continue
            gt = _normalize(row.get(attr))
            pred = _normalize(llm_pred.get(attr))
            rows.append({
                "category": base_cat,
                "code": str(row["code"]),
                "attr": attr,
                "gt": gt,
                "pred": pred,
                "predicted_non_null": int(pred is not None),
                "gt_non_null": int(gt is not None),
                "correct_when_both_present": int(
                    gt is not None and pred is not None and gt == pred
                ),
            })

        if (i + 1) % 50 == 0:
            pd.DataFrame(rows).to_parquet(checkpoint_path, index=False)
            elapsed = time.time() - t0
            logger.info("  [%s] %d/%d, %.1fs (%.1fs/product) — checkpoint",
                        category_stratified, i + 1, len(test),
                        elapsed, elapsed / (i + 1))

    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    out = []
    for (cat, attr), sub in rows.groupby(["category", "attr"]):
        n_total = len(sub)
        n_gt = int(sub["gt_non_null"].sum())
        n_pred = int(sub["predicted_non_null"].sum())
        # точность на пересечении (gt и pred не None)
        both = sub[(sub["gt_non_null"] == 1) & (sub["predicted_non_null"] == 1)]
        n_both = len(both)
        n_correct_both = int(both["correct_when_both_present"].sum())
        # абсолютная точность: совпадение pred=gt при любом значении (None=None считается верно)
        n_correct_abs = int(((sub["pred"].fillna("__none__") == sub["gt"].fillna("__none__"))).sum())
        out.append({
            "category": cat,
            "attr": attr,
            "n_total": n_total,
            "n_with_gt": n_gt,
            "n_predicted_by_llm": n_pred,
            "n_both_present": n_both,
            "accuracy_on_intersection": (n_correct_both / n_both) if n_both else None,
            "accuracy_with_none_match": n_correct_abs / n_total,
            "llm_coverage": n_pred / n_total,
        })
    return pd.DataFrame(out)


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--category",
                   choices=["pasta_stratified", "chocolate_stratified", "beverages_stratified",
                            "cheeses_stratified", "cereals_stratified", "cosmetics_stratified"])
    p.add_argument("--all", action="store_true")
    p.add_argument("--model", default="anthropic/claude-haiku-4.5")
    p.add_argument("--max-products", type=int, default=None,
                   help="Ограничить число тестовых продуктов на категорию")
    p.add_argument("--output-suffix", default="",
                   help="Suffix appended to output filename, e.g. 'gptoss' → direct_llm_eval_{cat}_stratified_gptoss.parquet")
    p.add_argument("--exclude-attrs", nargs="*", default=None,
                   help="Exclude specific attributes from LLM eval (e.g. --exclude-attrs nova_group). "
                        "Filtered from schema before prompt construction. Useful for known "
                        "labelspace mismatches.")
    args = p.parse_args()

    cats = []
    if args.all:
        cats = ["pasta_stratified", "chocolate_stratified", "beverages_stratified",
                "cheeses_stratified", "cereals_stratified", "cosmetics_stratified"]
    elif args.category:
        cats = [args.category]
    else:
        p.error("укажите --category или --all")

    suffix = f"_{args.output_suffix}" if args.output_suffix else ""

    all_rows = []
    for cat in cats:
        rows = evaluate(cat, model=args.model, max_products=args.max_products,
                        output_suffix=args.output_suffix,
                        exclude_attrs=args.exclude_attrs)
        if rows.empty:
            continue
        out_path = os.path.join(PROCESSED_DIR, f"direct_llm_eval_{cat}{suffix}.parquet")
        rows.to_parquet(out_path, index=False)
        logger.info("[%s] -> %s (%d rows)", cat, out_path, len(rows))
        all_rows.append(rows)

    if all_rows:
        full = pd.concat(all_rows, ignore_index=True)
        summary = summarize(full)
        sum_path = os.path.join(PROCESSED_DIR, f"direct_llm_summary{suffix}.parquet")
        summary.to_parquet(sum_path, index=False)
        logger.info("Summary -> %s\n%s", sum_path, summary.to_string(index=False))


if __name__ == "__main__":
    main()
