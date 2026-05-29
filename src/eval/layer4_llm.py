"""
Оценка Layer 4 (LLM-fallback) на хвосте каскада.

Берёт `experiment_per_product_<cat>_stratified.parquet` (config=regex_ml_bayes),
выбирает строки `layer='none'` и `gt` определён, прогоняет LLM на этих
продуктах и сравнивает с серебром.

Записывает:
- llm_fallback_eval_<cat>_stratified.parquet — построчные результаты
- llm_fallback_summary.parquet — сводка по category × attr × accuracy

Usage:
    python src/eval/eval_layer4_llm.py --category pasta_stratified
    python src/eval/eval_layer4_llm.py --all                   # все три
    python src/eval/eval_layer4_llm.py --all --model gpt-4o-mini --max-per-cat 100
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd

from src.common import PROCESSED_DIR, setup_logging
from src.pipeline.schemas import (
    BEVERAGE_SCHEMA,
    CHEESES_SCHEMA,
    CEREALS_SCHEMA,
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
    "cosmetics": COSMETICS_SCHEMA,
    "cheeses": CHEESES_SCHEMA,
    "cereals": CEREALS_SCHEMA,
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
             enforce_json: bool = True) -> pd.DataFrame:
    """Запускает LLM на хвосте regex_ml_bayes и возвращает результаты."""
    base_cat = category_stratified.replace("_stratified", "")
    schema = SCHEMA_BY_CATEGORY[base_cat]

    pp_path = os.path.join(PROCESSED_DIR, f"experiment_per_product_{category_stratified}.parquet")
    if not os.path.exists(pp_path):
        raise FileNotFoundError(
            f"{pp_path} не найден. Сначала перезапустите run_experiments на stratified."
        )
    pp = pd.read_parquet(pp_path)

    if "config" not in pp.columns:
        raise ValueError(f"{pp_path} без колонки config — пересоберите run_experiments.")
    if "regex_ml_bayes" not in pp["config"].unique():
        raise ValueError(
            f"{pp_path} не содержит config=regex_ml_bayes — нужен пропатченный run_experiments."
        )

    pp = pp[pp["config"] == "regex_ml_bayes"].copy()
    tail = pp[(pp["layer"] == "none") & pp["gt"].notna() & (pp["gt"] != "None")]
    if tail.empty:
        logger.warning("[%s] хвост пустой — всё закрылось предыдущими слоями", category_stratified)
        return pd.DataFrame()

    silver_path = os.path.join(PROCESSED_DIR, f"{category_stratified}_silver_standard.parquet")
    silver = pd.read_parquet(silver_path)
    silver["code"] = silver["code"].astype(str)
    silver_indexed = silver.set_index("code")

    unique_products = tail["code"].drop_duplicates().tolist()
    if max_products is not None:
        unique_products = unique_products[:max_products]

    logger.info("[%s] хвост: %d уникальных продуктов × %d атрибутов",
                category_stratified, len(unique_products), tail["attr"].nunique())

    rows = []
    t0 = time.time()
    checkpoint_every = 50
    checkpoint_path = os.path.join(PROCESSED_DIR, f"llm_fallback_eval_{category_stratified}.parquet")
    for i, code in enumerate(unique_products):
        if code not in silver_indexed.index:
            continue
        row = silver_indexed.loc[code]
        product = {f: row.get(f) for f in INPUT_FIELDS if f in row.index}
        product["code"] = code
        try:
            llm_pred = enrich_product(
                product, schema, backend="openrouter",
                model=model, enforce_json=enforce_json,
            )
        except Exception as exc:
            logger.warning("  [%d/%d] %s — LLM error: %s", i + 1, len(unique_products), code, exc)
            continue

        prod_tail = tail[tail["code"] == code]
        for _, t in prod_tail.iterrows():
            attr = t["attr"]
            gt = _normalize(t["gt"])
            pred = _normalize(llm_pred.get(attr))
            rows.append({
                "category": base_cat,
                "code": code,
                "attr": attr,
                "gt": gt,
                "pred": pred,
                "correct": int(gt == pred) if pred is not None else 0,
                "predicted_non_null": int(pred is not None),
            })

        if (i + 1) % checkpoint_every == 0:
            pd.DataFrame(rows).to_parquet(checkpoint_path, index=False)
            elapsed = time.time() - t0
            logger.info("  [%s] %d/%d products, %.1fs elapsed (%.1fs/product) — checkpoint -> %s",
                        category_stratified, i + 1, len(unique_products),
                        elapsed, elapsed / (i + 1), checkpoint_path)
        elif (i + 1) == len(unique_products):
            elapsed = time.time() - t0
            logger.info("  [%s] %d/%d products, %.1fs elapsed (%.1fs/product)",
                        category_stratified, i + 1, len(unique_products),
                        elapsed, elapsed / (i + 1))

    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """Группирует результаты по (category, attr) и считает α + покрытие."""
    if rows.empty:
        return rows
    out = []
    for (cat, attr), sub in rows.groupby(["category", "attr"]):
        n = len(sub)
        n_pred = int(sub["predicted_non_null"].sum())
        n_correct = int(sub["correct"].sum())
        out.append({
            "category": cat,
            "attr": attr,
            "n_fallback_products": n,
            "n_predicted_by_llm": n_pred,
            "n_correct": n_correct,
            "llm_alpha_on_predicted": (n_correct / n_pred) if n_pred else None,
            "llm_alpha_overall": n_correct / n,
            "llm_coverage": n_pred / n,
        })
    return pd.DataFrame(out)


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--category",
                   choices=["pasta_stratified", "chocolate_stratified",
                            "beverages_stratified",
                            "cosmetics_stratified",
                            "cheeses_stratified", "cereals_stratified"])
    p.add_argument("--all", action="store_true",
                   help="Прогнать все 6 stratified-категорий")
    p.add_argument("--all-new", action="store_true",
                   help="Только новые домены (cosmetics/cheeses/cereals)")
    p.add_argument("--model", default="anthropic/claude-haiku-4.5",
                   help="OpenRouter model id (default: claude-haiku-4.5)")
    p.add_argument("--max-per-cat", type=int, default=None,
                   help="Ограничить количество продуктов в хвосте на категорию (для отладки)")
    args = p.parse_args()

    cats = []
    if args.all:
        cats = ["pasta_stratified", "chocolate_stratified", "beverages_stratified",
                "cosmetics_stratified",
                "cheeses_stratified", "cereals_stratified"]
    elif args.all_new:
        cats = ["cosmetics_stratified", "cheeses_stratified", "cereals_stratified"]
    elif args.category:
        cats = [args.category]
    else:
        p.error("укажите --category или --all")

    all_rows = []
    for cat in cats:
        rows = evaluate(cat, model=args.model, max_products=args.max_per_cat)
        if rows.empty:
            continue
        out = os.path.join(PROCESSED_DIR, f"llm_fallback_eval_{cat}.parquet")
        rows.to_parquet(out, index=False)
        logger.info("[%s] -> %s (%d rows)", cat, out, len(rows))
        all_rows.append(rows)

    if all_rows:
        full = pd.concat(all_rows, ignore_index=True)
        summary = summarize(full)
        sum_path = os.path.join(PROCESSED_DIR, "llm_fallback_summary.parquet")
        summary.to_parquet(sum_path, index=False)
        logger.info("Summary -> %s", sum_path)
        logger.info("\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
