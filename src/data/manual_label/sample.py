"""
Готовит CSV для ручной gold-разметки.

Берёт стратифицированный test split (RANDOM_STATE=42, TEST_SIZE=0.2 — те же 250
продуктов, что во всех остальных метриках), сэмплирует N продуктов на категорию
и экспортирует CSV с колонками для ручного заполнения.

Принцип:
- Колонки `manual_*` пусты — заполняются вручную.
- Колонки `silver_*` содержат текущий эталон из тегов OFF — для сверки на этапе
  разметки (если разметчик согласен — копирует, если нет — ставит свой ответ).
- Колонки с входами (`product_name`, `brands`, `ingredients_text`, `quantity`)
  показываются как справка.
- Сохранение в `datasets/manual_label/<category>_to_label.csv`.

Usage:
    python scripts/sample_for_manual_label.py --all --n 50
    python scripts/sample_for_manual_label.py --category pasta_stratified --n 30
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.common import PROCESSED_DIR, RANDOM_STATE, TEST_SIZE
from src.pipeline.schemas import (
    BEVERAGE_SCHEMA, CHEESES_SCHEMA, CEREALS_SCHEMA,
    CHOCOLATE_SCHEMA, COSMETICS_SCHEMA, PASTA_SCHEMA
)

# Baby schema not yet available in reorganized structure
try:
    from src.pipeline.schemas import BABY_SCHEMA
except ImportError:
    BABY_SCHEMA = None

SCHEMAS = {
    "pasta_stratified": PASTA_SCHEMA,
    "chocolate_stratified": CHOCOLATE_SCHEMA,
    "beverages_stratified": BEVERAGE_SCHEMA,
    "cosmetics_stratified": COSMETICS_SCHEMA,
    "cheeses_stratified": CHEESES_SCHEMA,
    "cereals_stratified": CEREALS_SCHEMA,
}

if BABY_SCHEMA is not None:
    SCHEMAS["baby_stratified"] = BABY_SCHEMA

INPUT_FIELDS = ["product_name", "brands", "ingredients_text", "quantity"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets", "manual_label")


def values_hint(spec: dict) -> str:
    """Подсказка для разметчика: какие значения допустимы."""
    if spec["type"] == "bool":
        return "True / False / None"
    if spec["type"] == "enum":
        vals = "|".join(spec["values"])
        if spec.get("nullable"):
            vals += "|None"
        return vals
    if spec["type"] == "number":
        return "число или None"
    return spec["type"]


def sample_one(category: str, n: int) -> pd.DataFrame:
    silver_path = os.path.join(PROCESSED_DIR, f"{category}_silver_standard.parquet")
    silver = pd.read_parquet(silver_path)
    silver["code"] = silver["code"].astype(str)

    _, test_idx = train_test_split(
        np.arange(len(silver)), test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    test = silver.iloc[test_idx].copy().reset_index(drop=True)

    rng = np.random.default_rng(seed=2026)
    pick = rng.choice(len(test), size=min(n, len(test)), replace=False)
    sub = test.iloc[pick].copy().reset_index(drop=True)

    schema = SCHEMAS[category]
    cols = ["code"] + INPUT_FIELDS
    out = sub[[c for c in cols if c in sub.columns]].copy()
    # silver_* колонки и manual_* шаблоны
    for attr, spec in schema.items():
        out[f"silver_{attr}"] = sub[attr].astype(str) if attr in sub.columns else ""
        out[f"manual_{attr}"] = ""
    # подсказка по допустимым значениям — как заголовок-комментарий первой строки
    return out, schema


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--category", choices=list(SCHEMAS.keys()))
    p.add_argument("--all", action="store_true")
    p.add_argument("--n", type=int, default=50, help="Число продуктов на категорию")
    args = p.parse_args()

    cats = list(SCHEMAS.keys()) if args.all else ([args.category] if args.category else [])
    if not cats:
        p.error("укажите --category или --all")

    os.makedirs(OUT_DIR, exist_ok=True)

    for cat in cats:
        df, schema = sample_one(cat, args.n)
        # короткая инструкция первой строкой (комментарий-памятка)
        legend_row = {col: "" for col in df.columns}
        legend_row["code"] = "## ЛЕГЕНДА (не заполнять — удалить эту строку перед сохранением)"
        for attr, spec in schema.items():
            legend_row[f"manual_{attr}"] = values_hint(spec)
            legend_row[f"silver_{attr}"] = "(текущий эталон)"
        df = pd.concat([pd.DataFrame([legend_row]), df], ignore_index=True)

        out_path = os.path.join(OUT_DIR, f"{cat.replace('_stratified','')}_to_label.csv")
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"{cat}: -> {out_path} ({len(df) - 1} продуктов)")

    print(f"\nЗаполните вручную колонки `manual_*`. Когда готово, запустите:")
    print(f"  python scripts/eval_manual_vs_silver.py --all")


if __name__ == "__main__":
    main()
