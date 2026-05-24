"""Selective gold v2: per-(cat, attr) выбор лучшего источника.

Стратегия:
- На manual gold (250 codes per cat) измерить accuracy silver vs consolidated.
- Для каждого (cat, attr): выбрать тот источник, что лучше на gold.
- Применить выбор глобально (на все ~14k labelled codes per cat).

Это даёт максимум из обоих миров:
- Silver wins на tag-derived attrs (is_organic, is_pdo, etc) — perfect accuracy.
- Consolidated wins на text-derived (pasta_shape, chocolate_extra, cocoa_percentage)
  и calibrated TYPE_C (fat_class).

Output: datasets/processed/{cat}_gold_v2_selective.parquet
Columns: code, attr, value, source ('silver' / 'consolidated'), confidence
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from src.common import MAIN_CATEGORIES, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

PROCESSED = Path(PROCESSED_DIR)


def _norm(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "__null__"
    return str(v).lower().strip()


def measure_acc(df_pred: pd.DataFrame, pred_col: str, gold: pd.DataFrame) -> dict:
    """Per-attr accuracy of df_pred vs manual gold."""
    if "code" not in df_pred.columns or "attr" not in df_pred.columns:
        raise ValueError(f"df_pred must have code, attr columns; got {df_pred.columns}")
    df_pred = df_pred.copy()
    df_pred["code"] = df_pred["code"].astype(str)
    gold = gold.copy()
    gold["code"] = gold["code"].astype(str)
    m = df_pred.merge(gold[["code", "attr", "gold_value", "gold_is_null"]],
                       on=["code", "attr"], how="inner")
    m = m[~m.gold_is_null]
    m["p"] = m[pred_col].apply(_norm)
    m["g"] = m["gold_value"].apply(_norm)
    m = m[m.p != "__null__"]
    accs = {}
    for attr in m.attr.unique():
        sub = m[m.attr == attr]
        accs[attr] = {"n": len(sub), "acc": (sub.p == sub.g).mean()}
    return accs


def select_source_per_attr(cat: str, gold: pd.DataFrame) -> dict[str, str]:
    """Для (cat, attr) выбрать лучший источник по gold validation."""
    silver = pd.read_parquet(PROCESSED / f"{cat}_stratified_silver_standard.parquet")
    silver["code"] = silver["code"].astype(str)
    # Silver columns → long format
    raw_cols = {"code", "product_name", "brands", "categories_tags", "countries_tags",
                "labels_tags", "ingredients_text", "ingredients_analysis_tags",
                "traces_tags", "quantity", "fat_100g", "sugars_100g", "proteins_100g",
                "carbohydrates_100g", "alcohol_100g", "nutriscore_grade", "nova_group",
                "generic_name", "serving_size", "completeness", "image_url",
                "saturated-fat_100g", "salt_100g", "sodium_100g", "fiber_100g",
                "ingredients_tags"}
    attr_cols = [c for c in silver.columns if c not in raw_cols]
    silver_long = silver.melt(id_vars=["code"], value_vars=attr_cols,
                                var_name="attr", value_name="value")
    silver_long = silver_long.dropna(subset=["value"])

    # Consolidated v2 (with TYPE_C calibrated hard-override)
    cons = pd.read_parquet(PROCESSED / f"{cat}_consolidated_gold_v1.parquet")

    gold_cat = gold[gold.category == cat].copy()
    silver_acc = measure_acc(silver_long, "value", gold_cat)
    cons_acc = measure_acc(cons, "value", gold_cat)

    selection = {}
    logger.info("  %s per-attr selection:", cat)
    logger.info("  %-23s %8s %8s %8s   %s", "attr", "silver", "cons", "n_gold", "winner")
    all_attrs = sorted(set(silver_acc) | set(cons_acc))
    for attr in all_attrs:
        s = silver_acc.get(attr, {"n": 0, "acc": float("nan")})
        c = cons_acc.get(attr, {"n": 0, "acc": float("nan")})
        # Decision rule: silver wins by default (deterministic, no LLM noise).
        # Consolidated wins only if it improves accuracy by ≥1 п.п. (significant).
        # If silver missing → consolidated. If cons missing → silver.
        s_acc, c_acc, n = s["acc"], c["acc"], max(s["n"], c["n"])
        if pd.isna(s_acc):
            winner = "consolidated"
        elif pd.isna(c_acc):
            winner = "silver"
        elif c_acc - s_acc >= 0.01:
            winner = "consolidated"
        else:
            winner = "silver"
        selection[attr] = winner
        marker = " " if winner == "silver" else "*"
        logger.info("  %s%-22s %8.3f %8.3f %8d   %s",
                    marker, attr, s_acc, c_acc, n, winner)
    return selection


def build_gold_v2(cat: str, selection: dict[str, str]) -> pd.DataFrame:
    """Применить per-attr выбор на полный labelled corpus."""
    silver = pd.read_parquet(PROCESSED / f"{cat}_stratified_silver_standard.parquet")
    silver["code"] = silver["code"].astype(str)
    raw_cols = {"code", "product_name", "brands", "categories_tags", "countries_tags",
                "labels_tags", "ingredients_text", "ingredients_analysis_tags",
                "traces_tags", "quantity", "fat_100g", "sugars_100g", "proteins_100g",
                "carbohydrates_100g", "alcohol_100g", "nutriscore_grade", "nova_group",
                "generic_name", "serving_size", "completeness", "image_url",
                "saturated-fat_100g", "salt_100g", "sodium_100g", "fiber_100g",
                "ingredients_tags"}
    attr_cols = [c for c in silver.columns if c not in raw_cols]
    silver_long = silver.melt(id_vars=["code"], value_vars=attr_cols,
                                var_name="attr", value_name="value").dropna(subset=["value"])
    cons = pd.read_parquet(PROCESSED / f"{cat}_consolidated_gold_v1.parquet")
    cons["code"] = cons["code"].astype(str)

    parts = []
    for attr, winner in selection.items():
        if winner == "silver":
            sub = silver_long[silver_long.attr == attr].copy()
            sub["source"] = "silver"
            sub["confidence"] = "silver_deterministic"
        else:
            sub = cons[cons.attr == attr][["code", "attr", "value", "confidence"]].copy()
            sub["source"] = "consolidated"
        if len(sub) > 0:
            parts.append(sub[["code", "attr", "value", "source", "confidence"]])
    out = pd.concat(parts, ignore_index=True)
    out["value"] = out["value"].astype(str).str.lower().str.strip()
    return out


def main():
    gold = pd.read_parquet(PROCESSED / "consensus_gold_v2_expanded.parquet")
    for cat in MAIN_CATEGORIES:
        logger.info("=== %s ===", cat.upper())
        selection = select_source_per_attr(cat, gold)
        v2 = build_gold_v2(cat, selection)
        out_path = PROCESSED / f"{cat}_gold_v2_selective.parquet"
        v2.to_parquet(out_path, index=False)
        logger.info("  saved: %s (%d (code, attr) rows)", out_path, len(v2))


if __name__ == "__main__":
    main()
