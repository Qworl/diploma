"""
Build expanded arbitrage CSV for a specific (category, attribute) pair.

Отличие от `consensus_arbitrage_candidates.csv`:
- Включает не только no_majority пары, но и пары где silver ≠ consensus_gt
  (LLM коррелированно сказали не то, что silver — самые опасные кейсы).
- Сортирует по приоритету: no_majority → silver_diff → agree.
- Добавляет колонку `enum_choices` для quick-select в UI.

Output: datasets/manual_label/arbitrage_{cat}_{attr}.csv

Usage:
    python -m src.eval.build_arbitrage_csv --category cereals --attr cereal_type
"""
from __future__ import annotations

import argparse
import logging
import os

import pandas as pd

from src.common import PROCESSED_DIR, setup_logging
from src.eval.validation_sources import get_source, Source

logger = logging.getLogger(__name__)

_SCHEMA_MODULES = {
    "pasta":      ("src.pipeline.schemas.pasta",      "PASTA_SCHEMA"),
    "chocolate":  ("src.pipeline.schemas.chocolate",  "CHOCOLATE_SCHEMA"),
    "beverages":  ("src.pipeline.schemas.beverages",  "BEVERAGE_SCHEMA"),
    "cheeses":    ("src.pipeline.schemas.cheeses",    "CHEESES_SCHEMA"),
    "cereals":    ("src.pipeline.schemas.cereals",    "CEREALS_SCHEMA"),
    "cosmetics":  ("src.pipeline.schemas.cosmetics",  "COSMETICS_SCHEMA"),
}


def get_enum_values(category: str, attr: str) -> list[str]:
    """Прочитать enum values из реальной schema (а не дублировать в коде)."""
    if category not in _SCHEMA_MODULES:
        return []
    mod_path, schema_name = _SCHEMA_MODULES[category]
    import importlib
    mod = importlib.import_module(mod_path)
    schema = getattr(mod, schema_name)
    if attr not in schema:
        return []
    return list(schema[attr].get("values", []))

# 2026-05-13: strong-seed LLMs (Sonnet 4.5 + GPT-4o + Gemini 2.5 Flash).
CONSENSUS_LLMS = [("_sonnet45",      "sonnet45"),
                  ("_gpt4o",         "gpt4o"),
                  ("_gemini25flash", "gemini25flash")]


def _norm(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip().lower()


def build(category: str, attr: str) -> pd.DataFrame:
    enum_choices = get_enum_values(category, attr)

    silver = pd.read_parquet(f"{PROCESSED_DIR}/{category}_stratified_silver_standard.parquet")
    silver["code"] = silver["code"].astype(str)

    rows: list[dict] = []
    # Собрать LLM голоса
    llm_dfs = {}
    for suffix, label in CONSENSUS_LLMS:
        path = f"{PROCESSED_DIR}/direct_llm_eval_{category}_stratified{suffix}.parquet"
        if not os.path.exists(path):
            logger.warning("Missing LLM eval %s", path)
            continue
        df = pd.read_parquet(path)
        df = df[df.attr == attr][["code", "pred"]].rename(columns={"pred": label})
        df["code"] = df["code"].astype(str)
        llm_dfs[label] = df

    # Join всех LLM на silver. generic_name — критически важная подсказка
    # (особенно для cereals/cereal_type), часто содержит правильный ответ.
    # categories_tags — нужно для classifier'а: OFF regulatory тег должен
    # override marketing keywords в product_name (e.g. "Granola alla frutta"
    # с тегом en:mueslis-with-fruits → muesli, не granola).
    cols_silver = ["code", "product_name", "brands", "ingredients_text",
                    "categories_tags", attr]
    if "generic_name" in silver.columns:
        cols_silver.insert(2, "generic_name")
    cols_silver = [c for c in cols_silver if c in silver.columns]
    base = silver[cols_silver].copy()
    base = base.rename(columns={attr: "silver"})
    for label, df in llm_dfs.items():
        base = base.merge(df, on="code", how="left")

    for _, label in CONSENSUS_LLMS:
        if label not in base.columns:
            base[label] = None

    # Нормализуем для сравнения, но в CSV сохраним оригинальный case
    llm_labels = [label for _, label in CONSENSUS_LLMS]
    for col in ("silver",) + tuple(llm_labels):
        base[f"_{col}_n"] = base[col].apply(_norm)

    def classify(row) -> str:
        s = row["_silver_n"]
        votes = [row[f"_{label}_n"] for label in llm_labels]
        non_null = [v for v in votes if v]
        if len(non_null) < 2:
            return "no_llm_data"
        # 2-of-3 majority
        from collections import Counter
        cnt = Counter(non_null)
        top, top_count = cnt.most_common(1)[0]
        if top_count < 2:
            return "no_majority"
        # есть majority
        if not s:
            return "silver_missing"  # silver не размечен, LLM есть
        if top == s:
            return "agree"
        return "silver_diff"  # LLM сошлись, но silver другое

    base["status"] = base.apply(classify, axis=1)

    # Сортировка по приоритету
    PRIORITY = {"no_majority": 0, "silver_diff": 1, "silver_missing": 2,
                "no_llm_data": 3, "agree": 4}
    base["_prio"] = base["status"].map(PRIORITY).fillna(99).astype(int)
    base = base.sort_values(["_prio", "code"]).reset_index(drop=True)

    # Финальный CSV
    out_cols = ["code", "product_name", "brands"]
    if "generic_name" in base.columns:
        out_cols.append("generic_name")
    out_cols += ["categories_tags", "ingredients_text",
                  "silver"] + llm_labels + ["status"]
    out_cols = [c for c in out_cols if c in base.columns]
    out = base[out_cols].copy()
    out["enum_choices"] = ", ".join(enum_choices)
    out["your_arbitrage"] = ""
    out["note"] = ""

    # Усечь длинные тексты для удобства просмотра в CSV
    out["ingredients_text"] = out["ingredients_text"].apply(
        lambda x: (str(x or "")[:300] + "...") if len(str(x or "")) > 300 else (str(x or ""))
    )

    return out


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True)
    p.add_argument("--attr", required=True)
    p.add_argument("--out-dir", default="datasets/manual_label")
    args = p.parse_args()

    df = build(args.category, args.attr)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"arbitrage_{args.category}_{args.attr}.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\nSaved {len(df)} rows -> {out_path}")
    print()
    print("Status distribution:")
    print(df["status"].value_counts().to_string())
    print()
    print("Schema enum values for this attribute:")
    print(f"  {get_enum_values(args.category, args.attr)}")


if __name__ == "__main__":
    main()
