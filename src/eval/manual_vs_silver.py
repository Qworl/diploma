"""
Сравнивает ручную gold-разметку с эталоном из тегов OFF и с предсказанием каскада.

Читает `datasets/manual_label/<category>_labeled.csv` (заполненный вручную CSV
из `sample_for_manual_label.py`) и считает три метрики:

1. **Согласие эталона с правдой** — accuracy(silver_attr, manual_attr) на каждом
   атрибуте. Если 90%+ — слабый надзор близок к правде, метрики OFF-эталона
   валидны. Если 70-80% — есть систематический разрыв.
2. **Точность каскада против правды** — accuracy(cascade_pred, manual_attr).
   Берётся из `experiment_per_product_<cat>_stratified.parquet` (конфигурация
   `regex_ml_bayes`).
3. **Точность direct LLM против правды** — accuracy(llm_pred, manual_attr).
   Берётся из `direct_llm_eval_<cat>_stratified.parquet`.

Записывает:
- `datasets/processed/manual_eval_summary.parquet` — основная сводка
- `datasets/processed/manual_eval_per_product.parquet` — построчно

Usage:
    python src/eval/eval_manual_vs_silver.py --all
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from src.common import PROCESSED_DIR
from src.pipeline.schemas import (
    BEVERAGE_SCHEMA, CEREALS_SCHEMA, CHEESES_SCHEMA, CHOCOLATE_SCHEMA,
    COSMETICS_SCHEMA, PASTA_SCHEMA,
)

SCHEMAS = {
    "pasta": PASTA_SCHEMA,
    "chocolate": CHOCOLATE_SCHEMA,
    "beverages": BEVERAGE_SCHEMA,
    "cheeses": CHEESES_SCHEMA,
    "cereals": CEREALS_SCHEMA,
    "cosmetics": COSMETICS_SCHEMA,
}
LABEL_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets", "manual_label")


def _normalize(value):
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in ("none", "nan", "null"):
        return None
    return s


def evaluate(category: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    csv = os.path.join(LABEL_DIR, f"{category}_labeled.csv")
    if not os.path.exists(csv):
        print(f"[{category}] {csv} не найден — пропуск")
        return None

    df = pd.read_csv(csv)
    df = df[~df["code"].astype(str).str.startswith("##")]  # убрать строку легенды
    df["code"] = df["code"].astype(str)
    schema_attrs = list(SCHEMAS[category].keys())

    # Cascade predictions (regex_ml_bayes)
    pp_path = os.path.join(PROCESSED_DIR,
                            f"experiment_per_product_{category}_stratified.parquet")
    if os.path.exists(pp_path):
        pp = pd.read_parquet(pp_path)
        pp = pp[pp["config"] == "regex_ml_bayes"][["code", "attr", "pred"]]
        pp = pp.rename(columns={"pred": "cascade_pred"})
        pp["code"] = pp["code"].astype(str)
        pp_wide = pp.pivot(index="code", columns="attr", values="cascade_pred")
        pp_wide.columns = [f"cascade_{c}" for c in pp_wide.columns]
    else:
        pp_wide = pd.DataFrame()

    # Direct LLM predictions
    llm_path = os.path.join(PROCESSED_DIR,
                             f"direct_llm_eval_{category}_stratified.parquet")
    if os.path.exists(llm_path):
        llm = pd.read_parquet(llm_path)[["code", "attr", "pred"]].copy()
        llm = llm.rename(columns={"pred": "llm_pred"})
        llm["code"] = llm["code"].astype(str)
        llm_wide = llm.pivot(index="code", columns="attr", values="llm_pred")
        llm_wide.columns = [f"llm_{c}" for c in llm_wide.columns]
    else:
        llm_wide = pd.DataFrame()

    merged = df.set_index("code")
    if not pp_wide.empty:
        merged = merged.join(pp_wide, how="left")
    if not llm_wide.empty:
        merged = merged.join(llm_wide, how="left")
    merged = merged.reset_index()

    rows = []
    summary = []
    for attr in schema_attrs:
        manual_col = f"manual_{attr}"
        silver_col = f"silver_{attr}"
        cascade_col = f"cascade_{attr}"
        llm_col = f"llm_{attr}"
        if manual_col not in merged.columns:
            continue
        sub = merged[[c for c in [
            "code", manual_col, silver_col, cascade_col, llm_col
        ] if c in merged.columns]].copy()

        sub["manual"] = sub[manual_col].apply(_normalize)
        sub["silver"] = sub[silver_col].apply(_normalize) if silver_col in sub.columns else None
        sub["cascade"] = sub[cascade_col].apply(_normalize) if cascade_col in sub.columns else None
        sub["llm"] = sub[llm_col].apply(_normalize) if llm_col in sub.columns else None

        # удалить незаполненные ручной разметкой строки (manual is None)
        labeled = sub[sub["manual"].notna()].copy()
        if labeled.empty:
            continue

        n = len(labeled)
        # эталон может быть None — считаем согласие двумя способами:
        # (а) полное совпадение строки (silver=None vs gold="X" → не совпадает)
        # (б) только на silver-покрытом срезе (silver не None) — это «насколько
        #     эталон точен там, где у него есть данные»
        silver_covered_mask = labeled["silver"].notna() if "silver" in labeled.columns else pd.Series([False]*n)
        silver_covered = int(silver_covered_mask.sum())
        agree_silver_full = (labeled["silver"] == labeled["manual"]).sum()
        agree_silver_on_covered = (
            (labeled.loc[silver_covered_mask, "silver"] == labeled.loc[silver_covered_mask, "manual"]).sum()
            if silver_covered else 0
        )
        agree_cascade = (
            (labeled["cascade"] == labeled["manual"]) &
            labeled["cascade"].notna()
        ).sum()
        cascade_covered = labeled["cascade"].notna().sum()
        agree_llm = (
            (labeled["llm"] == labeled["manual"]) &
            labeled["llm"].notna()
        ).sum()
        llm_covered = labeled["llm"].notna().sum()

        summary.append({
            "category": category,
            "attr": attr,
            "n_manual": n,
            "silver_coverage": silver_covered / n,
            "silver_vs_manual_acc_on_covered": (
                agree_silver_on_covered / silver_covered if silver_covered else None
            ),
            "silver_vs_manual_acc_overall": agree_silver_full / n,
            "cascade_vs_manual_acc_on_covered": (
                agree_cascade / cascade_covered if cascade_covered else None
            ),
            "cascade_coverage": cascade_covered / n,
            "llm_vs_manual_acc_on_covered": (
                agree_llm / llm_covered if llm_covered else None
            ),
            "llm_coverage": llm_covered / n,
        })
        for _, r in labeled.iterrows():
            rows.append({
                "category": category, "attr": attr, "code": r["code"],
                "manual": r["manual"], "silver": r.get("silver"),
                "cascade": r.get("cascade"), "llm": r.get("llm"),
            })

    return pd.DataFrame(summary), pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--category", choices=list(SCHEMAS.keys()))
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    cats = list(SCHEMAS.keys()) if args.all else ([args.category] if args.category else [])
    if not cats:
        p.error("укажите --category или --all")

    summaries, perprod = [], []
    for cat in cats:
        res = evaluate(cat)
        if res is None:
            continue
        s, pp = res
        summaries.append(s)
        perprod.append(pp)

    if not summaries:
        print("Нет размеченных CSV. Заполните datasets/manual_label/*_labeled.csv")
        return

    summary = pd.concat(summaries, ignore_index=True)
    perprod = pd.concat(perprod, ignore_index=True)
    s_path = os.path.join(PROCESSED_DIR, "manual_eval_summary.parquet")
    p_path = os.path.join(PROCESSED_DIR, "manual_eval_per_product.parquet")
    summary.to_parquet(s_path, index=False)
    perprod.to_parquet(p_path, index=False)

    print(f"Saved -> {s_path}")
    print(f"Saved -> {p_path}\n")
    show = summary.copy()
    for col in ["silver_coverage",
                "silver_vs_manual_acc_on_covered",
                "silver_vs_manual_acc_overall",
                "cascade_vs_manual_acc_on_covered",
                "cascade_coverage",
                "llm_vs_manual_acc_on_covered",
                "llm_coverage"]:
        show[col] = show[col].apply(
            lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—"
        )
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
