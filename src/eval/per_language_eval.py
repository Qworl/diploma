"""Experiment E7 — per-language eval-time analysis on existing cascade predictions.

Closes §7.1 п.6 (language coverage): без переобучения моделей считаем точность
каскада по языкам product_name для pasta/chocolate/cheeses на v2-gold test split.

Entry point:
    OMP_NUM_THREADS=1 python -m src.eval.per_language_eval
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect

# Воспроизводимость детектора языка (внутренний RNG)
DetectorFactory.seed = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "datasets" / "processed"

CATEGORIES = ("pasta", "chocolate", "cheeses")
EU_TOP5 = {"en", "fr", "de", "it", "es"}
OUT_PATH = PROCESSED / "per_language_eval.parquet"


def detect_language(text: str | None) -> str:
    """Детектируем язык; экзотические языки сворачиваем в 'other', неудачи — 'unknown'."""
    if text is None:
        return "unknown"
    text = str(text).strip()
    if len(text) < 2:
        return "unknown"
    try:
        lang = detect(text)
    except LangDetectException:
        return "unknown"
    except Exception:
        return "unknown"
    return lang if lang in EU_TOP5 else "other"


def load_codes_to_language(category: str, codes: set[str]) -> dict[str, str]:
    """Маппинг code -> language по product_name из silver_standard."""
    silver_path = PROCESSED / f"{category}_stratified_silver_standard.parquet"
    silver = pd.read_parquet(silver_path, columns=["code", "product_name"])
    silver["code"] = silver["code"].astype(str)
    silver = silver.drop_duplicates(subset=["code"])
    silver = silver[silver["code"].isin(codes)]
    silver["language"] = silver["product_name"].map(detect_language)
    return dict(zip(silver["code"], silver["language"]))


def build_per_language_table() -> pd.DataFrame:
    gold = pd.read_parquet(PROCESSED / "consensus_gold_v2_expanded.parquet")
    gold = gold[gold["gold_is_null"] == False].copy()  # noqa: E712
    gold["code"] = gold["code"].astype(str)

    rows: list[pd.DataFrame] = []
    for cat in CATEGORIES:
        cascade = pd.read_parquet(
            PROCESSED / f"cascade_preds_{cat}_v2_gold_hybrid_v3_fixed.parquet"
        )
        cascade["code"] = cascade["code"].astype(str)

        gold_cat = gold[gold["category"] == cat][["code", "attr", "gold_value"]]
        merged = cascade.merge(gold_cat, on=["code", "attr"], how="inner")
        # Сравнение predicted vs gold_value: оба нормализуем в строки для устойчивости
        merged["correct"] = (
            merged["predicted"].astype(str).str.strip().str.lower()
            == merged["gold_value"].astype(str).str.strip().str.lower()
        )

        code_lang = load_codes_to_language(cat, set(merged["code"].unique()))
        merged["language"] = merged["code"].map(code_lang).fillna("unknown")
        merged["category"] = cat
        rows.append(merged[["category", "code", "attr", "predicted", "gold_value", "correct", "language"]])

    full = pd.concat(rows, ignore_index=True)
    return full


def summarise(full: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        full.groupby(["category", "attr", "language"], dropna=False)
        .agg(n_cells=("correct", "size"), accuracy=("correct", "mean"))
        .reset_index()
    )
    return grouped


def print_report(full: pd.DataFrame, grouped: pd.DataFrame) -> None:
    print("\n=== E7: Per-language cascade evaluation ===")
    print(f"Total merged cells: {len(full)}")
    print("\nRows per category (sanity check vs E1 baseline ~4350):")
    print(full.groupby("category").size())

    print("\nLanguage distribution (n_cells per language, all categories):")
    lang_totals = full.groupby("language").size().sort_values(ascending=False)
    print(lang_totals)

    print("\nGrand accuracy per language (across all attrs/cats):")
    grand = (
        full.groupby("language")
        .agg(n_cells=("correct", "size"), accuracy=("correct", "mean"))
        .sort_values("n_cells", ascending=False)
    )
    print(grand.to_string(float_format=lambda x: f"{x:.4f}"))

    # Поиск (cat, attr) где есть язык, отстающий >=10 pp от max среди языков с n>=10
    print("\nCalibration gaps: (cat, attr, lang) where lang acc ≥10pp below best lang in same (cat,attr):")
    gaps: list[tuple[str, str, str, float, float, int]] = []
    min_lang_n = 10
    for (cat, attr), sub in grouped.groupby(["category", "attr"]):
        sub_filt = sub[sub["n_cells"] >= min_lang_n]
        if len(sub_filt) < 2:
            continue
        best_acc = sub_filt["accuracy"].max()
        for _, row in sub_filt.iterrows():
            gap = best_acc - row["accuracy"]
            if gap >= 0.10:
                gaps.append(
                    (cat, attr, row["language"], row["accuracy"], best_acc, int(row["n_cells"]))
                )
    gaps.sort(key=lambda r: r[4] - r[3], reverse=True)
    if not gaps:
        print("  (none with n_cells>=10 and gap>=10pp)")
    else:
        print(f"  total problematic cells found: {len(gaps)}; top up to 5:")
        for cat, attr, lang, acc, best, n in gaps[:5]:
            print(
                f"  {cat:>10s} | {attr:<20s} | lang={lang:<7s} | "
                f"acc={acc:.3f} (n={n}) vs best={best:.3f} | gap={(best - acc) * 100:5.1f}pp"
            )


def main() -> int:
    full = build_per_language_table()
    grouped = summarise(full)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_parquet(OUT_PATH, index=False)
    print(f"Saved per-language eval to: {OUT_PATH}")
    print(f"  rows={len(grouped)}, cols={list(grouped.columns)}")

    print_report(full, grouped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
