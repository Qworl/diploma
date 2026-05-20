"""Bayes-validator threshold sweep, calibrated on **gold** instead of silver.

Гипотеза: Bayes-сеть выучена на 15 K silver-строк (нужна для устойчивости
структуры и охвата брендов), но пороги «p < thr → флажок» сейчас тоже взяты
с silver — а silver шумный, поэтому 5-й перцентиль смещён и валидатор
ошибается на остальных атрибутах. Здесь сеть остаётся та же, но пороги
пересчитываем на consensus_gold_v2_expanded (~888 строк × 7 атрибутов).

Затем — тот же sweep по q ∈ {0.01, 0.02, 0.03, 0.05, 0.10} на brand-disjoint
gold-тесте: для каждой (cat, attr) считаем demote_precision / lift / Δacc.

Вход:
  • models/{internal_cat}_bayesian.pkl
  • datasets/processed/consensus_gold_v2_expanded.parquet  (калибровка + eval)
  • datasets/processed/{internal_cat}_raw.parquet           (brand lookup)
  • datasets/processed/cascade_preds_{cat}_v2_gold_hybrid_v3_fixed.parquet
  • datasets/processed/direct_llm_eval_{internal_cat}_sonnet45.parquet

Выход:
  • datasets/processed/bayes_validator_gold_sweep.parquet

Запуск:
  OMP_NUM_THREADS=1 python -m src.eval.bayes_validator_gold_sweep
"""
from __future__ import annotations

import pickle
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination

from src.pipeline.bayes.validate import (
    attribute_likelihood,
    calibrate_thresholds,
)

CATEGORIES = [
    ("pasta", "pasta_stratified"),
    ("chocolate", "chocolate_stratified"),
    ("cheeses", "cheeses_stratified"),
]
Q_VALUES = (0.01, 0.02, 0.03, 0.05, 0.10)
PROCESSED = "datasets/processed"
MODELS = "models"
OUT_PATH = f"{PROCESSED}/bayes_validator_gold_sweep.parquet"

_ORGANIC_RE = re.compile(r"\b(bio|organic|organique|eco|ecol[oó]gico|ekol|öko)\b", re.I)


def _normalize(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, str) and v.strip().lower() in {"", "none", "null", "nan"}:
        return None
    return v


def _eq(a: Any, b: Any) -> bool:
    a, b = _normalize(a), _normalize(b)
    if a is None or b is None:
        return False
    return str(a).strip().lower() == str(b).strip().lower()


def _brand_first(raw_brand: str) -> str:
    if not raw_brand:
        return "unknown"
    s = str(raw_brand).split(",")[0].strip().lower()
    return s or "unknown"


def _brand_has_organic_marker(raw_brand: str) -> str:
    return "True" if _ORGANIC_RE.search(str(raw_brand or "")) else "False"


def _build_gold_wide(short_cat: str, internal_cat: str) -> pd.DataFrame:
    """Pivot consensus_gold_v2_expanded → wide DataFrame for calibration.

    Каждая строка — продукт; колонки = атрибуты bayes-сети. Только не-null
    gold-значения попадают (gold_is_null=False).
    """
    gold = pd.read_parquet(f"{PROCESSED}/consensus_gold_v2_expanded.parquet")
    gold = gold[(gold["category"] == short_cat) & (~gold["gold_is_null"])].copy()
    gold["code"] = gold["code"].astype(str)

    wide = gold.pivot_table(
        index="code", columns="attr", values="gold_value", aggfunc="first"
    )

    raw = pd.read_parquet(f"{PROCESSED}/{internal_cat}_raw.parquet",
                          columns=["code", "brands"])
    raw["code"] = raw["code"].astype(str)
    raw["brands"] = raw["brands"].fillna("").astype(str)
    raw["brand"] = raw["brands"].apply(_brand_first)
    raw["brand_has_organic_marker"] = raw["brands"].apply(_brand_has_organic_marker)
    raw = raw.set_index("code")

    merged = wide.join(raw[["brand", "brand_has_organic_marker"]], how="left")
    # Привести типы к тому формату, который ждёт bucketize (silver всё хранит
    # как str — особенно булевы поля: "True"/"False" вместо True/False).
    for col in merged.columns:
        if col in {"brand"}:
            merged[col] = merged[col].fillna("unknown").astype(str)
            continue
        merged[col] = merged[col].map(
            lambda v: None if v is None or (isinstance(v, float) and np.isnan(v))
            else str(v)
        )
    return merged.reset_index()


def _load_brand_lookup(internal_cat: str) -> dict[str, str]:
    df = pd.read_parquet(f"{PROCESSED}/{internal_cat}_raw.parquet",
                         columns=["code", "brands"])
    df["code"] = df["code"].astype(str)
    df["brands"] = df["brands"].fillna("").astype(str)
    return dict(zip(df["code"], df["brands"]))


def _load_llm_acc(internal_cat: str) -> dict[str, float]:
    df = pd.read_parquet(f"{PROCESSED}/direct_llm_eval_{internal_cat}_sonnet45.parquet")
    mask = (df["predicted_non_null"] == 1) & (df["gt_non_null"] == 1)
    return df[mask].groupby("attr")["correct_when_both_present"].mean().to_dict()


def _sweep_one_q(short_cat, internal_cat, q, bayes, inference,
                 gold_wide, cascade_ml, gold_long, base_ev, llm_acc) -> list[dict]:
    thresholds = calibrate_thresholds(bayes, gold_wide, inference, q=q)
    merged = cascade_ml.merge(gold_long, on=["code", "attr"], how="inner")
    cells: list[dict] = []
    for _, r in merged.iterrows():
        attr = r["attr"]
        if attr not in bayes.nodes():
            continue
        thr = thresholds.get(attr)
        if thr is None:
            continue
        ev = dict(base_ev.get(r["code"], {}))
        ev.pop(attr, None)
        p = attribute_likelihood(attr, r["predicted"], ev, bayes, inference)
        if p is None:
            continue
        cells.append({
            "attr": attr,
            "flagged": bool(p < thr),
            "correct": _eq(r["predicted"], r["gold_value"]),
            "thr": thr,
        })
    if not cells:
        return []
    cdf = pd.DataFrame(cells)
    out: list[dict] = []
    for attr, sub in cdf.groupby("attr"):
        n_ml = len(sub)
        n_flag = int(sub["flagged"].sum())
        tp = int(((sub["flagged"]) & (~sub["correct"])).sum())
        fp = int(((sub["flagged"]) & (sub["correct"])).sum())
        tn = int(((~sub["flagged"]) & (sub["correct"])).sum())
        fn = int(((~sub["flagged"]) & (~sub["correct"])).sum())
        flag_rate = n_flag / n_ml if n_ml else 0.0
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        cascade_acc = sub["correct"].mean() if n_ml else float("nan")
        baseline = 1.0 - cascade_acc
        l_acc = llm_acc.get(attr, float("nan"))
        if not np.isnan(l_acc):
            exp_dacc = (tp * l_acc + fp * (l_acc - 1.0)) / n_ml if n_ml else 0.0
        else:
            exp_dacc = float("nan")
        out.append({
            "category": short_cat, "attr": attr, "q": q,
            "threshold": float(sub["thr"].iloc[0]),
            "n_ml": n_ml, "n_flagged": n_flag, "flag_rate": flag_rate,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "cascade_acc": cascade_acc, "random_baseline": baseline,
            "demote_precision": prec,
            "demote_precision_lift": (prec - baseline) if not np.isnan(prec) else float("nan"),
            "demote_recall": rec,
            "llm_acc_on_attr": l_acc,
            "expected_delta_acc_if_demote": exp_dacc,
            "expected_delta_llm_cost": (tp + fp) / n_ml if n_ml else 0.0,
        })
    return out


def evaluate_category(short_cat: str, internal_cat: str) -> list[dict]:
    print(f"\n=== {short_cat} ({internal_cat}) ===", flush=True)
    with open(f"{MODELS}/{internal_cat}_bayesian.pkl", "rb") as f:
        bayes = pickle.load(f)
    inference = VariableElimination(bayes)

    gold_wide = _build_gold_wide(short_cat, internal_cat)
    print(f"  gold_wide rows for calibration: {len(gold_wide):,}", flush=True)
    print(f"  gold_wide columns vs bayes nodes: "
          f"present={sorted(set(gold_wide.columns) & set(bayes.nodes()))}; "
          f"missing={sorted(set(bayes.nodes()) - set(gold_wide.columns))}", flush=True)

    cascade = pd.read_parquet(
        f"{PROCESSED}/cascade_preds_{short_cat}_v2_gold_hybrid_v3_fixed.parquet"
    )
    cascade["code"] = cascade["code"].astype(str)
    brands = _load_brand_lookup(internal_cat)

    base_ev: dict[str, dict[str, Any]] = {}
    for code, grp in cascade.groupby("code"):
        ev: dict[str, Any] = {}
        if code in brands and brands[code]:
            ev["brand"] = brands[code]
        for _, row in grp.iterrows():
            if row["layer"] == "abstain":
                continue
            ev[row["attr"]] = row["predicted"]
        base_ev[code] = ev

    cascade_ml = cascade[cascade["layer"] == "ml"].copy()
    gold_long = pd.read_parquet(f"{PROCESSED}/consensus_gold_v2_expanded.parquet")
    gold_long = gold_long[(gold_long["category"] == short_cat)
                          & (~gold_long["gold_is_null"])].copy()
    gold_long["code"] = gold_long["code"].astype(str)
    gold_long = gold_long[["code", "attr", "gold_value"]]
    llm_acc = _load_llm_acc(internal_cat)

    rows: list[dict] = []
    for q in Q_VALUES:
        print(f"  q={q:.2f} ...", flush=True)
        rows.extend(_sweep_one_q(
            short_cat, internal_cat, q, bayes, inference,
            gold_wide, cascade_ml, gold_long, base_ev, llm_acc,
        ))
    return rows


def main() -> int:
    all_rows: list[dict] = []
    for short, internal in CATEGORIES:
        all_rows.extend(evaluate_category(short, internal))
    if not all_rows:
        print("No rows produced", file=sys.stderr)
        return 1
    df = pd.DataFrame(all_rows)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows → {OUT_PATH}", flush=True)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:+.3f}" if isinstance(x, float) else str(x))

    cols = ["category", "attr", "q", "n_flagged", "flag_rate",
            "demote_precision_lift", "expected_delta_acc_if_demote",
            "expected_delta_llm_cost", "demote_recall", "tp", "fp"]

    print("\n=== USEFUL PAIRS (gold-calibrated; lift>0 AND Δacc>0) ===")
    useful = df[(df["demote_precision_lift"] > 0)
                & (df["expected_delta_acc_if_demote"] > 0)].copy()
    if useful.empty:
        print("(none)")
    else:
        print(useful[cols].sort_values(["category", "attr", "q"]).to_string(index=False))

    print("\n=== BEST q PER (category, attr) BY Δacc (gold-calibrated) ===")
    best_idx = df.groupby(["category", "attr"])["expected_delta_acc_if_demote"].idxmax()
    best = df.loc[best_idx]
    print(best[cols].sort_values(
        ["category", "expected_delta_acc_if_demote"], ascending=[True, False]
    ).to_string(index=False))

    print("\n=== DIFF vs SILVER SWEEP (best q per pair) ===")
    try:
        silver = pd.read_parquet(f"{PROCESSED}/bayes_validator_threshold_sweep.parquet")
        silver_best = silver.loc[
            silver.groupby(["category", "attr"])["expected_delta_acc_if_demote"].idxmax()
        ][["category", "attr", "q", "demote_precision_lift",
           "expected_delta_acc_if_demote", "expected_delta_llm_cost"]].rename(
            columns={"q": "q_silver",
                     "demote_precision_lift": "lift_silver",
                     "expected_delta_acc_if_demote": "dacc_silver",
                     "expected_delta_llm_cost": "cost_silver"}
        )
        gold_best = best[["category", "attr", "q", "demote_precision_lift",
                          "expected_delta_acc_if_demote", "expected_delta_llm_cost"]].rename(
            columns={"q": "q_gold",
                     "demote_precision_lift": "lift_gold",
                     "expected_delta_acc_if_demote": "dacc_gold",
                     "expected_delta_llm_cost": "cost_gold"}
        )
        diff = silver_best.merge(gold_best, on=["category", "attr"], how="outer")
        diff["dacc_delta"] = diff["dacc_gold"] - diff["dacc_silver"]
        print(diff.sort_values(
            ["category", "dacc_delta"], ascending=[True, False]
        ).to_string(index=False))
    except Exception as e:
        print(f"(no silver baseline to diff against: {e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
