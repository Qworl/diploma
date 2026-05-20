"""Bayes-validator threshold sweep.

Гипотеза: исходный порог q=0.05 калибровался на шумном silver, поэтому на
большинстве (cat, attr) валидатор флагает в основном правильные ячейки.
Перепроверяем: сжимаем порог (q меньше → строже флажок), смотрим, появятся ли
дополнительные пары, где validator реально находит ошибки cascade-ML и
улучшает Δacc при demote → LLM.

Что считается «полезной» парой:
  • demote_precision_lift > 0  (точнее случайного demote)
  • expected_delta_acc_if_demote > 0  (вес ошибок × llm_acc оправдывает
                                       демоушн правильных ячеек)

Вход:
  • models/{cat}_stratified_bayesian.pkl
  • datasets/processed/{cat}_stratified_silver_standard.parquet   (для калибровки)
  • datasets/processed/cascade_preds_{cat}_v2_gold_hybrid_v3_fixed.parquet
  • datasets/processed/consensus_gold_v2_expanded.parquet
  • datasets/processed/direct_llm_eval_{cat}_stratified_sonnet45.parquet

Выход:
  • datasets/processed/bayes_validator_threshold_sweep.parquet
    (по строке на (category, attr, q))

Запуск:
  OMP_NUM_THREADS=1 python -m src.eval.bayes_validator_threshold_sweep
"""

from __future__ import annotations

import os
import pickle
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
OUT_PATH = f"{PROCESSED}/bayes_validator_threshold_sweep.parquet"


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


def _load_brand_lookup(internal_cat: str) -> dict[str, str]:
    path = f"{PROCESSED}/{internal_cat}_raw.parquet"
    df = pd.read_parquet(path, columns=["code", "brands"])
    df["code"] = df["code"].astype(str)
    df["brands"] = df["brands"].fillna("").astype(str)
    return dict(zip(df["code"], df["brands"]))


def _load_llm_acc(internal_cat: str) -> dict[str, float]:
    path = f"{PROCESSED}/direct_llm_eval_{internal_cat}_sonnet45.parquet"
    df = pd.read_parquet(path)
    mask = (df["predicted_non_null"] == 1) & (df["gt_non_null"] == 1)
    return df[mask].groupby("attr")["correct_when_both_present"].mean().to_dict()


def _evaluate_one_q(
    short_cat: str,
    internal_cat: str,
    q: float,
    bayes,
    inference,
    silver_df: pd.DataFrame,
    cascade_ml: pd.DataFrame,
    gold_df: pd.DataFrame,
    base_ev_by_code: dict[str, dict[str, Any]],
    llm_acc: dict[str, float],
) -> list[dict]:
    thresholds = calibrate_thresholds(bayes, silver_df, inference, q=q)

    merged = cascade_ml.merge(gold_df, on=["code", "attr"], how="inner")
    cells: list[dict] = []
    for _, r in merged.iterrows():
        attr = r["attr"]
        if attr not in bayes.nodes():
            continue
        thr = thresholds.get(attr)
        if thr is None:
            continue
        code = r["code"]
        pred = r["predicted"]
        gold_v = r["gold_value"]

        ev = dict(base_ev_by_code.get(code, {}))
        ev.pop(attr, None)
        p = attribute_likelihood(attr, pred, ev, bayes, inference)
        if p is None:
            continue
        flagged = bool(p < thr)
        correct = _eq(pred, gold_v)
        cells.append({"attr": attr, "flagged": flagged, "correct": correct, "thr": thr})

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
        demote_prec = tp / (tp + fp) if (tp + fp) else float("nan")
        demote_rec = tp / (tp + fn) if (tp + fn) else float("nan")
        cascade_acc = sub["correct"].mean() if n_ml else float("nan")
        baseline = 1.0 - cascade_acc

        l_acc = llm_acc.get(attr, float("nan"))
        if not np.isnan(l_acc):
            exp_dacc = (tp * l_acc + fp * (l_acc - 1.0)) / n_ml if n_ml else 0.0
        else:
            exp_dacc = float("nan")
        exp_dcost = (tp + fp) / n_ml if n_ml else 0.0

        out.append({
            "category": short_cat,
            "attr": attr,
            "q": q,
            "threshold": float(sub["thr"].iloc[0]),
            "n_ml": n_ml,
            "n_flagged": n_flag,
            "flag_rate": flag_rate,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "cascade_acc": cascade_acc,
            "random_baseline": baseline,
            "demote_precision": demote_prec,
            "demote_precision_lift": (demote_prec - baseline) if not np.isnan(demote_prec) else float("nan"),
            "demote_recall": demote_rec,
            "llm_acc_on_attr": l_acc,
            "expected_delta_acc_if_demote": exp_dacc,
            "expected_delta_llm_cost": exp_dcost,
        })
    return out


def evaluate_category(short_cat: str, internal_cat: str) -> list[dict]:
    print(f"\n=== {short_cat} ({internal_cat}) ===", flush=True)
    bayes_path = f"{MODELS}/{internal_cat}_bayesian.pkl"
    silver_path = f"{PROCESSED}/{internal_cat}_silver_standard.parquet"
    cascade_path = f"{PROCESSED}/cascade_preds_{short_cat}_v2_gold_hybrid_v3_fixed.parquet"
    gold_path = f"{PROCESSED}/consensus_gold_v2_expanded.parquet"

    with open(bayes_path, "rb") as f:
        bayes = pickle.load(f)
    silver = pd.read_parquet(silver_path)
    inference = VariableElimination(bayes)

    cascade = pd.read_parquet(cascade_path)
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

    gold = pd.read_parquet(gold_path)
    gold = gold[(gold["category"] == short_cat) & (~gold["gold_is_null"])].copy()
    gold["code"] = gold["code"].astype(str)
    gold = gold[["code", "attr", "gold_value"]]

    llm_acc = _load_llm_acc(internal_cat)

    print(f"  silver: {len(silver):,} | cascade_ml: {len(cascade_ml):,} | gold: {len(gold):,}", flush=True)
    rows: list[dict] = []
    for q in Q_VALUES:
        print(f"  q={q:.2f} ...", flush=True)
        rows.extend(_evaluate_one_q(
            short_cat, internal_cat, q, bayes, inference,
            silver, cascade_ml, gold, base_ev, llm_acc,
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

    print("\n=== FULL TABLE ===")
    cols = ["category", "attr", "q", "n_flagged", "flag_rate",
            "demote_precision_lift", "expected_delta_acc_if_demote",
            "expected_delta_llm_cost", "demote_recall", "tp", "fp"]
    print(df[cols].to_string(index=False))

    print("\n=== USEFUL PAIRS (lift > 0 AND Δacc > 0) ===")
    useful = df[(df["demote_precision_lift"] > 0) &
                (df["expected_delta_acc_if_demote"] > 0)].copy()
    if useful.empty:
        print("(none)")
    else:
        print(useful[cols].sort_values(
            ["category", "attr", "q"]
        ).to_string(index=False))

    print("\n=== BEST q PER (category, attr) BY Δacc ===")
    best_idx = df.groupby(["category", "attr"])["expected_delta_acc_if_demote"].idxmax()
    best = df.loc[best_idx]
    print(best[cols].sort_values(
        ["category", "expected_delta_acc_if_demote"], ascending=[True, False]
    ).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
