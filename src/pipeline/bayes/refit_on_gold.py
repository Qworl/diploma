"""Переобучение Bayes-сети на gold-данных с сохранением silver-структуры.

Подход:
  1. Загружаем существующую сеть `models/{cat}_bayesian.pkl` (структура DAG
     выучена на 15K silver — устойчивая, охватывает все бренды).
  2. Берём её рёбра как структуру.
  3. Заново фитим CPD-таблицы на gold (≈ 888 продуктов × 7 атрибутов из
     `consensus_gold_v2_expanded.parquet`) через BayesianEstimator с BDeu prior.
  4. Сохраняем как `models/{cat}_bayesian_gold.pkl`.
  5. Калибруем пороги q=0.02 на тех же gold-данных, сохраняем как
     `models/{cat}_validation_thresholds_gold_refit.json`.

Идея: structure от silver (надёжность), parameters от gold (чистота).
Это позволяет валидатору ловить ML-ошибки, которые ML унаследовал от silver-шума.

Запуск:
  OMP_NUM_THREADS=1 python -m src.pipeline.bayes.refit_on_gold
"""
from __future__ import annotations

import json
import pickle
import re
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

from src.pipeline.bayes.validate import calibrate_thresholds

PROCESSED = Path("datasets/processed")
MODELS = Path("models")
Q = 0.02

CATS = [
    ("pasta", "pasta_stratified"),
    ("chocolate", "chocolate_stratified"),
    ("cheeses", "cheeses_stratified"),
]

_ORGANIC_RE = re.compile(r"\b(bio|organic|organique|eco|ecol[oó]gico|ekol|öko)\b", re.I)


def _brand_first(s: str) -> str:
    if not s:
        return "unknown"
    s = str(s).split(",")[0].strip().lower()
    return s or "unknown"


def _brand_org(s: str) -> str:
    return "True" if _ORGANIC_RE.search(str(s or "")) else "False"


def build_gold_wide(short: str, internal: str) -> pd.DataFrame:
    gold = pd.read_parquet(PROCESSED / "consensus_gold_v2_expanded.parquet")
    gold = gold[(gold["category"] == short) & (~gold["gold_is_null"])].copy()
    gold["code"] = gold["code"].astype(str)
    wide = gold.pivot_table(index="code", columns="attr",
                            values="gold_value", aggfunc="first")

    raw = pd.read_parquet(PROCESSED / f"{internal}_raw.parquet",
                          columns=["code", "brands"])
    raw["code"] = raw["code"].astype(str)
    raw["brands"] = raw["brands"].fillna("").astype(str)
    raw["brand"] = raw["brands"].apply(_brand_first)
    raw["brand_has_organic_marker"] = raw["brands"].apply(_brand_org)

    merged = wide.merge(raw[["code", "brand", "brand_has_organic_marker"]],
                        on="code", how="left").set_index("code")
    for col in merged.columns:
        if col == "brand":
            merged[col] = merged[col].fillna("unknown").astype(str)
            continue
        merged[col] = merged[col].map(
            lambda v: None if v is None or (isinstance(v, float) and np.isnan(v))
            else str(v)
        )
    return merged


def restrict_brand_to_silver_support(gold: pd.DataFrame, silver_bayes) -> pd.DataFrame:
    """Свернуть бренды gold в silver-известные states + 'other'."""
    if "brand" not in silver_bayes.nodes():
        return gold
    cpd = silver_bayes.get_cpds("brand")
    silver_states = set(str(s) for s in cpd.state_names["brand"])
    out = gold.copy()
    out["brand"] = out["brand"].apply(
        lambda b: b if b in silver_states else ("other" if "other" in silver_states else b)
    )
    return out


def refit_one(short: str, internal: str) -> dict:
    silver_path = MODELS / f"{internal}_bayesian.pkl"
    with open(silver_path, "rb") as f:
        silver_bayes = pickle.load(f)
    edges = list(silver_bayes.edges())
    print(f"\n=== {internal} ===")
    print(f"  silver structure: {len(edges)} edges, {len(silver_bayes.nodes())} nodes")

    gold_wide = build_gold_wide(short, internal)
    print(f"  gold rows: {len(gold_wide):,}")

    # Свёртка бренда в silver-support, чтобы gold не плодил unseen states
    gold_train = restrict_brand_to_silver_support(gold_wide.reset_index(drop=True),
                                                  silver_bayes)

    # Drop rows where ANY needed node is NaN (BayesianEstimator не любит NaN)
    needed_nodes = list(silver_bayes.nodes())
    fit_df = gold_train[needed_nodes].dropna().copy()
    for c in fit_df.columns:
        fit_df[c] = fit_df[c].astype(str)
    print(f"  rows after dropna on {len(needed_nodes)} nodes: {len(fit_df):,}")

    if len(fit_df) < 100:
        print(f"  WARN: too few rows ({len(fit_df)}); skipping")
        return {"skipped": True}

    model = BayesianNetwork(edges)
    est = BayesianEstimator(model, fit_df)
    for node in model.nodes():
        cpd = est.estimate_cpd(node, prior_type="BDeu", equivalent_sample_size=10)
        model.add_cpds(cpd)
    model.check_model()
    print(f"  fitted CPDs for {len(model.nodes())} nodes")

    out_bayes = MODELS / f"{internal}_bayesian_gold.pkl"
    with open(out_bayes, "wb") as f:
        pickle.dump(model, f)
    print(f"  saved {out_bayes.name}")

    # Calibrate thresholds at q=0.02 on the same gold data
    inference = VariableElimination(model)
    thresholds = calibrate_thresholds(model, fit_df, inference, q=Q)
    print(f"  calibrated thresholds (q={Q}):")
    for attr, thr in sorted(thresholds.items()):
        print(f"    {attr}: {thr:.6f}")

    out_thr = MODELS / f"{internal}_validation_thresholds_gold_refit.json"
    payload = {
        "category": internal,
        "source": "gold (consensus_gold_v2_expanded) + silver DAG (structure)",
        "q": Q,
        "n_train_rows": int(len(fit_df)),
        "n_edges": len(edges),
        "thresholds": thresholds,
    }
    with open(out_thr, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"  saved {out_thr.name}")
    return {"thresholds": thresholds, "n_train": len(fit_df)}


def main() -> int:
    for short, internal in CATS:
        refit_one(short, internal)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
