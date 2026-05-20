"""Деплой финальной конфигурации accuracy-squeeze в production.

Действия:
  1. Резервная копия текущих файлов в `models/`.
  2. Обучить hybrid Bayes (silver brand-disjoint + gold replicated × 10),
     сохранить как `models/{cat}_bayesian.pkl`.
  3. Калибровать пороги Bayes по chosen_A.bayes_q (per-attr), сохранить
     selective `_validation_thresholds.json` только для атрибутов из конфига.
  4. Обновить `_thresholds.pkl` — заменить значения для (cat, attr) из chosen_E.
  5. Sanity-check через ValidatorService.

Источник конфига:
  `datasets/processed/accuracy_squeeze_chosen_config.json`

Запуск:
  OMP_NUM_THREADS=1 python -m src.experiments.deploy_squeeze_config
"""
from __future__ import annotations

import json
import pickle
import re
import shutil
import sys
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
GOLD_REPLICATIONS = 10

CATS = [
    ("pasta", "pasta_stratified"),
    ("chocolate", "chocolate_stratified"),
    ("cheeses", "cheeses_stratified"),
]

_ORG = re.compile(r"\b(bio|organic|organique|eco|ecol[oó]gico|ekol|öko)\b", re.I)


def _bf(s): return (str(s).split(",")[0].strip().lower() or "unknown") if s else "unknown"
def _bo(s): return "True" if _ORG.search(str(s or "")) else "False"


def get_test_codes(short):
    cp = pd.read_parquet(PROCESSED / f"cascade_preds_{short}_after_fix.parquet",
                          columns=["code"])
    return set(cp["code"].astype(str).unique())


def build_gold_wide(short, internal):
    g = pd.read_parquet(PROCESSED / "consensus_gold_v2_expanded.parquet")
    g = g[(g.category == short) & ~g.gold_is_null].copy()
    g["code"] = g["code"].astype(str)
    wide = g.pivot_table(index="code", columns="attr", values="gold_value", aggfunc="first")
    raw = pd.read_parquet(PROCESSED / f"{internal}_raw.parquet", columns=["code", "brands"])
    raw["code"] = raw["code"].astype(str); raw["brands"] = raw["brands"].fillna("").astype(str)
    raw["brand"] = raw["brands"].apply(_bf); raw["brand_has_organic_marker"] = raw["brands"].apply(_bo)
    m = wide.merge(raw[["code","brand","brand_has_organic_marker"]], on="code", how="left").set_index("code")
    for c in m.columns:
        if c == "brand":
            m[c] = m[c].fillna("unknown").astype(str); continue
        m[c] = m[c].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
    return m


def silver_brand_disjoint(internal, nodes, silver_bayes, test_codes):
    silver_df = pd.read_parquet(PROCESSED / f"{internal}_silver_standard.parquet")
    silver_df["code"] = silver_df["code"].astype(str)
    silver_df = silver_df[~silver_df["code"].isin(test_codes)].reset_index(drop=True)
    sub = pd.DataFrame()
    sub["brand_raw"] = silver_df.get("brands", pd.Series([""] * len(silver_df)))
    sub["brand"] = sub["brand_raw"].fillna("unknown").apply(_bf)
    cpd_states = set(str(s) for s in silver_bayes.get_cpds("brand").state_names["brand"]) \
        if "brand" in silver_bayes.nodes() else set()
    sub["brand"] = sub["brand"].apply(lambda b: b if b in cpd_states else ("other" if "other" in cpd_states else b))
    sub["brand_has_organic_marker"] = sub["brand_raw"].apply(_bo)
    for node in nodes:
        if node in ("brand", "brand_has_organic_marker"): continue
        sub[node] = silver_df.get(node)
    out = sub[nodes].copy()
    for c in out.columns:
        out[c] = out[c].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
    return out.dropna(subset=nodes)


def fit_and_get_hybrid(short, internal):
    """Учим финальную hybrid Bayes сеть (silver brand-disjoint vs test + gold × 10)."""
    with open(MODELS / f"{internal}_bayesian.silver_backup.pkl", "rb") as f:
        silver_bayes = pickle.load(f)
    edges = list(silver_bayes.edges()); nodes = list(silver_bayes.nodes())
    test_codes = get_test_codes(short)
    gold_wide = build_gold_wide(short, internal)
    train_gold = gold_wide[~gold_wide.index.isin(test_codes)].copy()
    if "brand" in silver_bayes.nodes():
        states = set(str(s) for s in silver_bayes.get_cpds("brand").state_names["brand"])
        train_gold["brand"] = train_gold["brand"].apply(lambda b: b if b in states else ("other" if "other" in states else b))
    gold_fit = train_gold[nodes].dropna().astype(str)
    silver_fit = silver_brand_disjoint(internal, nodes, silver_bayes, test_codes)
    hybrid = pd.concat([silver_fit] + [gold_fit] * GOLD_REPLICATIONS, ignore_index=True)
    model = BayesianNetwork(edges)
    est = BayesianEstimator(model, hybrid)
    for node in model.nodes():
        cpd = est.estimate_cpd(node, prior_type="BDeu", equivalent_sample_size=10)
        model.add_cpds(cpd)
    model.check_model()
    return model, gold_fit


def main():
    # Read config
    with open(PROCESSED / "accuracy_squeeze_chosen_config.json") as f:
        config = json.load(f)
    E_thr = {tuple(k.split("/")): v for k, v in config["E_ml_thresholds"].items()}
    A_q = {tuple(k.split("/")): v for k, v in config["A_bayes_q"].items()}

    # Step 1: BACKUP (если ещё не сделан)
    for short, internal in CATS:
        for fname, backup_suffix in [
            (f"{internal}_bayesian.pkl", ".silver_backup.pkl"),
            (f"{internal}_validation_thresholds.json", ".scenario_c_backup.json"),
            (f"{internal}_thresholds.pkl", ".scenario_c_backup.pkl"),
        ]:
            src = MODELS / fname
            backup = src.with_suffix(backup_suffix)
            if src.exists() and not backup.exists():
                shutil.copy(src, backup)
                print(f"  backup: {src.name} → {backup.name}")
            elif backup.exists():
                print(f"  backup already exists: {backup.name}")

    # Step 2: fit and save Bayes models
    print("\n=== TRAINING HYBRID BAYES (silver brand-disjoint + gold × 10) ===")
    for short, internal in CATS:
        print(f"\n  {internal}")
        model, gold_fit = fit_and_get_hybrid(short, internal)
        out_bayes = MODELS / f"{internal}_bayesian.pkl"
        with open(out_bayes, "wb") as f: pickle.dump(model, f)
        print(f"    saved {out_bayes}")

        # Step 3: calibrate selective thresholds per chosen A_q
        inference = VariableElimination(model)
        selective_thr = {}
        # Group chosen attrs by category
        chosen_attrs_for_cat = {a: q for (c, a), q in A_q.items() if c == short}
        unique_qs = set(chosen_attrs_for_cat.values())
        # Compute thresholds per q (since calibrate returns ALL attrs)
        thr_per_q = {q: calibrate_thresholds(model, gold_fit, inference, q=q) for q in unique_qs}
        for attr, q in chosen_attrs_for_cat.items():
            thr = thr_per_q[q].get(attr)
            if thr is not None:
                selective_thr[attr] = float(thr)
        out_thr = MODELS / f"{internal}_validation_thresholds.json"
        payload = {
            "category": internal,
            "source": "hybrid (silver brand-disjoint + gold × 10), per-attr q from accuracy-squeeze config",
            "n_train_rows": int(len(gold_fit)),
            "per_attr_q": chosen_attrs_for_cat,
            "thresholds": selective_thr,
        }
        with open(out_thr, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"    saved {out_thr.name} with {len(selective_thr)} attrs: {sorted(selective_thr.keys())}")

    # Step 4: update per-attr ML thresholds
    print("\n=== UPDATING ML THRESHOLDS ===")
    for short, internal in CATS:
        thr_path = MODELS / f"{internal}_thresholds.pkl"
        with open(thr_path, "rb") as f:
            current_thresholds = pickle.load(f)
        cat_E = {a: v for (c, a), v in E_thr.items() if c == short}
        changes = {}
        for attr, new_thr in cat_E.items():
            old = current_thresholds.get(attr)
            current_thresholds[attr] = float(new_thr)
            changes[attr] = (old, new_thr)
        with open(thr_path, "wb") as f: pickle.dump(current_thresholds, f)
        print(f"\n  {internal}: updated {len(changes)} attrs")
        for attr, (old, new) in changes.items():
            old_s = f"{float(old):.2f}" if old is not None else "—"
            print(f"    {attr}: {old_s} → {new:.2f}")

    # Step 5: sanity-check
    print("\n=== SANITY CHECK ===")
    sys.path.insert(0, str(Path("demo/ml_service")))
    from validator import ValidatorService
    v = ValidatorService(models_dir=str(MODELS),
                          internal_categories=[ic for _, ic in CATS])
    print(f"  Validator ready: {v.ready()}")
    for cat, thr in v.thresholds.items():
        print(f"  {cat}: {len(thr)} attrs with thresholds")

    print("\nDeployment complete.")


if __name__ == "__main__":
    sys.exit(main() or 0)
