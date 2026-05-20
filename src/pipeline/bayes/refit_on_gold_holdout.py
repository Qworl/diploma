"""Brand-disjoint holdout эксперимент: gold-refit Bayes + честный test.

  • Bayes обучается на gold-строках с TRAIN-брендами (бренды cascade-train)
  • Валидируется на cascade-ML предсказаниях для TEST-брендов
  • Это устраняет in-sample bias предыдущего эксперимента

Запуск:
  OMP_NUM_THREADS=1 python -m src.pipeline.bayes.refit_on_gold_holdout
"""
from __future__ import annotations

import pickle
import re
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

from src.pipeline.bayes.validate import (
    attribute_likelihood,
    calibrate_thresholds,
)

PROCESSED = Path("datasets/processed")
MODELS = Path("models")
Q = 0.02

CATS = [
    ("pasta", "pasta_stratified"),
    ("chocolate", "chocolate_stratified"),
    ("cheeses", "cheeses_stratified"),
]

_ORGANIC_RE = re.compile(r"\b(bio|organic|organique|eco|ecol[oó]gico|ekol|öko)\b", re.I)


def _bf(s): return (str(s).split(",")[0].strip().lower() or "unknown") if s else "unknown"
def _bo(s): return "True" if _ORGANIC_RE.search(str(s or "")) else "False"


def _normalize(v):
    if v is None: return None
    if isinstance(v, float) and np.isnan(v): return None
    if isinstance(v, str) and v.strip().lower() in {"", "none", "null", "nan"}: return None
    return v


def _eq(a, b):
    a, b = _normalize(a), _normalize(b)
    if a is None or b is None: return False
    return str(a).strip().lower() == str(b).strip().lower()


def build_gold_with_brand(short, internal):
    """Pivot gold to wide; добавить brand + brand_has_organic_marker."""
    g = pd.read_parquet(PROCESSED / "consensus_gold_v2_expanded.parquet")
    g = g[(g["category"] == short) & (~g["gold_is_null"])].copy()
    g["code"] = g["code"].astype(str)
    wide = g.pivot_table(index="code", columns="attr", values="gold_value", aggfunc="first")

    raw = pd.read_parquet(PROCESSED / f"{internal}_raw.parquet", columns=["code", "brands"])
    raw["code"] = raw["code"].astype(str)
    raw["brands"] = raw["brands"].fillna("").astype(str)
    raw["brand_raw"] = raw["brands"]
    raw["brand"] = raw["brands"].apply(_bf)
    raw["brand_has_organic_marker"] = raw["brands"].apply(_bo)

    merged = wide.merge(raw[["code", "brand", "brand_has_organic_marker"]],
                        on="code", how="left").set_index("code")
    for col in merged.columns:
        if col == "brand":
            merged[col] = merged[col].fillna("unknown").astype(str)
            continue
        merged[col] = merged[col].map(
            lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
        )
    return merged


def get_test_codes(short):
    """Test codes — те, что фигурируют в cascade_preds_after_fix (это brand-disjoint test)."""
    cp = pd.read_parquet(PROCESSED / f"cascade_preds_{short}_after_fix.parquet",
                          columns=["code"])
    cp["code"] = cp["code"].astype(str)
    return set(cp["code"].unique())


def restrict_brand_to_silver(gold, silver_bayes):
    if "brand" not in silver_bayes.nodes(): return gold
    cpd = silver_bayes.get_cpds("brand")
    states = set(str(s) for s in cpd.state_names["brand"])
    out = gold.copy()
    out["brand"] = out["brand"].apply(lambda b: b if b in states else ("other" if "other" in states else b))
    return out


def llm_acc(internal):
    df = pd.read_parquet(PROCESSED / f"direct_llm_eval_{internal}_sonnet45.parquet")
    m = (df.predicted_non_null == 1) & (df.gt_non_null == 1)
    return df[m].groupby("attr").correct_when_both_present.mean().to_dict()


def main():
    print(f"{'category/attr':38s} {'n_ml':5s} {'n_flag':7s} {'TP':4s} {'FP':4s} "
          f"{'lift':>7s} {'Δacc':>7s} {'Δcost':>7s}")
    print("-" * 96)

    total_results = []
    n_total_test_cells = 0

    for short, internal in CATS:
        with open(MODELS / f"{internal}_bayesian.pkl", "rb") as f:
            silver_bayes = pickle.load(f)
        edges = list(silver_bayes.edges())

        gold = build_gold_with_brand(short, internal)
        test_codes = get_test_codes(short)
        train_mask = ~gold.index.isin(test_codes)
        train_gold = gold[train_mask].copy()
        train_gold = restrict_brand_to_silver(train_gold.reset_index(drop=True),
                                              silver_bayes)
        nodes = list(silver_bayes.nodes())
        train_fit = train_gold[nodes].dropna().astype(str)
        print(f"\n{internal}: gold total {len(gold)}, "
              f"train brand-disjoint {len(train_gold)}, "
              f"after dropna {len(train_fit)}")

        if len(train_fit) < 100:
            print("  too few rows, skip")
            continue

        model = BayesianNetwork(edges)
        est = BayesianEstimator(model, train_fit)
        for node in model.nodes():
            cpd = est.estimate_cpd(node, prior_type="BDeu", equivalent_sample_size=10)
            model.add_cpds(cpd)
        model.check_model()
        inference = VariableElimination(model)
        thresholds = calibrate_thresholds(model, train_fit, inference, q=Q)

        # Eval on test cascade
        cascade = pd.read_parquet(PROCESSED / f"cascade_preds_{short}_after_fix.parquet")
        cascade["code"] = cascade["code"].astype(str)
        raw = pd.read_parquet(PROCESSED / f"{internal}_raw.parquet",
                               columns=["code", "brands"])
        raw["code"] = raw["code"].astype(str)
        raw["brands"] = raw["brands"].fillna("").astype(str)
        brands = dict(zip(raw["code"], raw["brands"]))

        base_ev = {}
        for code, grp in cascade.groupby("code"):
            ev = {}
            if code in brands and brands[code]:
                ev["brand"] = brands[code]
            for _, r in grp.iterrows():
                if r["layer"] == "abstain": continue
                ev[r["attr"]] = r["predicted"]
            base_ev[code] = ev

        gold_long = pd.read_parquet(PROCESSED / "consensus_gold_v2_expanded.parquet")
        gold_long = gold_long[(gold_long.category == short) & ~gold_long.gold_is_null]
        gold_long["code"] = gold_long["code"].astype(str)
        ml = cascade[cascade.layer == "ml"].copy()
        m = ml.merge(gold_long[["code", "attr", "gold_value"]],
                     on=["code", "attr"], how="inner")
        la = llm_acc(internal)
        n_total_test_cells += len(m)

        for attr in sorted(set(m.attr.unique()) & set(model.nodes())):
            if attr == "brand": continue
            sub = m[m.attr == attr]
            n_ml = len(sub)
            if n_ml == 0: continue
            thr = thresholds.get(attr)
            if thr is None: continue
            flagged = tp = fp = wrong = 0
            for _, r in sub.iterrows():
                ev = dict(base_ev.get(r["code"], {}))
                ev.pop(attr, None)
                p = attribute_likelihood(attr, r["predicted"], ev, model, inference)
                if p is None: continue
                w = not _eq(r["predicted"], r["gold_value"])
                if w: wrong += 1
                if p < thr:
                    flagged += 1
                    if w: tp += 1
                    else: fp += 1
            if n_ml == 0: continue
            ml_acc_ = (n_ml - wrong) / n_ml
            base_rand = 1 - ml_acc_
            prec = tp / (tp + fp) if (tp + fp) else float("nan")
            lift = (prec - base_rand) if (tp + fp) else float("nan")
            l = la.get(attr, 0.70)
            dacc = (tp * l + fp * (l - 1)) / 4350
            dcost = (tp + fp) / 4350
            useful = (not np.isnan(lift)) and (lift > 0) and (dacc > 0)
            if flagged > 0:
                ann = " *" if useful else ""
                print(f"{short+'/'+attr:38s} {n_ml:5d} {flagged:7d} {tp:4d} {fp:4d} "
                      f"{('+' + f'{lift*100:.1f}') if not np.isnan(lift) else '—':>7s} "
                      f"{f'{dacc*100:+.3f}':>7s} {f'{dcost*100:+.2f}%':>7s}{ann}")
            total_results.append({"cat":short,"attr":attr,"n_ml":n_ml,
                                  "n_flag":flagged,"tp":tp,"fp":fp,
                                  "lift":lift,"dacc":dacc,"dcost":dcost,
                                  "l_acc":l,"useful":useful})

    df = pd.DataFrame(total_results)
    print("\n" + "=" * 96)
    useful_df = df[df.useful]
    print(f"\nUSEFUL PAIRS ({len(useful_df)}):")
    print(useful_df[["cat","attr","n_flag","tp","fp","lift","dacc","dcost"]].to_string(index=False))

    if len(useful_df):
        agg_tp = useful_df.tp.sum()
        agg_fp = useful_df.fp.sum()
        # Use per-attr l_acc weighted
        agg_dacc = sum(r.tp * r.l_acc + r.fp * (r.l_acc - 1) for _, r in useful_df.iterrows()) / 4350
        agg_dcost = (agg_tp + agg_fp) / 4350
        print(f"\nAggregate on full 4350 (selective inclusion of {len(useful_df)} useful pairs):")
        print(f"  TP: {agg_tp}  FP: {agg_fp}")
        print(f"  Δacc: {agg_dacc*100:+.3f} пп")
        print(f"  Δcost: {agg_dcost*100:+.2f}% LLM calls")

    # All-in baseline
    agg_tp_all = df.tp.sum(); agg_fp_all = df.fp.sum()
    agg_dacc_all = sum(r.tp * r.l_acc + r.fp * (r.l_acc - 1) for _, r in df.iterrows()) / 4350
    agg_dcost_all = (agg_tp_all + agg_fp_all) / 4350
    print(f"\nAll-pairs baseline (naive always-on across {len(df)} pairs):")
    print(f"  TP: {agg_tp_all}  FP: {agg_fp_all}")
    print(f"  Δacc: {agg_dacc_all*100:+.3f} пп")
    print(f"  Δcost: {agg_dcost_all*100:+.2f}%")

    df.to_parquet(PROCESSED / "bayes_gold_refit_holdout.parquet", index=False)
    print(f"\nSaved {PROCESSED / 'bayes_gold_refit_holdout.parquet'}")


if __name__ == "__main__":
    sys.exit(main() or 0)
