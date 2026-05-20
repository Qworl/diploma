"""Hybrid-обучение Bayes-валидатора: silver (объём) + gold (чистота).

Schema:
  • DAG (структура) — от silver-bayes (учен на 15K, охватывает все бренды).
  • CPD-таблицы — fit на CONCAT(silver-train, gold-train × N) где N — параметр.
  • Калибровка порогов — на gold-train (q=0.02).
  • Eval — на cascade-after_fix предсказаниях, brand-disjoint test.

Гипотеза: gold с replicated weight ≈ silver-volume даёт сети «чистый» сигнал
там, где silver шумный, не теряя широкого покрытия брендов.

Sweep по N ∈ {1, 5, 10, 20, 40} → отбираем оптимум по агрегированному Δacc
с selective inclusion полезных пар.

Запуск:
  OMP_NUM_THREADS=1 python -m src.pipeline.bayes.refit_hybrid_holdout
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

from src.pipeline.bayes.validate import attribute_likelihood, calibrate_thresholds

PROCESSED = Path("datasets/processed")
MODELS = Path("models")
Q = 0.02
GOLD_REPLICATIONS = [1, 5, 10, 20, 40]

CATS = [
    ("pasta", "pasta_stratified"),
    ("chocolate", "chocolate_stratified"),
    ("cheeses", "cheeses_stratified"),
]

_ORG = re.compile(r"\b(bio|organic|organique|eco|ecol[oó]gico|ekol|öko)\b", re.I)


def _bf(s): return (str(s).split(",")[0].strip().lower() or "unknown") if s else "unknown"
def _bo(s): return "True" if _ORG.search(str(s or "")) else "False"


def _norm(v):
    if v is None: return None
    if isinstance(v, float) and np.isnan(v): return None
    if isinstance(v, str) and v.strip().lower() in {"", "none", "null", "nan"}: return None
    return v


def _eq(a, b):
    a, b = _norm(a), _norm(b)
    if a is None or b is None: return False
    return str(a).strip().lower() == str(b).strip().lower()


def get_test_codes(short):
    cp = pd.read_parquet(PROCESSED / f"cascade_preds_{short}_after_fix.parquet",
                          columns=["code"])
    return set(cp["code"].astype(str).unique())


def build_gold_wide_with_brand(short, internal):
    g = pd.read_parquet(PROCESSED / "consensus_gold_v2_expanded.parquet")
    g = g[(g.category == short) & ~g.gold_is_null].copy()
    g["code"] = g["code"].astype(str)
    wide = g.pivot_table(index="code", columns="attr", values="gold_value", aggfunc="first")
    raw = pd.read_parquet(PROCESSED / f"{internal}_raw.parquet", columns=["code", "brands"])
    raw["code"] = raw["code"].astype(str)
    raw["brands"] = raw["brands"].fillna("").astype(str)
    raw["brand"] = raw["brands"].apply(_bf)
    raw["brand_has_organic_marker"] = raw["brands"].apply(_bo)
    m = wide.merge(raw[["code", "brand", "brand_has_organic_marker"]],
                   on="code", how="left").set_index("code")
    for c in m.columns:
        if c == "brand":
            m[c] = m[c].fillna("unknown").astype(str)
            continue
        m[c] = m[c].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
    return m


def silver_train_data(internal, nodes, silver_bayes):
    """Загрузить silver_standard и подготовить под Bayes — те же ноды, что есть в сети."""
    silver_df = pd.read_parquet(PROCESSED / f"{internal}_silver_standard.parquet")
    # Используем тот же препроцессинг, что в pipeline.bayes.train — это извлекается
    # из существующего bayes-моделя через данные, на которых он учен. Проще
    # сконструировать колонки самим:
    sub = pd.DataFrame()
    sub["brand_raw"] = silver_df.get("brands", pd.Series([""] * len(silver_df)))
    sub["brand"] = sub["brand_raw"].fillna("unknown").apply(_bf)
    # Свернём brand к silver-states (которые в trained CPD)
    cpd_states = set(str(s) for s in silver_bayes.get_cpds("brand").state_names["brand"]) \
        if "brand" in silver_bayes.nodes() else set()
    sub["brand"] = sub["brand"].apply(
        lambda b: b if b in cpd_states else ("other" if "other" in cpd_states else b))
    sub["brand_has_organic_marker"] = sub["brand_raw"].apply(_bo)
    # Скопируем остальные ноды напрямую из silver_df, если они есть.
    for node in nodes:
        if node in ("brand", "brand_has_organic_marker"):
            continue
        if node not in silver_df.columns:
            sub[node] = None
            continue
        sub[node] = silver_df[node]
    out = sub[nodes].copy()
    # Привести типы к str (Bayes-движок требует категориальные значения как str)
    for c in out.columns:
        out[c] = out[c].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
    return out


def silver_brand_disjoint_train(internal, nodes, silver_bayes, test_codes):
    df = silver_train_data(internal, nodes, silver_bayes)
    # Привязка по code: загрузим code-список из silver_standard, потом фильтруем
    silver_raw = pd.read_parquet(PROCESSED / f"{internal}_silver_standard.parquet",
                                  columns=["code"])
    silver_raw["code"] = silver_raw["code"].astype(str)
    keep_mask = ~silver_raw["code"].isin(test_codes)
    df = df.loc[keep_mask].copy()
    return df.dropna(subset=nodes)


def llm_acc(internal):
    df = pd.read_parquet(PROCESSED / f"direct_llm_eval_{internal}_sonnet45.parquet")
    m = (df.predicted_non_null == 1) & (df.gt_non_null == 1)
    return df[m].groupby("attr").correct_when_both_present.mean().to_dict()


def evaluate_holdout(model, inference, thresholds, short, internal, la):
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

    gold = pd.read_parquet(PROCESSED / "consensus_gold_v2_expanded.parquet")
    gold = gold[(gold.category == short) & ~gold.gold_is_null]
    gold["code"] = gold["code"].astype(str)
    ml = cascade[cascade.layer == "ml"].copy()
    m = ml.merge(gold[["code", "attr", "gold_value"]], on=["code", "attr"], how="inner")

    out_rows = []
    for attr in sorted(set(m.attr.unique()) & set(model.nodes())):
        if attr == "brand": continue
        sub = m[m.attr == attr]
        n_ml = len(sub)
        if n_ml == 0: continue
        thr = thresholds.get(attr)
        if thr is None: continue
        flag = tp = fp = wrong = 0
        for _, r in sub.iterrows():
            ev = dict(base_ev.get(r["code"], {}))
            ev.pop(attr, None)
            p = attribute_likelihood(attr, r["predicted"], ev, model, inference)
            if p is None: continue
            w = not _eq(r["predicted"], r["gold_value"])
            if w: wrong += 1
            if p < thr:
                flag += 1
                if w: tp += 1
                else: fp += 1
        ml_acc_ = (n_ml - wrong) / n_ml
        base = 1 - ml_acc_
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        lift = (prec - base) if (tp + fp) else float("nan")
        l = la.get(attr, 0.70)
        dacc = (tp * l + fp * (l - 1)) / 4350
        out_rows.append({
            "cat": short, "attr": attr, "n_ml": n_ml,
            "n_flag": flag, "tp": tp, "fp": fp,
            "lift": lift, "dacc": dacc, "l_acc": l,
            "useful": (not np.isnan(lift)) and (lift > 0) and (dacc > 0),
        })
    return out_rows


def run_one(N, all_results):
    print(f"\n{'#'*90}\n#  GOLD REPLICATIONS = {N}\n{'#'*90}")
    for short, internal in CATS:
        with open(MODELS / f"{internal}_bayesian.pkl", "rb") as f:
            silver_bayes = pickle.load(f)
        edges = list(silver_bayes.edges())
        nodes = list(silver_bayes.nodes())

        test_codes = get_test_codes(short)
        gold_wide = build_gold_wide_with_brand(short, internal)
        train_gold = gold_wide[~gold_wide.index.isin(test_codes)].copy()
        # Свернуть brand к silver-states
        if "brand" in silver_bayes.nodes():
            states = set(str(s) for s in silver_bayes.get_cpds("brand").state_names["brand"])
            train_gold["brand"] = train_gold["brand"].apply(
                lambda b: b if b in states else ("other" if "other" in states else b))
        gold_fit = train_gold[nodes].dropna().astype(str)

        silver_fit = silver_brand_disjoint_train(internal, nodes, silver_bayes, test_codes)

        # Replicate gold × N
        gold_rep = pd.concat([gold_fit] * N, ignore_index=True) if N > 0 else pd.DataFrame()
        hybrid = pd.concat([silver_fit, gold_rep], ignore_index=True)
        print(f"\n  {internal}: silver={len(silver_fit):,}, gold×{N}={len(gold_rep):,}, "
              f"hybrid={len(hybrid):,}")

        model = BayesianNetwork(edges)
        est = BayesianEstimator(model, hybrid)
        for node in model.nodes():
            cpd = est.estimate_cpd(node, prior_type="BDeu", equivalent_sample_size=10)
            model.add_cpds(cpd)
        model.check_model()

        inference = VariableElimination(model)
        # Калибруем пороги на gold-train, не на hybrid — мы хотим, чтобы границы
        # отражали именно gold-распределение, а не среднее silver+gold.
        thresholds = calibrate_thresholds(model, gold_fit, inference, q=Q)

        la = llm_acc(internal)
        rows = evaluate_holdout(model, inference, thresholds, short, internal, la)
        for r in rows:
            r["N"] = N
            all_results.append(r)


def main():
    all_rows = []
    for N in GOLD_REPLICATIONS:
        run_one(N, all_rows)

    df = pd.DataFrame(all_rows)

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 220)

    # Aggregated summary per N
    print("\n" + "=" * 96)
    print("SUMMARY across gold replication factor N")
    print("=" * 96)
    rows = []
    for N, sub in df.groupby("N"):
        u = sub[sub.useful]
        tp_u = u.tp.sum()
        fp_u = u.fp.sum()
        dacc_u = sum(r.tp * r.l_acc + r.fp * (r.l_acc - 1) for _, r in u.iterrows()) / 4350
        dcost_u = (tp_u + fp_u) / 4350
        tp_a = sub.tp.sum()
        fp_a = sub.fp.sum()
        dacc_a = sum(r.tp * r.l_acc + r.fp * (r.l_acc - 1) for _, r in sub.iterrows()) / 4350
        rows.append({
            "N": N,
            "useful_pairs": len(u),
            "TP_useful": tp_u, "FP_useful": fp_u,
            "Δacc_useful_pp": dacc_u * 100,
            "Δcost_useful_pct": dcost_u * 100,
            "TP_all": tp_a, "FP_all": fp_a,
            "Δacc_all_pp": dacc_a * 100,
        })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    # Detail of useful pairs per N
    print("\nUseful pairs per N (lift>0 AND Δacc>0):")
    for N in GOLD_REPLICATIONS:
        sub = df[(df.N == N) & df.useful][["cat", "attr", "tp", "fp", "lift", "dacc"]]
        if not len(sub): continue
        print(f"\n  N={N}:")
        print(sub.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    df.to_parquet(PROCESSED / "bayes_hybrid_refit_sweep.parquet", index=False)
    print(f"\nSaved {PROCESSED / 'bayes_hybrid_refit_sweep.parquet'}")


if __name__ == "__main__":
    sys.exit(main() or 0)
