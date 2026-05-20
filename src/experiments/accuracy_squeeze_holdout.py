"""Out-of-sample валидация комбинации A+E.

Брэнды test-фолда делятся 50/50 на val / held-out test (seed 0):
  • Val: подбираем per-attr ML-threshold (E) и Bayes-q (A) по Δacc
  • Held-out: применяем выбранные параметры, меряем итоговый Δheadline

Это защищает от test-set tuning bias предыдущего эксперимента.

Запуск:
  OMP_NUM_THREADS=1 python -m src.experiments.accuracy_squeeze_holdout
"""
from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

from src.pipeline.bayes.validate import attribute_likelihood, calibrate_thresholds

PROCESSED = Path("datasets/processed")
MODELS = Path("models")
SEED = 0

CATS = [
    ("pasta", "pasta_stratified"),
    ("chocolate", "chocolate_stratified"),
    ("cheeses", "cheeses_stratified"),
]

Q_SWEEP = [0.005, 0.010, 0.020, 0.050, 0.100, 0.150]
ML_THR_SWEEP = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
GOLD_REPLICATIONS = 10

HYBRID_MODELS = {("chocolate", "chocolate_type"), ("chocolate", "contains_nuts")}

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


def llm_acc(internal):
    df = pd.read_parquet(PROCESSED / f"direct_llm_eval_{internal}_sonnet45.parquet")
    m = (df.predicted_non_null == 1) & (df.gt_non_null == 1)
    return df[m].groupby("attr").correct_when_both_present.mean().to_dict()


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
    raw["code"] = raw["code"].astype(str); raw["brands"] = raw["brands"].fillna("").astype(str)
    raw["brand"] = raw["brands"].apply(_bf); raw["brand_has_organic_marker"] = raw["brands"].apply(_bo)
    m = wide.merge(raw[["code","brand","brand_has_organic_marker"]], on="code", how="left").set_index("code")
    for c in m.columns:
        if c == "brand": m[c] = m[c].fillna("unknown").astype(str); continue
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


def fit_hybrid_bayes(short, internal):
    with open(MODELS / f"{internal}_bayesian.pkl", "rb") as f:
        silver_bayes = pickle.load(f)
    edges = list(silver_bayes.edges()); nodes = list(silver_bayes.nodes())
    test_codes = get_test_codes(short)
    gold_wide = build_gold_wide_with_brand(short, internal)
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


def split_test_brands(short, internal, seed=SEED):
    """Делим test-codes на val/test по брендам, чтобы внутри test-folda бренды не пересекались."""
    test_codes = get_test_codes(short)
    raw = pd.read_parquet(PROCESSED / f"{internal}_raw.parquet", columns=["code","brands"])
    raw["code"] = raw["code"].astype(str); raw["brands"] = raw["brands"].fillna("").astype(str)
    raw["brand"] = raw["brands"].apply(_bf)
    raw_t = raw[raw["code"].isin(test_codes)].copy()
    brands = sorted(raw_t["brand"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(brands)
    half = len(brands) // 2
    val_brands = set(brands[:half])
    held_brands = set(brands[half:])
    val_codes = set(raw_t[raw_t["brand"].isin(val_brands)].code)
    held_codes = set(raw_t[raw_t["brand"].isin(held_brands)].code)
    return val_codes, held_codes


def predict_ml(short, internal, codes_subset):
    """ML-предсказания для подмножества кодов."""
    silver = pd.read_parquet(PROCESSED / f"{internal}_silver_standard.parquet")
    silver["code"] = silver["code"].astype(str); silver["_pos"] = np.arange(len(silver))
    emb_all = np.load(PROCESSED / f"{internal}_embeddings.npy")
    sub = silver[silver["code"].isin(codes_subset)].copy().reset_index(drop=True)
    emb_sub = emb_all[sub["_pos"].values]
    codes_sub = sub["code"].tolist()

    X_hybrid = None
    if short == "chocolate":
        try:
            with open(MODELS / f"{internal}_hybrid_tfidf.pkl", "rb") as f:
                hybrid_vec = pickle.load(f)
            texts = []
            for col in ["product_name", "brands", "ingredients_text", "quantity"]:
                texts.append(sub.get(col, pd.Series([""] * len(sub))).astype(str).fillna(""))
            text_concat = texts[0].str.cat(texts[1:], sep=" ", na_rep="").tolist()
            X_tfidf = hybrid_vec.transform(text_concat)
            X_hybrid = hstack([csr_matrix(emb_sub), X_tfidf]).tocsr()
        except FileNotFoundError:
            pass

    # For each attr, predict
    cascade_old = pd.read_parquet(PROCESSED / f"cascade_preds_{short}_after_fix.parquet")
    attrs = cascade_old["attr"].unique()
    pred_table = {attr: {} for attr in attrs}  # code -> (pred, conf)
    for attr in attrs:
        use_hybrid = (short, attr) in HYBRID_MODELS
        if use_hybrid and X_hybrid is not None:
            clf_path = MODELS / f"{internal}_hybrid_{attr}_xgb.pkl"
            le_path = MODELS / f"{internal}_hybrid_{attr}_le.pkl"
            X_in = X_hybrid
        else:
            clf_path = MODELS / f"{internal}_{attr}_xgb_hybrid.pkl"
            le_path = MODELS / f"{internal}_{attr}_le_hybrid.pkl"
            X_in = emb_sub
        if not clf_path.exists(): continue
        with open(clf_path, "rb") as f: clf = pickle.load(f)
        try:
            with open(le_path, "rb") as f: le = pickle.load(f)
        except FileNotFoundError: le = None
        proba = clf.predict_proba(X_in)
        max_conf = proba.max(axis=1); max_idx = proba.argmax(axis=1)
        if le is not None:
            preds = [str(le.inverse_transform([i])[0]) for i in max_idx]
        else:
            preds = [str(bool(i)) for i in max_idx]
        for code, pred, conf in zip(codes_sub, preds, max_conf):
            pred_table[attr][code] = (pred, float(conf))
    return pred_table


def evaluate_config(short, internal, codes_subset, ml_thr_per_attr, bayes_q_per_attr,
                    bayes_model, base_ev_lookup, gold_lookup, llm_a):
    """Для каждого attr пройти cells, применить конфиг E+A, вернуть per-attr summary."""
    inference = VariableElimination(bayes_model)
    # Per-q calibrated thresholds (only those we need)
    needed_qs = set(bayes_q_per_attr.values())
    gold_fit = base_ev_lookup["__gold_fit__"]
    thr_per_q = {q: calibrate_thresholds(bayes_model, gold_fit, inference, q=q) for q in needed_qs}

    cascade_old = pd.read_parquet(PROCESSED / f"cascade_preds_{short}_after_fix.parquet")
    cascade_old["code"] = cascade_old["code"].astype(str)
    cascade_old = cascade_old[cascade_old.code.isin(codes_subset)].copy()
    cascade_old["pred_norm"] = cascade_old.predicted.astype(str).str.lower()

    pred_table = predict_ml(short, internal, codes_subset)

    rows = []
    for attr in cascade_old.attr.unique():
        sub = cascade_old[cascade_old.attr == attr].copy()
        sub = sub.merge(gold_lookup[gold_lookup.attr == attr][["code", "gold_norm"]],
                        on="code", how="inner")
        if not len(sub): continue
        sub["cascade_correct"] = (sub.pred_norm == sub.gold_norm) & (sub.layer != "abstain")
        l = llm_a.get(attr, 0.70)
        n_total = len(sub)
        regex_cells = sub[sub.layer == "regex"]
        n_regex_correct = int(regex_cells.cascade_correct.sum())
        ml_pool = sub[sub.layer.isin(["ml", "abstain"])]

        # CURRENT cascade
        n_abs_old = int((sub.layer == "abstain").sum())
        e2e_current = (sub.cascade_correct.sum() + n_abs_old * l) / n_total

        # NEW cascade with E + A
        ml_thr = ml_thr_per_attr.get(attr)  # None = use old layer
        bayes_q = bayes_q_per_attr.get(attr)
        bayes_thr = thr_per_q.get(bayes_q, {}).get(attr) if bayes_q else None

        n_ml_correct = 0
        n_abstain = 0
        n_bayes_flag = n_bayes_tp = n_bayes_fp = 0
        for _, row in ml_pool.iterrows():
            code = str(row.code); gold_n = row.gold_norm
            if code not in pred_table.get(attr, {}): continue
            pred, conf = pred_table[attr][code]
            ml_correct = (pred.lower() == gold_n)

            if ml_thr is not None:
                ml_keeps = (conf >= ml_thr)
            else:
                ml_keeps = (row.layer == "ml")
            if not ml_keeps:
                n_abstain += 1
                continue

            if bayes_thr is not None and attr in bayes_model.nodes():
                ev = dict(base_ev_lookup.get(code, {})); ev.pop(attr, None)
                p = attribute_likelihood(attr, pred, ev, bayes_model, inference)
                if p is not None and p < bayes_thr:
                    n_bayes_flag += 1
                    if not ml_correct: n_bayes_tp += 1
                    else: n_bayes_fp += 1
                    n_abstain += 1
                    continue

            if ml_correct: n_ml_correct += 1

        e2e_combined = (n_regex_correct + n_ml_correct + n_abstain * l) / n_total
        rows.append({
            "cat": short, "attr": attr, "n": n_total,
            "ml_thr": ml_thr, "bayes_q": bayes_q,
            "e2e_current": e2e_current, "e2e_combined": e2e_combined,
            "delta_pp": (e2e_combined - e2e_current) * 100,
            "abs_gain_cells": (e2e_combined - e2e_current) * n_total,
            "n_abstain_new": n_abstain, "n_bayes_flag": n_bayes_flag,
            "n_bayes_tp": n_bayes_tp, "n_bayes_fp": n_bayes_fp,
        })
    return rows


def main():
    # Подготовка inputs
    bayes_models = {}
    gold_fits = {}
    base_ev_per_cat = {}
    gold_per_cat = {}
    llm_a_per_cat = {}
    for short, internal in CATS:
        bayes, gold_fit = fit_hybrid_bayes(short, internal)
        bayes_models[short] = bayes
        gold_fits[short] = gold_fit
        cascade_old = pd.read_parquet(PROCESSED / f"cascade_preds_{short}_after_fix.parquet")
        cascade_old["code"] = cascade_old["code"].astype(str)
        raw = pd.read_parquet(PROCESSED / f"{internal}_raw.parquet", columns=["code","brands"])
        raw["code"] = raw["code"].astype(str); raw["brands"] = raw["brands"].fillna("").astype(str)
        brands_lookup = dict(zip(raw["code"], raw["brands"]))
        base_ev = {}
        for code, grp in cascade_old.groupby("code"):
            ev = {}
            if code in brands_lookup and brands_lookup[code]: ev["brand"] = brands_lookup[code]
            for _, r in grp.iterrows():
                if r["layer"] == "abstain": continue
                ev[r["attr"]] = r["predicted"]
            base_ev[code] = ev
        base_ev["__gold_fit__"] = gold_fit
        base_ev_per_cat[short] = base_ev
        gold = pd.read_parquet(PROCESSED / "consensus_gold_v2_expanded.parquet")
        gold = gold[(gold.category == short) & ~gold.gold_is_null].copy()
        gold["code"] = gold["code"].astype(str)
        gold["gold_norm"] = gold.gold_value.astype(str).str.lower()
        gold_per_cat[short] = gold[["code","attr","gold_norm"]]
        llm_a_per_cat[short] = llm_acc(internal)

    # Splits
    splits = {}
    for short, internal in CATS:
        val, held = split_test_brands(short, internal)
        splits[short] = (val, held)
        print(f"  {short}: val={len(val)} codes, held={len(held)} codes")

    # ---- Phase 1: подобрать параметры на val ----
    print("\n=== PHASE 1: PICKING PARAMETERS ON VAL ===")
    chosen_E = {}  # (cat, attr) -> ml_thr
    chosen_A = {}  # (cat, attr) -> bayes_q
    for short, internal in CATS:
        val_codes, _ = splits[short]
        bayes = bayes_models[short]
        gold_fit = gold_fits[short]
        inference = VariableElimination(bayes)
        thr_per_q = {q: calibrate_thresholds(bayes, gold_fit, inference, q=q) for q in Q_SWEEP}

        pred_table = predict_ml(short, internal, val_codes)
        cascade_old = pd.read_parquet(PROCESSED / f"cascade_preds_{short}_after_fix.parquet")
        cascade_old["code"] = cascade_old["code"].astype(str)
        cascade_old = cascade_old[cascade_old.code.isin(val_codes)].copy()
        cascade_old["pred_norm"] = cascade_old.predicted.astype(str).str.lower()
        la = llm_a_per_cat[short]
        gold = gold_per_cat[short]
        base_ev = base_ev_per_cat[short]

        for attr in cascade_old.attr.unique():
            sub = cascade_old[cascade_old.attr == attr].merge(
                gold[gold.attr == attr][["code","gold_norm"]], on="code", how="inner")
            if not len(sub): continue
            sub["cascade_correct"] = (sub.pred_norm == sub.gold_norm) & (sub.layer != "abstain")
            l = la.get(attr, 0.70)
            n_total = len(sub)
            ml_pool = sub[sub.layer.isin(["ml","abstain"])]
            regex_cells = sub[sub.layer == "regex"]
            n_regex_correct = int(regex_cells.cascade_correct.sum())

            # Current e2e on val
            n_abs_old = int((sub.layer == "abstain").sum())
            e2e_cur = (sub.cascade_correct.sum() + n_abs_old * l) / n_total

            # Try each (ml_thr, bayes_q) combo (включая None)
            best = (0.0, None, None)  # (gain, ml_thr, bayes_q)
            ml_thr_options = [None] + ML_THR_SWEEP
            bayes_q_options = [None] + Q_SWEEP
            for ml_thr in ml_thr_options:
                for bayes_q in bayes_q_options:
                    bayes_thr = thr_per_q.get(bayes_q, {}).get(attr) if bayes_q else None
                    n_ml_correct = 0; n_abstain = 0
                    for _, row in ml_pool.iterrows():
                        code = str(row.code); gold_n = row.gold_norm
                        if code not in pred_table.get(attr, {}): continue
                        pred, conf = pred_table[attr][code]
                        ml_correct = (pred.lower() == gold_n)
                        if ml_thr is not None:
                            ml_keeps = (conf >= ml_thr)
                        else:
                            ml_keeps = (row.layer == "ml")
                        if not ml_keeps:
                            n_abstain += 1; continue
                        if bayes_thr is not None and attr in bayes.nodes():
                            ev = dict(base_ev.get(code, {})); ev.pop(attr, None)
                            p = attribute_likelihood(attr, pred, ev, bayes, inference)
                            if p is not None and p < bayes_thr:
                                n_abstain += 1; continue
                        if ml_correct: n_ml_correct += 1
                    e2e_new = (n_regex_correct + n_ml_correct + n_abstain * l) / n_total
                    gain = (e2e_new - e2e_cur) * n_total
                    if gain > best[0]:
                        best = (gain, ml_thr, bayes_q)
            if best[1] is not None: chosen_E[(short, attr)] = best[1]
            if best[2] is not None: chosen_A[(short, attr)] = best[2]
    print(f"  Chosen E: {len(chosen_E)} attrs; Chosen A: {len(chosen_A)} attrs")

    # ---- Phase 2: apply chosen config on HELD-OUT test ----
    print("\n=== PHASE 2: HELD-OUT EVAL ===")
    all_rows = []
    for short, internal in CATS:
        _, held_codes = splits[short]
        ml_thr_map = {attr: thr for (cat, attr), thr in chosen_E.items() if cat == short}
        bayes_q_map = {attr: q for (cat, attr), q in chosen_A.items() if cat == short}
        rows = evaluate_config(
            short, internal, held_codes, ml_thr_map, bayes_q_map,
            bayes_models[short], base_ev_per_cat[short],
            gold_per_cat[short], llm_a_per_cat[short]
        )
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)

    pd.set_option("display.width", 240); pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:+.3f}")
    print("\nPer-attr на HELD-OUT (defended numbers):")
    df_sorted = df.sort_values("delta_pp", ascending=False)
    print(df_sorted[["cat","attr","n","ml_thr","bayes_q","n_bayes_tp","n_bayes_fp",
                     "e2e_current","e2e_combined","delta_pp","abs_gain_cells"]]
          .to_string(index=False))

    n_total = df.n.sum()
    n_gain = df.abs_gain_cells.sum()
    print(f"\nDEFENDED HEADLINE GAIN:")
    print(f"  Held-out cells: {n_total}")
    print(f"  Δheadline: {n_gain/n_total*100:+.3f} пп (на 4350-shaped тесте: ≈{n_gain/4350*100*4350/n_total:.3f} пп)")
    print(f"  Bayes total: TP={df.n_bayes_tp.sum()}, FP={df.n_bayes_fp.sum()}")

    df.to_parquet(PROCESSED / "accuracy_squeeze_holdout.parquet", index=False)
    print(f"\nSaved {PROCESSED / 'accuracy_squeeze_holdout.parquet'}")

    # Also save the chosen config
    config = {"E_ml_thresholds": {f"{c}/{a}": v for (c,a), v in chosen_E.items()},
              "A_bayes_q":       {f"{c}/{a}": v for (c,a), v in chosen_A.items()}}
    import json
    with open(PROCESSED / "accuracy_squeeze_chosen_config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Saved chosen config to accuracy_squeeze_chosen_config.json")


if __name__ == "__main__":
    sys.exit(main() or 0)
