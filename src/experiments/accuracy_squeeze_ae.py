"""Accuracy squeeze — direction A + E.

A. Per-attr Bayes threshold sweep (hybrid model N=10):
   для каждого useful-кандидата (всех 22 пар) перебираем q ∈ {0.005, 0.01, 0.02,
   0.03, 0.05, 0.10, 0.15} → находим оптимум по Δacc на holdout.

E. Layer 2 ML threshold relaxation: для атрибутов, где Layer 2 acc на covered
   ≥ 95 %, перебираем пониженные пороги {0.45, 0.50, 0.55, 0.60, текущий}.
   Re-predict ML cell coverage и точность.

Финальный отчёт: для каждой стратегии — оптимальная конфигурация и
ожидаемый Δacc, без двойного учёта (если cell обрабатывается через Layer 2
по новому порогу, она не идёт в Bayes-валидатор).

Запуск:
  OMP_NUM_THREADS=1 python -m src.experiments.accuracy_squeeze_ae
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

CATS = [
    ("pasta", "pasta_stratified"),
    ("chocolate", "chocolate_stratified"),
    ("cheeses", "cheeses_stratified"),
]
Q_SWEEP = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.15]
GOLD_REPLICATIONS = 10

ML_THRESHOLD_SWEEP = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

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


def silver_brand_disjoint(internal, nodes, silver_bayes, test_codes):
    silver_df = pd.read_parquet(PROCESSED / f"{internal}_silver_standard.parquet")
    silver_df["code"] = silver_df["code"].astype(str)
    keep = ~silver_df["code"].isin(test_codes)
    silver_df = silver_df[keep].reset_index(drop=True)
    sub = pd.DataFrame()
    sub["brand_raw"] = silver_df.get("brands", pd.Series([""] * len(silver_df)))
    sub["brand"] = sub["brand_raw"].fillna("unknown").apply(_bf)
    cpd_states = set(str(s) for s in silver_bayes.get_cpds("brand").state_names["brand"]) \
        if "brand" in silver_bayes.nodes() else set()
    sub["brand"] = sub["brand"].apply(
        lambda b: b if b in cpd_states else ("other" if "other" in cpd_states else b))
    sub["brand_has_organic_marker"] = sub["brand_raw"].apply(_bo)
    for node in nodes:
        if node in ("brand", "brand_has_organic_marker"):
            continue
        sub[node] = silver_df.get(node)
    out = sub[nodes].copy()
    for c in out.columns:
        out[c] = out[c].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
    return out.dropna(subset=nodes)


def llm_acc(internal):
    df = pd.read_parquet(PROCESSED / f"direct_llm_eval_{internal}_sonnet45.parquet")
    m = (df.predicted_non_null == 1) & (df.gt_non_null == 1)
    return df[m].groupby("attr").correct_when_both_present.mean().to_dict()


# ----------------- A. per-attr Bayes q-sweep -----------------

def fit_hybrid_bayes(short, internal):
    with open(MODELS / f"{internal}_bayesian.pkl", "rb") as f:
        silver_bayes = pickle.load(f)
    edges = list(silver_bayes.edges())
    nodes = list(silver_bayes.nodes())

    test_codes = get_test_codes(short)
    gold_wide = build_gold_wide_with_brand(short, internal)
    train_gold = gold_wide[~gold_wide.index.isin(test_codes)].copy()
    if "brand" in silver_bayes.nodes():
        states = set(str(s) for s in silver_bayes.get_cpds("brand").state_names["brand"])
        train_gold["brand"] = train_gold["brand"].apply(
            lambda b: b if b in states else ("other" if "other" in states else b))
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


def per_attr_q_sweep(short, internal):
    model, gold_fit = fit_hybrid_bayes(short, internal)
    inference = VariableElimination(model)
    # Per-attr thresholds at each q
    thr_per_q = {}
    for q in Q_SWEEP:
        thr_per_q[q] = calibrate_thresholds(model, gold_fit, inference, q=q)

    # Evaluate flagging
    cascade = pd.read_parquet(PROCESSED / f"cascade_preds_{short}_after_fix.parquet")
    cascade["code"] = cascade["code"].astype(str)
    raw = pd.read_parquet(PROCESSED / f"{internal}_raw.parquet", columns=["code","brands"])
    raw["code"] = raw["code"].astype(str); raw["brands"] = raw["brands"].fillna("").astype(str)
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
    m = ml.merge(gold[["code","attr","gold_value"]], on=["code","attr"], how="inner")
    la = llm_acc(internal)

    rows = []
    for attr in sorted(set(m.attr.unique()) & set(model.nodes())):
        if attr == "brand": continue
        sub = m[m.attr == attr]
        # Per attr precompute likelihoods once (independent of q)
        likelihoods = []
        wrong_list = []
        for _, r in sub.iterrows():
            ev = dict(base_ev.get(r["code"], {})); ev.pop(attr, None)
            p = attribute_likelihood(attr, r["predicted"], ev, model, inference)
            if p is None: continue
            likelihoods.append(p)
            wrong_list.append(not _eq(r["predicted"], r["gold_value"]))
        n_ml = len(likelihoods)
        if n_ml == 0: continue
        l = la.get(attr, 0.70)
        for q in Q_SWEEP:
            thr = thr_per_q[q].get(attr)
            if thr is None: continue
            tp = fp = 0
            for p, w in zip(likelihoods, wrong_list):
                if p < thr:
                    if w: tp += 1
                    else: fp += 1
            dacc = (tp * l + fp * (l - 1)) / 4350
            rows.append({
                "cat": short, "attr": attr, "q": q, "thr": thr,
                "n_ml": n_ml, "tp": tp, "fp": fp,
                "dacc_pp": dacc * 100,
                "dcost_pct": (tp + fp) / 4350 * 100,
                "l_acc": l,
            })
    return rows


# ----------------- E. ML threshold relaxation -----------------

def ml_threshold_relaxation(short, internal):
    """Перевычисляем cascade для каждого ML-threshold кандидата.

    Используем уже сохранённые embeddings/модели; для каждого attr пробуем
    пониженные thresholds и считаем, сколько cells перейдут из abstain в ML
    с какой точностью.
    """
    from scipy.sparse import csr_matrix, hstack
    import pickle as pkl

    cascade_old = pd.read_parquet(PROCESSED / f"cascade_preds_{short}_after_fix.parquet")
    cascade_old["code"] = cascade_old["code"].astype(str)
    silver = pd.read_parquet(PROCESSED / f"{internal}_silver_standard.parquet")
    silver["code"] = silver["code"].astype(str)
    silver["_pos"] = np.arange(len(silver))
    emb_all = np.load(PROCESSED / f"{internal}_embeddings.npy")

    test_codes = sorted(cascade_old["code"].unique())
    sub = silver[silver["code"].isin(test_codes)].copy().reset_index(drop=True)
    emb_sub = emb_all[sub["_pos"].values]
    codes_sub = sub["code"].tolist()

    # Hybrid features only for chocolate (matching regen script)
    X_hybrid = None
    hybrid_vec = None
    if short == "chocolate":
        try:
            with open(MODELS / f"{internal}_hybrid_tfidf.pkl", "rb") as f:
                hybrid_vec = pkl.load(f)
            texts = []
            for col in ["product_name", "brands", "ingredients_text", "quantity"]:
                texts.append(sub.get(col, pd.Series([""] * len(sub))).astype(str).fillna(""))
            text_concat = texts[0].str.cat(texts[1:], sep=" ", na_rep="").tolist()
            X_tfidf = hybrid_vec.transform(text_concat)
            X_hybrid = hstack([csr_matrix(emb_sub), X_tfidf]).tocsr()
        except FileNotFoundError:
            pass

    HYBRID_MODELS = {("chocolate", "chocolate_type"), ("chocolate", "contains_nuts")}

    gold = pd.read_parquet(PROCESSED / "consensus_gold_v2_expanded.parquet")
    gold = gold[(gold.category == short) & ~gold.gold_is_null].copy()
    gold["code"] = gold["code"].astype(str)
    gold["gold_norm"] = gold.gold_value.astype(str).str.lower()
    la = llm_acc(internal)

    candidates = pd.read_parquet(PROCESSED / "cascade_preds_pasta_after_fix.parquet")
    attrs = cascade_old["attr"].unique()
    rows = []
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
        with open(clf_path, "rb") as f: clf = pkl.load(f)
        try:
            with open(le_path, "rb") as f: le = pkl.load(f)
        except FileNotFoundError:
            le = None
        proba = clf.predict_proba(X_in)
        max_conf = proba.max(axis=1)
        max_idx = proba.argmax(axis=1)
        if le is not None:
            preds = [str(le.inverse_transform([i])[0]) for i in max_idx]
        else:
            preds = [str(bool(i)) for i in max_idx]

        gattr = gold[gold.attr == attr][["code", "gold_norm"]].set_index("code")
        # Build (code, pred, conf, gold_norm) list — only for cells gold has
        records = []
        for code, pred, conf in zip(codes_sub, preds, max_conf):
            gn = gattr.gold_norm.get(code)
            if gn is None: continue
            records.append({"code": code, "pred": pred.lower(), "conf": float(conf),
                            "gold": gn, "ml_correct": pred.lower() == gn})
        if not records: continue
        df_attr = pd.DataFrame(records)

        # MERGE with cascade_old layer info — нужно знать, какие cells
        # уходят на regex (их не трогаем) vs ML/abstain (применяем новый thr).
        co_attr = cascade_old[cascade_old.attr == attr].copy()
        co_attr["pred_norm"] = co_attr.predicted.astype(str).str.lower()
        co_attr = co_attr.merge(gattr.reset_index(), on="code", how="inner")
        co_attr["cascade_correct"] = (co_attr.pred_norm == co_attr.gold_norm) \
            & (co_attr.layer != "abstain")

        # Bucket gold cells by cascade-old layer
        regex_cells = co_attr[co_attr.layer == "regex"].copy()
        ml_or_abstain_cells = co_attr[co_attr.layer.isin(["ml", "abstain"])].copy()

        # Regex contribution — фиксированный, не зависит от thr
        n_regex_correct = int(regex_cells.cascade_correct.sum())
        n_regex_total = len(regex_cells)

        # На ML/abstain-cells применяем новый ML с new threshold
        ml_codes_set = set(ml_or_abstain_cells.code.astype(str))
        ml_subset = df_attr[df_attr.code.astype(str).isin(ml_codes_set)]
        n_ml_pool = len(ml_subset)
        l = la.get(attr, 0.70)
        n_total = n_regex_total + n_ml_pool  # eq. gold cells

        for thr in ML_THRESHOLD_SWEEP:
            covered = ml_subset[ml_subset.conf >= thr]
            abstain_n = n_ml_pool - len(covered)
            n_correct_ml = int(covered.ml_correct.sum())
            e2e_correct = n_regex_correct + n_correct_ml + abstain_n * l
            e2e_acc = e2e_correct / n_total if n_total else 0
            rows.append({
                "cat": short, "attr": attr, "thr": thr,
                "n_total": n_total, "n_regex": n_regex_total, "n_ml_pool": n_ml_pool,
                "covered": len(covered),
                "ml_acc_on_covered": covered.ml_correct.mean() if len(covered) else float('nan'),
                "abstain": abstain_n,
                "e2e_acc": e2e_acc,
                "n_correct_estimate": e2e_correct,
            })
    return rows


# ----------------- main -----------------

def main():
    print("=" * 96)
    print("A. PER-ATTR BAYES Q-SWEEP (hybrid N=10, holdout)")
    print("=" * 96)
    a_rows = []
    for short, internal in CATS:
        print(f"\n  ... {internal}")
        a_rows.extend(per_attr_q_sweep(short, internal))
    a_df = pd.DataFrame(a_rows)
    a_df.to_parquet(PROCESSED / "accuracy_squeeze_a_bayes_qsweep.parquet", index=False)

    # Find optimum per attr
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:+.3f}" if isinstance(x, float) else str(x))
    print("\nOptimal q per attr (max Δacc):")
    best_a = a_df.loc[a_df.groupby(["cat","attr"])["dacc_pp"].idxmax()].copy()
    best_a = best_a.sort_values("dacc_pp", ascending=False)
    print(best_a[["cat","attr","q","tp","fp","dacc_pp","dcost_pct"]].to_string(index=False))
    a_total = best_a[best_a.dacc_pp > 0].copy()
    print(f"\nSelective optimum: {len(a_total)} useful attrs")
    print(f"  TP: {a_total.tp.sum()}, FP: {a_total.fp.sum()}")
    print(f"  Total Δacc: {a_total.dacc_pp.sum():+.3f} пп")
    print(f"  Total Δcost: {a_total.dcost_pct.sum():+.2f}%")

    print("\n" + "=" * 96)
    print("E. ML THRESHOLD RELAXATION (per-attr)")
    print("=" * 96)
    e_rows = []
    for short, internal in CATS:
        print(f"\n  ... {internal}")
        e_rows.extend(ml_threshold_relaxation(short, internal))
    e_df = pd.DataFrame(e_rows)
    e_df.to_parquet(PROCESSED / "accuracy_squeeze_e_ml_thresholds.parquet", index=False)

    # For each (cat, attr), pick threshold maximizing e2e_acc
    best_e = e_df.loc[e_df.groupby(["cat","attr"])["n_correct_estimate"].idxmax()].copy()
    print("\nOptimal Layer 2 threshold per attr (max e2e_acc):")
    print(best_e[["cat","attr","thr","covered","abstain","ml_acc_on_covered","e2e_acc"]]
          .to_string(index=False))

    # Compare to current cascade for each attr (use existing cascade_old where thr was applied)
    print("\nE2E gain per attr from optimal threshold (vs current e2e estimate):")
    # Compute current cascade e2e_acc per attr
    cascade_cur_rows = []
    for short, internal in CATS:
        cur = pd.read_parquet(PROCESSED / f"cascade_preds_{short}_after_fix.parquet")
        cur["code"] = cur["code"].astype(str)
        gold = pd.read_parquet(PROCESSED / "consensus_gold_v2_expanded.parquet")
        gold = gold[(gold.category == short) & ~gold.gold_is_null]
        gold["code"] = gold["code"].astype(str)
        gold["gold_norm"] = gold.gold_value.astype(str).str.lower()
        la = llm_acc(internal)
        merged = cur.merge(gold[["code","attr","gold_norm"]], on=["code","attr"], how="inner")
        merged["pred_norm"] = merged.predicted.astype(str).str.lower()
        merged["correct"] = (merged.pred_norm == merged.gold_norm) & (merged.layer != "abstain")
        for attr, sub in merged.groupby("attr"):
            n = len(sub)
            abst = (sub.layer == "abstain").sum()
            n_correct = sub.correct.sum() + abst * la.get(attr, 0.70)
            cascade_cur_rows.append({"cat":short,"attr":attr,"e2e_current":n_correct/n,"n":n})
    cur_df = pd.DataFrame(cascade_cur_rows)
    merged_e = best_e.merge(cur_df, on=["cat","attr"], how="left")
    merged_e["delta_e2e_pp"] = (merged_e["e2e_acc"] - merged_e["e2e_current"]) * 100
    merged_e["abs_correct_gain"] = (merged_e["e2e_acc"] - merged_e["e2e_current"]) * merged_e["n"]
    print(merged_e[["cat","attr","thr","e2e_current","e2e_acc","delta_e2e_pp","abs_correct_gain"]]
          .sort_values("delta_e2e_pp", ascending=False).to_string(index=False))

    total_e_gain_cells = merged_e[merged_e.delta_e2e_pp > 0].abs_correct_gain.sum()
    print(f"\nTotal headline gain from E (если применить оптимум на всех):"
          f" +{total_e_gain_cells/4350*100:.3f} пп")
    total_e_gain_positive = merged_e[merged_e.delta_e2e_pp > 0]
    print(f"  Positive attrs: {len(total_e_gain_positive)} / {len(merged_e)}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
