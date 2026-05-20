"""Комбинированный эффект A2 + E с честным учётом перекрытия.

Сценарий:
  1. E применяется первым: для каждого (cat, attr) выбираем оптимальный ML
     threshold; cells с conf < thr уходят в abstain → LLM.
  2. A применяется к оставшимся ML cells: Bayes с per-attr оптимальным q
     (cost ≤ 0.5%); флаги уходят в LLM.
  3. Подсчёт e2e: regex-cells → их correct/incorrect; ML cells → ML
     correct/incorrect, минус Bayes-flagged → LLM acc; abstain-cells → LLM acc.

Запуск:
  OMP_NUM_THREADS=1 python -m src.experiments.accuracy_squeeze_combined
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

CATS = [
    ("pasta", "pasta_stratified"),
    ("chocolate", "chocolate_stratified"),
    ("cheeses", "cheeses_stratified"),
]

# A2 Pareto-selected per-attr Bayes q (Δacc > 0 AND dcost ≤ 0.5% per attr)
BAYES_PER_ATTR_Q = {
    ("cheeses", "is_ultra_processed"): 0.005,
    ("chocolate", "contains_nuts"): 0.020,
    ("pasta", "is_gluten_free"): 0.010,
    ("chocolate", "is_organic"): 0.050,
    ("chocolate", "protein_class"): 0.100,
    ("cheeses", "texture"): 0.005,
    ("cheeses", "is_pdo"): 0.005,
    ("chocolate", "chocolate_type"): 0.005,
    # cocoa_percentage skipped — Δacc 0
}

# E per-attr optimal thresholds (only those with positive delta_e2e_pp)
ML_THR_PER_ATTR = {
    ("cheeses", "is_ultra_processed"): 0.75,
    ("chocolate", "chocolate_extra"): 0.45,
    ("chocolate", "nutri_score_grade"): 0.60,
    ("cheeses", "is_pdo"): 0.75,
    ("cheeses", "country_of_origin"): 0.75,
    ("chocolate", "chocolate_type"): 0.75,
    ("cheeses", "is_organic"): 0.55,
    ("pasta", "is_organic"): 0.75,
    ("pasta", "pasta_shape"): 0.75,
    ("chocolate", "cocoa_percentage"): 0.60,
    ("cheeses", "fat_class"): 0.75,
    ("cheeses", "texture"): 0.50,
    ("pasta", "is_filled"): 0.45,
    ("pasta", "is_gluten_free"): 0.45,
}

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
    raw = pd.read_parquet(PROCESSED / f"{internal}_raw.parquet", columns=["code","brands"])
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
    hybrid = pd.concat([silver_fit] + [gold_fit] * 10, ignore_index=True)
    model = BayesianNetwork(edges)
    est = BayesianEstimator(model, hybrid)
    for node in model.nodes():
        cpd = est.estimate_cpd(node, prior_type="BDeu", equivalent_sample_size=10)
        model.add_cpds(cpd)
    model.check_model()
    return model, gold_fit


def llm_acc(internal):
    df = pd.read_parquet(PROCESSED / f"direct_llm_eval_{internal}_sonnet45.parquet")
    m = (df.predicted_non_null == 1) & (df.gt_non_null == 1)
    return df[m].groupby("attr").correct_when_both_present.mean().to_dict()


HYBRID_MODELS = {("chocolate", "chocolate_type"), ("chocolate", "contains_nuts")}


def reapply_cascade(short, internal):
    """Применяем E (новые ML thr) + A (Bayes flags) и считаем e2e."""
    cascade_old = pd.read_parquet(PROCESSED / f"cascade_preds_{short}_after_fix.parquet")
    cascade_old["code"] = cascade_old["code"].astype(str)
    silver = pd.read_parquet(PROCESSED / f"{internal}_silver_standard.parquet")
    silver["code"] = silver["code"].astype(str); silver["_pos"] = np.arange(len(silver))
    emb_all = np.load(PROCESSED / f"{internal}_embeddings.npy")
    test_codes = sorted(cascade_old["code"].unique())
    sub = silver[silver["code"].isin(test_codes)].copy().reset_index(drop=True)
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

    gold = pd.read_parquet(PROCESSED / "consensus_gold_v2_expanded.parquet")
    gold = gold[(gold.category == short) & ~gold.gold_is_null].copy()
    gold["code"] = gold["code"].astype(str)
    gold["gold_norm"] = gold.gold_value.astype(str).str.lower()
    la = llm_acc(internal)

    # Reload bayes model for A
    bayes, _ = fit_hybrid_bayes(short, internal)
    inference = VariableElimination(bayes)
    bayes_thr_per_q = {}
    for q in {0.005, 0.010, 0.020, 0.050, 0.100}:
        bayes_thr_per_q[q] = calibrate_thresholds(bayes, _, inference, q=q)

    # Build base evidence per code (from cascade_old)
    raw = pd.read_parquet(PROCESSED / f"{internal}_raw.parquet", columns=["code","brands"])
    raw["code"] = raw["code"].astype(str); raw["brands"] = raw["brands"].fillna("").astype(str)
    brands = dict(zip(raw["code"], raw["brands"]))
    base_ev = {}
    for code, grp in cascade_old.groupby("code"):
        ev = {}
        if code in brands and brands[code]: ev["brand"] = brands[code]
        for _, r in grp.iterrows():
            if r["layer"] == "abstain": continue
            ev[r["attr"]] = r["predicted"]
        base_ev[code] = ev

    results_per_attr = {}
    attrs = cascade_old["attr"].unique()
    for attr in attrs:
        # Get gold for this attr
        gattr = gold[gold.attr == attr][["code","gold_norm"]].set_index("code")
        if len(gattr) == 0: continue
        co_attr = cascade_old[cascade_old.attr == attr].merge(
            gattr.reset_index(), on="code", how="inner")
        co_attr["pred_norm"] = co_attr.predicted.astype(str).str.lower()
        co_attr["cascade_correct"] = (co_attr.pred_norm == co_attr.gold_norm) & (co_attr.layer != "abstain")

        l = la.get(attr, 0.70)
        n_total = len(co_attr)

        # 1) REGEX cells — fixed contribution (not touched by E or A)
        regex_cells = co_attr[co_attr.layer == "regex"]
        n_regex_correct = int(regex_cells.cascade_correct.sum())
        n_regex_total = len(regex_cells)

        # 2) ML/abstain cells — apply E threshold
        ml_pool_codes = set(co_attr[co_attr.layer.isin(["ml","abstain"])].code.astype(str))
        # Predict ML
        use_hybrid = (short, attr) in HYBRID_MODELS
        if use_hybrid and X_hybrid is not None:
            clf_path = MODELS / f"{internal}_hybrid_{attr}_xgb.pkl"
            le_path = MODELS / f"{internal}_hybrid_{attr}_le.pkl"
            X_in = X_hybrid
        else:
            clf_path = MODELS / f"{internal}_{attr}_xgb_hybrid.pkl"
            le_path = MODELS / f"{internal}_{attr}_le_hybrid.pkl"
            X_in = emb_sub
        if not clf_path.exists():
            # Without model — keep cascade_old behavior
            e2e_correct = co_attr.cascade_correct.sum() + (co_attr.layer == "abstain").sum() * l
            results_per_attr[attr] = {"cat":short, "attr":attr, "n":n_total,
                                       "e2e_combined": e2e_correct/n_total,
                                       "e2e_current": e2e_correct/n_total,
                                       "n_ml": 0, "n_abstain": 0, "n_bayes_flag": 0}
            continue
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
        code_to_pred = dict(zip(codes_sub, zip(preds, max_conf)))

        # New ML thr (E)
        ml_new_thr = ML_THR_PER_ATTR.get((short, attr))
        # If E not configured for this attr, use a sentinel (no change → use cascade_old layers)
        # Simplest: rerun threshold even at default thr (= old behavior approx for ML/abstain pool)

        # Apply Bayes (A)
        bayes_q = BAYES_PER_ATTR_Q.get((short, attr))
        bayes_thr = bayes_thr_per_q.get(bayes_q, {}).get(attr) if bayes_q else None

        n_ml_correct = n_ml_total = 0
        n_abstain = 0
        n_bayes_flag = 0
        n_bayes_tp = n_bayes_fp = 0
        for _, row in co_attr[co_attr.layer.isin(["ml","abstain"])].iterrows():
            code = str(row.code)
            gold_n = row.gold_norm
            if code not in code_to_pred:
                continue
            pred, conf = code_to_pred[code]
            pred_low = pred.lower()
            ml_correct = (pred_low == gold_n)

            # Step 1: E — if conf < ml_new_thr OR if no E config and conf < cascade_default
            if ml_new_thr is not None:
                ml_keeps = (conf >= ml_new_thr)
            else:
                # No E config — keep cascade_old layer behavior
                ml_keeps = (row.layer == "ml")

            if not ml_keeps:
                n_abstain += 1
                continue

            # Step 2: A — if Bayes flagged, demote to LLM
            if bayes_thr is not None and attr in bayes.nodes():
                ev = dict(base_ev.get(code, {})); ev.pop(attr, None)
                p = attribute_likelihood(attr, pred, ev, bayes, inference)
                if p is not None and p < bayes_thr:
                    n_bayes_flag += 1
                    if not ml_correct: n_bayes_tp += 1
                    else: n_bayes_fp += 1
                    # Demote: cell goes to LLM
                    n_abstain += 1
                    continue

            n_ml_total += 1
            if ml_correct: n_ml_correct += 1

        e2e_correct = n_regex_correct + n_ml_correct + n_abstain * l
        e2e_combined = e2e_correct / n_total if n_total else 0
        # Current cascade e2e
        n_abs_old = (co_attr.layer == "abstain").sum()
        e2e_current = (co_attr.cascade_correct.sum() + n_abs_old * l) / n_total

        results_per_attr[attr] = {
            "cat": short, "attr": attr, "n": n_total,
            "n_regex_correct": n_regex_correct, "n_regex_total": n_regex_total,
            "n_ml_correct": n_ml_correct, "n_ml_total": n_ml_total,
            "n_abstain": n_abstain,
            "n_bayes_flag": n_bayes_flag, "n_bayes_tp": n_bayes_tp, "n_bayes_fp": n_bayes_fp,
            "ml_thr_E": ml_new_thr, "bayes_q_A": bayes_q,
            "e2e_combined": e2e_combined,
            "e2e_current": e2e_current,
            "delta_pp": (e2e_combined - e2e_current) * 100,
            "abs_correct_gain": (e2e_combined - e2e_current) * n_total,
        }
    return results_per_attr


def main():
    all_rows = []
    for short, internal in CATS:
        print(f"\n--- {short} ---")
        d = reapply_cascade(short, internal)
        for k, v in d.items(): all_rows.append(v)

    df = pd.DataFrame(all_rows).sort_values("delta_pp", ascending=False)
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:+.3f}")
    print("\nPer-attr combined A+E effect:")
    print(df[["cat","attr","n","ml_thr_E","bayes_q_A","n_abstain","n_bayes_flag",
              "n_bayes_tp","n_bayes_fp","e2e_current","e2e_combined","delta_pp","abs_correct_gain"]]
          .to_string(index=False))

    total_gain = df.abs_correct_gain.sum()
    total_cells = df.n.sum()
    print(f"\nTotal cells: {total_cells}")
    print(f"Total absolute correct gain: {total_gain:.2f}")
    print(f"Combined Δheadline (vs after_fix baseline): {total_gain/4350*100:+.3f} пп")

    # LLM cost: total abstain change
    cascade_full = pd.concat([pd.read_parquet(PROCESSED / f"cascade_preds_{cat}_after_fix.parquet")
                               for cat, _ in CATS], ignore_index=True)
    cascade_full["code"] = cascade_full["code"].astype(str)
    gold = pd.read_parquet(PROCESSED / "consensus_gold_v2_expanded.parquet")
    gold = gold[~gold.gold_is_null]
    gold["code"] = gold["code"].astype(str)
    m_old = cascade_full.merge(gold[["code","attr","category"]], on=["code","attr"], how="inner")
    n_abs_old = (m_old.layer == "abstain").sum()
    n_abs_new = df.n_abstain.sum()
    n_total = len(m_old)
    print(f"\nLLM calls: {n_abs_old} → {n_abs_new}  "
          f"(Δ = {n_abs_new - n_abs_old:+d}, "
          f"{(n_abs_new-n_abs_old)/n_total*100:+.2f}% от 4350)")

    df.to_parquet(PROCESSED / "accuracy_squeeze_combined.parquet", index=False)
    print(f"\nSaved {PROCESSED / 'accuracy_squeeze_combined.parquet'}")


if __name__ == "__main__":
    sys.exit(main() or 0)
