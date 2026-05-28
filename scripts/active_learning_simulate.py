"""
Active Learning Simulation on 3-LLM consensus gold ($0 — no new labels).

For ticket 2026-05-28-active-learning-pilot-run.

Idea
----
We don't pay Opus for blind labelling. Instead, we use existing
`manual_gold_consensus.parquet` (22 207 cells × 3 cats, extended rerun
2026-05-27) as the source of "gold labels to add to training". The
simulation answers: if we add N more labelled cells to the training
data, does ACTIVE selection (confident errors) outperform RANDOM?

Pipeline per (category, attr)
------------------------------
1. Load silver standard + embeddings (Layer 2 training data, ~13–21k rows).
2. Load consensus gold (the cells with reliable labels).
3. Load cascade_raw_with_conf for ml_pred / ml_conf on gold cells.
4. Compute confidence quartile P75 of ml_conf on gold pool.
5. AL pool = gold cells with ml_pred != consensus_gold AND ml_conf >= P75.
6. Hold-out = 30% of gold codes (code-disjoint), not used for either pool.
7. For each N in {50, 100, 200, 500}, two strategies (active, random):
   - Sample N (code, attr) pairs stratified by attr (from AL pool / from
     full gold pool excluding holdout).
   - Train XGBoost on (silver-train + N extra gold labels).
   - Eval accuracy on the holdout subset of gold.
8. Baseline (N=0) accuracy = train on silver only, eval on holdout.

Writes `datasets/processed/active_learning_results.parquet` with columns:
  category, attr, strategy, n_added, accuracy, n_holdout, delta_vs_base,
  baseline_acc, seed.

Plus a summary JSON at `datasets/processed/active_learning_summary.json`.

Run (on VM):
  source .venv/bin/activate
  python -m scripts.active_learning_simulate
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("active_learning_simulate")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
PROCESSED = Path("datasets/processed")

CATEGORIES = ["pasta", "chocolate", "cheeses"]
N_VALUES = [50, 100, 200, 500]
SEED = 42
HOLDOUT_FRAC = 0.30  # 30% gold codes reserved for eval

# Attributes per category (mirror production schema from train.py CATEGORY_CONFIG).
# Type: 'binary' (true/false) or 'multiclass'.
CAT_ATTRS = {
    "pasta": {
        "grain_type": "multiclass",
        "pasta_shape": "multiclass",
        "is_filled": "binary",
        "is_organic": "binary",
        "is_gluten_free": "binary",
        "is_vegan": "binary",
        "cuisine_origin": "multiclass",
    },
    "chocolate": {
        "chocolate_type": "multiclass",
        "chocolate_extra": "multiclass",
        "is_filled": "binary",
        "contains_nuts": "binary",
        "is_organic": "binary",
        "flavor_profile": "multiclass",
    },
    "cheeses": {
        "milk_source": "multiclass",
        "texture": "multiclass",
        "country_of_origin": "multiclass",
        "is_pdo": "binary",
        "is_organic": "binary",
        "is_ultra_processed": "binary",
        "aging": "multiclass",
    },
}

# XGBoost params (mirror production train.py)
MC_PARAMS = dict(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3, gamma=0.1,
    reg_alpha=0.1, reg_lambda=1.0, verbosity=0, n_jobs=2,
    tree_method="hist",
)
BIN_PARAMS = dict(
    n_estimators=250, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3, gamma=0.1,
    reg_alpha=0.1, reg_lambda=1.0, verbosity=0, n_jobs=2,
    tree_method="hist",
)
GOLD_WEIGHT = 5.0


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_cat(cat: str):
    """Load silver (with attrs as columns) + embeddings + consensus gold for one cat."""
    silver = pd.read_parquet(PROCESSED / f"{cat}_stratified_silver_standard.parquet")
    silver["code"] = silver["code"].astype(str)
    silver = silver.reset_index(drop=True)
    emb = np.load(PROCESSED / f"{cat}_stratified_embeddings.npy")
    assert len(silver) == emb.shape[0], (
        f"silver/emb mismatch: {len(silver)} vs {emb.shape[0]}"
    )
    # consensus_gold long format: code, attr, gold_value (+ category)
    gold_all = pd.read_parquet(PROCESSED / "manual_gold_consensus.parquet")
    gold = gold_all[gold_all["category"] == cat].copy()
    gold["code"] = gold["code"].astype(str)
    # Use cells where consensus reached non-null value AND not disputed
    # gold_value can be NaN if all voters said null
    gold = gold[gold["gold_value"].notna()].copy()
    # Cast values to lower-case strings (matches train.py normalization)
    gold["gold_value"] = gold["gold_value"].astype(str).str.lower()

    # cascade preds give us ml_pred + ml_conf for each (code, attr) on gold
    cascade = pd.read_parquet(PROCESSED / f"cascade_raw_with_conf_{cat}.parquet")
    cascade["code"] = cascade["code"].astype(str)
    return silver, emb, gold, cascade


# -----------------------------------------------------------------------------
# AL pool selection
# -----------------------------------------------------------------------------

def build_pools(gold: pd.DataFrame, cascade: pd.DataFrame, holdout_codes: set,
                attrs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (al_pool_df, full_merged_df).

    AL pool: "confident errors" — cells где ml_pred != gold_value AND
             ml_conf >= P75 среди ошибок этого атрибута.

             P75 берётся conditional на множестве ошибок (а не на всех
             cells), потому что распределение conf на всех cells крайне
             перекошено вправо (P75 ≈ 0.98+ — модель супер-уверена), и
             unconditional порог оставляет ≤ 5 cells на весь пул. P75 на
             ошибках — корректная семантика "верхний квартиль уверенности
             среди реальных ошибок", и это самые ценные labels для
             добавления (модель уверенно врёт — есть чему учиться).

             Cells must NOT be in holdout. Layer 1 hits (rule_tier == "high")
             исключаются, т.к. решение принимает regex/OFF rule, не ML.

    full_merged_df: pool gold cells после merge с cascade (для диагностики).
    """
    g = gold[gold["attr"].isin(attrs)].copy()
    g = g[~g["code"].isin(holdout_codes)].copy()
    c = cascade[cascade["attr"].isin(attrs)].copy()
    if "rule_tier" in c.columns:
        c = c[c["rule_tier"].isna() | (c["rule_tier"].astype(str) != "high")].copy()
    merged = g.merge(
        c[["code", "attr", "ml_pred", "ml_conf"]],
        on=["code", "attr"], how="left",
    )
    errors = merged[
        merged["ml_pred"].notna()
        & merged["ml_conf"].notna()
        & (merged["ml_pred"].astype(str).str.lower()
           != merged["gold_value"].astype(str).str.lower())
    ].copy()
    # P75 conf среди ошибок — per attr
    p75_err = errors.groupby("attr")["ml_conf"].quantile(0.75).to_dict()
    errors["p75_err"] = errors["attr"].map(p75_err).fillna(0.0)
    al = errors[errors["ml_conf"] >= errors["p75_err"]].copy()
    return al, merged


# -----------------------------------------------------------------------------
# Training & eval
# -----------------------------------------------------------------------------

def _stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample n rows stratified by 'attr' column."""
    if n >= len(df):
        return df.copy()
    rng = np.random.RandomState(seed)
    per_attr = df.groupby("attr")
    # Proportional allocation, with floor of 1 per attr present
    attrs = list(per_attr.groups.keys())
    total = len(df)
    quotas = {}
    for a in attrs:
        prop = len(per_attr.get_group(a)) / total
        quotas[a] = max(1, int(round(prop * n)))
    # Adjust to exactly n
    diff = n - sum(quotas.values())
    while diff != 0:
        if diff > 0:
            # Add to largest pools
            for a in sorted(attrs, key=lambda x: -len(per_attr.get_group(x))):
                if quotas[a] < len(per_attr.get_group(a)):
                    quotas[a] += 1
                    diff -= 1
                    if diff == 0:
                        break
        else:
            for a in sorted(attrs, key=lambda x: -quotas[x]):
                if quotas[a] > 1:
                    quotas[a] -= 1
                    diff += 1
                    if diff == 0:
                        break
            else:
                break
    picks = []
    for a in attrs:
        grp = per_attr.get_group(a)
        k = min(quotas[a], len(grp))
        idxs = rng.choice(len(grp), size=k, replace=False)
        picks.append(grp.iloc[idxs])
    return pd.concat(picks, axis=0).reset_index(drop=True)


def _train_eval_attr(
    cat: str,
    attr: str,
    attr_type: str,
    silver: pd.DataFrame,
    emb: np.ndarray,
    code_to_silver_idx: dict,
    holdout_gold: pd.DataFrame,
    extra_gold: pd.DataFrame | None,
) -> tuple[float, int]:
    """Train Layer 2 model for this attr (silver+extra), eval on holdout.

    Returns (accuracy, n_holdout).
    """
    # Build training set from silver
    if attr not in silver.columns:
        return float("nan"), 0
    sil = silver[silver[attr].notna()].copy()
    if len(sil) < 30:
        return float("nan"), 0

    # Exclude any silver row whose code appears in holdout (defense in depth —
    # gold codes may overlap silver, must not let labels leak)
    holdout_codes = set(holdout_gold["code"].astype(str))
    sil = sil[~sil["code"].astype(str).isin(holdout_codes)].copy()
    sil_idxs = sil.index.to_numpy()
    X_sil = emb[sil_idxs]
    if attr_type == "binary":
        y_sil = sil[attr].astype(bool).astype(int).to_numpy()
    else:
        y_sil = sil[attr].astype(str).str.lower().to_numpy()

    w_sil = np.ones(len(y_sil), dtype=np.float32)

    # Extra gold (for active or random strategy)
    X_extra = np.empty((0, emb.shape[1]), dtype=emb.dtype)
    y_extra_raw = []
    if extra_gold is not None and len(extra_gold) > 0:
        eg = extra_gold[extra_gold["attr"] == attr].copy()
        if len(eg) > 0:
            # Map gold codes to silver embeddings (most gold codes exist in silver)
            eg["sidx"] = eg["code"].astype(str).map(code_to_silver_idx)
            eg = eg[eg["sidx"].notna()].copy()
            if len(eg) > 0:
                eg_idxs = eg["sidx"].astype(int).to_numpy()
                X_extra = emb[eg_idxs]
                if attr_type == "binary":
                    y_extra_raw = (
                        eg["gold_value"].astype(str).str.lower()
                        .isin(["true", "1", "yes"])
                        .astype(int).to_numpy()
                    )
                else:
                    y_extra_raw = eg["gold_value"].astype(str).str.lower().to_numpy()

    if len(X_extra) > 0:
        X_tr = np.vstack([X_sil, X_extra])
        if attr_type == "binary":
            y_tr = np.concatenate([y_sil, y_extra_raw])
        else:
            y_tr = np.concatenate([y_sil, y_extra_raw])
        w_tr = np.concatenate([w_sil, GOLD_WEIGHT * np.ones(len(y_extra_raw), dtype=np.float32)])
    else:
        X_tr, y_tr, w_tr = X_sil, y_sil, w_sil

    # Holdout
    hg = holdout_gold[holdout_gold["attr"] == attr].copy()
    hg["sidx"] = hg["code"].astype(str).map(code_to_silver_idx)
    hg = hg[hg["sidx"].notna()].copy()
    if len(hg) < 5:
        return float("nan"), len(hg)
    h_idxs = hg["sidx"].astype(int).to_numpy()
    X_te = emb[h_idxs]
    if attr_type == "binary":
        y_te = (
            hg["gold_value"].astype(str).str.lower()
            .isin(["true", "1", "yes"])
            .astype(int).to_numpy()
        )
    else:
        y_te = hg["gold_value"].astype(str).str.lower().to_numpy()

    # Train
    if attr_type == "binary":
        if y_tr.sum() < 5 or (y_tr == 0).sum() < 5:
            return float("nan"), len(hg)
        spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
        clf = XGBClassifier(eval_metric="logloss", scale_pos_weight=spw, **BIN_PARAMS)
        clf.fit(X_tr, y_tr, sample_weight=w_tr)
        y_pred = clf.predict(X_te)
        acc = float((y_pred == y_te).mean())
    else:
        classes = sorted(set(y_tr.tolist()) | set(y_te.tolist()))
        if len(classes) < 2:
            return float("nan"), len(hg)
        le = LabelEncoder()
        le.fit(classes)
        # Drop test samples whose class is not seen in train (acc=0 contribution
        # is misleading — they cannot be learned in a single retrain). Keep as
        # part of the denominator but predict wrong by construction.
        y_tr_enc = le.transform(y_tr)
        # Test samples: if class not in train, label_encoder still maps it (we
        # fit on union); but XGB can only predict classes present in training.
        train_classes = sorted(set(y_tr.tolist()))
        train_classes_enc = le.transform(train_classes)
        clf = XGBClassifier(
            objective=("binary:logistic" if len(train_classes) == 2 else "multi:softmax"),
            num_class=(None if len(train_classes) == 2 else len(train_classes)),
            eval_metric=("logloss" if len(train_classes) == 2 else "mlogloss"),
            **MC_PARAMS,
        )
        # Re-encode train labels to dense 0..K-1 within train_classes for XGB
        train_le = LabelEncoder()
        train_le.fit(train_classes)
        y_tr_enc_dense = train_le.transform(y_tr)
        clf.fit(X_tr, y_tr_enc_dense, sample_weight=w_tr)
        y_pred_dense = clf.predict(X_te)
        # Map predictions back to string labels
        y_pred = train_le.inverse_transform(y_pred_dense.astype(int))
        acc = float((y_pred == y_te).mean())
    return acc, int(len(hg))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="+", default=CATEGORIES)
    ap.add_argument("--n-values", nargs="+", type=int, default=N_VALUES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(PROCESSED / "active_learning_results.parquet"))
    ap.add_argument("--summary-out", default=str(PROCESSED / "active_learning_summary.json"))
    args = ap.parse_args()

    rows: list[dict] = []
    summary_per_cat: dict[str, dict] = {}

    for cat in args.cats:
        logger.info("===== %s =====", cat)
        silver, emb, gold, cascade = load_cat(cat)
        attrs_map = CAT_ATTRS[cat]
        attrs = list(attrs_map.keys())
        # Restrict gold to scope attrs
        gold = gold[gold["attr"].isin(attrs)].copy()
        logger.info(
            "[%s] silver=%d, emb=%s, gold cells=%d (codes=%d), cascade rows=%d",
            cat, len(silver), emb.shape, len(gold), gold["code"].nunique(), len(cascade),
        )

        # 70/30 split by gold code
        rng = np.random.RandomState(args.seed)
        gold_codes = sorted(gold["code"].astype(str).unique().tolist())
        rng.shuffle(gold_codes)
        n_hold = int(len(gold_codes) * HOLDOUT_FRAC)
        holdout_codes = set(gold_codes[:n_hold])
        pool_codes = set(gold_codes[n_hold:])
        holdout_gold = gold[gold["code"].astype(str).isin(holdout_codes)].copy()
        pool_gold = gold[gold["code"].astype(str).isin(pool_codes)].copy()
        logger.info(
            "[%s] holdout codes=%d (cells=%d) | pool codes=%d (cells=%d)",
            cat, len(holdout_codes), len(holdout_gold), len(pool_codes), len(pool_gold),
        )

        al_pool, full_pool = build_pools(pool_gold, cascade, holdout_codes, attrs)
        # full_pool may include rows where ml_pred is NaN — random sampling
        # should still allow these (they're still valid gold labels we can add).
        # Use original pool_gold (long format) as the random pool source.
        random_pool = pool_gold.copy()
        logger.info(
            "[%s] AL pool size=%d (confident errors, P75 conf threshold per attr) | "
            "random pool size=%d (all non-holdout gold)",
            cat, len(al_pool), len(random_pool),
        )

        # Silver code → embedding row index map (for matching gold codes to embeddings)
        code_to_silver_idx = {
            c: i for i, c in enumerate(silver["code"].astype(str).tolist())
        }

        # -- Baseline (N=0)
        logger.info("[%s] baseline (silver only) ...", cat)
        base_accs: dict[str, tuple[float, int]] = {}
        for attr in attrs:
            attr_type = attrs_map[attr]
            acc, nh = _train_eval_attr(
                cat, attr, attr_type, silver, emb, code_to_silver_idx,
                holdout_gold, extra_gold=None,
            )
            base_accs[attr] = (acc, nh)
            logger.info("  [%s/%s] baseline acc=%.4f (n_holdout=%d)", cat, attr, acc, nh)
            rows.append({
                "category": cat, "attr": attr, "strategy": "baseline",
                "n_added": 0, "accuracy": acc, "n_holdout": nh,
                "baseline_acc": acc, "delta_vs_base": 0.0, "seed": args.seed,
            })

        # -- For each N, both strategies
        for n_add in args.n_values:
            for strategy, source in [("active", al_pool), ("random", random_pool)]:
                if len(source) == 0:
                    logger.warning("[%s/%s/N=%d] empty source pool, skipping", cat, strategy, n_add)
                    continue
                n_actual = min(n_add, len(source))
                seed_run = args.seed + n_add + (0 if strategy == "active" else 1000)
                picked = _stratified_sample(source, n_actual, seed=seed_run)
                logger.info(
                    "[%s/%s/N=%d] sampled %d cells, attrs=%s",
                    cat, strategy, n_add, len(picked),
                    dict(picked["attr"].value_counts()),
                )
                for attr in attrs:
                    attr_type = attrs_map[attr]
                    extra = picked[picked["attr"] == attr][["code", "gold_value", "attr"]].copy()
                    acc, nh = _train_eval_attr(
                        cat, attr, attr_type, silver, emb, code_to_silver_idx,
                        holdout_gold, extra_gold=extra if len(extra) > 0 else None,
                    )
                    base_acc, _ = base_accs.get(attr, (float("nan"), 0))
                    delta = acc - base_acc if not (np.isnan(acc) or np.isnan(base_acc)) else float("nan")
                    logger.info(
                        "  [%s/%s/%s/N=%d] acc=%.4f delta=%+.4f (n_extra_attr=%d, n_hold=%d)",
                        cat, attr, strategy, n_add, acc, delta, len(extra), nh,
                    )
                    rows.append({
                        "category": cat, "attr": attr, "strategy": strategy,
                        "n_added": n_add, "accuracy": acc, "n_holdout": nh,
                        "baseline_acc": base_acc, "delta_vs_base": delta,
                        "n_extra_attr": int(len(extra)),
                        "seed": seed_run,
                    })

        summary_per_cat[cat] = {
            "n_silver": int(len(silver)),
            "n_gold_codes": int(len(gold_codes)),
            "n_gold_cells": int(len(gold)),
            "n_holdout_codes": int(len(holdout_codes)),
            "n_pool_codes": int(len(pool_codes)),
            "n_al_pool": int(len(al_pool)),
            "n_random_pool": int(len(random_pool)),
        }

    # -----------------------------------------------------------------------
    # Persist
    # -----------------------------------------------------------------------
    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), out_path)

    # Summary: cells-weighted mean accuracy by (strategy, n_added, category)
    grouped = (
        df[df["strategy"].isin(["active", "random"])]
        .dropna(subset=["accuracy", "n_holdout"])
        .assign(
            num=lambda x: x["accuracy"] * x["n_holdout"],
        )
        .groupby(["category", "strategy", "n_added"])
        .agg(
            n_cells_eval=("n_holdout", "sum"),
            acc_weighted=("num", "sum"),
        )
    )
    grouped["accuracy_weighted_mean"] = grouped["acc_weighted"] / grouped["n_cells_eval"]
    grouped = grouped.drop(columns=["acc_weighted"]).reset_index()
    # Compare to baseline
    base_per_cat = (
        df[df["strategy"] == "baseline"]
        .dropna(subset=["accuracy", "n_holdout"])
        .assign(num=lambda x: x["accuracy"] * x["n_holdout"])
        .groupby("category")
        .agg(n_cells_eval=("n_holdout", "sum"), acc_weighted=("num", "sum"))
    )
    base_per_cat["baseline_accuracy_weighted"] = (
        base_per_cat["acc_weighted"] / base_per_cat["n_cells_eval"]
    )
    grouped = grouped.merge(
        base_per_cat[["baseline_accuracy_weighted"]].reset_index(),
        on="category", how="left",
    )
    grouped["delta_vs_base_pp"] = (
        (grouped["accuracy_weighted_mean"] - grouped["baseline_accuracy_weighted"]) * 100
    )
    summary = {
        "per_category_pools": summary_per_cat,
        "learning_curve": grouped.to_dict(orient="records"),
        "n_values": args.n_values,
        "seed": args.seed,
        "holdout_frac": HOLDOUT_FRAC,
        "gold_weight": GOLD_WEIGHT,
    }
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Wrote summary to %s", args.summary_out)

    # Pretty print
    print("\n" + "=" * 80)
    print("Active Learning Simulation — Results")
    print("=" * 80)
    for cat in args.cats:
        sub = grouped[grouped["category"] == cat]
        if len(sub) == 0:
            continue
        base = sub["baseline_accuracy_weighted"].iloc[0]
        print(f"\n[{cat}] baseline (silver only) = {base*100:.2f}%")
        for n in args.n_values:
            row_a = sub[(sub["strategy"] == "active") & (sub["n_added"] == n)]
            row_r = sub[(sub["strategy"] == "random") & (sub["n_added"] == n)]
            if len(row_a) and len(row_r):
                a = row_a.iloc[0]
                r = row_r.iloc[0]
                print(
                    f"  N={n:4d}: active={a['accuracy_weighted_mean']*100:6.2f}% "
                    f"({a['delta_vs_base_pp']:+5.2f}pp) | "
                    f"random={r['accuracy_weighted_mean']*100:6.2f}% "
                    f"({r['delta_vs_base_pp']:+5.2f}pp) | "
                    f"active-vs-random={(a['accuracy_weighted_mean']-r['accuracy_weighted_mean'])*100:+5.2f}pp"
                )


if __name__ == "__main__":
    main()
