"""P2-EXP11 Step 3: Eval AL vs Random vs Baseline (pure XGBoost + LightGBM).

No torch/sentence-transformers. Uses pre-computed *_stratified_embeddings.npy.
New codes (AL + control) don't need embeddings for eval since they're only
added to TRAINING set; their embeddings come from a separate npy built by
a separate torch process (run_al_embed.py).

If new-code embeddings aren't available, we skip adding new codes to
XGB training (LGBM TF-IDF still works without embeddings).

Usage:
    OMP_NUM_THREADS=2 python scripts/run_al_eval.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

WORKTREE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKTREE_ROOT))

PROCESSED = WORKTREE_ROOT / "datasets" / "processed"
MANUAL_LABEL = WORKTREE_ROOT / "datasets" / "manual_label"

CATEGORIES = ["pasta", "chocolate", "cheeses"]
SEED = 42
TEST_SIZE = 0.2
GOLD_WEIGHT = 5.0

GOLD_PATH = PROCESSED / "consensus_gold_v2_expanded.parquet"
OUT_PATH = PROCESSED / "active_learning_pilot.parquet"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_text(row: pd.Series) -> str:
    parts = []
    for col in ["product_name", "ingredients_text", "brands"]:
        v = row.get(col)
        if pd.notna(v) and str(v).strip():
            parts.append(str(v).strip())
    return " ".join(parts)


def _load_long_annotations(parquet_path: Path, cat: str) -> pd.DataFrame:
    """Load wide annotation parquet and convert to long format."""
    if not parquet_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(parquet_path)
    if len(df) == 0:
        return pd.DataFrame()
    rows = []
    for _, row in df.iterrows():
        code = str(row["code"])
        try:
            parsed = json.loads(row["parsed_json"]) if row.get("parsed_json") else {}
        except (json.JSONDecodeError, TypeError):
            parsed = {}
        for attr, val in parsed.items():
            if val is None:
                continue
            rows.append({
                "category": cat,
                "code": code,
                "attr": attr,
                "gold_value": str(val),
                "gold_is_null": False,
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _eval_attr(
    cat: str,
    attr: str,
    cat_gold: pd.DataFrame,
    silver: pd.DataFrame,
    emb_all: np.ndarray,
    code_to_idx: dict,
    train_codes_set: set,
    test_codes_set: set,
    extra_gold: pd.DataFrame | None,
    extra_emb: np.ndarray | None,
    extra_code_to_idx: dict | None,
    extra_silver_df: pd.DataFrame | None,
) -> float:
    """Return R_ml accuracy (XGB+LGBM soft-vote) for one attr."""
    # Gold for this attr
    ag = cat_gold[(cat_gold["attr"] == attr) & ~cat_gold["gold_is_null"]].copy()
    ag["code"] = ag["code"].astype(str)
    ag = ag[ag["code"].isin(code_to_idx)]

    train_gold = ag[ag["code"].isin(train_codes_set)]
    test_gold = ag[ag["code"].isin(test_codes_set)]

    if len(train_gold) < 5 or len(test_gold) < 3:
        return float("nan")

    test_idx = np.array([code_to_idx[c] for c in test_gold["code"]])
    X_te_emb = emb_all[test_idx]
    y_te = test_gold["gold_value"].astype(str).values

    # Silver train data (excl. test/train codes)
    sil_indexed = silver.set_index("code")
    X_sil_emb = np.empty((0, emb_all.shape[1]))
    y_sil = np.array([], dtype=str)
    sil_texts: list[str] = []

    if attr in silver.columns:
        s = silver[silver[attr].notna()].copy()
        s["code"] = s["code"].astype(str)
        s = s[~s["code"].isin(test_codes_set) & ~s["code"].isin(train_codes_set)]
        s = s[s["code"].isin(code_to_idx)]
        if len(s) > 0:
            sidx = np.array([code_to_idx[c] for c in s["code"]])
            X_sil_emb = emb_all[sidx]
            y_sil = s[attr].astype(str).values
            sil_texts = [_build_text(sil_indexed.loc[c]) if c in sil_indexed.index else "" for c in s["code"]]

    # Base gold train
    tr_idx = np.array([code_to_idx[c] for c in train_gold["code"]])
    X_gold_emb = emb_all[tr_idx]
    y_gold = train_gold["gold_value"].astype(str).values
    gold_texts = [_build_text(sil_indexed.loc[c]) if c in sil_indexed.index else "" for c in train_gold["code"]]

    # Extra gold (AL or random)
    X_eg_emb = np.empty((0, emb_all.shape[1]))
    y_eg = np.array([], dtype=str)
    eg_texts: list[str] = []

    if extra_gold is not None and len(extra_gold) > 0:
        eg = extra_gold[
            (extra_gold["category"] == cat) & (extra_gold["attr"] == attr) & ~extra_gold["gold_is_null"]
        ].copy()
        eg["code"] = eg["code"].astype(str)

        if extra_emb is not None and extra_code_to_idx is not None:
            eg_valid = eg[eg["code"].isin(extra_code_to_idx)]
            if len(eg_valid) > 0:
                eg_eidx = np.array([extra_code_to_idx[c] for c in eg_valid["code"]])
                X_eg_emb = extra_emb[eg_eidx]
                y_eg = eg_valid["gold_value"].astype(str).values
                if extra_silver_df is not None:
                    es_idx = extra_silver_df.set_index("code")
                    eg_texts = [_build_text(es_idx.loc[c]) if c in es_idx.index else "" for c in eg_valid["code"]]
                else:
                    eg_texts = [""] * len(eg_valid)
        else:
            # No embeddings for extra: use LGBM only for extra (skip XGB part)
            # Still include in y_eg for LGBM
            if extra_silver_df is not None:
                es_idx = extra_silver_df.set_index("code")
                for _, row in eg.iterrows():
                    c = str(row["code"])
                    t = _build_text(es_idx.loc[c]) if c in es_idx.index else ""
                    eg_texts.append(t)
                    y_eg_list = y_eg.tolist() if len(y_eg) else []
                    y_eg_list.append(str(row["gold_value"]))
                y_eg = np.array(y_eg_list, dtype=str)

    # Combine all train
    X_parts = [x for x in [X_sil_emb, X_gold_emb, X_eg_emb] if len(x) > 0]
    y_parts = [y_sil, y_gold, y_eg]
    w_parts = [np.ones(len(y_sil)), GOLD_WEIGHT * np.ones(len(y_gold)), GOLD_WEIGHT * np.ones(len(y_eg))]
    tr_texts = sil_texts + gold_texts + eg_texts

    if not X_parts:
        return float("nan")

    X_tr_emb = np.vstack(X_parts)
    y_tr = np.concatenate([y for y in y_parts if len(y) > 0])
    w_tr = np.concatenate([w for w in w_parts if len(w) > 0])

    # Truncate tr_texts if mismatched due to no-embedding path
    if len(tr_texts) != len(y_tr):
        tr_texts = tr_texts[:len(y_tr)]

    all_classes = sorted(set(y_tr.tolist()))
    if len(all_classes) < 2:
        return float("nan")

    n_classes = len(all_classes)

    # XGB
    le_xgb = LabelEncoder()
    le_xgb.fit(all_classes)
    y_enc = le_xgb.transform(y_tr[:len(X_tr_emb)])  # only rows with embeddings

    common = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8,
                  reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
                  tree_method="hist", verbosity=0)
    if n_classes == 2:
        pos = int((y_enc == 1).sum())
        neg = int((y_enc == 0).sum())
        clf_xgb = xgb.XGBClassifier(scale_pos_weight=max(neg / max(pos, 1), 0.5), **common)
    else:
        clf_xgb = xgb.XGBClassifier(objective="multi:softmax", num_class=n_classes, **common)
    clf_xgb.fit(X_tr_emb, y_enc, sample_weight=w_tr[:len(X_tr_emb)])

    p_xgb = clf_xgb.predict_proba(X_te_emb)
    cls_to_col = {c: i for i, c in enumerate(all_classes)}
    p_xgb_a = np.zeros((len(X_te_emb), n_classes), dtype=np.float32)
    for i, cls in enumerate(le_xgb.classes_.tolist()):
        j = cls_to_col.get(str(cls), -1)
        if j >= 0:
            p_xgb_a[:, j] = p_xgb[:, i]

    # LGBM on TF-IDF
    te_texts = [_build_text(sil_indexed.loc[c]) if c in sil_indexed.index else "" for c in test_gold["code"]]
    p_lgbm_a = np.zeros((len(X_te_emb), n_classes), dtype=np.float32)
    lgbm_ok = False
    try:
        le_lgbm = LabelEncoder()
        le_lgbm.fit(all_classes)
        y_lgbm_enc = le_lgbm.transform(y_tr)

        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=10000)
        X_tfidf_tr = vec.fit_transform(tr_texts)
        lgbm_clf = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=5,
            objective=("binary" if n_classes == 2 else "multiclass"),
            **({"num_class": n_classes} if n_classes > 2 else {}),
            verbose=-1,
        )
        lgbm_clf.fit(X_tfidf_tr, y_lgbm_enc, sample_weight=w_tr)
        X_tfidf_te = vec.transform(te_texts)
        p_lgbm = lgbm_clf.predict_proba(X_tfidf_te)
        for i, cls in enumerate(le_lgbm.classes_.tolist()):
            j = cls_to_col.get(str(cls), -1)
            if j >= 0:
                p_lgbm_a[:, j] = p_lgbm[:, i]
        lgbm_ok = True
    except Exception as ex:
        logger.debug("LGBM fail %s/%s: %s", cat, attr, ex)

    p_r_ml = (p_xgb_a + p_lgbm_a) / 2.0 if lgbm_ok else p_xgb_a
    preds = [all_classes[i] for i in p_r_ml.argmax(axis=1)]
    return float(sum(p == g for p, g in zip(preds, y_te)) / len(y_te))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    gold = pd.read_parquet(GOLD_PATH)
    gold["code"] = gold["code"].astype(str)
    logger.info("Gold: %d rows, %d codes", len(gold), gold["code"].nunique())

    # Check for new-code embeddings (optional — from separate torch process)
    new_emb_path = PROCESSED / "al_new_codes_embeddings.npy"
    new_codes_path = PROCESSED / "al_new_codes_list.json"
    extra_emb: dict[str, np.ndarray] = {}
    extra_code_to_idx: dict[str, dict] = {}

    if new_emb_path.exists() and new_codes_path.exists():
        emb_all_new = np.load(new_emb_path)
        all_new_codes = json.loads(new_codes_path.read_text())
        logger.info("Loaded new-code embeddings: %d codes × %d dims",
                    len(all_new_codes), emb_all_new.shape[1])
        # All cats share same embedding matrix with combined codes
        base_idx = 0
        for cat in CATEGORIES:
            al = pd.read_csv(MANUAL_LABEL / f"al_codes_{cat}.csv")["code"].astype(str).tolist()
            ctrl = pd.read_csv(MANUAL_LABEL / f"al_control_codes_{cat}.csv")["code"].astype(str).tolist()
            cat_new = list(set(al) | set(ctrl))
            cat_idx_map = {}
            for c in cat_new:
                if c in all_new_codes:
                    cat_idx_map[c] = all_new_codes.index(c)
            extra_code_to_idx[cat] = cat_idx_map
            extra_emb[cat] = emb_all_new
    else:
        logger.warning("No new-code embeddings found at %s — XGB won't use extra gold", new_emb_path)
        for cat in CATEGORIES:
            extra_emb[cat] = None
            extra_code_to_idx[cat] = None

    all_rows: list[dict] = []
    total_cost_usd = 0.0

    for cat in CATEGORIES:
        logger.info("=== %s ===", cat)

        silver = pd.read_parquet(PROCESSED / f"{cat}_stratified_silver_standard.parquet")
        silver["code"] = silver["code"].astype(str)
        emb_all = np.load(PROCESSED / f"{cat}_stratified_embeddings.npy")
        code_to_idx = {c: i for i, c in enumerate(silver["code"].tolist())}

        cat_gold = gold[gold["category"] == cat].copy()
        unique_codes = sorted(cat_gold["code"].unique().tolist())
        train_codes, test_codes = train_test_split(unique_codes, test_size=TEST_SIZE, random_state=SEED)
        train_set = set(train_codes)
        test_set = set(test_codes)
        logger.info("[%s] Split: %d train / %d test", cat, len(train_codes), len(test_codes))

        # Load annotations
        al_long = _load_long_annotations(PROCESSED / "al_gpt55_uncertain" / f"{cat}_gold.parquet", cat)
        ctrl_long = _load_long_annotations(PROCESSED / "al_gpt55_random" / f"{cat}_gold.parquet", cat)
        logger.info("[%s] AL annotations: %d rows, ctrl: %d rows", cat, len(al_long), len(ctrl_long))

        # Track costs
        for dir_name in ["al_gpt55_uncertain", "al_gpt55_random"]:
            p = PROCESSED / dir_name / f"{cat}_gold.parquet"
            if p.exists():
                df_c = pd.read_parquet(p)
                if "cost_usd" in df_c.columns:
                    total_cost_usd += float(df_c["cost_usd"].sum())

        # Load extra silver pool for texts
        al_codes = pd.read_csv(MANUAL_LABEL / f"al_codes_{cat}.csv")["code"].astype(str).tolist()
        ctrl_codes = pd.read_csv(MANUAL_LABEL / f"al_control_codes_{cat}.csv")["code"].astype(str).tolist()
        all_new_cat = list(set(al_codes) | set(ctrl_codes))

        # Build extra_silver_df from OFF parquet (product text fields only)
        cols = ["code", "product_name", "brands", "ingredients_text", "quantity"]
        off_pool = pd.read_parquet(
            WORKTREE_ROOT / "datasets" / "raw" / "en.openfoodfacts.org.products.parquet",
            columns=cols,
        )
        off_pool["code"] = off_pool["code"].astype(str)
        extra_silver_df = off_pool[off_pool["code"].isin(all_new_cat)].copy().reset_index(drop=True)

        attrs = sorted(cat_gold["attr"].unique().tolist())

        for attr in attrs:
            acc_base = _eval_attr(
                cat, attr, cat_gold, silver, emb_all, code_to_idx,
                train_set, test_set,
                extra_gold=None,
                extra_emb=None, extra_code_to_idx=None,
                extra_silver_df=None,
            )
            acc_al = _eval_attr(
                cat, attr, cat_gold, silver, emb_all, code_to_idx,
                train_set, test_set,
                extra_gold=al_long if len(al_long) > 0 else None,
                extra_emb=extra_emb.get(cat),
                extra_code_to_idx=extra_code_to_idx.get(cat),
                extra_silver_df=extra_silver_df,
            )
            acc_rand = _eval_attr(
                cat, attr, cat_gold, silver, emb_all, code_to_idx,
                train_set, test_set,
                extra_gold=ctrl_long if len(ctrl_long) > 0 else None,
                extra_emb=extra_emb.get(cat),
                extra_code_to_idx=extra_code_to_idx.get(cat),
                extra_silver_df=extra_silver_df,
            )
            logger.info("[%s/%s] base=%.3f AL=%.3f rand=%.3f",
                        cat, attr, acc_base, acc_al, acc_rand)

            al_lift = (acc_al - acc_base) * 100 if not (np.isnan(acc_al) or np.isnan(acc_base)) else float("nan")
            rand_lift = (acc_rand - acc_base) * 100 if not (np.isnan(acc_rand) or np.isnan(acc_base)) else float("nan")
            al_vs_rand = (acc_al - acc_rand) * 100 if not (np.isnan(acc_al) or np.isnan(acc_rand)) else float("nan")
            all_rows.append({
                "category": cat, "attr": attr,
                "baseline_R_ml": acc_base,
                "AL_R_ml": acc_al,
                "random_R_ml": acc_rand,
                "AL_lift_pp": al_lift,
                "random_lift_pp": rand_lift,
                "AL_vs_random_pp": al_vs_rand,
            })

    result = pd.DataFrame(all_rows)
    result.to_parquet(OUT_PATH, index=False)
    logger.info("Wrote %d rows to %s", len(result), OUT_PATH)

    # Print summary
    print("\n" + "=" * 72)
    print("P2-EXP11: Active Learning Pilot — Results")
    print("=" * 72)
    print(f"Total annotation cost: ${total_cost_usd:.3f}")
    valid = result.dropna(subset=["baseline_R_ml", "AL_R_ml", "random_R_ml"])
    if len(valid) == 0:
        print("No valid rows!")
        return
    grand = valid[["baseline_R_ml", "AL_R_ml", "random_R_ml", "AL_lift_pp", "random_lift_pp", "AL_vs_random_pp"]].mean()
    print(f"  baseline_R_ml : {grand['baseline_R_ml']*100:.2f}%")
    print(f"  AL_R_ml       : {grand['AL_R_ml']*100:.2f}%  ({grand['AL_lift_pp']:+.2f} pp vs baseline)")
    print(f"  random_R_ml   : {grand['random_R_ml']*100:.2f}%  ({grand['random_lift_pp']:+.2f} pp vs baseline)")
    print(f"  AL vs random  : {grand['AL_vs_random_pp']:+.2f} pp")
    verdict = "AL WINS" if grand["AL_vs_random_pp"] > 0.3 else (
        "MARGINAL" if grand["AL_vs_random_pp"] > 0 else "RANDOM WINS / NO CLEAR WINNER"
    )
    print(f"  VERDICT: {verdict}")
    print()
    for cat in CATEGORIES:
        cr = valid[valid["category"] == cat]
        if len(cr) == 0:
            continue
        m = cr[["baseline_R_ml", "AL_R_ml", "random_R_ml", "AL_vs_random_pp"]].mean()
        print(f"  [{cat}] base={m['baseline_R_ml']*100:.2f}% AL={m['AL_R_ml']*100:.2f}% "
              f"rand={m['random_R_ml']*100:.2f}% AL_vs_rand={m['AL_vs_random_pp']:+.2f}pp")
    print()
    print(result.set_index(["category", "attr"])[
        ["baseline_R_ml", "AL_R_ml", "random_R_ml", "AL_vs_random_pp"]
    ].round(4).to_string())


if __name__ == "__main__":
    main()
