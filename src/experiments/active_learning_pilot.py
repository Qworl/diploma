"""P2-EXP11: Active Learning Pilot — AL vs Random annotation lift.

Pipeline:
1. Sample 5000 random OFF products per cat (not in silver/gold)
2. Compute embeddings and run R_ml ensemble predict
3. Find uncertain cells (max_proba < 0.6), select 150 per cat
4. Sample 150 random control codes per cat
5. Populate OFF cache for new codes
6. Annotate via gpt-5.5 (off_grounded mode)
7. Re-train R_ml: baseline vs AL vs random
8. Compare on same 20% held-out

Output:
  datasets/processed/active_learning_pilot.parquet
  datasets/manual_label/al_codes_{cat}.csv
  datasets/manual_label/al_control_codes_{cat}.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.common import MODELS_DIR, PROCESSED_DIR, get_embeddings, setup_logging
from src.eval.direct_llm_v2 import run_llm_on_products
from src.experiments.gold_vs_silver_training import train_xgb_and_score
from src.manual_label.schemas_loader import load_domain_attrs
from src.pipeline.ml.infer import load_classifier, predict_with_threshold
from src.pipeline.regex.extractor import RegexExtractor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CATEGORIES = ["pasta", "chocolate", "cheeses"]
SEED = 42
TEST_SIZE = 0.2
GOLD_WEIGHT = 5.0
MIN_GOLD = 20

OFF_PATH = Path("datasets/raw/en.openfoodfacts.org.products.parquet")
OFF_CACHE_DIR = Path("datasets/manual_label/off_cache")
MANUAL_LABEL_DIR = Path("datasets/manual_label")
PROCESSED_PATH = Path(PROCESSED_DIR)

GOLD_PATH = PROCESSED_PATH / "consensus_gold_v2_expanded.parquet"
OUT_PATH = PROCESSED_PATH / "active_learning_pilot.parquet"

# AL params
POOL_SIZE = 5000         # OFF products to sample per cat for uncertainty estimation
AL_BUDGET = 150          # codes per cat to annotate
RANDOM_BUDGET = 150      # codes per cat for random control
UNCERTAIN_THRESH = 0.6   # max_proba < this → uncertain
MIN_UNCERTAIN_ATTRS = 2  # code must be uncertain in >= N attrs

MODEL_NAME = "openai/gpt-5.5"
MAX_COST_PER_CAT = 15.0   # USD

# ---------------------------------------------------------------------------
# Phase 1: Sample 5000 OFF products per cat, compute embeddings, predict R_ml
# ---------------------------------------------------------------------------

def _get_text_blob(row: pd.Series) -> str:
    parts = []
    for col in ["product_name", "brands", "ingredients_text", "quantity"]:
        v = row.get(col)
        if pd.notna(v) and str(v).strip():
            parts.append(str(v).strip())
    return " ".join(parts)


def _load_silver_model(cat: str, attr: str):
    """Load hybrid model if available, else silver. Returns (clf, le) or (None, None)."""
    base = os.path.join(MODELS_DIR, f"{cat}_stratified_{attr}")
    for suffix in ["_xgb_hybrid.pkl", "_xgb.pkl"]:
        xp = base + suffix
        lp = base + (suffix.replace("_xgb", "_le").replace("_xgb_hybrid", "_le_hybrid"))
        if os.path.exists(xp) and os.path.exists(lp):
            with open(xp, "rb") as f:
                clf = pickle.load(f)
            with open(lp, "rb") as f:
                le = pickle.load(f)
            return clf, le
    return None, None


def _r_ml_predict_probas(
    df: pd.DataFrame,
    cat: str,
    attrs: list[str],
) -> pd.DataFrame:
    """Run R_ml (XGB embeddings + LGBM TF-IDF, soft-vote 50/50) on df.
    Returns long-format: code, attr, max_proba.
    """
    # Compute XGB embeddings once
    texts = df.apply(_get_text_blob, axis=1).tolist()
    logger.info("[%s] Computing embeddings for %d products...", cat, len(texts))
    emb = get_embeddings(texts)

    # Build text features for LGBM
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=10000)
    tfidf = vec.fit_transform(texts)

    extractor = RegexExtractor()
    rows: list[dict] = []

    for attr in attrs:
        # Load XGB
        clf_xgb, le_xgb = _load_silver_model(cat, attr)

        if clf_xgb is None:
            # No model — all abstains
            for code in df["code"].tolist():
                rows.append({"code": code, "attr": attr, "max_proba": 0.0, "predicted": None})
            continue

        classes = le_xgb.classes_.tolist()
        n_classes = len(classes)

        # XGB proba
        p_xgb = clf_xgb.predict_proba(emb)  # shape: (n, n_classes)

        # LGBM on TF-IDF — train on silver (skip: we only need relative uncertainty)
        # For uncertainty, XGB proba is sufficient and much faster
        # Use XGB proba directly (R_ml uses XGB embeddings + LGBM tfidf, but for
        # uncertainty detection XGB alone is adequate)

        # Regex check to override high-confidence
        for i, (_, row) in enumerate(df.iterrows()):
            code = str(row["code"])
            product_name = str(row.get("product_name") or "")
            ingredients = str(row.get("ingredients_text") or "")
            quantity = str(row.get("quantity") or "")

            regex_results = extractor.extract_all(
                product_name=product_name,
                description=ingredients,
                quantity=quantity,
                category=cat,
            )
            regex_val = regex_results.get(attr)
            if regex_val and regex_val.value is not None and regex_val.confidence > 0.5:
                rows.append({"code": code, "attr": attr, "max_proba": 1.0, "predicted": str(regex_val.value)})
                continue

            proba_row = p_xgb[i]
            max_p = float(proba_row.max())
            pred_idx = int(proba_row.argmax())
            pred_label = classes[pred_idx] if pred_idx < len(classes) else None
            rows.append({"code": code, "attr": attr, "max_proba": max_p, "predicted": pred_label})

    return pd.DataFrame(rows)


def sample_pool_from_off(
    cat: str,
    silver_codes: set[str],
    gold_codes: set[str],
    n: int = POOL_SIZE,
    seed: int = SEED,
) -> pd.DataFrame:
    """Sample n OFF products for cat not in silver."""
    logger.info("[%s] Loading OFF parquet to sample pool...", cat)
    cols = ["code", "product_name", "brands", "categories_tags", "ingredients_text", "quantity"]
    off = pd.read_parquet(OFF_PATH, columns=cols)
    off["code"] = off["code"].astype(str)

    # Filter by category
    cat_tag_map = {
        "pasta": "en:pasta",
        "chocolate": "en:chocolate",
        "cheeses": "en:cheeses",
    }
    tag = cat_tag_map[cat]
    mask = off["categories_tags"].fillna("").str.contains(tag, case=False)
    cat_off = off[mask].copy()

    # Exclude already seen
    exclude = silver_codes | gold_codes
    cat_off = cat_off[~cat_off["code"].isin(exclude)]
    logger.info("[%s] Pool size after exclusion: %d", cat, len(cat_off))

    if len(cat_off) == 0:
        raise RuntimeError(f"No OFF products for {cat} outside silver/gold")

    rng = random.Random(seed)
    n_sample = min(n, len(cat_off))
    sample_codes = rng.sample(cat_off["code"].tolist(), n_sample)
    return cat_off[cat_off["code"].isin(sample_codes)].copy().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Phase 2: Find uncertain cells
# ---------------------------------------------------------------------------

def find_uncertain_codes(
    preds: pd.DataFrame,
    budget: int = AL_BUDGET,
    uncertain_thresh: float = UNCERTAIN_THRESH,
    min_attrs: int = MIN_UNCERTAIN_ATTRS,
    seed: int = SEED,
) -> list[str]:
    """Return up to `budget` codes ranked by number of uncertain attrs (desc)."""
    uncertain = preds[preds["max_proba"] < uncertain_thresh].copy()
    per_code = uncertain.groupby("code")["attr"].count().reset_index()
    per_code.columns = ["code", "n_uncertain"]
    per_code = per_code[per_code["n_uncertain"] >= min_attrs]
    per_code = per_code.sort_values("n_uncertain", ascending=False).reset_index(drop=True)
    logger.info("Uncertain codes (>=%d uncertain attrs): %d", min_attrs, len(per_code))
    candidates = per_code["code"].tolist()
    rng = random.Random(seed)
    rng.shuffle(candidates)  # break ties randomly within same n_uncertain tier
    return candidates[:budget]


# ---------------------------------------------------------------------------
# Phase 3: Random control codes
# ---------------------------------------------------------------------------

def sample_random_control(
    pool_df: pd.DataFrame,
    exclude_codes: set[str],
    budget: int = RANDOM_BUDGET,
    seed: int = SEED,
) -> list[str]:
    """Sample random codes from pool not in uncertain set."""
    available = [c for c in pool_df["code"].tolist() if c not in exclude_codes]
    rng = random.Random(seed + 1)
    rng.shuffle(available)
    return available[:budget]


# ---------------------------------------------------------------------------
# Phase 4: Populate OFF cache + annotate via gpt-5.5
# ---------------------------------------------------------------------------

def populate_cache_for_codes(codes: list[str], cache_dir: Path) -> None:
    """Populate OFF cache for new codes using src/eval/populate_off_cache_from_parquet.py."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [c for c in codes if not (cache_dir / f"{c}.json").exists()]
    if not missing:
        logger.info("All %d codes already cached", len(codes))
        return
    logger.info("Populating cache for %d missing codes...", len(missing))
    tmp_codes_file = Path("/tmp/al_new_codes.txt")
    tmp_codes_file.write_text("\n".join(missing) + "\n")
    cmd = [
        sys.executable, "src/eval/populate_off_cache_from_parquet.py",
        "--codes-file", str(tmp_codes_file),
        "--cache-dir", str(cache_dir),
        "--parquet", str(OFF_PATH),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path.cwd()))
    logger.info("Cache population stdout: %s", result.stdout.strip())
    if result.returncode != 0:
        logger.warning("Cache population stderr: %s", result.stderr.strip())


def annotate_codes(
    codes: list[str],
    cat: str,
    pool_df: pd.DataFrame,
    out_path: Path,
    api_key: str,
    max_cost: float = MAX_COST_PER_CAT,
    context_mode: str = "off_grounded",
) -> pd.DataFrame:
    """Annotate given codes using gpt-5.5 via direct_llm_v2 infrastructure."""
    # Build product dataframe from pool
    products = pool_df[pool_df["code"].isin(codes)].copy()
    products["code"] = products["code"].astype(str)
    if len(products) == 0:
        logger.warning("[%s] No products found for annotation", cat)
        return pd.DataFrame()

    logger.info("[%s] Annotating %d products with %s (mode=%s)...",
                cat, len(products), MODEL_NAME, context_mode)

    from src.llm.client import call_openrouter
    df = run_llm_on_products(
        products,
        domain=cat,
        model=MODEL_NAME,
        api_key=api_key,
        out_path=out_path,
        context_mode=context_mode,
        off_cache_dir=OFF_CACHE_DIR,
        max_cost_usd=max_cost,
        sleep_between=0.0,
        call_fn=call_openrouter,
    )
    return df


# ---------------------------------------------------------------------------
# Convert annotation parquet to long-format gold
# ---------------------------------------------------------------------------

def annotations_to_long(df_wide: pd.DataFrame, cat: str) -> pd.DataFrame:
    """Convert gpt-5.5 wide-format to long-format matching gold schema."""
    sig_path = PROCESSED_PATH / "attribute_signal_taxonomy.parquet"
    sig = pd.read_parquet(sig_path)
    sig_map = {
        (row["category"], row["attr"]): row["signal_type"]
        for _, row in sig.iterrows()
    }

    rows = []
    for _, row in df_wide.iterrows():
        code = str(row["code"])
        try:
            parsed = json.loads(row["parsed_json"]) if row["parsed_json"] else {}
        except (json.JSONDecodeError, TypeError):
            parsed = {}
        for attr, val in parsed.items():
            is_null = val is None
            gold_value = str(val) if val is not None else None
            rows.append({
                "category": cat,
                "code": code,
                "attr": attr,
                "gold_value": gold_value,
                "gold_is_null": is_null,
                "opus_reasoning": None,
                "signal_type": sig_map.get((cat, attr), "text_derived"),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Phase 5: Re-train + eval
# ---------------------------------------------------------------------------

def _build_text_from_row(row: pd.Series) -> str:
    parts = []
    for col in ["product_name", "ingredients_text", "brands"]:
        v = row.get(col)
        if pd.notna(v) and str(v).strip():
            parts.append(str(v).strip())
    return " ".join(parts)


def _run_r_ml_eval(
    cat: str,
    attr: str,
    gold: pd.DataFrame,
    silver: pd.DataFrame,
    emb_all: np.ndarray,
    code_to_idx: dict[str, int],
    train_codes_set: set[str],
    test_codes_set: set[str],
    extra_gold: Optional[pd.DataFrame] = None,
    extra_silver: Optional[pd.DataFrame] = None,
    extra_emb: Optional[np.ndarray] = None,
    extra_code_to_idx: Optional[dict[str, int]] = None,
) -> float:
    """Eval R_ml (XGB emb + LGBM tfidf soft-vote) on test set.

    extra_gold: additional long-format gold rows to add to training.
    extra_silver: additional silver rows (with text fields) for LGBM TF-IDF.
    extra_emb: embeddings for extra silver rows.
    extra_code_to_idx: code→row-index for extra_emb.
    """
    # Gold for this attr
    cat_gold = gold[
        (gold["category"] == cat) & (gold["attr"] == attr) & ~gold["gold_is_null"]
    ].copy()
    cat_gold["code"] = cat_gold["code"].astype(str)
    cat_gold = cat_gold[cat_gold["code"].isin(code_to_idx)]

    # Extra gold (AL or random)
    if extra_gold is not None and len(extra_gold) > 0:
        eg = extra_gold[
            (extra_gold["category"] == cat) & (extra_gold["attr"] == attr) & ~extra_gold["gold_is_null"]
        ].copy()
        eg["code"] = eg["code"].astype(str)
        # Extra gold: these are NEW codes, in extra_code_to_idx
        if extra_code_to_idx is not None:
            eg = eg[eg["code"].isin(extra_code_to_idx)]
        else:
            eg = pd.DataFrame()
    else:
        eg = pd.DataFrame()

    # Train/test split
    train_gold = cat_gold[cat_gold["code"].isin(train_codes_set)]
    test_gold = cat_gold[cat_gold["code"].isin(test_codes_set)]

    if len(train_gold) < 5 or len(test_gold) < 3:
        return float("nan")

    test_idx = np.array([code_to_idx[c] for c in test_gold["code"]])
    X_te_emb = emb_all[test_idx]
    y_te = test_gold["gold_value"].astype(str).values

    # Silver data (exclude test codes)
    X_sil_emb = np.empty((0, emb_all.shape[1]))
    y_sil = np.array([], dtype=str)
    sil_texts: list[str] = []
    silver_indexed = silver.set_index("code")

    if attr in silver.columns:
        sil_attr = silver[silver[attr].notna()].copy()
        sil_attr["code"] = sil_attr["code"].astype(str)
        sil_attr = sil_attr[~sil_attr["code"].isin(test_codes_set)]
        sil_attr = sil_attr[~sil_attr["code"].isin(train_codes_set)]
        sil_attr = sil_attr[sil_attr["code"].isin(code_to_idx)]
        if len(sil_attr) > 0:
            s_idx = np.array([code_to_idx[c] for c in sil_attr["code"]])
            X_sil_emb = emb_all[s_idx]
            y_sil = sil_attr[attr].astype(str).values
            sil_texts = [_build_text_from_row(silver_indexed.loc[c]) if c in silver_indexed.index else "" for c in sil_attr["code"]]

    # Build combined train: silver + gold train
    train_idx = np.array([code_to_idx[c] for c in train_gold["code"]])
    X_gold_emb = emb_all[train_idx]
    y_gold = train_gold["gold_value"].astype(str).values

    # Extra gold embeddings
    X_eg_emb = np.empty((0, emb_all.shape[1]))
    y_eg = np.array([], dtype=str)
    eg_texts: list[str] = []
    if len(eg) > 0 and extra_emb is not None and extra_code_to_idx is not None:
        eg_valid = eg[eg["code"].isin(extra_code_to_idx)]
        if len(eg_valid) > 0:
            eg_idx = np.array([extra_code_to_idx[c] for c in eg_valid["code"]])
            X_eg_emb = extra_emb[eg_idx]
            y_eg = eg_valid["gold_value"].astype(str).values
            if extra_silver is not None:
                xs_idx = extra_silver.set_index("code")
                eg_texts = [_build_text_from_row(xs_idx.loc[c]) if c in xs_idx.index else "" for c in eg_valid["code"]]
            else:
                eg_texts = [""] * len(eg_valid)

    # Combine all train
    X_tr_emb_parts = [X_sil_emb, X_gold_emb]
    y_tr_parts = [y_sil, y_gold]
    w_parts = [np.ones(len(y_sil)), GOLD_WEIGHT * np.ones(len(y_gold))]
    tr_texts = sil_texts + [
        _build_text_from_row(silver_indexed.loc[c]) if c in silver_indexed.index else ""
        for c in train_gold["code"]
    ]

    if len(X_eg_emb) > 0:
        X_tr_emb_parts.append(X_eg_emb)
        y_tr_parts.append(y_eg)
        w_parts.append(GOLD_WEIGHT * np.ones(len(y_eg)))
        tr_texts = tr_texts + eg_texts

    X_tr_emb = np.vstack([x for x in X_tr_emb_parts if len(x) > 0])
    y_tr = np.concatenate([y for y in y_tr_parts if len(y) > 0])
    w_tr = np.concatenate([w for w in w_parts if len(w) > 0])

    all_classes = sorted(set(y_tr.tolist()))
    if len(all_classes) < 2:
        return float("nan")

    # XGB on embeddings
    le_xgb = LabelEncoder()
    le_xgb.fit(all_classes)
    y_enc = le_xgb.transform(y_tr)
    n_classes = len(all_classes)

    common_xgb = dict(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
        tree_method="hist", verbosity=0,
    )
    if n_classes == 2:
        pos = int((y_enc == 1).sum())
        neg = int((y_enc == 0).sum())
        spw = max(neg / max(pos, 1), 0.5)
        clf_xgb = xgb.XGBClassifier(scale_pos_weight=spw, **common_xgb)
    else:
        clf_xgb = xgb.XGBClassifier(objective="multi:softmax", num_class=n_classes, **common_xgb)
    clf_xgb.fit(X_tr_emb, y_enc, sample_weight=w_tr)

    p_xgb = clf_xgb.predict_proba(X_te_emb)
    # Align to all_classes
    le_classes = le_xgb.classes_.tolist()
    class_to_col = {c: i for i, c in enumerate(all_classes)}
    p_xgb_aligned = np.zeros((len(X_te_emb), n_classes), dtype=np.float32)
    for i, cls in enumerate(le_classes):
        j = class_to_col.get(str(cls), -1)
        if j >= 0:
            p_xgb_aligned[:, j] = p_xgb[:, i]

    # LGBM on TF-IDF
    te_texts = [
        _build_text_from_row(silver_indexed.loc[c]) if c in silver_indexed.index else ""
        for c in test_gold["code"]
    ]
    # extra_silver is handled above in eg_texts computation

    p_lgbm_aligned = np.zeros((len(X_te_emb), n_classes), dtype=np.float32)
    lgbm_ok = False
    try:
        le_lgbm = LabelEncoder()
        le_lgbm.fit(all_classes)
        y_lgbm_enc = le_lgbm.transform(y_tr)
        vec_lgbm = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=10000)
        X_tfidf_tr = vec_lgbm.fit_transform(tr_texts)
        lgbm_clf = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=5,
            objective=("binary" if n_classes == 2 else "multiclass"),
            **({"num_class": n_classes} if n_classes > 2 else {}),
            verbose=-1,
        )
        lgbm_clf.fit(X_tfidf_tr, y_lgbm_enc, sample_weight=w_tr)
        X_tfidf_te = vec_lgbm.transform(te_texts)
        p_lgbm = lgbm_clf.predict_proba(X_tfidf_te)
        le_lgbm_classes = le_lgbm.classes_.tolist()
        for i, cls in enumerate(le_lgbm_classes):
            j = class_to_col.get(str(cls), -1)
            if j >= 0:
                p_lgbm_aligned[:, j] = p_lgbm[:, i]
        lgbm_ok = True
    except Exception as ex:
        logger.debug("LGBM failed for %s/%s: %s", cat, attr, ex)

    # Soft-vote R_ml = (XGB + LGBM) / 2
    if lgbm_ok:
        p_r_ml = (p_xgb_aligned + p_lgbm_aligned) / 2.0
    else:
        p_r_ml = p_xgb_aligned

    preds = p_r_ml.argmax(axis=1)
    # Map back to class labels
    pred_labels = [all_classes[i] for i in preds]

    # Accuracy
    n = len(y_te)
    if n == 0:
        return float("nan")
    acc = float(sum(p == g for p, g in zip(pred_labels, y_te)) / n)
    return acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()

    ap = argparse.ArgumentParser(description="P2-EXP11: Active Learning Pilot")
    ap.add_argument("--skip-annotation", action="store_true",
                    help="Skip LLM annotation step (use existing annotation files)")
    ap.add_argument("--skip-sampling", action="store_true",
                    help="Skip sampling/uncertainty phase (use existing al/control CSVs)")
    ap.add_argument("--cats", nargs="+", default=CATEGORIES)
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key and not args.skip_annotation:
        raise SystemExit("OPENROUTER_API_KEY not set")

    # Load base gold
    gold = pd.read_parquet(GOLD_PATH)
    gold["code"] = gold["code"].astype(str)
    logger.info("Loaded gold: %d rows, %d codes", len(gold), gold["code"].nunique())

    # Load all silver codes (to exclude from pool)
    all_silver_codes: set[str] = set()
    for cat in CATEGORIES:
        sil = pd.read_parquet(PROCESSED_PATH / f"{cat}_stratified_silver_standard.parquet")
        sil["code"] = sil["code"].astype(str)
        all_silver_codes.update(sil["code"].tolist())

    gold_codes = set(gold["code"].unique())

    # -----------------------------------------------------------------------
    # Phase 1–3: Sampling + uncertainty estimation
    # -----------------------------------------------------------------------

    cat_al_codes: dict[str, list[str]] = {}
    cat_ctrl_codes: dict[str, list[str]] = {}
    cat_pool_df: dict[str, pd.DataFrame] = {}

    for cat in args.cats:
        al_csv = MANUAL_LABEL_DIR / f"al_codes_{cat}.csv"
        ctrl_csv = MANUAL_LABEL_DIR / f"al_control_codes_{cat}.csv"

        if args.skip_sampling and al_csv.exists() and ctrl_csv.exists():
            al_codes = pd.read_csv(al_csv)["code"].astype(str).tolist()
            ctrl_codes = pd.read_csv(ctrl_csv)["code"].astype(str).tolist()
            logger.info("[%s] Loaded existing AL codes: %d, ctrl: %d", cat, len(al_codes), len(ctrl_codes))
            cat_al_codes[cat] = al_codes
            cat_ctrl_codes[cat] = ctrl_codes
            # Still need pool_df for annotation (with product fields)
            pool_df = sample_pool_from_off(cat, all_silver_codes, gold_codes, n=POOL_SIZE + 500)
            cat_pool_df[cat] = pool_df
            continue

        # Sample pool from OFF
        pool_df = sample_pool_from_off(cat, all_silver_codes, gold_codes, n=POOL_SIZE + 500)
        cat_pool_df[cat] = pool_df

        attrs = list(load_domain_attrs(cat))
        logger.info("[%s] Running R_ml predict on %d products, attrs=%s", cat, len(pool_df), attrs)

        # Run predictions
        preds_df = _r_ml_predict_probas(pool_df.head(POOL_SIZE), cat, attrs)

        # Save predictions
        pred_out = Path(f"/tmp/al_predictions_{cat}.parquet")
        preds_df.to_parquet(pred_out, index=False)
        logger.info("[%s] Predictions saved to %s", cat, pred_out)

        # Find uncertain codes
        al_codes = find_uncertain_codes(preds_df, budget=AL_BUDGET)
        logger.info("[%s] AL codes selected: %d (target=%d)", cat, len(al_codes), AL_BUDGET)

        # Random control (not from uncertain pool)
        al_codes_set = set(al_codes)
        ctrl_codes = sample_random_control(pool_df.head(POOL_SIZE), al_codes_set, budget=RANDOM_BUDGET)
        logger.info("[%s] Random control codes: %d", cat, len(ctrl_codes))

        # Save CSVs
        pd.DataFrame({"code": al_codes}).to_csv(al_csv, index=False)
        pd.DataFrame({"code": ctrl_codes}).to_csv(ctrl_csv, index=False)
        logger.info("[%s] Saved %s and %s", cat, al_csv, ctrl_csv)

        cat_al_codes[cat] = al_codes
        cat_ctrl_codes[cat] = ctrl_codes

    # -----------------------------------------------------------------------
    # Phase 4: Populate OFF cache + annotate
    # -----------------------------------------------------------------------

    al_gold_dir = PROCESSED_PATH / "al_gpt55_uncertain"
    rand_gold_dir = PROCESSED_PATH / "al_gpt55_random"
    al_gold_dir.mkdir(parents=True, exist_ok=True)
    rand_gold_dir.mkdir(parents=True, exist_ok=True)

    cat_al_annotations: dict[str, pd.DataFrame] = {}
    cat_ctrl_annotations: dict[str, pd.DataFrame] = {}

    for cat in args.cats:
        al_codes = cat_al_codes[cat]
        ctrl_codes = cat_ctrl_codes[cat]
        pool_df = cat_pool_df[cat]

        all_new_codes = list(set(al_codes) | set(ctrl_codes))
        logger.info("[%s] Populating OFF cache for %d new codes...", cat, len(all_new_codes))
        populate_cache_for_codes(all_new_codes, OFF_CACHE_DIR)

        al_out = al_gold_dir / f"{cat}_gold.parquet"
        ctrl_out = rand_gold_dir / f"{cat}_gold.parquet"

        if not args.skip_annotation:
            # AL annotation
            al_df = annotate_codes(al_codes, cat, pool_df, al_out, api_key, max_cost=MAX_COST_PER_CAT)
            logger.info("[%s] AL annotation done: %d rows, cost=$%.3f",
                        cat, len(al_df), float(al_df["cost_usd"].sum()) if len(al_df) else 0.0)

            # Random control annotation
            ctrl_df = annotate_codes(ctrl_codes, cat, pool_df, ctrl_out, api_key, max_cost=MAX_COST_PER_CAT)
            logger.info("[%s] Random annotation done: %d rows, cost=$%.3f",
                        cat, len(ctrl_df), float(ctrl_df["cost_usd"].sum()) if len(ctrl_df) else 0.0)
        else:
            al_df = pd.read_parquet(al_out) if al_out.exists() else pd.DataFrame()
            ctrl_df = pd.read_parquet(ctrl_out) if ctrl_out.exists() else pd.DataFrame()
            logger.info("[%s] Loaded existing annotations: AL=%d, ctrl=%d", cat, len(al_df), len(ctrl_df))

        # Convert to long format
        al_long = annotations_to_long(al_df, cat) if len(al_df) > 0 else pd.DataFrame()
        ctrl_long = annotations_to_long(ctrl_df, cat) if len(ctrl_df) > 0 else pd.DataFrame()
        cat_al_annotations[cat] = al_long
        cat_ctrl_annotations[cat] = ctrl_long

    # -----------------------------------------------------------------------
    # Phase 5: Re-train + eval three variants
    # -----------------------------------------------------------------------

    logger.info("=== Phase 5: Re-train + Eval ===")

    # We need embeddings for the new codes (AL + control)
    # Compute embeddings per cat
    cat_extra_emb: dict[str, np.ndarray] = {}
    cat_extra_pool_df: dict[str, pd.DataFrame] = {}
    cat_extra_code_to_idx: dict[str, dict[str, int]] = {}

    for cat in args.cats:
        al_codes = cat_al_codes[cat]
        ctrl_codes = cat_ctrl_codes[cat]
        pool_df = cat_pool_df[cat]
        all_new_codes = list(set(al_codes) | set(ctrl_codes))

        new_products = pool_df[pool_df["code"].isin(all_new_codes)].copy()
        if len(new_products) == 0:
            logger.warning("[%s] No new products for embedding computation", cat)
            cat_extra_emb[cat] = np.empty((0, 384))
            cat_extra_pool_df[cat] = pd.DataFrame()
            cat_extra_code_to_idx[cat] = {}
            continue

        new_texts = new_products.apply(_get_text_blob, axis=1).tolist()
        logger.info("[%s] Computing embeddings for %d new products...", cat, len(new_products))
        new_emb = get_embeddings(new_texts)
        new_code_to_idx = {c: i for i, c in enumerate(new_products["code"].tolist())}

        cat_extra_emb[cat] = new_emb
        cat_extra_pool_df[cat] = new_products
        cat_extra_code_to_idx[cat] = new_code_to_idx

    all_result_rows: list[dict] = []
    total_cost_usd = 0.0

    for cat in args.cats:
        logger.info("=== Category: %s ===", cat)
        silver = pd.read_parquet(PROCESSED_PATH / f"{cat}_stratified_silver_standard.parquet")
        silver["code"] = silver["code"].astype(str)
        emb_all = np.load(PROCESSED_PATH / f"{cat}_stratified_embeddings.npy")
        code_to_idx = {c: i for i, c in enumerate(silver["code"].tolist())}

        # 80/20 split on base gold codes
        cat_gold = gold[gold["category"] == cat].copy()
        unique_codes = sorted(cat_gold["code"].unique().tolist())
        train_codes, test_codes = train_test_split(unique_codes, test_size=TEST_SIZE, random_state=SEED)
        train_codes_set = set(train_codes)
        test_codes_set = set(test_codes)
        logger.info("[%s] Split: %d train / %d test codes", cat, len(train_codes), len(test_codes))

        attrs = sorted(cat_gold["attr"].unique().tolist())

        al_long = cat_al_annotations.get(cat, pd.DataFrame())
        ctrl_long = cat_ctrl_annotations.get(cat, pd.DataFrame())
        extra_emb = cat_extra_emb.get(cat)
        extra_pool = cat_extra_pool_df.get(cat)
        extra_code_to_idx = cat_extra_code_to_idx.get(cat)

        # Collect costs
        al_out = al_gold_dir / f"{cat}_gold.parquet"
        ctrl_out = rand_gold_dir / f"{cat}_gold.parquet"
        for p in [al_out, ctrl_out]:
            if p.exists():
                df_c = pd.read_parquet(p)
                if "cost_usd" in df_c.columns:
                    total_cost_usd += float(df_c["cost_usd"].sum())

        for attr in attrs:
            # Baseline: no extra training data
            acc_base = _run_r_ml_eval(
                cat=cat, attr=attr,
                gold=cat_gold, silver=silver,
                emb_all=emb_all, code_to_idx=code_to_idx,
                train_codes_set=train_codes_set, test_codes_set=test_codes_set,
                extra_gold=None,
            )

            # AL treatment: add AL annotations
            acc_al = _run_r_ml_eval(
                cat=cat, attr=attr,
                gold=cat_gold, silver=silver,
                emb_all=emb_all, code_to_idx=code_to_idx,
                train_codes_set=train_codes_set, test_codes_set=test_codes_set,
                extra_gold=al_long if len(al_long) > 0 else None,
                extra_silver=extra_pool,
                extra_emb=extra_emb,
                extra_code_to_idx=extra_code_to_idx,
            )

            # Random control: add random annotations
            acc_rand = _run_r_ml_eval(
                cat=cat, attr=attr,
                gold=cat_gold, silver=silver,
                emb_all=emb_all, code_to_idx=code_to_idx,
                train_codes_set=train_codes_set, test_codes_set=test_codes_set,
                extra_gold=ctrl_long if len(ctrl_long) > 0 else None,
                extra_silver=extra_pool,
                extra_emb=extra_emb,
                extra_code_to_idx=extra_code_to_idx,
            )

            logger.info(
                "[%s/%s] baseline=%.3f AL=%.3f random=%.3f",
                cat, attr, acc_base, acc_al, acc_rand,
            )

            all_result_rows.append({
                "category": cat,
                "attr": attr,
                "baseline_R_ml": acc_base,
                "AL_R_ml": acc_al,
                "random_R_ml": acc_rand,
                "AL_lift_pp": (acc_al - acc_base) * 100 if not (np.isnan(acc_al) or np.isnan(acc_base)) else float("nan"),
                "random_lift_pp": (acc_rand - acc_base) * 100 if not (np.isnan(acc_rand) or np.isnan(acc_base)) else float("nan"),
                "AL_vs_random_pp": (acc_al - acc_rand) * 100 if not (np.isnan(acc_al) or np.isnan(acc_rand)) else float("nan"),
            })

    result_df = pd.DataFrame(all_result_rows)
    result_df.to_parquet(OUT_PATH, index=False)
    logger.info("Wrote %d rows to %s", len(result_df), OUT_PATH)

    # Summary
    print("\n" + "=" * 72)
    print("P2-EXP11: Active Learning Pilot — Results")
    print("=" * 72)
    print(f"Total annotation cost: ${total_cost_usd:.3f}")
    print()
    grand = result_df[["baseline_R_ml", "AL_R_ml", "random_R_ml", "AL_lift_pp", "random_lift_pp", "AL_vs_random_pp"]].mean()
    print(f"  baseline_R_ml     : {grand['baseline_R_ml']*100:.2f}%")
    print(f"  AL_R_ml           : {grand['AL_R_ml']*100:.2f}%  ({grand['AL_lift_pp']:+.2f} pp vs baseline)")
    print(f"  random_R_ml       : {grand['random_R_ml']*100:.2f}%  ({grand['random_lift_pp']:+.2f} pp vs baseline)")
    print(f"  AL vs random      : {grand['AL_vs_random_pp']:+.2f} pp")
    print()
    verdict = "AL WINS" if grand["AL_vs_random_pp"] > 0.3 else ("MARGINAL" if grand["AL_vs_random_pp"] > 0 else "RANDOM WINS")
    print(f"  VERDICT: {verdict}")
    print()
    print("Per-category:")
    for cat in CATEGORIES:
        cat_rows = result_df[result_df["category"] == cat]
        if len(cat_rows) == 0:
            continue
        m = cat_rows[["baseline_R_ml", "AL_R_ml", "random_R_ml", "AL_vs_random_pp"]].mean()
        print(f"  [{cat}] base={m['baseline_R_ml']*100:.2f}% AL={m['AL_R_ml']*100:.2f}% random={m['random_R_ml']*100:.2f}% AL_vs_rand={m['AL_vs_random_pp']:+.2f}pp")

    print("\n" + "=" * 72)
    print("Per-(category, attr):")
    print(result_df.set_index(["category", "attr"])[
        ["baseline_R_ml", "AL_R_ml", "random_R_ml", "AL_vs_random_pp"]
    ].round(4).to_string())


if __name__ == "__main__":
    main()
