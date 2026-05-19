"""Partial-partner-data evaluation for OCR and CLIP signals.

Tests the user's hypothesis: when partner sends MINIMAL data
(product_name + brands + quantity, NO ingredients_text), how much do
OCR text augmentation and CLIP visual features actually help?

Variants per (cat, attr), 80/20 split (seed=42):

  baseline_min : LightGBM(TF-IDF on name+brand+qty)
  +ocr_min     : LightGBM(TF-IDF on name+brand+qty+OCR)
  +clip_min    : 0.5 * LightGBM(TF-IDF on name+brand+qty)
                 + 0.5 * XGB(CLIP visual 512-dim)
  +ocr+clip    : 0.5 * LightGBM(TF-IDF on name+brand+qty+OCR)
                 + 0.5 * XGB(CLIP visual)

Reference (full-partner control):
  baseline_full: LightGBM(TF-IDF on name+brand+qty+ingredients_text)

Output:
  datasets/processed/partial_partner_eval.parquet
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

WORKTREE_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = WORKTREE_ROOT / "datasets" / "processed"
OCR_CACHE_PATH = PROCESSED_DIR / "ocr_text_cache.json"
OUT_PATH = PROCESSED_DIR / "partial_partner_eval.parquet"

CATEGORIES = ["pasta", "cheeses", "chocolate"]
RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_TRAIN_SAMPLES = 10
MIN_CLASSES = 2


# ---------------------------------------------------------------------------
# Text builders
# ---------------------------------------------------------------------------

def build_minimal_text(row: pd.Series) -> str:
    """Partial-partner text: product_name + brands + quantity ONLY."""
    parts: list[str] = []
    for col in ["product_name", "brands", "quantity"]:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    return " ".join(parts)


def build_full_text(row: pd.Series) -> str:
    """Full-partner text (includes ingredients_text)."""
    parts: list[str] = []
    for col in ["product_name", "brands", "ingredients_text", "quantity"]:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    return " ".join(parts)


def build_minimal_plus_ocr(row: pd.Series, ocr_cache: dict[str, str]) -> str:
    base = build_minimal_text(row)
    code = str(row.get("code", "")).strip()
    code_stripped = code.lstrip("0") or code
    ocr = ocr_cache.get(code, "") or ocr_cache.get(code_stripped, "")
    return f"{base} {ocr}".strip() if ocr else base


# ---------------------------------------------------------------------------
# Classifier helpers
# ---------------------------------------------------------------------------

def train_lgbm(
    texts: list[str], y: list[str],
) -> tuple[Optional[lgb.LGBMClassifier], Optional[TfidfVectorizer], Optional[LabelEncoder]]:
    classes = sorted(set(y))
    if len(classes) < MIN_CLASSES or len(y) < MIN_TRAIN_SAMPLES:
        return None, None, None
    le = LabelEncoder()
    le.fit(classes)
    y_enc = le.transform(y)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=10_000, sublinear_tf=True)
    X = vec.fit_transform(texts)
    n = len(classes)
    kwargs: dict = dict(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        num_leaves=31, min_child_samples=5,
        objective="binary" if n == 2 else "multiclass", verbose=-1,
    )
    if n > 2:
        kwargs["num_class"] = n
    clf = lgb.LGBMClassifier(**kwargs)
    clf.fit(X, y_enc)
    return clf, vec, le


def train_xgb_on_visual(
    X: np.ndarray, y: list[str],
) -> tuple[Optional[xgb.XGBClassifier], Optional[LabelEncoder]]:
    classes = sorted(set(y))
    if len(classes) < MIN_CLASSES or len(y) < MIN_TRAIN_SAMPLES:
        return None, None
    le = LabelEncoder()
    le.fit(classes)
    y_enc = le.transform(y)
    n = len(classes)
    common: dict = dict(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
        tree_method="hist", verbosity=0,
    )
    if n == 2:
        pos = int((y_enc == 1).sum())
        neg = int((y_enc == 0).sum())
        spw = max(neg / max(pos, 1), 0.5)
        clf = xgb.XGBClassifier(scale_pos_weight=spw, **common)
    else:
        clf = xgb.XGBClassifier(objective="multi:softmax", num_class=n, **common)
    clf.fit(X, y_enc)
    return clf, le


def align_probas(
    lgbm_probas: np.ndarray, lgbm_le: LabelEncoder,
    other_probas: np.ndarray, other_le: LabelEncoder,
) -> tuple[np.ndarray, list[str]]:
    a = list(lgbm_le.classes_)
    b = list(other_le.classes_)
    merged = sorted(set(a) | set(b))
    n, k = lgbm_probas.shape[0], len(merged)
    A = np.zeros((n, k))
    for j, c in enumerate(a):
        A[:, merged.index(c)] = lgbm_probas[:, j]
    B = np.zeros((n, k))
    for j, c in enumerate(b):
        B[:, merged.index(c)] = other_probas[:, j]
    return 0.5 * A + 0.5 * B, merged


def lgbm_predict_acc(clf, vec, le, texts: list[str], y_true: list[str]) -> float:
    if clf is None:
        return float("nan")
    X = vec.transform(texts)
    enc = clf.predict(X)
    preds = le.inverse_transform(enc).tolist()
    return float(sum(p == g for p, g in zip(preds, y_true)) / len(y_true))


# ---------------------------------------------------------------------------
# Per-(cat, attr) evaluation
# ---------------------------------------------------------------------------

def evaluate_attr(
    cat: str, attr: str, df: pd.DataFrame,
    ocr_cache: dict[str, str],
    clip_emb: Optional[np.ndarray], clip_code_idx: Optional[dict[str, int]],
) -> Optional[dict]:
    if attr not in df.columns:
        return None
    sub = df[df[attr].notna()].copy()
    sub["code"] = sub["code"].astype(str)
    if len(sub) < MIN_TRAIN_SAMPLES * 2 or sub[attr].nunique() < MIN_CLASSES:
        return None

    y = sub[attr].astype(str).values.tolist()
    codes = sub["code"].tolist()

    try:
        tr, te = train_test_split(
            range(len(sub)), test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
        )
    except ValueError:
        tr, te = train_test_split(
            range(len(sub)), test_size=TEST_SIZE, random_state=RANDOM_STATE,
        )

    tr_sub, te_sub = sub.iloc[list(tr)], sub.iloc[list(te)]
    y_tr = [y[i] for i in tr]
    y_te = [y[i] for i in te]

    # Build all text variants
    tr_min = [build_minimal_text(r) for _, r in tr_sub.iterrows()]
    te_min = [build_minimal_text(r) for _, r in te_sub.iterrows()]
    tr_full = [build_full_text(r) for _, r in tr_sub.iterrows()]
    te_full = [build_full_text(r) for _, r in te_sub.iterrows()]
    tr_min_ocr = [build_minimal_plus_ocr(r, ocr_cache) for _, r in tr_sub.iterrows()]
    te_min_ocr = [build_minimal_plus_ocr(r, ocr_cache) for _, r in te_sub.iterrows()]

    # --- LightGBM-only variants (no ensemble) ---
    clf_min, vec_min, le_min = train_lgbm(tr_min, y_tr)
    acc_min = lgbm_predict_acc(clf_min, vec_min, le_min, te_min, y_te)

    clf_full, vec_full, le_full = train_lgbm(tr_full, y_tr)
    acc_full = lgbm_predict_acc(clf_full, vec_full, le_full, te_full, y_te)

    clf_ocr, vec_ocr, le_ocr = train_lgbm(tr_min_ocr, y_tr)
    acc_ocr = lgbm_predict_acc(clf_ocr, vec_ocr, le_ocr, te_min_ocr, y_te)

    # --- CLIP ensemble (LGBM minimal + XGB clip) ---
    acc_clip = float("nan")
    acc_ocr_clip = float("nan")
    n_test_with_clip = 0
    if clip_emb is not None and clip_code_idx is not None:
        tr_codes = [codes[i] for i in tr]
        te_codes = [codes[i] for i in te]
        tr_has_clip = [c in clip_code_idx for c in tr_codes]
        te_has_clip = [c in clip_code_idx for c in te_codes]
        n_test_with_clip = sum(te_has_clip)

        if sum(tr_has_clip) >= MIN_TRAIN_SAMPLES and sum(te_has_clip) >= 3:
            X_tr_clip = clip_emb[[clip_code_idx[c] for c in tr_codes if c in clip_code_idx]]
            y_tr_clip = [y_tr[i] for i, h in enumerate(tr_has_clip) if h]
            X_te_clip = clip_emb[[clip_code_idx[c] for c in te_codes if c in clip_code_idx]]
            te_min_clip = [t for t, h in zip(te_min, te_has_clip) if h]
            te_min_ocr_clip = [t for t, h in zip(te_min_ocr, te_has_clip) if h]
            y_te_clip = [y_te[i] for i, h in enumerate(te_has_clip) if h]

            clf_v, le_v = train_xgb_on_visual(X_tr_clip, y_tr_clip)
            if clf_v is not None and clf_min is not None:
                visual_probas = clf_v.predict_proba(X_te_clip)
                text_probas = clf_min.predict_proba(vec_min.transform(te_min_clip))
                merged, cls = align_probas(text_probas, le_min, visual_probas, le_v)
                preds = [cls[i] for i in np.argmax(merged, axis=1)]
                acc_clip = float(sum(p == g for p, g in zip(preds, y_te_clip)) / len(y_te_clip))

                if clf_ocr is not None:
                    text_ocr_probas = clf_ocr.predict_proba(vec_ocr.transform(te_min_ocr_clip))
                    merged2, cls2 = align_probas(text_ocr_probas, le_ocr, visual_probas, le_v)
                    preds2 = [cls2[i] for i in np.argmax(merged2, axis=1)]
                    acc_ocr_clip = float(
                        sum(p == g for p, g in zip(preds2, y_te_clip)) / len(y_te_clip)
                    )

    return {
        "category": cat,
        "attr": attr,
        "n_train": len(tr),
        "n_test": len(te),
        "n_test_with_clip": n_test_with_clip,
        "acc_baseline_full": acc_full,        # reference (partner sends everything)
        "acc_baseline_min": acc_min,           # baseline (no ingredients)
        "acc_min_plus_ocr": acc_ocr,           # OCR augmentation
        "acc_min_plus_clip": acc_clip,         # CLIP ensemble (clip-subset)
        "acc_min_plus_ocr_clip": acc_ocr_clip, # both signals
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not OCR_CACHE_PATH.exists():
        logger.error("OCR cache missing: %s", OCR_CACHE_PATH)
        return
    with open(OCR_CACHE_PATH, encoding="utf-8") as f:
        ocr_cache = json.load(f)
    logger.info("OCR cache: %d entries (%d non-empty)",
                len(ocr_cache), sum(1 for v in ocr_cache.values() if v.strip()))

    all_results: list[dict] = []
    for cat in CATEGORIES:
        silver_path = PROCESSED_DIR / f"{cat}_stratified_silver_standard.parquet"
        clip_emb_path = PROCESSED_DIR / f"clip_embeddings_{cat}.npy"
        clip_idx_path = PROCESSED_DIR / f"clip_code_index_{cat}.json"

        if not silver_path.exists():
            logger.warning("Missing silver for %s — skipping", cat)
            continue
        df = pd.read_parquet(silver_path)
        df["code"] = df["code"].astype(str)

        clip_emb = None
        clip_idx = None
        if clip_emb_path.exists() and clip_idx_path.exists():
            clip_emb = np.load(clip_emb_path)
            with open(clip_idx_path, encoding="utf-8") as f:
                raw_idx = json.load(f)
            clip_idx = {str(k): int(v) for k, v in raw_idx.items()}
            logger.info("CLIP for %s: %d codes, emb shape %s", cat, len(clip_idx), clip_emb.shape)
        else:
            logger.warning("CLIP missing for %s — only OCR variants will run", cat)

        meta_cols = {
            "code", "product_name", "brands", "ingredients_text", "quantity",
            "categories_tags", "labels_tags", "ingredients_analysis_tags",
            "traces_tags", "countries_tags",
        }
        attrs = [c for c in df.columns if c not in meta_cols]
        logger.info("=== %s: %d attrs ===", cat, len(attrs))

        for attr in attrs:
            res = evaluate_attr(cat, attr, df, ocr_cache, clip_emb, clip_idx)
            if res is None:
                continue
            all_results.append(res)
            logger.info(
                "  [%s/%s] full=%.3f min=%.3f +ocr=%.3f +clip=%.3f +both=%.3f n=%d clip_n=%d",
                cat, attr,
                res["acc_baseline_full"], res["acc_baseline_min"],
                res["acc_min_plus_ocr"], res["acc_min_plus_clip"],
                res["acc_min_plus_ocr_clip"], res["n_test"], res["n_test_with_clip"],
            )

    if not all_results:
        logger.error("No results")
        return

    out = pd.DataFrame(all_results)
    out.to_parquet(OUT_PATH, index=False)
    logger.info("Saved %d rows to %s", len(out), OUT_PATH)

    # --- Summary ---
    print(f"\n{'='*78}")
    print("PARTIAL-PARTNER EVAL — what does OCR/CLIP add when ingredients_text missing?")
    print(f"{'='*78}")

    def nanmean(col: str) -> float:
        return float(out[col].dropna().mean()) if out[col].notna().any() else float("nan")

    full = nanmean("acc_baseline_full")
    mn = nanmean("acc_baseline_min")
    ocr = nanmean("acc_min_plus_ocr")
    clip_only_rows = out[out["acc_min_plus_clip"].notna()]
    clip = float(clip_only_rows["acc_min_plus_clip"].mean()) if len(clip_only_rows) else float("nan")
    clip_base_subset = (
        float(clip_only_rows["acc_baseline_min"].mean()) if len(clip_only_rows) else float("nan")
    )
    both = float(clip_only_rows["acc_min_plus_ocr_clip"].mean()) if len(clip_only_rows) else float("nan")

    print(f"\nMean accuracy across {len(out)} (cat, attr):")
    print(f"  baseline_full (partner sends ingredients) : {full*100:.2f}%")
    print(f"  baseline_min  (no ingredients)            : {mn*100:.2f}%   "
          f"(drop {(full-mn)*100:+.2f}pp from full)")
    print(f"  +ocr (text augmentation)                  : {ocr*100:.2f}%   "
          f"({(ocr-mn)*100:+.2f}pp vs min)")
    print(f"\nOn CLIP-available subset (n_attrs={len(clip_only_rows)}):")
    print(f"  baseline_min (CLIP-subset only)           : {clip_base_subset*100:.2f}%")
    print(f"  +clip  (visual ensemble)                  : {clip*100:.2f}%   "
          f"({(clip-clip_base_subset)*100:+.2f}pp vs min)")
    print(f"  +ocr +clip                                : {both*100:.2f}%   "
          f"({(both-clip_base_subset)*100:+.2f}pp vs min)")

    print(f"\nPer-attr breakdown (sorted by +ocr lift desc):")
    out["ocr_lift"] = (out["acc_min_plus_ocr"] - out["acc_baseline_min"]) * 100
    out["clip_lift"] = (out["acc_min_plus_clip"] - out["acc_baseline_min"]) * 100
    show = out.sort_values("ocr_lift", ascending=False)[
        ["category", "attr", "acc_baseline_full", "acc_baseline_min",
         "acc_min_plus_ocr", "acc_min_plus_clip", "ocr_lift", "clip_lift"]
    ].round(4)
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
