"""R_ml v3 — stacked 4-way ensemble with auto-learned weights.

Combines four signals per (cat, attr) via logistic regression meta-learner:
  M1: LightGBM(TF-IDF on full text)        — current production member
  M2: XGBoost(sentence embeddings)          — current production member
  M3: LightGBM(TF-IDF on OCR text only)     — augmentation
  M4: XGBoost(CLIP 512-dim visual)          — augmentation

Stacking: 5-fold CV on TRAIN set produces out-of-fold (OOF) probability
predictions for each model. Logistic regression is fit on the concatenated
OOF probas to predict y_train. At test time, each base model predicts on
the test set, probas are concatenated, and the meta-learner produces the
final prediction.

Missing OCR / CLIP coverage handled by imputing uniform distribution for
those rows — meta-learner naturally learns to discount uninformative slots.

Output:
  datasets/processed/cascade_v3_eval.parquet  — per-(cat,attr) accuracies
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

WORKTREE_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = WORKTREE_ROOT / "datasets" / "processed"
OCR_CACHE_PATH = PROCESSED_DIR / "ocr_text_cache.json"
DEFAULT_OUT = PROCESSED_DIR / "cascade_v3_eval.parquet"
WEIGHTS_PATH = PROCESSED_DIR / "cascade_v3_weights.json"

CATEGORIES = ["pasta", "cheeses", "chocolate"]
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5
MIN_TRAIN_SAMPLES = 20
MIN_CLASSES = 2
MAX_CLASSES = 30  # skip numeric-as-class attrs (fat_100g etc — these are regression-like)


# ---------------------------------------------------------------------------
# Text builders
# ---------------------------------------------------------------------------

def build_full_text(row: pd.Series) -> str:
    parts: list[str] = []
    for col in ["product_name", "brands", "ingredients_text", "quantity"]:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    return " ".join(parts)


def lookup_ocr(code: str, ocr_cache: dict[str, str]) -> str:
    if not code:
        return ""
    return ocr_cache.get(code, "") or ocr_cache.get(code.lstrip("0") or code, "")


# ---------------------------------------------------------------------------
# Base model wrappers — return (oof_probas, fit_callable, predict_proba_callable)
# ---------------------------------------------------------------------------

def fit_lgbm_tfidf(
    train_texts: list[str], y_train: list[str], classes: list[str],
) -> tuple[Optional[object], Optional[object]]:
    le = LabelEncoder().fit(classes)
    y_enc = le.transform(y_train)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=10_000, sublinear_tf=True)
    X = vec.fit_transform(train_texts)
    n = len(classes)
    kwargs: dict = dict(
        n_estimators=300, max_depth=6, learning_rate=0.05, num_leaves=31,
        min_child_samples=5, verbose=-1,
        objective="binary" if n == 2 else "multiclass",
    )
    if n > 2:
        kwargs["num_class"] = n
    clf = lgb.LGBMClassifier(**kwargs)
    clf.fit(X, y_enc)
    return (clf, vec), le


def predict_lgbm_tfidf(model_pack, le, texts: list[str], classes: list[str]) -> np.ndarray:
    clf, vec = model_pack
    X = vec.transform(texts)
    p = clf.predict_proba(X)
    # LGBM returns one column per class it ACTUALLY saw in training (clf.classes_),
    # which can be a subset of le.classes_ when a fold misses some classes.
    actual_enc = clf.classes_
    actual_labels = le.inverse_transform(actual_enc)
    return align_probas_to_classes(p, actual_labels, classes)


def fit_xgb_dense(
    X_train: np.ndarray, y_train: list[str], classes: list[str],
) -> tuple[Optional[object], Optional[object]]:
    """Train LightGBM on dense numeric features (renamed for API parity; uses LGBM internally
    to avoid XGBoost's strict requirement that y values be contiguous [0..n-1] — partial folds
    can yield non-contiguous label sets which XGBoost rejects)."""
    le = LabelEncoder().fit(classes)
    y_enc = le.transform(y_train)
    n = len(classes)
    kwargs: dict = dict(
        n_estimators=300, max_depth=4, learning_rate=0.05, num_leaves=31,
        min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, verbose=-1,
        objective="binary" if n == 2 else "multiclass",
    )
    if n > 2:
        kwargs["num_class"] = n
    clf = lgb.LGBMClassifier(**kwargs)
    clf.fit(X_train, y_enc)
    return clf, le


def predict_xgb_dense(model, le, X: np.ndarray, classes: list[str]) -> np.ndarray:
    p = model.predict_proba(X)
    actual_enc = model.classes_
    actual_labels = le.inverse_transform(actual_enc)
    return align_probas_to_classes(p, actual_labels, classes)


def align_probas_to_classes(
    probas: np.ndarray, model_classes: np.ndarray, target_classes: list[str],
) -> np.ndarray:
    """Re-order probability matrix columns to match target_classes."""
    n = probas.shape[0]
    k = len(target_classes)
    out = np.zeros((n, k))
    model_cls_list = list(model_classes)
    for j, c in enumerate(target_classes):
        if c in model_cls_list:
            out[:, j] = probas[:, model_cls_list.index(c)]
    return out


def uniform_probas(n_rows: int, n_classes: int) -> np.ndarray:
    return np.full((n_rows, n_classes), 1.0 / n_classes)


# ---------------------------------------------------------------------------
# OOF generation per model
# ---------------------------------------------------------------------------

def oof_lgbm_tfidf(
    train_texts: list[str], y_train: list[str], classes: list[str], n_folds: int,
) -> np.ndarray:
    n = len(y_train)
    k = len(classes)
    oof = uniform_probas(n, k)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    y_arr = np.array(y_train)
    try:
        splits = list(skf.split(np.zeros(n), y_arr))
    except ValueError:
        return oof
    for tr_idx, va_idx in splits:
        tr_texts = [train_texts[i] for i in tr_idx]
        tr_y = [y_train[i] for i in tr_idx]
        va_texts = [train_texts[i] for i in va_idx]
        if len(set(tr_y)) < 2:
            continue
        pack, le = fit_lgbm_tfidf(tr_texts, tr_y, classes)
        oof[va_idx] = predict_lgbm_tfidf(pack, le, va_texts, classes)
    return oof


def oof_xgb_dense(
    X_train: np.ndarray, y_train: list[str], classes: list[str],
    mask: np.ndarray, n_folds: int,
) -> np.ndarray:
    """OOF preds for XGB on dense features. mask: bool array — True where features available."""
    n = len(y_train)
    k = len(classes)
    oof = uniform_probas(n, k)
    if mask.sum() < N_FOLDS * MIN_CLASSES:
        return oof
    # Restrict CV to rows with available features
    sub_idx = np.where(mask)[0]
    X_sub = X_train[sub_idx]
    y_sub = [y_train[i] for i in sub_idx]
    skf = StratifiedKFold(n_splits=min(n_folds, 5), shuffle=True, random_state=RANDOM_STATE)
    y_arr = np.array(y_sub)
    try:
        splits = list(skf.split(np.zeros(len(sub_idx)), y_arr))
    except ValueError:
        return oof
    for tr_idx, va_idx in splits:
        tr_y = [y_sub[i] for i in tr_idx]
        if len(set(tr_y)) < 2:
            continue
        clf, le = fit_xgb_dense(X_sub[tr_idx], tr_y, classes)
        oof[sub_idx[va_idx]] = predict_xgb_dense(clf, le, X_sub[va_idx], classes)
    return oof


# ---------------------------------------------------------------------------
# Per-(cat, attr) evaluation
# ---------------------------------------------------------------------------

def evaluate_attr(
    cat: str, attr: str, df: pd.DataFrame, emb_full: np.ndarray, code_to_idx: dict[str, int],
    ocr_cache: dict[str, str],
    clip_emb: Optional[np.ndarray], clip_code_idx: Optional[dict[str, int]],
) -> Optional[dict]:
    if attr not in df.columns:
        return None
    sub = df[df[attr].notna()].copy()
    sub["code"] = sub["code"].astype(str)
    sub = sub[sub["code"].isin(code_to_idx)]
    if len(sub) < MIN_TRAIN_SAMPLES or sub[attr].nunique() < MIN_CLASSES:
        return None
    if sub[attr].nunique() > MAX_CLASSES:
        return None  # skip numeric/regression-like attrs (fat_100g, proteins_100g, ...)

    y_all = sub[attr].astype(str).values.tolist()
    codes_all = sub["code"].tolist()
    classes = sorted(set(y_all))
    n_classes = len(classes)

    try:
        tr_idx, te_idx = train_test_split(
            range(len(sub)), test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_all,
        )
    except ValueError:
        tr_idx, te_idx = train_test_split(
            range(len(sub)), test_size=TEST_SIZE, random_state=RANDOM_STATE,
        )
    tr_idx, te_idx = list(tr_idx), list(te_idx)
    tr_sub, te_sub = sub.iloc[tr_idx], sub.iloc[te_idx]
    y_tr = [y_all[i] for i in tr_idx]
    y_te = [y_all[i] for i in te_idx]
    tr_codes = [codes_all[i] for i in tr_idx]
    te_codes = [codes_all[i] for i in te_idx]

    # --- Build feature matrices for each model ---
    tr_texts_full = [build_full_text(r) for _, r in tr_sub.iterrows()]
    te_texts_full = [build_full_text(r) for _, r in te_sub.iterrows()]

    tr_texts_ocr = [lookup_ocr(c, ocr_cache) for c in tr_codes]
    te_texts_ocr = [lookup_ocr(c, ocr_cache) for c in te_codes]
    tr_has_ocr = np.array([bool(t.strip()) for t in tr_texts_ocr])
    te_has_ocr = np.array([bool(t.strip()) for t in te_texts_ocr])

    X_tr_emb = emb_full[[code_to_idx[c] for c in tr_codes]]
    X_te_emb = emb_full[[code_to_idx[c] for c in te_codes]]

    if clip_emb is not None and clip_code_idx is not None:
        tr_has_clip = np.array([c in clip_code_idx for c in tr_codes])
        te_has_clip = np.array([c in clip_code_idx for c in te_codes])
        n_dim = clip_emb.shape[1]
        X_tr_clip = np.zeros((len(tr_codes), n_dim))
        X_te_clip = np.zeros((len(te_codes), n_dim))
        for i, c in enumerate(tr_codes):
            if c in clip_code_idx:
                X_tr_clip[i] = clip_emb[clip_code_idx[c]]
        for i, c in enumerate(te_codes):
            if c in clip_code_idx:
                X_te_clip[i] = clip_emb[clip_code_idx[c]]
    else:
        tr_has_clip = np.zeros(len(tr_codes), dtype=bool)
        te_has_clip = np.zeros(len(te_codes), dtype=bool)
        X_tr_clip = np.zeros((len(tr_codes), 1))
        X_te_clip = np.zeros((len(te_codes), 1))

    # --- OOF predictions on TRAIN ---
    oof_full = oof_lgbm_tfidf(tr_texts_full, y_tr, classes, N_FOLDS)
    oof_emb = oof_xgb_dense(X_tr_emb, y_tr, classes, np.ones(len(y_tr), dtype=bool), N_FOLDS)

    if tr_has_ocr.sum() >= MIN_TRAIN_SAMPLES:
        # Train LGBM-OCR only on rows that have OCR; impute uniform for others
        oof_ocr = uniform_probas(len(y_tr), n_classes)
        sub_idx = np.where(tr_has_ocr)[0]
        sub_texts = [tr_texts_ocr[i] for i in sub_idx]
        sub_y = [y_tr[i] for i in sub_idx]
        sub_oof = oof_lgbm_tfidf(sub_texts, sub_y, classes, min(N_FOLDS, max(2, len(set(sub_y)))))
        for j, idx in enumerate(sub_idx):
            oof_ocr[idx] = sub_oof[j]
    else:
        oof_ocr = uniform_probas(len(y_tr), n_classes)

    if tr_has_clip.sum() >= MIN_TRAIN_SAMPLES and clip_emb is not None:
        oof_clip = oof_xgb_dense(X_tr_clip, y_tr, classes, tr_has_clip, N_FOLDS)
    else:
        oof_clip = uniform_probas(len(y_tr), n_classes)

    # --- Fit base models on full TRAIN ---
    pack_full, le_full = fit_lgbm_tfidf(tr_texts_full, y_tr, classes)
    te_p_full = predict_lgbm_tfidf(pack_full, le_full, te_texts_full, classes)

    clf_emb, le_emb = fit_xgb_dense(X_tr_emb, y_tr, classes)
    te_p_emb = predict_xgb_dense(clf_emb, le_emb, X_te_emb, classes)

    if tr_has_ocr.sum() >= MIN_TRAIN_SAMPLES:
        pack_ocr, le_ocr = fit_lgbm_tfidf(
            [t for t, m in zip(tr_texts_ocr, tr_has_ocr) if m],
            [yy for yy, m in zip(y_tr, tr_has_ocr) if m],
            classes,
        )
        te_p_ocr = uniform_probas(len(y_te), n_classes)
        sub_te_idx = np.where(te_has_ocr)[0]
        if len(sub_te_idx) > 0:
            te_p_ocr[sub_te_idx] = predict_lgbm_tfidf(
                pack_ocr, le_ocr, [te_texts_ocr[i] for i in sub_te_idx], classes,
            )
    else:
        te_p_ocr = uniform_probas(len(y_te), n_classes)

    if tr_has_clip.sum() >= MIN_TRAIN_SAMPLES and clip_emb is not None:
        sub_idx = np.where(tr_has_clip)[0]
        clf_clip, le_clip = fit_xgb_dense(X_tr_clip[sub_idx], [y_tr[i] for i in sub_idx], classes)
        te_p_clip = uniform_probas(len(y_te), n_classes)
        sub_te_idx = np.where(te_has_clip)[0]
        if len(sub_te_idx) > 0:
            te_p_clip[sub_te_idx] = predict_xgb_dense(
                clf_clip, le_clip, X_te_clip[sub_te_idx], classes,
            )
    else:
        te_p_clip = uniform_probas(len(y_te), n_classes)

    # --- Meta-learner: logistic regression on stacked OOF probas ---
    X_meta_tr = np.hstack([oof_full, oof_emb, oof_ocr, oof_clip])
    X_meta_te = np.hstack([te_p_full, te_p_emb, te_p_ocr, te_p_clip])
    y_tr_enc = LabelEncoder().fit(classes).transform(y_tr)

    if len(set(y_tr)) >= 2:
        meta = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        meta.fit(X_meta_tr, y_tr_enc)
        preds_enc = meta.predict(X_meta_te)
        preds_meta = [classes[i] for i in preds_enc]
        acc_meta = float(sum(p == g for p, g in zip(preds_meta, y_te)) / len(y_te))
    else:
        acc_meta = float("nan")

    def acc_of(probas: np.ndarray) -> float:
        preds = [classes[i] for i in np.argmax(probas, axis=1)]
        return float(sum(p == g for p, g in zip(preds, y_te)) / len(y_te))

    acc_full_only = acc_of(te_p_full)
    acc_emb_only = acc_of(te_p_emb)
    acc_ocr_only = acc_of(te_p_ocr) if tr_has_ocr.sum() >= MIN_TRAIN_SAMPLES else float("nan")
    acc_clip_only = acc_of(te_p_clip) if tr_has_clip.sum() >= MIN_TRAIN_SAMPLES else float("nan")

    # Current production R_ml = (full + emb) / 2
    avg2 = 0.5 * te_p_full + 0.5 * te_p_emb
    acc_v2 = acc_of(avg2)

    # Meta coefficients indicate which model the meta learned to trust
    coef_norms: dict[str, float] = {}
    if "meta" in locals() and hasattr(meta, "coef_"):
        try:
            coef = meta.coef_
            for j, name in enumerate(["full", "emb", "ocr", "clip"]):
                slice_ = coef[:, j * n_classes : (j + 1) * n_classes]
                coef_norms[f"meta_norm_{name}"] = float(np.linalg.norm(slice_))
        except Exception:  # noqa: BLE001
            pass

    return {
        "category": cat,
        "attr": attr,
        "n_train": len(tr_idx),
        "n_test": len(te_idx),
        "n_classes": n_classes,
        "n_tr_with_ocr": int(tr_has_ocr.sum()),
        "n_tr_with_clip": int(tr_has_clip.sum()),
        "acc_full_only": acc_full_only,
        "acc_emb_only": acc_emb_only,
        "acc_ocr_only": acc_ocr_only,
        "acc_clip_only": acc_clip_only,
        "acc_v2_baseline": acc_v2,           # current production
        "acc_v3_stacked": acc_meta,           # new 4-way stacked
        "v3_lift_pp": (acc_meta - acc_v2) * 100,
        **coef_norms,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import sys
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout, force=True,
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="+", default=CATEGORIES,
                    help="Categories to evaluate (smoke test: --cats pasta)")
    ap.add_argument("--output", default=str(DEFAULT_OUT),
                    help="Output parquet path (use separate paths for parallel runs)")
    args = ap.parse_args()
    out_path = Path(args.output)

    if not OCR_CACHE_PATH.exists():
        logger.error("OCR cache missing: %s", OCR_CACHE_PATH)
        return
    with open(OCR_CACHE_PATH, encoding="utf-8") as f:
        ocr_cache = json.load(f)
    logger.info("OCR cache: %d entries (%d non-empty)",
                len(ocr_cache), sum(1 for v in ocr_cache.values() if v.strip()))

    all_rows: list[dict] = []
    for cat in args.cats:
        silver_path = PROCESSED_DIR / f"{cat}_stratified_silver_standard.parquet"
        emb_path = PROCESSED_DIR / f"{cat}_stratified_embeddings.npy"
        clip_emb_path = PROCESSED_DIR / f"clip_embeddings_{cat}.npy"
        clip_idx_path = PROCESSED_DIR / f"clip_code_index_{cat}.json"

        if not silver_path.exists() or not emb_path.exists():
            logger.warning("Missing data for %s — skip", cat)
            continue
        df = pd.read_parquet(silver_path)
        df["code"] = df["code"].astype(str)
        emb = np.load(emb_path)
        code_to_idx = {c: i for i, c in enumerate(df["code"].tolist())}

        clip_emb = None
        clip_idx = None
        if clip_emb_path.exists() and clip_idx_path.exists():
            clip_emb = np.load(clip_emb_path)
            with open(clip_idx_path, encoding="utf-8") as f:
                raw = json.load(f)
            clip_idx = {str(k): int(v) for k, v in raw.items()}

        meta_cols = {
            "code", "product_name", "brands", "ingredients_text", "quantity",
            "categories_tags", "labels_tags", "ingredients_analysis_tags",
            "traces_tags", "countries_tags",
        }
        attrs = [c for c in df.columns if c not in meta_cols]
        logger.info("=== %s: %d attrs ===", cat, len(attrs))

        for attr in attrs:
            try:
                res = evaluate_attr(cat, attr, df, emb, code_to_idx, ocr_cache, clip_emb, clip_idx)
            except Exception as exc:  # noqa: BLE001
                logger.warning("FAIL %s/%s: %s", cat, attr, exc)
                continue
            if res is None:
                continue
            all_rows.append(res)
            logger.info(
                "  [%s/%s] v2=%.3f v3=%.3f lift=%+.2fpp",
                cat, attr, res["acc_v2_baseline"], res["acc_v3_stacked"], res["v3_lift_pp"],
            )

        # Incremental save after each category
        if all_rows:
            pd.DataFrame(all_rows).to_parquet(out_path, index=False)
            logger.info("[%s] checkpoint: %d rows saved to %s", cat, len(all_rows), out_path)

    if not all_rows:
        logger.error("No results")
        return

    out = pd.DataFrame(all_rows)
    out.to_parquet(out_path, index=False)
    logger.info("Saved %d rows to %s", len(out), out_path)

    # Per-cat means
    print(f"\n{'='*78}")
    print("Cascade v3 — stacked 4-way ensemble (auto-learned weights)")
    print(f"{'='*78}")
    g = out.groupby("category")[["acc_v2_baseline", "acc_v3_stacked", "v3_lift_pp"]].mean()
    print(g.round(4).to_string())
    print()
    grand = out[["acc_v2_baseline", "acc_v3_stacked", "v3_lift_pp"]].mean()
    print(f"Grand mean across {len(out)} (cat, attr):")
    print(f"  v2 baseline (LGBM+XGB / 2): {grand['acc_v2_baseline']*100:.2f}%")
    print(f"  v3 stacked (LGBM+XGB+OCR+CLIP): {grand['acc_v3_stacked']*100:.2f}%  "
          f"({grand['v3_lift_pp']:+.2f}pp)")
    print()
    print("Per-attr top-10 lifts:")
    print(out.sort_values("v3_lift_pp", ascending=False).head(10)[
        ["category", "attr", "acc_v2_baseline", "acc_v3_stacked", "v3_lift_pp",
         "n_tr_with_ocr", "n_tr_with_clip"]
    ].round(3).to_string(index=False))
    print()
    print("Per-attr bottom-5 (regressions):")
    print(out.sort_values("v3_lift_pp").head(5)[
        ["category", "attr", "acc_v2_baseline", "acc_v3_stacked", "v3_lift_pp"]
    ].round(3).to_string(index=False))

    # Save meta norms (proxy for learned weights) per attr
    norm_cols = [c for c in out.columns if c.startswith("meta_norm_")]
    if norm_cols:
        weights_summary = {}
        for _, row in out.iterrows():
            key = f"{row['category']}/{row['attr']}"
            total = sum(row[c] for c in norm_cols) or 1.0
            weights_summary[key] = {c.replace("meta_norm_", ""): row[c] / total for c in norm_cols}
        with open(WEIGHTS_PATH, "w", encoding="utf-8") as f:
            json.dump(weights_summary, f, indent=2)
        logger.info("Saved meta weights to %s", WEIGHTS_PATH)


if __name__ == "__main__":
    main()
