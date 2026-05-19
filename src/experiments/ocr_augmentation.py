"""P2-EXP12: EasyOCR text augmentation pilot.

Reads product images downloaded from OpenFoodFacts, extracts text via EasyOCR,
and evaluates whether augmenting LightGBM TF-IDF features with OCR text improves
the R_ml ensemble (LightGBM[TF-IDF+OCR] + XGB[embeddings]) / 2.

Three variants evaluated on 80/20 split (seed=42) per (cat, attr):
  R_baseline : LightGBM(baseline text) + XGB / 2     — replicated EXP9 result
  R_ocr      : LightGBM(baseline + OCR) + XGB / 2   — full dataset
  R_ocr_subset : R_ocr scored ONLY on codes where ingredients_text was short

Output:
  datasets/processed/ocr_text_cache.json
  datasets/processed/ocr_augmentation_eval.parquet
"""
from __future__ import annotations

import json
import logging
import os
import time
from glob import glob
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKTREE_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = WORKTREE_ROOT / "datasets" / "processed"
IMAGES_DIR = WORKTREE_ROOT / "datasets" / "raw" / "off_images"
OCR_CACHE_PATH = PROCESSED_DIR / "ocr_text_cache.json"
EVAL_OUT_PATH = PROCESSED_DIR / "ocr_augmentation_eval.parquet"

CATEGORIES = ["pasta", "cheeses", "beverages"]
SHORT_INGREDIENTS_THRESH = 20   # chars — defines "empty/short ingredients"
MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB — skip very large files
OCR_CONF_THRESH = 0.3
RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_TRAIN_SAMPLES = 10
MIN_CLASSES = 2


# ---------------------------------------------------------------------------
# Phase 1: Run OCR on images and build cache
# ---------------------------------------------------------------------------

def run_ocr(images_dir: Path, cache_path: Path, force: bool = False) -> dict[str, str]:
    """Run EasyOCR on all .jpg files in images_dir.

    Returns dict {code: ocr_text}. Saves to cache_path (JSON).
    Skips images > 5MB. Filters detections by confidence >= OCR_CONF_THRESH.
    """
    if cache_path.exists() and not force:
        logger.info("Loading OCR cache from %s", cache_path)
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    img_files = sorted(glob(str(images_dir / "*.jpg")))
    logger.info("Found %d images in %s", len(img_files), images_dir)

    if not img_files:
        logger.warning("No images found — OCR cache will be empty")
        return {}

    # Lazy import — easyocr pulls torch, takes a second
    import easyocr  # noqa: PLC0415
    reader = easyocr.Reader(["en", "fr", "de", "it", "es"], gpu=False, verbose=False)
    logger.info("EasyOCR reader initialized")

    cache: dict[str, str] = {}
    t_start = time.time()
    skipped_size = 0

    for i, img_path in enumerate(img_files, 1):
        code = Path(img_path).stem
        # Skip very large files (likely bad downloads)
        file_size = os.path.getsize(img_path)
        if file_size > MAX_IMAGE_SIZE_BYTES:
            logger.debug("Skipping %s — %.1f MB", code, file_size / 1e6)
            skipped_size += 1
            continue

        try:
            results = reader.readtext(img_path, detail=1)
            ocr_text = " ".join(
                text for _, text, conf in results if conf >= OCR_CONF_THRESH
            )
            cache[code] = ocr_text.strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR failed for %s: %s", code, exc)
            cache[code] = ""

        if i % 50 == 0:
            elapsed = time.time() - t_start
            rate = i / elapsed
            eta = (len(img_files) - i) / rate if rate > 0 else 0
            logger.info("  OCR %d/%d (%.1f img/s, ETA %.0fs)", i, len(img_files), rate, eta)

    elapsed = time.time() - t_start
    logger.info(
        "OCR done: %d images processed, %d skipped (size), %.1fs total",
        len(cache), skipped_size, elapsed,
    )

    # Save cache
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    logger.info("Saved OCR cache to %s", cache_path)
    return cache


# ---------------------------------------------------------------------------
# Text builders
# ---------------------------------------------------------------------------

def build_baseline_text(row: pd.Series) -> str:
    """Baseline text: product_name + brands + ingredients_text + quantity."""
    parts: list[str] = []
    for col in ["product_name", "brands", "ingredients_text", "quantity"]:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    return " ".join(parts)


def build_ocr_text(row: pd.Series, ocr_cache: dict[str, str]) -> str:
    """Augmented text: baseline + OCR text (when code has an entry)."""
    base = build_baseline_text(row)
    code = str(row.get("code", "")).strip().lstrip("0") or str(row.get("code", ""))
    # try both zero-padded and raw
    ocr = ocr_cache.get(str(row.get("code", "")), "") or ocr_cache.get(code, "")
    if ocr:
        return f"{base} {ocr}".strip()
    return base


# ---------------------------------------------------------------------------
# Classifier helpers
# ---------------------------------------------------------------------------

def train_lgbm(
    train_texts: list[str],
    y_train: list[str],
) -> tuple[Optional[lgb.LGBMClassifier], Optional[TfidfVectorizer], Optional[LabelEncoder]]:
    """Train LightGBM on TF-IDF bi-gram features. Returns (None, None, None) if degenerate."""
    classes = sorted(set(y_train))
    if len(classes) < MIN_CLASSES or len(y_train) < MIN_TRAIN_SAMPLES:
        return None, None, None

    le = LabelEncoder()
    le.fit(classes)
    y_enc = le.transform(y_train)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=10_000, sublinear_tf=True)
    X = vec.fit_transform(train_texts)

    n_classes = len(classes)
    objective = "binary" if n_classes == 2 else "multiclass"
    clf_kwargs: dict = dict(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        num_leaves=31, min_child_samples=5,
        objective=objective, verbose=-1,
    )
    if n_classes > 2:
        clf_kwargs["num_class"] = n_classes

    clf = lgb.LGBMClassifier(**clf_kwargs)
    clf.fit(X, y_enc)
    return clf, vec, le


def train_xgb(
    X_emb: np.ndarray,
    y: list[str],
    sample_weights: Optional[np.ndarray] = None,
) -> tuple[Optional[xgb.XGBClassifier], Optional[LabelEncoder]]:
    """Train XGBoost on sentence embeddings."""
    classes = sorted(set(y))
    if len(classes) < MIN_CLASSES or len(y) < MIN_TRAIN_SAMPLES:
        return None, None

    le = LabelEncoder()
    le.fit(classes)
    y_enc = le.transform(y)

    n_classes = len(classes)
    common: dict = dict(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
        tree_method="hist", verbosity=0,
    )
    if n_classes == 2:
        pos = int((y_enc == 1).sum())
        neg = int((y_enc == 0).sum())
        spw = max(neg / max(pos, 1), 0.5)
        clf = xgb.XGBClassifier(scale_pos_weight=spw, **common)
    else:
        clf = xgb.XGBClassifier(objective="multi:softmax", num_class=n_classes, **common)

    clf.fit(X_emb, y_enc, sample_weight=sample_weights)
    return clf, le


def align_probas(
    lgbm_probas: np.ndarray, lgbm_le: LabelEncoder,
    xgb_probas: np.ndarray, xgb_le: LabelEncoder,
) -> tuple[np.ndarray, list[str]]:
    """Average LightGBM and XGB probability arrays over shared label space."""
    lgbm_cls = list(lgbm_le.classes_)
    xgb_cls = list(xgb_le.classes_)
    all_cls = sorted(set(lgbm_cls) | set(xgb_cls))
    n, k = lgbm_probas.shape[0], len(all_cls)

    lgbm_full = np.zeros((n, k))
    for j, c in enumerate(lgbm_cls):
        lgbm_full[:, all_cls.index(c)] = lgbm_probas[:, j]

    xgb_full = np.zeros((n, k))
    for j, c in enumerate(xgb_cls):
        xgb_full[:, all_cls.index(c)] = xgb_probas[:, j]

    return 0.5 * lgbm_full + 0.5 * xgb_full, all_cls


# ---------------------------------------------------------------------------
# Per-(cat, attr) evaluation
# ---------------------------------------------------------------------------

def evaluate_attr(
    cat: str,
    attr: str,
    df: pd.DataFrame,
    emb: np.ndarray,
    code_to_idx: dict[str, int],
    ocr_cache: dict[str, str],
    short_codes: set[str],
) -> Optional[dict]:
    """Evaluate baseline, R_ocr, and R_ocr_subset for one (cat, attr).

    Returns dict with accuracy metrics, or None if skipped.
    """
    if attr not in df.columns:
        return None

    sub = df[df[attr].notna()].copy()
    sub["code"] = sub["code"].astype(str)
    sub = sub[sub["code"].isin(code_to_idx)]

    if len(sub) < MIN_TRAIN_SAMPLES * 2 or sub[attr].nunique() < MIN_CLASSES:
        return None

    y = sub[attr].astype(str).values.tolist()
    codes = sub["code"].tolist()

    try:
        train_idx, test_idx = train_test_split(
            range(len(sub)), test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
        )
    except ValueError:
        # stratify fails when class too rare
        train_idx, test_idx = train_test_split(
            range(len(sub)), test_size=TEST_SIZE, random_state=RANDOM_STATE,
        )

    train_sub = sub.iloc[list(train_idx)]
    test_sub = sub.iloc[list(test_idx)]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]

    # Embeddings for train/test
    train_codes = [codes[i] for i in train_idx]
    test_codes = [codes[i] for i in test_idx]
    X_train_emb = emb[np.array([code_to_idx[c] for c in train_codes])]
    X_test_emb = emb[np.array([code_to_idx[c] for c in test_codes])]

    # Train XGB once (embeddings do not use OCR)
    clf_xgb, le_xgb = train_xgb(X_train_emb, y_train)
    if clf_xgb is None or le_xgb is None:
        return None

    xgb_train_probas = clf_xgb.predict_proba(X_train_emb)  # noqa: unused — for sanity
    xgb_test_probas = clf_xgb.predict_proba(X_test_emb)

    results: dict[str, object] = {
        "category": cat,
        "attr": attr,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_test_short": sum(1 for c in test_codes if c in short_codes),
    }

    for variant, text_builder in [
        ("baseline", lambda row: build_baseline_text(row)),
        ("ocr", lambda row: build_ocr_text(row, ocr_cache)),
    ]:
        train_texts = [text_builder(row) for _, row in train_sub.iterrows()]
        test_texts = [text_builder(row) for _, row in test_sub.iterrows()]

        clf_lgbm, vec_lgbm, le_lgbm = train_lgbm(train_texts, y_train)
        if clf_lgbm is None or vec_lgbm is None or le_lgbm is None:
            # XGB-only fallback
            enc_preds = np.argmax(xgb_test_probas, axis=1)
            preds = le_xgb.inverse_transform(enc_preds).tolist()
            acc_all = float(sum(p == g for p, g in zip(preds, y_test)) / len(y_test))
            results[f"acc_{variant}"] = acc_all

            # Subset (short ingredients_text)
            short_mask = [c in short_codes for c in test_codes]
            if sum(short_mask) > 0:
                preds_sub = [p for p, m in zip(preds, short_mask) if m]
                y_sub = [g for g, m in zip(y_test, short_mask) if m]
                results[f"acc_{variant}_subset"] = float(
                    sum(p == g for p, g in zip(preds_sub, y_sub)) / len(y_sub)
                )
            else:
                results[f"acc_{variant}_subset"] = float("nan")
            continue

        X_test_tfidf = vec_lgbm.transform(test_texts)
        lgbm_probas = clf_lgbm.predict_proba(X_test_tfidf)

        # Ensemble
        avg_probas, merged_cls = align_probas(lgbm_probas, le_lgbm, xgb_test_probas, le_xgb)
        preds = [merged_cls[i] for i in np.argmax(avg_probas, axis=1)]
        acc_all = float(sum(p == g for p, g in zip(preds, y_test)) / len(y_test))
        results[f"acc_{variant}"] = acc_all

        # Subset accuracy (codes where ingredients_text was short)
        short_mask = [c in short_codes for c in test_codes]
        if sum(short_mask) > 0:
            preds_sub = [p for p, m in zip(preds, short_mask) if m]
            y_sub = [g for g, m in zip(y_test, short_mask) if m]
            results[f"acc_{variant}_subset"] = float(
                sum(p == g for p, g in zip(preds_sub, y_sub)) / len(y_sub)
            )
        else:
            results[f"acc_{variant}_subset"] = float("nan")

    return results


# ---------------------------------------------------------------------------
# Sample OCR outputs
# ---------------------------------------------------------------------------

def print_ocr_samples(ocr_cache: dict[str, str], n: int = 6) -> None:
    """Print n sample OCR outputs, sorted by text length."""
    items = [(k, v) for k, v in ocr_cache.items() if v.strip()]
    # sort by text length descending — shows both good (long) and short results
    items.sort(key=lambda x: len(x[1]), reverse=True)
    print(f"\n{'='*70}")
    print("OCR SAMPLE OUTPUTS")
    print(f"{'='*70}")
    for code, text in items[:n]:
        truncated = text[:120] + "..." if len(text) > 120 else text
        print(f"  [{code}] {truncated!r}")
    # also show shortest (potentially garbled)
    items_short = sorted(items, key=lambda x: len(x[1]))
    print(f"\n--- Shortest OCR results (potentially garbled) ---")
    for code, text in items_short[:3]:
        print(f"  [{code}] {text!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # --- Phase 1: OCR ---
    img_count = len(list(IMAGES_DIR.glob("*.jpg"))) if IMAGES_DIR.exists() else 0
    if img_count == 0:
        logger.error(
            "No images found in %s — EXP10 has not downloaded images yet. "
            "STATUS: BLOCKED (images required for OCR augmentation).",
            IMAGES_DIR,
        )
        print("\nSTATUS: BLOCKED — no images in datasets/raw/off_images/")
        print("EXP10 must download product images first.")
        return

    logger.info("Found %d images — starting OCR", img_count)
    ocr_cache = run_ocr(IMAGES_DIR, OCR_CACHE_PATH)

    print_ocr_samples(ocr_cache, n=6)

    # --- Phase 2: Prepare short-ingredients codes ---
    short_codes: set[str] = set()
    for cat in CATEGORIES:
        silver_path = PROCESSED_DIR / f"{cat}_stratified_silver_standard.parquet"
        if not silver_path.exists():
            continue
        df_cat = pd.read_parquet(silver_path)
        mask = df_cat["ingredients_text"].fillna("").str.len() < SHORT_INGREDIENTS_THRESH
        short_codes.update(df_cat.loc[mask, "code"].astype(str).tolist())

    logger.info(
        "Short ingredients_text (<20 chars) codes: %d across %s",
        len(short_codes), CATEGORIES,
    )

    n_ocr_available = sum(1 for c in ocr_cache if ocr_cache[c].strip())
    n_in_short = sum(1 for c in short_codes if c in ocr_cache and ocr_cache[c].strip())
    logger.info(
        "OCR text available: %d codes (non-empty), %d of those are short-ingredients codes",
        n_ocr_available, n_in_short,
    )

    # --- Phase 3: Evaluate per (cat, attr) ---
    all_results: list[dict] = []

    for cat in CATEGORIES:
        silver_path = PROCESSED_DIR / f"{cat}_stratified_silver_standard.parquet"
        emb_path = PROCESSED_DIR / f"{cat}_stratified_embeddings.npy"

        if not silver_path.exists() or not emb_path.exists():
            logger.warning("Missing data for %s — skipping", cat)
            continue

        df = pd.read_parquet(silver_path)
        df["code"] = df["code"].astype(str)
        emb = np.load(emb_path)
        code_to_idx = {c: i for i, c in enumerate(df["code"].tolist())}

        # Identify attribute columns (exclude metadata)
        meta_cols = {
            "code", "product_name", "brands", "ingredients_text", "quantity",
            "categories_tags", "labels_tags", "ingredients_analysis_tags",
            "traces_tags", "countries_tags",
        }
        attrs = [c for c in df.columns if c not in meta_cols]

        logger.info("=== %s: %d attrs ===", cat, len(attrs))

        for attr in attrs:
            res = evaluate_attr(cat, attr, df, emb, code_to_idx, ocr_cache, short_codes)
            if res is not None:
                all_results.append(res)
                logger.info(
                    "  [%s/%s] baseline=%.3f ocr=%.3f subset_baseline=%.3f subset_ocr=%.3f n=%d",
                    cat, attr,
                    res.get("acc_baseline", float("nan")),
                    res.get("acc_ocr", float("nan")),
                    res.get("acc_baseline_subset", float("nan")),
                    res.get("acc_ocr_subset", float("nan")),
                    res.get("n_test", 0),
                )

    if not all_results:
        logger.error("No results — cannot compute summary")
        return

    # --- Phase 4: Summary ---
    results_df = pd.DataFrame(all_results)
    results_df.to_parquet(EVAL_OUT_PATH, index=False)
    logger.info("Saved eval results to %s", EVAL_OUT_PATH)

    # Compute aggregate means (over attrs with valid values)
    def nanmean(col: str) -> float:
        vals = results_df[col].dropna()
        return float(vals.mean()) if len(vals) > 0 else float("nan")

    mean_baseline = nanmean("acc_baseline")
    mean_ocr = nanmean("acc_ocr")
    mean_baseline_subset = nanmean("acc_baseline_subset")
    mean_ocr_subset = nanmean("acc_ocr_subset")

    n_total_short = len(short_codes)
    n_test_with_short = int(results_df["n_test_short"].sum())

    print(f"\n{'='*70}")
    print("EXP12: EasyOCR AUGMENTATION RESULTS")
    print(f"{'='*70}")
    print(f"Images processed : {img_count}")
    print(f"OCR cache entries (non-empty): {n_ocr_available}")
    print(f"Codes with short ingredients_text: {n_total_short} total, "
          f"{n_in_short} with OCR available")
    print(f"Test rows with short ingredients : {n_test_with_short} (across all attrs)")
    print()
    print(f"{'Variant':<25} {'Mean Accuracy':>15} {'Delta vs Baseline':>20}")
    print(f"{'-'*65}")
    print(f"{'R_baseline':<25} {mean_baseline:>15.4f} {'—':>20}")
    print(f"{'R_ocr (all)':<25} {mean_ocr:>15.4f} "
          f"{(mean_ocr - mean_baseline)*100:>+19.2f}pp")
    print(f"{'R_ocr subset only':<25} {mean_ocr_subset:>15.4f} "
          f"[vs baseline_sub {mean_baseline_subset:.4f} "
          f"delta={(mean_ocr_subset - mean_baseline_subset)*100:+.2f}pp]")

    print(f"\n{'Per-attr breakdown':}")
    print(f"{'cat':<12} {'attr':<25} {'baseline':>10} {'ocr':>10} {'delta_pp':>10} {'sub_delta':>10}")
    print(f"{'-'*85}")
    for _, row in results_df.sort_values(["category", "attr"]).iterrows():
        base = row.get("acc_baseline", float("nan"))
        ocr_v = row.get("acc_ocr", float("nan"))
        sub_base = row.get("acc_baseline_subset", float("nan"))
        sub_ocr = row.get("acc_ocr_subset", float("nan"))
        delta = (ocr_v - base) * 100 if not (np.isnan(base) or np.isnan(ocr_v)) else float("nan")
        sub_delta = (
            (sub_ocr - sub_base) * 100
            if not (np.isnan(sub_base) or np.isnan(sub_ocr))
            else float("nan")
        )
        delta_s = f"{delta:+.1f}" if not np.isnan(delta) else "n/a"
        sub_delta_s = f"{sub_delta:+.1f}" if not np.isnan(sub_delta) else "n/a"
        print(
            f"{row['category']:<12} {row['attr']:<25} {base:>10.4f} {ocr_v:>10.4f} "
            f"{delta_s:>10} {sub_delta_s:>10}"
        )

    print(f"\n{'VERDICT':}")
    ocr_helps_global = mean_ocr > mean_baseline
    ocr_helps_subset = mean_ocr_subset > mean_baseline_subset
    if ocr_helps_global:
        print(f"  OCR helps GLOBALLY (+{(mean_ocr - mean_baseline)*100:.2f}pp)")
    else:
        print(f"  OCR does NOT help globally ({(mean_ocr - mean_baseline)*100:.2f}pp)")
    if ocr_helps_subset:
        print(f"  OCR helps on SHORT-INGREDIENTS SUBSET "
              f"(+{(mean_ocr_subset - mean_baseline_subset)*100:.2f}pp)")
    else:
        print(f"  OCR does NOT help on subset "
              f"({(mean_ocr_subset - mean_baseline_subset)*100:.2f}pp)")

    deploy_recommendation = (
        "DEPLOY for empty-ingredients products"
        if ocr_helps_subset and not ocr_helps_global
        else "DEPLOY for all products"
        if ocr_helps_global
        else "DO NOT DEPLOY — OCR adds noise without accuracy gain"
    )
    print(f"\n  Recommendation: {deploy_recommendation}")


if __name__ == "__main__":
    main()
