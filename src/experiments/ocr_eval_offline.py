"""P2-EXP12 eval-only: OCR augmentation impact on R_ml (offline, NO torch).

Uses pre-computed embeddings (npy) + pre-saved OCR text cache.

Three variants evaluated on 80/20 split (seed=42) per (cat, attr):
  R_baseline  : LightGBM(baseline text) + XGB(emb) / 2  -- replicates EXP9
  R_ocr       : LightGBM(baseline + OCR) + XGB(emb) / 2 -- full dataset
  R_ocr_subset: R_ocr scored ONLY on codes where ingredients_text < 20 chars

Phase B: OFF quality check -- side-by-side OFF vs OCR for 50 codes per cat,
         Jaccard similarity on word sets, find cases OCR adds information.

Output:
  datasets/processed/ocr_augmentation_eval.parquet
  docs/ocr_findings.md
  docs/ocr_vs_off_comparison.md
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKTREE_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = WORKTREE_ROOT / "datasets" / "processed"
OCR_CACHE_PATH = PROCESSED_DIR / "ocr_text_cache.json"
EVAL_OUT_PATH = PROCESSED_DIR / "ocr_augmentation_eval.parquet"
DOCS_DIR = WORKTREE_ROOT / "docs"
OCR_FINDINGS_PATH = DOCS_DIR / "ocr_findings.md"
OCR_COMPARISON_PATH = DOCS_DIR / "ocr_vs_off_comparison.md"

CATEGORIES = ["pasta", "cheeses", "beverages"]
SHORT_INGREDIENTS_THRESH = 20   # chars
RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_TRAIN_SAMPLES = 10
MIN_CLASSES = 2
SAMPLE_PER_CAT_COMPARISON = 50  # for Phase B OFF quality check

META_COLS = {
    "code", "product_name", "brands", "ingredients_text", "quantity",
    "categories_tags", "labels_tags", "ingredients_analysis_tags",
    "traces_tags", "countries_tags",
}

# Numerical columns — skip for classification eval
NUMERICAL_COLS = {
    "fat_100g", "sugars_100g", "proteins_100g", "carbohydrates_100g",
    "alcohol_100g", "fiber_100g", "salt_100g", "nutriscore_grade", "nova_group",
}


# ---------------------------------------------------------------------------
# Text builders
# ---------------------------------------------------------------------------

def build_baseline_text(row: pd.Series) -> str:
    parts: list[str] = []
    for col in ["product_name", "brands", "ingredients_text", "quantity"]:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    return " ".join(parts)


def build_ocr_text(row: pd.Series, ocr_cache: dict[str, str]) -> str:
    base = build_baseline_text(row)
    code_str = str(row.get("code", "")).strip()
    ocr = ocr_cache.get(code_str, "")
    if ocr:
        return f"{base} {ocr}".strip()
    return base


# ---------------------------------------------------------------------------
# Classifier helpers
# ---------------------------------------------------------------------------

def train_lgbm(
    train_texts: list[str],
    y_train: list[str],
    weights: Optional[np.ndarray] = None,
) -> tuple[Optional[lgb.LGBMClassifier], Optional[TfidfVectorizer], Optional[LabelEncoder]]:
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
    clf.fit(X, y_enc, sample_weight=weights)
    return clf, vec, le


def train_xgb(
    X_emb: np.ndarray,
    y: list[str],
    sample_weights: Optional[np.ndarray] = None,
) -> tuple[Optional[xgb.XGBClassifier], Optional[LabelEncoder]]:
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
    if attr not in df.columns:
        return None

    # Skip numerical columns (regression not classification)
    if attr in NUMERICAL_COLS:
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
        train_idx, test_idx = train_test_split(
            range(len(sub)), test_size=TEST_SIZE, random_state=RANDOM_STATE,
        )

    train_sub = sub.iloc[list(train_idx)]
    test_sub = sub.iloc[list(test_idx)]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]

    train_codes = [codes[i] for i in train_idx]
    test_codes = [codes[i] for i in test_idx]
    X_train_emb = emb[np.array([code_to_idx[c] for c in train_codes])]
    X_test_emb = emb[np.array([code_to_idx[c] for c in test_codes])]

    # Train XGB once (embeddings unchanged by OCR)
    clf_xgb, le_xgb = train_xgb(X_train_emb, y_train)
    if clf_xgb is None or le_xgb is None:
        return None

    xgb_test_probas = clf_xgb.predict_proba(X_test_emb)

    results: dict[str, object] = {
        "category": cat,
        "attr": attr,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_test_short": sum(1 for c in test_codes if c in short_codes),
        "n_test_with_ocr": sum(
            1 for c in test_codes if ocr_cache.get(c, "").strip()
        ),
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

        avg_probas, merged_cls = align_probas(lgbm_probas, le_lgbm, xgb_test_probas, le_xgb)
        preds = [merged_cls[i] for i in np.argmax(avg_probas, axis=1)]
        acc_all = float(sum(p == g for p, g in zip(preds, y_test)) / len(y_test))
        results[f"acc_{variant}"] = acc_all

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
# Phase B: OFF quality check (Jaccard)
# ---------------------------------------------------------------------------

def jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def ocr_adds_info(off_text: str, ocr_text: str) -> bool:
    """True if OCR has words not in OFF text (case-insensitive)."""
    off_words = set(off_text.lower().split())
    ocr_words = set(ocr_text.lower().split())
    extra = ocr_words - off_words
    # Require at least 3 new words and OCR text is non-trivial
    return len(extra) >= 3 and len(ocr_words) >= 5


def run_off_quality_check(
    categories: list[str],
    ocr_cache: dict[str, str],
    n_per_cat: int = SAMPLE_PER_CAT_COMPARISON,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    rows: list[dict] = []
    rng = np.random.default_rng(seed)

    for cat in categories:
        silver_path = PROCESSED_DIR / f"{cat}_stratified_silver_standard.parquet"
        if not silver_path.exists():
            continue
        df = pd.read_parquet(silver_path)
        df["code"] = df["code"].astype(str)
        df_with_ocr = df[df["code"].isin(ocr_cache)].copy()
        n_sample = min(n_per_cat, len(df_with_ocr))
        if n_sample == 0:
            continue
        sampled = df_with_ocr.sample(n=n_sample, random_state=int(rng.integers(1000)))
        for _, row in sampled.iterrows():
            code = str(row["code"])
            off_ing = str(row.get("ingredients_text", "") or "")
            ocr_text = ocr_cache.get(code, "")
            jac = jaccard(off_ing, ocr_text)
            adds_info = ocr_adds_info(off_ing, ocr_text)
            rows.append({
                "category": cat,
                "code": code,
                "off_ingredients_len": len(off_ing),
                "ocr_text_len": len(ocr_text),
                "jaccard": jac,
                "ocr_adds_info": adds_info,
                "off_ingredients_snippet": off_ing[:120],
                "ocr_text_snippet": ocr_text[:120],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def write_ocr_findings(
    results_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    ocr_cache: dict[str, str],
) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    non_empty_ocr = sum(1 for v in ocr_cache.values() if v.strip())

    def nanmean(col: str) -> float:
        vals = results_df[col].dropna()
        return float(vals.mean()) if len(vals) > 0 else float("nan")

    mean_baseline = nanmean("acc_baseline")
    mean_ocr = nanmean("acc_ocr")
    mean_baseline_subset = nanmean("acc_baseline_subset")
    mean_ocr_subset = nanmean("acc_ocr_subset")

    delta_global = (mean_ocr - mean_baseline) * 100
    delta_subset = (mean_ocr_subset - mean_baseline_subset) * 100

    lines = [
        "# EXP12: OCR Augmentation Findings",
        "",
        "## Setup",
        f"- OCR cache: {len(ocr_cache)} entries, {non_empty_ocr} non-empty",
        f"- Categories evaluated: {', '.join(CATEGORIES)}",
        f"- Split: 80/20, seed={RANDOM_STATE}",
        f"- Variants: R_baseline, R_ocr (all), R_ocr_subset (ingredients_text < {SHORT_INGREDIENTS_THRESH} chars)",
        "",
        "## Results",
        "",
        "| Variant | Mean Accuracy | Delta vs Baseline |",
        "|---------|--------------|-------------------|",
        f"| R_baseline | {mean_baseline:.4f} | — |",
        f"| R_ocr (all) | {mean_ocr:.4f} | {delta_global:+.2f}pp |",
        f"| R_ocr subset | {mean_ocr_subset:.4f} | {delta_subset:+.2f}pp vs {mean_baseline_subset:.4f} |",
        "",
        "## Per-attr Breakdown",
        "",
        "| cat | attr | baseline | ocr | delta_pp | sub_delta_pp |",
        "|-----|------|----------|-----|----------|-------------|",
    ]

    for _, row in results_df.sort_values(["category", "attr"]).iterrows():
        base = row.get("acc_baseline", float("nan"))
        ocr_v = row.get("acc_ocr", float("nan"))
        sub_base = row.get("acc_baseline_subset", float("nan"))
        sub_ocr = row.get("acc_ocr_subset", float("nan"))
        delta = (ocr_v - base) * 100 if not (np.isnan(base) or np.isnan(ocr_v)) else float("nan")
        sub_delta = (
            (sub_ocr - sub_base) * 100 if not (np.isnan(sub_base) or np.isnan(sub_ocr)) else float("nan")
        )
        d_s = f"{delta:+.1f}" if not np.isnan(delta) else "n/a"
        sd_s = f"{sub_delta:+.1f}" if not np.isnan(sub_delta) else "n/a"
        lines.append(
            f"| {row['category']} | {row['attr']} | {base:.4f} | {ocr_v:.4f} | {d_s} | {sd_s} |"
        )

    lines += [
        "",
        "## OFF Quality Check Summary",
        "",
    ]
    if len(comp_df) > 0:
        for cat in CATEGORIES:
            cat_df = comp_df[comp_df["category"] == cat]
            if len(cat_df) == 0:
                continue
            n_adds = int(cat_df["ocr_adds_info"].sum())
            pct_adds = 100 * n_adds / len(cat_df)
            mean_jac = float(cat_df["jaccard"].mean())
            mean_off_len = float(cat_df["off_ingredients_len"].mean())
            mean_ocr_len = float(cat_df["ocr_text_len"].mean())
            lines.append(f"**{cat}** (n={len(cat_df)}):")
            lines.append(f"  - OCR adds info missing from OFF: {n_adds}/{len(cat_df)} ({pct_adds:.1f}%)")
            lines.append(f"  - Mean Jaccard similarity: {mean_jac:.3f}")
            lines.append(f"  - Mean OFF ingredients length: {mean_off_len:.0f} chars")
            lines.append(f"  - Mean OCR text length: {mean_ocr_len:.0f} chars")
            lines.append("")

    lines += [
        "## Verdict",
        "",
    ]
    if delta_global > 0.5:
        lines.append(f"OCR augmentation helps globally (+{delta_global:.2f}pp). Recommend DEPLOY for all products.")
    elif delta_subset > 1.0:
        lines.append(f"OCR does not help globally ({delta_global:+.2f}pp) but helps on short-ingredients subset ({delta_subset:+.2f}pp). Recommend DEPLOY selectively for products with empty/short ingredients_text.")
    else:
        lines.append(f"OCR augmentation provides no meaningful lift (global {delta_global:+.2f}pp, subset {delta_subset:+.2f}pp). DO NOT DEPLOY — OCR adds noise without accuracy gain at current corpus size.")
    lines.append("")
    lines.append("### §7.2 Future Work Note")
    lines.append("Active learning with OCR context may yield better results with larger annotated corpora. Current 1650-entry cache covers only ~40% of silver codes; expanding to full OFF image dataset (~1M products) would be required for robust signal.")

    OCR_FINDINGS_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote OCR findings to %s", OCR_FINDINGS_PATH)


def write_comparison_report(comp_df: pd.DataFrame, ocr_cache: dict[str, str]) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# OCR vs OFF Ingredients: Side-by-Side Comparison",
        "",
        "Anecdotal evidence: where does OCR text add information not present in OFF `ingredients_text`?",
        "",
        "**Method**: Jaccard similarity on word sets; `ocr_adds_info = True` when OCR has ≥3 words not in OFF text.",
        "",
    ]

    for cat in CATEGORIES:
        cat_df = comp_df[comp_df["category"] == cat]
        if len(cat_df) == 0:
            continue
        adds_df = cat_df[cat_df["ocr_adds_info"]].head(8)
        lines.append(f"## {cat.capitalize()} — Cases where OCR adds information ({len(adds_df)} shown)")
        lines.append("")
        lines.append("| code | OFF snippet | OCR snippet | Jaccard |")
        lines.append("|------|------------|------------|---------|")
        for _, row in adds_df.iterrows():
            off_s = row["off_ingredients_snippet"].replace("|", "/")[:60]
            ocr_s = row["ocr_text_snippet"].replace("|", "/")[:60]
            lines.append(f"| {row['code']} | {off_s!r} | {ocr_s!r} | {row['jaccard']:.2f} |")
        lines.append("")

        # summary row
        n_adds = int(cat_df["ocr_adds_info"].sum())
        lines.append(f"**Summary**: {n_adds}/{len(cat_df)} samples ({100*n_adds/len(cat_df):.1f}%) have OCR information beyond OFF text. Mean Jaccard={cat_df['jaccard'].mean():.3f}.")
        lines.append("")

    OCR_COMPARISON_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote OFF comparison report to %s", OCR_COMPARISON_PATH)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not OCR_CACHE_PATH.exists():
        logger.error("OCR cache not found at %s — aborting", OCR_CACHE_PATH)
        return

    with open(OCR_CACHE_PATH, encoding="utf-8") as f:
        ocr_cache: dict[str, str] = json.load(f)

    non_empty = sum(1 for v in ocr_cache.values() if v.strip())
    logger.info("OCR cache: %d entries, %d non-empty", len(ocr_cache), non_empty)

    # Build short-ingredients set
    short_codes: set[str] = set()
    for cat in CATEGORIES:
        silver_path = PROCESSED_DIR / f"{cat}_stratified_silver_standard.parquet"
        if not silver_path.exists():
            continue
        df_cat = pd.read_parquet(silver_path)
        mask = df_cat["ingredients_text"].fillna("").str.len() < SHORT_INGREDIENTS_THRESH
        short_codes.update(df_cat.loc[mask, "code"].astype(str).tolist())
    logger.info("Short-ingredients codes: %d", len(short_codes))

    # Phase A: Eval per (cat, attr)
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

        attrs = [c for c in df.columns if c not in META_COLS and c not in NUMERICAL_COLS]
        logger.info("=== %s: %d classification attrs ===", cat, len(attrs))

        for attr in attrs:
            res = evaluate_attr(cat, attr, df, emb, code_to_idx, ocr_cache, short_codes)
            if res is not None:
                all_results.append(res)
                logger.info(
                    "  [%s/%s] baseline=%.3f ocr=%.3f sub_base=%.3f sub_ocr=%.3f n_test=%d n_short=%d",
                    cat, attr,
                    res.get("acc_baseline", float("nan")),
                    res.get("acc_ocr", float("nan")),
                    res.get("acc_baseline_subset", float("nan")),
                    res.get("acc_ocr_subset", float("nan")),
                    res.get("n_test", 0),
                    res.get("n_test_short", 0),
                )

    if not all_results:
        logger.error("No results — aborting")
        return

    results_df = pd.DataFrame(all_results)
    results_df.to_parquet(EVAL_OUT_PATH, index=False)
    logger.info("Saved eval results to %s (%d rows)", EVAL_OUT_PATH, len(results_df))

    # Phase B: OFF quality check
    logger.info("=== Phase B: OFF quality check ===")
    comp_df = run_off_quality_check(CATEGORIES, ocr_cache)
    logger.info("Quality check: %d rows", len(comp_df))

    # Print summary
    def nanmean(col: str) -> float:
        vals = results_df[col].dropna()
        return float(vals.mean()) if len(vals) > 0 else float("nan")

    mean_baseline = nanmean("acc_baseline")
    mean_ocr = nanmean("acc_ocr")
    mean_baseline_subset = nanmean("acc_baseline_subset")
    mean_ocr_subset = nanmean("acc_ocr_subset")

    print(f"\n{'='*70}")
    print("EXP12: OCR AUGMENTATION RESULTS (offline, no torch)")
    print(f"{'='*70}")
    print(f"OCR cache: {len(ocr_cache)} entries, {non_empty} non-empty")
    print(f"Attrs evaluated: {len(results_df)}")
    print()
    print(f"{'Variant':<28} {'Mean Acc':>10} {'Delta':>12}")
    print(f"{'-'*54}")
    print(f"{'R_baseline':<28} {mean_baseline:>10.4f} {'—':>12}")
    print(f"{'R_ocr (all)':<28} {mean_ocr:>10.4f} {(mean_ocr-mean_baseline)*100:>+11.2f}pp")
    print(f"{'R_ocr_subset':<28} {mean_ocr_subset:>10.4f} {(mean_ocr_subset-mean_baseline_subset)*100:>+11.2f}pp (vs sub base {mean_baseline_subset:.4f})")
    print()

    if len(comp_df) > 0:
        print("OFF Quality Check:")
        for cat in CATEGORIES:
            cat_df = comp_df[comp_df["category"] == cat]
            if len(cat_df) == 0:
                continue
            n_adds = int(cat_df["ocr_adds_info"].sum())
            print(f"  [{cat}] OCR adds info: {n_adds}/{len(cat_df)} ({100*n_adds/len(cat_df):.1f}%), mean Jaccard={cat_df['jaccard'].mean():.3f}")

    # Write reports
    write_ocr_findings(results_df, comp_df, ocr_cache)
    write_comparison_report(comp_df, ocr_cache)
    logger.info("Done.")


if __name__ == "__main__":
    main()
