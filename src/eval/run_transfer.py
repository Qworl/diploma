"""
Cross-category transfer learning experiments.

For shared attributes (is_organic, nutri_score_grade, protein_class),
applies a source-trained classifier to target test set without retraining
(zero-shot) and reports per-attribute accuracy.

Categories: pasta, chocolate, beverages.

Usage:
    python scripts/run_transfer.py
    python scripts/run_transfer.py --source pasta --target chocolate
"""

import argparse
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.common import MODELS_DIR, PROCESSED_DIR, RANDOM_STATE, TEST_SIZE, setup_logging, wilson_ci

logger = logging.getLogger(__name__)

CATEGORY_FILES = {
    "pasta": ("pasta_silver_standard.parquet", "pasta_embeddings.npy"),
    "chocolate": ("chocolate_silver_standard.parquet", "chocolate_embeddings.npy"),
    "beverages": ("beverages_silver_standard.parquet", "beverages_embeddings.npy"),
}

SHARED_ATTRS = ["is_organic", "nutri_score_grade", "protein_class"]

ALL_PAIRS = [
    ("pasta", "chocolate"),
    ("pasta", "beverages"),
    ("chocolate", "beverages"),
    ("chocolate", "pasta"),
    ("beverages", "pasta"),
    ("beverages", "chocolate"),
]


def load_target_test(category: str):
    ss_file, emb_file = CATEGORY_FILES[category]
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, ss_file))
    emb = np.load(os.path.join(PROCESSED_DIR, emb_file))
    _, test_idx = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return df.iloc[test_idx].reset_index(drop=True), emb[test_idx]


def zero_shot(source: str, target: str, attr: str) -> dict | None:
    xgb_path = os.path.join(MODELS_DIR, f"{source}_{attr}_xgb.pkl")
    le_path = os.path.join(MODELS_DIR, f"{source}_{attr}_le.pkl")
    if not os.path.exists(xgb_path):
        return None

    with open(xgb_path, "rb") as f:
        clf = pickle.load(f)
    le = None
    if os.path.exists(le_path):
        with open(le_path, "rb") as f:
            le = pickle.load(f)

    target_df, target_emb = load_target_test(target)
    if attr not in target_df.columns:
        return None

    mask = target_df[attr].notna()
    if mask.sum() == 0:
        return None

    X = target_emb[mask.values]
    y_true = target_df.loc[mask, attr].astype(str).values

    proba = clf.predict_proba(X)
    pred_idx = proba.argmax(axis=1)
    if le is not None:
        try:
            y_pred = le.inverse_transform(pred_idx).astype(str)
        except Exception:
            return None
    else:
        y_pred = np.array([str(bool(i)) for i in pred_idx])

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Majority-class baseline на target distribution — без него accuracy на
    # binary/skewed-multiclass атрибутах не интерпретируется (76% is_organic
    # = просто предсказывать False каждый раз, если 76% данных = False).
    y_series = pd.Series(y_true)
    majority_class = y_series.value_counts().idxmax()
    majority_acc = (y_series == majority_class).mean()
    lift = acc - majority_acc

    n_correct = int((y_true == y_pred).sum())
    ci_lo, ci_hi = wilson_ci(n_correct, int(mask.sum()))

    return {
        "n": int(mask.sum()),
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "majority_class": str(majority_class),
        "majority_acc": float(majority_acc),
        "lift_over_majority": float(lift),
        "acc_ci_lo": float(ci_lo),
        "acc_ci_hi": float(ci_hi),
    }


def run_transfer_experiments(source: str | None = None, target: str | None = None):
    pairs = ALL_PAIRS
    if source and target:
        pairs = [(source, target)]

    results = []
    for src, tgt in pairs:
        if src == tgt:
            continue
        for attr in SHARED_ATTRS:
            r = zero_shot(src, tgt, attr)
            if r is None:
                continue
            results.append({"source": src, "target": tgt, "attr": attr, **r})

    if not results:
        logger.info("No results — check source models / target data exist")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    logger.info("=" * 80)
    logger.info("CROSS-CATEGORY TRANSFER (zero-shot)")
    logger.info("=" * 80)
    logger.info("%-11s %-11s %-22s %7s %5s %5s %5s %15s",
                "source", "target", "attr", "acc", "[ci]", "majo", "lift", "(95% CI)")
    for _, r in df.iterrows():
        logger.info("%-11s %-11s %-22s %6.1f%% %5.1f%% %5.1f%% %+5.1fpp  [%5.1f, %5.1f]",
                    r["source"], r["target"], r["attr"],
                    r["acc"] * 100,
                    r["acc_ci_lo"] * 100,
                    r["majority_acc"] * 100,
                    r["lift_over_majority"] * 100,
                    r["acc_ci_lo"] * 100, r["acc_ci_hi"] * 100)

    out_path = os.path.join(PROCESSED_DIR, "transfer_results.parquet")
    df.to_parquet(out_path, index=False)
    logger.info("Saved %s", out_path)
    return df


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=list(CATEGORY_FILES.keys()), default=None)
    parser.add_argument("--target", choices=list(CATEGORY_FILES.keys()), default=None)
    args = parser.parse_args()
    run_transfer_experiments(source=args.source, target=args.target)


if __name__ == "__main__":
    main()
