"""
Per-class precision / recall / F1 + macro-F1 for multiclass attributes.

Closes reviewer claim: accuracy alone hides class-specific failures on
unbalanced multiclass attributes.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

logger = logging.getLogger(__name__)


def compute_per_class_metrics(
    y_true: list | np.ndarray | pd.Series,
    y_pred: list | np.ndarray | pd.Series,
) -> dict[str, Any]:
    """Compute per-class precision/recall/F1 + macro-F1 + accuracy."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        str(lbl): {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, lbl in enumerate(labels)
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": per_class,
    }


def compute_for_all_categories(
    categories: list[str],
    processed_dir: str,
) -> pd.DataFrame:
    """For each (category, attr) in experiment_per_product_*: compute per-class metrics."""
    rows = []
    for cat in categories:
        path = f"{processed_dir}/experiment_per_product_{cat}_stratified.parquet"
        try:
            df = pd.read_parquet(path)
        except FileNotFoundError:
            logger.warning("Skipping %s: %s not found", cat, path)
            continue
        df = df[df["config"] == "regex_ml_bayes"].dropna(subset=["gt", "pred"])
        for attr in df["attr"].unique():
            sub = df[df["attr"] == attr]
            if sub["gt"].nunique() < 2:
                continue
            metrics = compute_per_class_metrics(sub["gt"].astype(str), sub["pred"].astype(str))
            for cls, m in metrics["per_class"].items():
                rows.append({
                    "category": cat,
                    "attr": attr,
                    "class": cls,
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "support": m["support"],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                })
    return pd.DataFrame(rows)


def main():
    import os
    from src.common import PROCESSED_DIR, setup_logging

    setup_logging()
    cats = ["pasta", "chocolate", "beverages", "cheeses", "cereals", "cosmetics"]
    df = compute_for_all_categories(cats, PROCESSED_DIR)
    out = os.path.join(PROCESSED_DIR, "per_class_metrics.parquet")
    df.to_parquet(out, index=False)
    logger.info("Saved %s (%d rows)", out, len(df))


if __name__ == "__main__":
    main()
