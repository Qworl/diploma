import pandas as pd
from src.diagnostics.ml.per_class_metrics import compute_per_class_metrics


def test_per_class_metrics_balanced_multiclass():
    """3-class balanced, all correct → precision=recall=f1=1 per class, macro_f1=1.0"""
    y_true = ["a", "b", "c", "a", "b", "c"]
    y_pred = ["a", "b", "c", "a", "b", "c"]
    result = compute_per_class_metrics(y_true, y_pred)
    assert result["macro_f1"] == 1.0
    assert result["per_class"]["a"]["precision"] == 1.0
    assert result["per_class"]["a"]["recall"] == 1.0
    assert result["per_class"]["a"]["f1"] == 1.0


def test_per_class_metrics_imbalanced_with_errors():
    """Top class dominant, model only predicts top → macro_f1 < accuracy"""
    y_true = ["a"] * 80 + ["b"] * 10 + ["c"] * 10
    y_pred = ["a"] * 100
    result = compute_per_class_metrics(y_true, y_pred)
    assert result["accuracy"] == 0.80
    assert result["per_class"]["b"]["recall"] == 0.0
    assert result["per_class"]["c"]["recall"] == 0.0
    assert result["macro_f1"] < 0.40


def test_per_class_metrics_handles_unseen_pred_class():
    """Predicted class not in y_true → precision=0, recall=undefined (zero_division=0)"""
    y_true = ["a", "a", "b"]
    y_pred = ["a", "c", "c"]
    result = compute_per_class_metrics(y_true, y_pred)
    assert "c" in result["per_class"]
    assert result["per_class"]["c"]["precision"] == 0.0
