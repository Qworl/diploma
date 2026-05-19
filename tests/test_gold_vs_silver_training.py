"""Module exposes a train_and_score() that returns dict with accuracy."""
import pandas as pd
import numpy as np
from src.experiments.gold_vs_silver_training import train_xgb_and_score


def test_train_xgb_returns_accuracy_in_unit_range():
    # 50 train + 20 test samples, 2 features, 2 classes
    rng = np.random.default_rng(0)
    X_tr = rng.normal(size=(50, 8))
    y_tr = (X_tr[:, 0] > 0).astype(str)
    X_te = rng.normal(size=(20, 8))
    y_te = (X_te[:, 0] > 0).astype(str)
    acc = train_xgb_and_score(X_tr, y_tr, X_te, y_te)
    assert 0.0 <= acc <= 1.0
