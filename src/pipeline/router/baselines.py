"""
Baseline routing strategies. Each baseline implements:
    decisions = strategy(df, **params)
    decisions[i] = True  ⇒ send to LLM
    decisions[i] = False ⇒ keep cascade answer
"""

from __future__ import annotations

import logging
from typing import Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def static_confidence_threshold(
    df: pd.DataFrame, threshold: float
) -> np.ndarray:
    """Send to LLM iff cascade_conf < threshold.

    At threshold=0 nothing routes to LLM; at threshold=1 everything does.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be in [0, 1]")
    if threshold >= 1.0:
        return np.ones(len(df), dtype=bool)
    if threshold <= 0.0:
        return np.zeros(len(df), dtype=bool)
    return (df["cascade_conf"].values < threshold)


def build_per_attr_table(
    train_df: pd.DataFrame,
) -> dict[tuple[str, str], bool]:
    """For each (cat, attr) pair seen in train, decide whether LLM > cascade on average.

    train_df must include columns: category, attr, cascade_correct, llm_correct.
    Returns {(cat, attr): use_llm_bool}.
    """
    required = {"category", "attr", "cascade_correct", "llm_correct"}
    missing = required - set(train_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    table: dict[tuple[str, str], bool] = {}
    for (cat, attr), grp in train_df.groupby(["category", "attr"]):
        c_acc = grp["cascade_correct"].mean()
        l_acc = grp["llm_correct"].mean()
        table[(cat, attr)] = bool(l_acc > c_acc)
    return table


def per_attr_static_table(
    df: pd.DataFrame,
    table: Mapping[tuple[str, str], bool],
    default_use_llm: bool = False,
) -> np.ndarray:
    """For each row, look up (cat, attr) → use_llm. Unseen pairs use default."""
    decisions = np.array([
        table.get((c, a), default_use_llm)
        for c, a in zip(df["category"].values, df["attr"].values)
    ])
    return decisions


def random_router(
    df: pd.DataFrame, llm_budget: float, seed: int = 42
) -> np.ndarray:
    """Send a random `llm_budget` fraction of rows to LLM."""
    if not (0.0 <= llm_budget <= 1.0):
        raise ValueError("llm_budget must be in [0, 1]")
    rng = np.random.default_rng(seed)
    return rng.uniform(size=len(df)) < llm_budget
