"""Brand-disjoint train/val/test split.

Назначение: на каждый бренд приходится строго один split. Закрывает blocker
brand leakage (см. spec 2026-05-13, blocker №2): random split допускает
overlap бренда между train и test, что для атрибутов вроде is_organic
(которые сильно тянутся брендом, см. §6.2 ablation) даёт оптимистично
смещённую accuracy.

Алгоритм: greedy bin-packing. Бренды сортируются по убыванию количества
продуктов; каждый кладётся в split с минимальной текущей долей относительно
target ratio.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd


def brand_disjoint_split(
    df: pd.DataFrame,
    brand_col: str = "brand",
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 42,
    check_class_col: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """Разбить df на train/val/test так, что один бренд = один split.

    Parameters
    ----------
    df : pd.DataFrame
        Любой DataFrame с колонкой бренда.
    brand_col : str
        Имя колонки бренда (по умолчанию "brand").
    ratios : tuple of 3 floats
        Целевые доли для train, val, test. Должны суммироваться к 1.0.
    seed : int
        Для tie-breaking между брендами с равными размерами.
    check_class_col : str or None
        Если задано — проверить, что каждый class каждого split'а
        присутствует ≥ 1 раз; если нет — выдать UserWarning.

    Returns
    -------
    dict
        {"train": pd.DataFrame, "val": pd.DataFrame, "test": pd.DataFrame}.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")

    brand_sizes = df.groupby(brand_col).size().sort_values(ascending=False)
    # tie-breaking
    rng = np.random.default_rng(seed)
    brands_list = list(brand_sizes.index)
    rng.shuffle(brands_list)
    brand_sizes = brand_sizes.reindex(brands_list).sort_values(
        ascending=False, kind="stable"
    )

    n_total = len(df)
    targets = {"train": ratios[0] * n_total,
               "val":   ratios[1] * n_total,
               "test":  ratios[2] * n_total}
    current = {"train": 0, "val": 0, "test": 0}
    assignment: dict[str, str] = {}  # brand -> split

    for brand, size in brand_sizes.items():
        # Бренд идёт в split с наименьшей долей выполнения target.
        ratio_progress = {s: current[s] / max(targets[s], 1e-9)
                          for s in ("train", "val", "test")}
        target_split = min(ratio_progress, key=ratio_progress.get)
        assignment[brand] = target_split
        current[target_split] += size

    out = {}
    for split in ("train", "val", "test"):
        brands_in_split = [b for b, s in assignment.items() if s == split]
        out[split] = df[df[brand_col].isin(brands_in_split)].copy()

    # Class coverage warning
    if check_class_col is not None and check_class_col in df.columns:
        all_classes = set(df[check_class_col].dropna().unique())
        for split in ("train", "val", "test"):
            split_classes = set(out[split][check_class_col].dropna().unique())
            missing = all_classes - split_classes
            if missing:
                warnings.warn(
                    f"class coverage incomplete in {split}: missing {missing}",
                    UserWarning,
                )

    return out
