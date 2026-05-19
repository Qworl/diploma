"""Brand-disjoint train/test split for category router."""
from __future__ import annotations

import numpy as np
import pandas as pd


def brand_disjoint_split(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df so that no brand appears in both halves.

    Rows with empty `brand` are distributed independently via random split.
    """
    rng = np.random.default_rng(seed)
    branded = df[df["brand"].astype(str) != ""].copy()
    unbranded = df[df["brand"].astype(str) == ""].copy()

    brands = np.array(sorted(branded["brand"].unique()))
    rng.shuffle(brands)
    n_test_brands = max(1, int(round(len(brands) * test_size)))
    test_brand_set = set(brands[:n_test_brands].tolist())

    branded_train = branded[~branded["brand"].isin(test_brand_set)]
    branded_test = branded[branded["brand"].isin(test_brand_set)]

    if len(unbranded) > 0:
        u = unbranded.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        cut = int(round(len(u) * test_size))
        unbranded_test = u.iloc[:cut]
        unbranded_train = u.iloc[cut:]
    else:
        unbranded_test = unbranded
        unbranded_train = unbranded

    train = pd.concat([branded_train, unbranded_train], ignore_index=True)
    test = pd.concat([branded_test, unbranded_test], ignore_index=True)
    train = train.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test = test.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train, test
