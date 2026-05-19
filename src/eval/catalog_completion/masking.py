"""Deterministic per-product attribute masking driven by missingness profile."""
from __future__ import annotations

import hashlib
import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def sentinel_for(dtype: str) -> object:
    """Value to write into a masked cell. We use None everywhere — cascade code
    already treats None / NaN / empty-string as missing."""
    return None


def _row_seed(global_seed: int, code: object) -> int:
    """Stable 32-bit int derived from (global_seed, code)."""
    h = hashlib.blake2b(f"{global_seed}:{code}".encode(), digest_size=4).digest()
    return int.from_bytes(h, "big")


def _attr_seed(row_seed: int, attr: str) -> int:
    h = hashlib.blake2b(f"{row_seed}:{attr}".encode(), digest_size=4).digest()
    return int.from_bytes(h, "big")


def _bernoulli(seed: int, p: float) -> bool:
    """Deterministic bernoulli draw in [0,1) from a 32-bit seed."""
    # 2**32 = 4294967296
    u = seed / 4294967296.0
    return u < p


def mask_row(row: pd.Series, profile: dict, *, global_seed: int) -> pd.Series:
    """Return a copy of `row` with attributes deleted according to profile.

    `row` must contain a `code` field for stable seeding.
    """
    code = row.get("code")
    rseed = _row_seed(global_seed, code)
    out = row.copy()
    for bucket in ("partner_attrs", "target_attrs"):
        for attr, p in profile.get(bucket, {}).items():
            if attr not in out.index:
                continue
            aseed = _attr_seed(rseed, attr)
            if _bernoulli(aseed, p):
                out[attr] = sentinel_for(str(out[attr].__class__.__name__))
    return out


def mask_dataframe(
    df: pd.DataFrame, profile: dict, *, global_seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply `mask_row` to every row.

    Returns (masked_df, mask_log) where mask_log is long-format with columns
    (code, attr, masked, original_value).
    """
    masked_rows = []
    log_rows = []
    all_attrs = list(profile.get("partner_attrs", {})) + list(profile.get("target_attrs", {}))
    for _, row in df.iterrows():
        m = mask_row(row, profile, global_seed=global_seed)
        masked_rows.append(m)
        code = row.get("code")
        for a in all_attrs:
            if a not in row.index:
                continue
            orig = row[a]
            new = m[a]
            was_masked = (
                (pd.notna(orig) and (not isinstance(orig, str) or orig.strip() != ""))
                and (pd.isna(new) or (isinstance(new, str) and new.strip() == ""))
            )
            log_rows.append({
                "code": code,
                "attr": a,
                "masked": bool(was_masked),
                "original_value": None if pd.isna(orig) else (
                    str(orig) if not isinstance(orig, (bool,)) else bool(orig)
                ),
            })
    return pd.DataFrame(masked_rows).reset_index(drop=True), pd.DataFrame(log_rows)
