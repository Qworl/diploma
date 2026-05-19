"""Stratified sampler for multi-domain manual-audit gold sets (Trek E).

Unlike `sample_pasta_gold.py` (which relies on pasta-only artefacts:
consensus_gold_v1_emulated.parquet + brand_disjoint_split.parquet), this module samples
from any domain's silver-standard parquet using three pools defined purely
in terms of silver content:

  Pool A (typical, default 60%): stratified random by `primary_attr`,
      taken from products where ALL schema attributes are non-null in silver.
      Represents "easy" rows where the silver-extractor was confident.

  Pool B (silver-empty, default 25%): at least one schema attribute is null.
      Represents the manual_only / coverage-gap case.

  Pool C (silver-hard, default 15%): at least three schema attributes are
      null. Represents the hardest rows where the silver-extractor failed
      multiple times. Substitute for the pasta "cascade-disagreement" pool
      when consensus_gold_v1_emulated parquet is unavailable.

Outputs the full Trek D gold schema:
    silver_<attr>, manual_<attr>, manual_<attr>_status, manual_<attr>_at,
    manual_<attr>_mode, manual_<attr>_note  --- one block per attribute.

Run::

    python -m src.manual_label.sample_domain_gold \\
        --domain chocolate \\
        --silver datasets/processed/chocolate_stratified_silver_standard.parquet \\
        --n-total 239 \\
        --out datasets/manual_label/chocolate_gold_239.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.manual_label.schemas_loader import load_domain_attrs

logger = logging.getLogger(__name__)


class SamplingError(RuntimeError):
    pass


_PASSTHROUGH_COLS = [
    "code", "product_name", "brands", "ingredients_text", "quantity", "lang",
]

# Per-domain primary stratifier (most discriminative attr for Pool A).
_PRIMARY_ATTR: dict[str, str] = {
    "chocolate": "chocolate_type",
    "cheeses": "texture",
    "beverages": "beverage_type",
    "cereals": "cereal_type",
    "cosmetics": "body_area",
}


def _select_pool_a_typical(silver: pd.DataFrame, attrs: list[str], primary: str,
                            n: int, seed: int) -> list[str]:
    """Stratified sample by `primary` from rows with zero silver-nulls."""
    cols_present = [a for a in attrs if a in silver.columns]
    mask_no_nulls = silver[cols_present].notna().all(axis=1)
    pool = silver[mask_no_nulls].copy()
    if primary not in pool.columns or pool[primary].isna().all():
        # Fallback: random.
        if len(pool) < n:
            raise SamplingError(
                f"Pool A: zero-null pool too small: {len(pool)} < {n} (no primary stratifier)"
            )
        return pool.sample(n=n, random_state=seed)["code"].astype(str).tolist()
    # Proportional allocation per primary value.
    counts = pool[primary].value_counts()
    if counts.empty or len(pool) < n:
        raise SamplingError(f"Pool A: zero-null pool too small: {len(pool)} < {n}")
    # Compute floor allocations + distribute remainder by largest fractional part.
    fractions = counts / counts.sum() * n
    floors = fractions.astype(int)
    remainder = n - int(floors.sum())
    fractional = (fractions - floors).sort_values(ascending=False)
    allocations = floors.to_dict()
    for v in list(fractional.index)[:remainder]:
        allocations[v] += 1
    picked: list[str] = []
    rng_state = seed
    for primary_val, k in allocations.items():
        if k <= 0:
            continue
        bucket = pool[pool[primary] == primary_val]
        take = min(k, len(bucket))
        picked.extend(
            bucket.sample(n=take, random_state=rng_state)["code"].astype(str).tolist()
        )
        rng_state += 1
    # Top-up if rounding left us short.
    if len(picked) < n:
        remaining = pool[~pool["code"].astype(str).isin(picked)]
        topup = remaining.sample(n=n - len(picked), random_state=seed + 999)
        picked.extend(topup["code"].astype(str).tolist())
    return picked[:n]


def _select_pool_b_silver_empty(silver: pd.DataFrame, attrs: list[str], exclude: set[str],
                                 n: int, seed: int) -> list[str]:
    """Random sample from rows with 1+ silver-null."""
    cols_present = [a for a in attrs if a in silver.columns]
    null_counts = silver[cols_present].isna().sum(axis=1)
    mask = (null_counts >= 1) & (null_counts <= 2)  # bias toward few-nulls in pool B
    pool = silver[mask & ~silver["code"].astype(str).isin(exclude)]
    if len(pool) < n:
        # Relax constraint: any null.
        mask = null_counts >= 1
        pool = silver[mask & ~silver["code"].astype(str).isin(exclude)]
    if len(pool) < n:
        raise SamplingError(f"Pool B: silver-empty pool too small: {len(pool)} < {n}")
    return pool.sample(n=n, random_state=seed)["code"].astype(str).tolist()


def _select_pool_c_hard(silver: pd.DataFrame, attrs: list[str], exclude: set[str],
                        n: int, seed: int) -> list[str]:
    """Random sample from rows with 3+ silver-nulls."""
    cols_present = [a for a in attrs if a in silver.columns]
    null_counts = silver[cols_present].isna().sum(axis=1)
    mask = null_counts >= 3
    pool = silver[mask & ~silver["code"].astype(str).isin(exclude)]
    if len(pool) < n:
        # Relax to 2+ nulls.
        mask = null_counts >= 2
        pool = silver[mask & ~silver["code"].astype(str).isin(exclude)]
    if len(pool) < n:
        raise SamplingError(f"Pool C: hard pool too small: {len(pool)} < {n}")
    return pool.sample(n=n, random_state=seed)["code"].astype(str).tolist()


def build_sample(
    *,
    domain: str,
    silver: pd.DataFrame,
    n_total: int = 239,
    pool_ratio: tuple[int, int, int] = (60, 25, 15),
    seed: int = 42,
) -> pd.DataFrame:
    """Build the gold-annotation seed dataframe for a domain.

    Pool ratios are interpreted as percentages summing to 100. The function
    raises if any pool can't fill its allocation (caller should pick smaller
    n_total or relax pool ratios).
    """
    if sum(pool_ratio) != 100:
        raise SamplingError(f"Pool ratios must sum to 100, got {pool_ratio}")
    n_a = round(n_total * pool_ratio[0] / 100)
    n_b = round(n_total * pool_ratio[1] / 100)
    n_c = n_total - n_a - n_b  # remainder absorbs rounding

    attrs = list(load_domain_attrs(domain))
    primary = _PRIMARY_ATTR.get(domain) or attrs[0]
    silver = silver.copy()
    silver["code"] = silver["code"].astype(str)

    a_codes = _select_pool_a_typical(silver, attrs, primary, n_a, seed)
    chosen = set(a_codes)
    b_codes = _select_pool_b_silver_empty(silver, attrs, chosen, n_b, seed + 1)
    chosen |= set(b_codes)
    c_codes = _select_pool_c_hard(silver, attrs, chosen, n_c, seed + 2)

    sources = (
        [("pool_a_typical", c) for c in a_codes]
        + [("pool_b_silver_empty", c) for c in b_codes]
        + [("pool_c_hard", c) for c in c_codes]
    )

    src_idx = silver.set_index("code")
    rows = []
    for source, code in sources:
        if code not in src_idx.index:
            continue
        s = src_idx.loc[code]
        row = {c: ("" if c not in s.index or pd.isna(s[c]) else s[c])
               for c in _PASSTHROUGH_COLS}
        row["code"] = code
        row["source"] = source
        for attr in attrs:
            v = s.get(attr, "")
            row[f"silver_{attr}"] = "" if (v is None or pd.isna(v)) else v
            row[f"manual_{attr}"] = ""
            row[f"manual_{attr}_status"] = "empty"
            row[f"manual_{attr}_at"] = ""
            row[f"manual_{attr}_mode"] = ""
            row[f"manual_{attr}_note"] = ""
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True, choices=list(_PRIMARY_ATTR))
    p.add_argument("--silver", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n-total", type=int, default=239)
    p.add_argument("--pool-ratio", default="60:25:15",
                   help="Pool A:B:C percentages (sum=100), e.g. 60:25:15")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    silver = pd.read_parquet(args.silver)
    ratio = tuple(int(x) for x in args.pool_ratio.split(":"))
    if len(ratio) != 3:
        raise SystemExit(f"pool-ratio must have 3 components, got {args.pool_ratio}")

    df = build_sample(
        domain=args.domain,
        silver=silver,
        n_total=args.n_total,
        pool_ratio=ratio,  # type: ignore[arg-type]
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows to {args.out}")
    print(df["source"].value_counts())


if __name__ == "__main__":
    main()
