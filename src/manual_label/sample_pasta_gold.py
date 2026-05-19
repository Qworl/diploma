"""Sample 250 pasta products for manual gold annotation.

Stratified pick:
- 150 from brand-disjoint test fold (pasta_gold_split.parquet, split=='test')
- 50 from disagreement subset (cascade != Sonnet/GPT-4o on pasta_shape/grain_type)
- 50 from gold-tier control (silver nutri_score_grade + protein_class both set)

Usage:
    python -m src.manual_label.sample_pasta_gold \\
        --out datasets/manual_label/pasta_gold_250.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd

from src.manual_label.schemas_loader import load_pasta_attrs


class SamplingError(RuntimeError):
    pass


_PASSTHROUGH_COLS = [
    "code", "product_name", "brands", "ingredients_text", "quantity", "lang",
]


def _select_disagreement_codes(
    disagreement: pd.DataFrame,
    n: int,
    seed: int,
    exclude: set[str] | None = None,
) -> list[str]:
    # Pool B intentionally truncates silently when the source has fewer than `n`
    # candidates — see plan §3.1 fallback policy. Pools A and C raise SamplingError.
    if disagreement.empty:
        return []
    exclude = exclude or set()
    rel = disagreement[disagreement["attr"].isin(["pasta_shape", "grain_type"])]
    codes = [c for c in rel["code"].drop_duplicates().tolist() if c not in exclude]
    if len(codes) > n:
        codes = (
            pd.Series(codes)
            .sample(n=n, random_state=seed)
            .tolist()
        )
    return codes


def _select_control_codes(silver_extended: pd.DataFrame, exclude: set[str], n: int, seed: int) -> list[str]:
    pool = silver_extended[
        silver_extended["silver_nutri_score_grade"].notna()
        & silver_extended["silver_protein_class"].notna()
        & ~silver_extended["code"].isin(exclude)
    ]
    if len(pool) < n:
        raise SamplingError(f"gold-tier control pool too small: {len(pool)} < {n}")
    return pool.sample(n=n, random_state=seed)["code"].tolist()


def _select_test_codes(
    silver_extended: pd.DataFrame,
    split: pd.DataFrame,
    n: int,
    seed: int,
    exclude: set[str] | None = None,
) -> list[str]:
    exclude = exclude or set()
    test_codes = set(split[split["split"] == "test"]["code"])
    pool = silver_extended[
        silver_extended["code"].isin(test_codes)
        & ~silver_extended["code"].isin(exclude)
    ]
    if len(pool) < n:
        raise SamplingError(f"brand-disjoint test pool too small: {len(pool)} < {n}")
    return pool.sample(n=n, random_state=seed)["code"].tolist()


def build_sample(
    *,
    silver_extended: pd.DataFrame,
    split: pd.DataFrame,
    disagreement: pd.DataFrame,
    n_total: int = 250,
    n_test: int = 150,
    n_disagreement: int = 50,
    n_control: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Build the gold-annotation seed dataframe."""
    if n_test + n_disagreement + n_control != n_total:
        raise SamplingError("n_test + n_disagreement + n_control must equal n_total")

    # Priority A > B > C on overlap. Fill A first, then B from remaining
    # disagreement codes, then C from remaining control pool.
    test_codes = _select_test_codes(silver_extended, split, n_test, seed)
    chosen = set(test_codes)

    dis_codes = _select_disagreement_codes(disagreement, n_disagreement, seed, exclude=chosen)
    chosen |= set(dis_codes)

    ctrl_codes = _select_control_codes(silver_extended, chosen, n_control, seed)
    chosen |= set(ctrl_codes)

    sources = (
        [("brand_disjoint_test", c) for c in test_codes]
        + [("disagreement", c) for c in dis_codes]
        + [("gold_tier_control", c) for c in ctrl_codes]
    )

    rows = []
    src_idx = silver_extended.set_index("code")
    attrs = load_pasta_attrs()
    for source, code in sources:
        if code not in src_idx.index:
            continue
        s = src_idx.loc[code]
        row = {c: (s[c] if c in s.index else "") for c in _PASSTHROUGH_COLS}
        row["code"] = code
        row["source"] = source
        for attr in attrs:
            v = s.get(f"silver_{attr}", "")
            row[f"silver_{attr}"] = "" if pd.isna(v) else v
            row[f"manual_{attr}"] = ""
            row[f"manual_{attr}_status"] = "empty"
            row[f"manual_{attr}_at"] = ""
            row[f"manual_{attr}_mode"] = ""
            row[f"manual_{attr}_note"] = ""
        rows.append(row)

    return pd.DataFrame(rows)


def _prepare_silver_extended(silver: pd.DataFrame) -> pd.DataFrame:
    """Normalise real `pasta_stratified_silver_extended.parquet` to the schema
    expected by `build_sample`: silver_<attr> columns + `lang` column.
    """
    attrs = list(load_pasta_attrs())
    rename_map = {a: f"silver_{a}" for a in attrs if a in silver.columns}
    out = silver.rename(columns=rename_map).copy()
    if "lang" not in out.columns:
        out["lang"] = ""
    return out


def _prepare_disagreement(consensus: pd.DataFrame) -> pd.DataFrame:
    """Derive a (code, attr, cascade_pred, llm_pred) frame from
    `consensus_gold_v1_emulated.parquet`. A row counts as disagreement when the cascade
    silver value differs from the LLM consensus on `pasta_shape` or
    `grain_type`.
    """
    df = consensus
    if "category" in df.columns:
        df = df[df["category"] == "pasta"]
    df = df[df["attr"].isin(["pasta_shape", "grain_type"])]
    df = df[df["silver_value"].notna() & df["gt_consensus"].notna()]
    df = df[df["silver_value"] != df["gt_consensus"]]
    return pd.DataFrame({
        "code": df["code"].astype(str).values,
        "attr": df["attr"].values,
        "cascade_pred": df["silver_value"].values,
        "llm_pred": df["gt_consensus"].values,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--silver",
        default="datasets/processed/pasta_stratified_silver_extended.parquet",
    )
    parser.add_argument(
        "--split",
        default="datasets/processed/pasta_gold_split.parquet",
    )
    parser.add_argument(
        "--disagreement",
        default="datasets/processed/consensus_gold_v1_emulated.parquet",
        help="Source of cascade<->LLM disagreement (filtered to category=pasta).",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-total", type=int, default=250)
    parser.add_argument("--n-test", type=int, default=150)
    parser.add_argument("--n-disagreement", type=int, default=50)
    parser.add_argument("--n-control", type=int, default=50)
    args = parser.parse_args()

    silver = _prepare_silver_extended(pd.read_parquet(args.silver))
    split = pd.read_parquet(args.split)
    try:
        consensus = pd.read_parquet(args.disagreement)
        dis = _prepare_disagreement(consensus)
    except FileNotFoundError:
        dis = pd.DataFrame(columns=["code", "attr", "cascade_pred", "llm_pred"])

    df = build_sample(
        silver_extended=silver,
        split=split,
        disagreement=dis,
        n_total=args.n_total,
        n_test=args.n_test,
        n_disagreement=args.n_disagreement,
        n_control=args.n_control,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows to {args.out}")
    print(df["source"].value_counts())


if __name__ == "__main__":
    main()
