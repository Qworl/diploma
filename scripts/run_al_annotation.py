"""P2-EXP11 Step 2: Annotate AL + random control codes via gpt-5.5 (off_grounded).

No torch/sentence-transformers imported here — pure LLM + pandas.

Usage:
    OMP_NUM_THREADS=2 python scripts/run_al_annotation.py \
        --cat pasta --variant uncertain
    OMP_NUM_THREADS=2 python scripts/run_al_annotation.py \
        --cat pasta --variant random
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Paths
WORKTREE_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = WORKTREE_ROOT / "datasets" / "processed"
MANUAL_LABEL = WORKTREE_ROOT / "datasets" / "manual_label"
OFF_CACHE_DIR = MANUAL_LABEL / "off_cache"
OFF_PARQUET = WORKTREE_ROOT / "datasets" / "raw" / "en.openfoodfacts.org.products.parquet"

MODEL = "openai/gpt-5.5"
MAX_COST_PER_RUN = 16.0  # USD per (cat, variant)


def get_codes(cat: str, variant: str) -> list[str]:
    if variant == "uncertain":
        csv_path = MANUAL_LABEL / f"al_codes_{cat}.csv"
    else:
        csv_path = MANUAL_LABEL / f"al_control_codes_{cat}.csv"
    return pd.read_csv(csv_path)["code"].astype(str).tolist()


def populate_cache(codes: list[str]) -> None:
    missing = [c for c in codes if not (OFF_CACHE_DIR / f"{c}.json").exists()]
    if not missing:
        print(f"  Cache: all {len(codes)} codes present")
        return
    print(f"  Cache: populating {len(missing)} missing codes...")
    tmp = Path("/tmp/al_codes_populate.txt")
    tmp.write_text("\n".join(missing) + "\n")
    cmd = [
        sys.executable,
        str(WORKTREE_ROOT / "scripts" / "populate_off_cache_from_parquet.py"),
        "--codes-file", str(tmp),
        "--cache-dir", str(OFF_CACHE_DIR),
        "--parquet", str(OFF_PARQUET),
    ]
    result = subprocess.run(cmd, cwd=str(WORKTREE_ROOT), capture_output=True, text=True)
    print(result.stdout.strip())
    if result.returncode != 0:
        print("WARN:", result.stderr.strip()[:500])


def get_pool_df(cat: str, codes: list[str]) -> pd.DataFrame:
    """Load product fields for given codes from OFF parquet."""
    print(f"  Loading pool products for {cat}...")
    cols = ["code", "product_name", "brands", "ingredients_text", "quantity", "categories_tags"]
    off = pd.read_parquet(OFF_PARQUET, columns=cols)
    off["code"] = off["code"].astype(str)
    pool = off[off["code"].isin(set(codes))].copy().reset_index(drop=True)
    print(f"  Found {len(pool)} / {len(codes)} codes in OFF parquet")
    return pool


def annotate(cat: str, variant: str, codes: list[str], pool_df: pd.DataFrame,
             api_key: str) -> None:
    out_dir = PROCESSED / (f"al_gpt55_uncertain" if variant == "uncertain" else "al_gpt55_random")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cat}_gold.parquet"

    # Check existing
    existing_codes: set[str] = set()
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        existing_codes = set(existing["code"].astype(str))
        print(f"  Resuming: {len(existing_codes)} already done")

    remaining = [c for c in codes if c not in existing_codes]
    print(f"  To annotate: {len(remaining)} codes (out of {len(codes)})")
    if not remaining:
        print("  All done — skipping")
        return

    products = pool_df[pool_df["code"].isin(remaining)].copy()
    if len(products) == 0:
        print(f"  WARN: no products in pool for remaining codes — skipping")
        return

    # Run annotation
    sys.path.insert(0, str(WORKTREE_ROOT))
    from src.eval.direct_llm_v2 import run_llm_on_products
    from src.llm.client import call_openrouter

    df = run_llm_on_products(
        products,
        domain=cat,
        model=MODEL,
        api_key=api_key,
        out_path=out_path,
        context_mode="off_grounded",
        off_cache_dir=OFF_CACHE_DIR,
        max_cost_usd=MAX_COST_PER_RUN,
        sleep_between=0.0,
        call_fn=call_openrouter,
    )
    total_cost = float(df["cost_usd"].sum()) if len(df) else 0.0
    print(f"  Done: {len(df)} rows annotated, cost=${total_cost:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", required=True, choices=["pasta", "chocolate", "cheeses"])
    ap.add_argument("--variant", required=True, choices=["uncertain", "random"])
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set")

    print(f"\n=== Annotating {args.cat} / {args.variant} ===")
    codes = get_codes(args.cat, args.variant)
    print(f"  Codes to annotate: {len(codes)}")

    # Step 1: Populate cache
    populate_cache(codes)

    # Step 2: Load pool products
    pool_df = get_pool_df(args.cat, codes)

    # Step 3: Annotate
    annotate(args.cat, args.variant, codes, pool_df, api_key)
    print(f"=== Done: {args.cat} / {args.variant} ===\n")


if __name__ == "__main__":
    main()
