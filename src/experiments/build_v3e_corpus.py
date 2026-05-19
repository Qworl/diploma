"""Build consensus_v3e.parquet = v3d + Tier 0 deterministic OFF-derived labels.

Tier 0 = OFF nutriments → deterministic Python computation of:
  - nutri_score_grade (from OFF's nutri_score, when present in dump)
  - protein_class (from proteins_100g)
  - fat_class (cheeses only, from fat_100g)

Why: Opus 4.5 promptfix labels NULL these attrs in 100% of cases when nutriments
aren't shown in prompt; gemini promptfix has noisier values; OFF nutriments
give us deterministic ground truth on 41k codes.
"""
from pathlib import Path
import pandas as pd

ROOT = Path("datasets/processed")
v3d = pd.read_parquet(ROOT / "consensus_v3d.parquet")
v3d["code"] = v3d["code"].astype(str)
print(f"v3d: {len(v3d):,} rows, {v3d['code'].nunique():,} codes")

off = pd.read_parquet(ROOT / "off_derived_truth.parquet")
off["code"] = off["code"].astype(str)
off = off[~off["gold_is_null"]].copy()
print(f"OFF-derived non-null: {len(off):,} rows, {off['code'].nunique():,} codes")

# Cast to v3d schema
off_tier0 = off.assign(
    opus_reasoning=None,
    tier="tier0_off_deterministic",
)[["code", "category", "attr", "gold_value", "gold_is_null", "opus_reasoning", "tier"]]

# Deduplicate vs v3d on (code, attr): prefer Tier 0 for derived attrs (it's deterministic)
DERIVED = {"nutri_score_grade", "protein_class", "fat_class"}
v3d_keep = v3d[~(
    v3d["attr"].isin(DERIVED) &
    v3d.set_index(["code", "attr"]).index.isin(
        off_tier0.set_index(["code", "attr"]).index
    )
)].copy()

print(f"\nv3d rows kept (derived attrs replaced where OFF available): {len(v3d_keep):,} (was {len(v3d):,})")
print(f"Tier 0 rows added: {len(off_tier0):,}")

v3e = pd.concat([v3d_keep, off_tier0], ignore_index=True)
print(f"\nv3e: {len(v3e):,} rows, {v3e['code'].nunique():,} codes")
print(f"\nTier composition:")
print(v3e.groupby("tier").size().sort_values(ascending=False).to_string())
print(f"\nDerived attrs by tier:")
print(v3e[v3e["attr"].isin(DERIVED)].groupby(["attr", "tier"]).size().unstack(fill_value=0))

v3e.to_parquet(ROOT / "consensus_v3e.parquet", index=False)
print(f"\nSaved → {ROOT / 'consensus_v3e.parquet'}")
