"""Stratified 80/20 split of v2_expanded by category (over codes).
Produces:
  - consensus_v2_train.parquet      (80% Tier1+2)
  - consensus_v3_train.parquet      (80% Tier1+2 + all Tier3 from hybrid_v3)
  - consensus_holdout.parquet       (20% Tier1+2, eval set)
"""
import numpy as np
import pandas as pd

RNG = np.random.RandomState(42)

v2 = pd.read_parquet("datasets/processed/consensus_gold_v2_expanded.parquet")
v2["code"] = v2["code"].astype(str)
# Tier assignment (Phase 1 lineage): opus_reasoning present → Tier1, иначе Tier2 (gpt55).
v2["tier"] = v2["opus_reasoning"].apply(lambda x: "tier1_opus" if x else "tier2_gpt55")
v3 = pd.read_parquet("datasets/processed/consensus_hybrid_v3.parquet")
v3["code"] = v3["code"].astype(str)

# Per-category 80/20 over codes
train_codes, holdout_codes = set(), set()
for cat, grp in v2.groupby("category"):
    codes = sorted(grp["code"].unique())
    shuf = RNG.permutation(codes)
    cut = int(len(shuf) * 0.8)
    train_codes.update(shuf[:cut])
    holdout_codes.update(shuf[cut:])
    print(f"{cat}: {cut} train / {len(shuf)-cut} holdout (of {len(shuf)})")

v2_train = v2[v2["code"].isin(train_codes)].copy()
v2_holdout = v2[v2["code"].isin(holdout_codes)].copy()

# v3 train = v2 train + all Tier3 (tier3 codes are guaranteed disjoint per merge script)
tier3 = v3[v3["tier"] == "tier3_gemini"]
v3_train = pd.concat([v2_train, tier3], ignore_index=True)

print(f"\nv2 train: {len(v2_train)} rows, {v2_train['code'].nunique()} codes")
print(f"v2 holdout: {len(v2_holdout)} rows, {v2_holdout['code'].nunique()} codes")
print(f"v3 train: {len(v3_train)} rows, {v3_train['code'].nunique()} codes")
print(f"  tier1+2 in v3_train: {v3_train[v3_train['tier'].isin(['tier1_opus','tier2_gpt55'])]['code'].nunique()}")
print(f"  tier3 in v3_train: {v3_train[v3_train['tier']=='tier3_gemini']['code'].nunique()}")

# Sanity: no holdout leakage into v3 train
leak = set(v3_train['code']) & holdout_codes
print(f"\nLEAK CHECK: holdout codes in v3_train = {len(leak)} (must be 0)")
assert len(leak) == 0, "HOLDOUT LEAK"

v2_train.to_parquet("datasets/processed/consensus_v2_train.parquet", index=False)
v3_train.to_parquet("datasets/processed/consensus_v3_train.parquet", index=False)
v2_holdout.to_parquet("datasets/processed/consensus_holdout.parquet", index=False)
print("\nSaved 3 parquets.")
