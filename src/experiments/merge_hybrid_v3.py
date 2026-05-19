"""Merge gemini-flash B3 (round 1+2) into long-format parquet matching v2 gold schema.
Output: consensus_hybrid_v3.parquet (Tier 1+2+3).
"""
import glob, json
import pandas as pd

# Tier 1 + Tier 2 = v2 expanded (existing)
v2 = pd.read_parquet("datasets/processed/consensus_gold_v2_expanded.parquet")
v2["code"] = v2["code"].astype(str)
v2["tier"] = v2["opus_reasoning"].apply(lambda x: "tier1_opus" if x else "tier2_gpt55")
print(f"v2 expanded: {len(v2):,} rows, {v2['code'].nunique():,} codes, "
      f"tier1={sum(v2['tier']=='tier1_opus'):,}, tier2={sum(v2['tier']=='tier2_gpt55'):,}")

# Tier 3 = B3 round 1 + round 2 gemini-flash
tier3_rows = []
for cat in ["pasta", "chocolate", "cheeses"]:
    files = sorted(glob.glob(f"datasets/processed/b3_full_gemini_{cat}*.parquet")
                   + glob.glob(f"datasets/processed/b3_r2_gemini_{cat}*.parquet"))
    for f in files:
        df = pd.read_parquet(f)
        df["code"] = df["code"].astype(str)
        for _, row in df.iterrows():
            try:
                parsed = json.loads(row["parsed_json"]) if row["parsed_json"] else {}
            except (json.JSONDecodeError, TypeError):
                continue
            for attr, value in parsed.items():
                # Skip unsure / null answers
                if value is None:
                    is_null = True
                    val_str = ""
                else:
                    is_null = False
                    val_str = str(value).strip()
                tier3_rows.append({
                    "category": cat, "code": row["code"], "attr": attr,
                    "gold_value": val_str, "gold_is_null": is_null,
                    "opus_reasoning": None, "signal_type": "weak_gemini",
                    "tier": "tier3_gemini",
                })
tier3 = pd.DataFrame(tier3_rows)
print(f"\nTier 3 (gemini B3): {len(tier3):,} rows, {tier3['code'].nunique():,} codes")
# Drop tier3 duplicates (same code/attr from round 1 and round 2)
tier3 = tier3.drop_duplicates(subset=["code", "attr"], keep="first")
print(f"After dedupe: {len(tier3):,} rows, {tier3['code'].nunique():,} codes")

# Exclude any tier3 codes that overlap with v2 expanded (shouldn't happen, but safety)
v2_codes = set(v2["code"])
tier3_clean = tier3[~tier3["code"].isin(v2_codes)]
print(f"After v2-overlap removal: {len(tier3_clean):,} rows, {tier3_clean['code'].nunique():,} codes")

# Merge
hybrid_v3 = pd.concat([v2, tier3_clean], ignore_index=True)
print(f"\nHybrid v3: {len(hybrid_v3):,} rows, {hybrid_v3['code'].nunique():,} codes")
print(f"Tier breakdown:")
print(hybrid_v3.groupby(["category", "tier"])["code"].nunique().unstack(fill_value=0))

out_path = "datasets/processed/consensus_hybrid_v3.parquet"
hybrid_v3.to_parquet(out_path, index=False)
print(f"\nSaved to {out_path}")
