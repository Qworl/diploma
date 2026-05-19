"""Build consensus_v3d.parquet — merged corpus after promptfix re-audit.

Sources:
  Tier 1 (Opus, OFF-grounded, promptfix prompt):
    - Phase 1 originals: opus_batches/promptfix_v2_full/{cat}_decisions.json (+_retry.json)
    - Expand 4.5: opus_batches/expand_to_2k/{cat}_decisions.json (+_retry.json)
  Tier 3 (gemini-2.5-flash, OFF-grounded, promptfix prompt):
    - b3_promptfix_gemini_{cat}_part{0,1,2}.parquet
    - b3_promptfix_gemini_{cat}_part{0,1,2}_retry.parquet

Tier 2 (gpt-5.5 old prompt) is DEPRECATED for v3d — its derived attrs are mostly null
(old prompt). Old Tier 1 (Phase 1 with old prompt) also dropped — superseded by promptfix.

Schema (long-format matching consensus_gold_v2_expanded.parquet):
  code, category, attr, gold_value, gold_is_null, opus_reasoning (None), tier
"""
import glob
import json
from pathlib import Path

import pandas as pd

OPUS_PHASE1_DIRS = {
    "full": Path("datasets/manual_label/opus_batches/promptfix_v2_full"),
}
OPUS_EXPAND_DIR = Path("datasets/manual_label/opus_batches/expand_to_2k")
GEMINI_GLOB = "datasets/processed/b3_promptfix_gemini_{cat}*.parquet"

CATS = ["pasta", "chocolate", "cheeses"]


def load_opus_decisions(json_path: Path, category: str, tier_label: str) -> list[dict]:
    """Parse Opus decisions JSON (dict[code] = {attr: {value, status_hint, reasoning}})."""
    if not json_path.exists():
        return []
    data = json.load(open(json_path, encoding="utf-8"))
    rows = []
    for code, attrs in data.items():
        if not isinstance(attrs, dict):
            continue
        for attr, payload in attrs.items():
            if not isinstance(payload, dict):
                continue
            val = payload.get("value")
            is_null = val is None or (isinstance(val, str) and val.strip().lower() in ("", "null", "none"))
            rows.append({
                "code": str(code), "category": category, "attr": attr,
                "gold_value": "" if is_null else str(val).strip(),
                "gold_is_null": is_null,
                "opus_reasoning": payload.get("reasoning"),
                "tier": tier_label,
            })
    return rows


tier1_rows = []
for cat in CATS:
    # Phase 1 promptfix (217 original + retry)
    for suffix in ["_decisions.json", "_decisions_retry.json"]:
        p = OPUS_PHASE1_DIRS["full"] / f"{cat}{suffix}"
        tier1_rows.extend(load_opus_decisions(p, cat, "tier1_opus_phase1"))
    # Expand 4.5 (1400-1700 codes per cat + retry)
    for suffix in ["_decisions.json", "_decisions_retry.json"]:
        p = OPUS_EXPAND_DIR / f"{cat}{suffix}"
        tier1_rows.extend(load_opus_decisions(p, cat, "tier1_opus_expand45"))

tier1 = pd.DataFrame(tier1_rows)
tier1["code"] = tier1["code"].astype(str)
# Dedupe within Tier 1 (a code may appear in both Phase 1 and Expand; Expand 4.5 wins because newer/cheaper model with same prompt)
tier1 = tier1.sort_values(["tier"], ascending=False)  # tier1_opus_phase1 < expand45 alphabetically → expand45 first
tier1 = tier1.drop_duplicates(subset=["code", "attr"], keep="first")
print(f"Tier 1 rows: {len(tier1):,}  ({tier1['code'].nunique():,} codes)")
print(f"  by sub-tier:\n{tier1.groupby('tier')['code'].nunique()}")

tier3_rows = []
for cat in CATS:
    for fn in sorted(glob.glob(GEMINI_GLOB.format(cat=cat))):
        df = pd.read_parquet(fn)
        df["code"] = df["code"].astype(str)
        for _, row in df.iterrows():
            try:
                parsed = json.loads(row["parsed_json"]) if row["parsed_json"] else {}
            except (json.JSONDecodeError, TypeError):
                continue
            for attr, value in parsed.items():
                is_null = value is None
                val_str = "" if is_null else str(value).strip()
                tier3_rows.append({
                    "code": row["code"], "category": cat, "attr": attr,
                    "gold_value": val_str, "gold_is_null": is_null,
                    "opus_reasoning": None, "tier": "tier3_gemini_promptfix",
                })
tier3 = pd.DataFrame(tier3_rows)
tier3["code"] = tier3["code"].astype(str)
# Dedupe (round1/round2/retry may overlap)
tier3 = tier3.drop_duplicates(subset=["code", "attr"], keep="first")
print(f"\nTier 3 rows: {len(tier3):,}  ({tier3['code'].nunique():,} codes)")

# Drop Tier 3 codes that overlap with Tier 1 (Tier 1 wins for shared codes)
tier1_codes = set(tier1["code"])
overlap_codes = set(tier3["code"]) & tier1_codes
tier3_clean = tier3[~tier3["code"].isin(tier1_codes)]
print(f"  Tier 1 ∩ Tier 3 codes: {len(overlap_codes):,}  → dropped from Tier 3")
print(f"  Tier 3 after dedupe: {len(tier3_clean):,} rows ({tier3_clean['code'].nunique():,} codes)")

merged = pd.concat([tier1, tier3_clean], ignore_index=True)
print(f"\n========== v3d merged ==========")
print(f"Total rows: {len(merged):,}")
print(f"Total codes: {merged['code'].nunique():,}")
print(f"Per category × tier (codes):")
print(merged.groupby(["category", "tier"])["code"].nunique().unstack(fill_value=0))

out_path = Path("datasets/processed/consensus_v3d.parquet")
merged.to_parquet(out_path, index=False)
print(f"\nSaved → {out_path}")
