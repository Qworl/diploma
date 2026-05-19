"""Build unified gold dataset combining best-of all sources.

Output: datasets/processed/unified_gold_v1.parquet (long-format).
Does NOT overwrite any existing file.

Sources (priority order, higher overrides lower):
  1. off_deterministic — OFF nutriments → bucket function for derived attrs
     (nutri_score_grade, protein_class, fat_class). 100% reliable.
  2. off_tag_derived — silver_standard columns derived from labels_tags /
     categories_tags. Factual when tag is present.
  3. opus_clean_diverse — blind_v2 + promptfix_v2_full (top_share ~0.74).
     Reliable for non-derived attrs.

EXCLUDED (broken):
  - opus fresh_prod (top_share=1.00 single-class everywhere)
  - opus expand_to_2k_opus4_partial (top_share=0.97, conservative default)

Per (cat, code, attr): take highest-priority available, record source.
"""
import json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import pandas as pd
pd.set_option("display.width", 240); pd.set_option("display.max_columns", None)

PROCESSED = Path("datasets/processed")

# === 1. OFF deterministic for derived attrs ===
off = pd.read_parquet(PROCESSED / "off_derived_truth.parquet")
off["code"] = off["code"].astype(str)
off = off[~off["gold_is_null"]].copy()
off["gold_norm"] = off["gold_value"].astype(str).str.strip().str.lower()
off["source"] = "off_deterministic"
off = off[["category","code","attr","gold_norm","source"]]
print(f"[1] OFF deterministic (derived): {len(off):,} rows")

# === 2. OFF tag-derived from silver_standard ===
EXCLUDE_COLS = {
    "code","product_name","brands","categories_tags","countries_tags","labels_tags",
    "ingredients_text","ingredients_analysis_tags","traces_tags","quantity",
    "fat_100g","sugars_100g","proteins_100g","carbohydrates_100g","alcohol_100g",
    "nutriscore_grade","nova_group",
    # Derived attrs handled by source 1:
    "nutri_score_grade","protein_class","fat_class",
}
tag_rows = []
for cat in ["pasta","chocolate","cheeses"]:
    s = pd.read_parquet(PROCESSED / f"{cat}_stratified_silver_standard.parquet")
    s["code"] = s["code"].astype(str)
    for attr in s.columns:
        if attr in EXCLUDE_COLS: continue
        sub = s[["code", attr]].dropna(subset=[attr]).copy()
        sub["gold_norm"] = sub[attr].astype(str).str.strip().str.lower()
        sub = sub[~sub["gold_norm"].isin(["", "nan", "none"])]
        sub["category"] = cat
        sub["attr"] = attr
        sub["source"] = "off_tag_derived"
        tag_rows.append(sub[["category","code","attr","gold_norm","source"]])
tag = pd.concat(tag_rows, ignore_index=True)
print(f"[2] OFF tag-derived (non-derived): {len(tag):,} rows, "
      f"{tag['attr'].nunique()} attrs")

# === 3. Opus clean (diverse batches only) ===
opus_rows = []
for batch_dir in [Path("datasets/manual_label/opus_batches/blind_v2"),
                  Path("datasets/manual_label/opus_batches/promptfix_v2_full")]:
    if not batch_dir.exists(): continue
    for f in batch_dir.rglob("*decisions*.json"):
        try: data = json.load(open(f, encoding="utf-8"))
        except Exception: continue
        if not isinstance(data, dict): continue
        cat = None
        for c in ["pasta","chocolate","cheeses"]:
            if c in f.name.lower() or c in str(f.parent).lower():
                cat = c; break
        for code, attrs in data.items():
            if not isinstance(attrs, dict): continue
            for attr, payload in attrs.items():
                if not isinstance(payload, dict): continue
                val = payload.get("value")
                is_null = val is None or (isinstance(val, str) and val.strip().lower() in ("","null","none"))
                if is_null: continue
                opus_rows.append({"category": cat, "code": str(code), "attr": attr,
                                  "gold_norm": str(val).strip().lower(),
                                  "source": "opus_clean_diverse"})
opus = pd.DataFrame(opus_rows).drop_duplicates(subset=["category","code","attr"], keep="last")
opus["code"] = opus["code"].astype(str)
print(f"[3] Opus clean (blind_v2 + promptfix_v2_full): {len(opus):,} rows")

# === 4. Combine with priority ===
all_sources = pd.concat([off, tag, opus], ignore_index=True)
# Priority: assign rank, sort, drop duplicates keeping first (highest priority)
priority = {"off_deterministic": 1, "off_tag_derived": 2, "opus_clean_diverse": 3}
all_sources["_p"] = all_sources["source"].map(priority)
all_sources = all_sources.sort_values("_p").drop_duplicates(
    subset=["category","code","attr"], keep="first").drop(columns=["_p"])

print(f"\n=== UNIFIED GOLD ===")
print(f"Total: {len(all_sources):,} rows, "
      f"{all_sources['code'].nunique():,} codes, "
      f"{all_sources['attr'].nunique()} attrs")
print(f"\nBy source:")
print(all_sources['source'].value_counts())
print(f"\nBy (category, attr) — coverage:")
counts = all_sources.groupby(['category','attr']).size().unstack(fill_value=0)
print(counts)

out = PROCESSED / "unified_gold_v1.parquet"
all_sources.to_parquet(out, index=False)
print(f"\nSaved → {out}  ({out.stat().st_size/1024:.1f} KB)")

# Also save per-source coverage table for the notebook
counts.to_parquet(PROCESSED / "unified_gold_v1_attr_coverage.parquet")
print(f"Per-attr coverage table → {PROCESSED}/unified_gold_v1_attr_coverage.parquet")
