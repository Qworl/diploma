"""Extend cheeses silver_standard with nutri_score_grade + protein_class columns.

These derived attrs were missing → v3e training skipped them for cheeses.
Source: OFF nutriments (same derivation as build_off_derived_truth).
"""
from pathlib import Path
import pandas as pd

from src.experiments.build_off_derived_truth import bucket_protein

PROCESSED = Path("datasets/processed")
src = PROCESSED / "cheeses_stratified_silver_standard.parquet"
s = pd.read_parquet(src)
print(f"cheeses silver: {len(s)} rows, columns: {len(s.columns)}")

def _nutri_grade(g):
    if pd.isna(g): return None
    g = str(g).strip().lower()
    if g in ("a", "b", "c", "d", "e"):
        return g
    return None

def _protein_class(p):
    if pd.isna(p): return None
    try:
        v = float(p)
    except (TypeError, ValueError):
        return None
    return bucket_protein(v)

# Add columns (preserve existing values if they were already there)
if "nutri_score_grade" not in s.columns:
    s["nutri_score_grade"] = s["nutriscore_grade"].apply(_nutri_grade)
    print(f"  added nutri_score_grade: {s['nutri_score_grade'].notna().sum()} non-null")
else:
    print(f"  nutri_score_grade already present ({s['nutri_score_grade'].notna().sum()} non-null)")

if "protein_class" not in s.columns:
    s["protein_class"] = s["proteins_100g"].apply(_protein_class)
    print(f"  added protein_class: {s['protein_class'].notna().sum()} non-null")
else:
    print(f"  protein_class already present ({s['protein_class'].notna().sum()} non-null)")

s.to_parquet(src, index=False)
print(f"Saved → {src} (cols now: {len(s.columns)})")
print(f"value distributions:")
print("  nutri_score_grade:", s["nutri_score_grade"].value_counts(dropna=False).to_dict())
print("  protein_class:    ", s["protein_class"].value_counts(dropna=False).to_dict())
