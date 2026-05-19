"""Build deterministic ground truth for derived attributes from OFF nutriments.

For attrs that are FUNCTIONS of nutrient values, we don't need LLMs — we can compute
the answer directly. This gives us 100× more eval observations than LLM-labeled holdout.

Derived attrs:
  - nutri_score_grade: A-E. OFF's own `nutriscore_grade` field is canonical truth.
  - protein_class: 0/low/med/high, binned from proteins_100g.
  - fat_class: low/medium/high/very_high, binned from fat_100g.

Output: datasets/processed/off_derived_truth.parquet (long format matching consensus_gold).
"""
import json
from pathlib import Path

import pandas as pd

CACHE_DIRS = [
    Path("datasets/manual_label/off_cache"),
    Path("datasets/manual_label/off_cache_b3_full"),
    Path("datasets/manual_label/off_cache_b3_r2"),
]


def bucket_protein(g: float) -> str:
    if g == 0: return "0"
    if g < 5: return "low"
    if g <= 15: return "med"
    return "high"


def bucket_fat(g: float) -> str:
    """cheeses/fat_class — matches prompt derivation rules used during gemini/Opus labeling.
    NOTE: schema description has DIFFERENT thresholds (<15/15-25/25-32/>32) but actual
    labels follow prompt rules (<5/5-20/20-50/>50). Eval must match training labels.
    """
    if g < 5: return "low"
    if g <= 20: return "medium"
    if g <= 50: return "high"
    return "very_high"


def infer_category(prod: dict) -> str | None:
    """Map OFF categories_tags → our domain (pasta/chocolate/cheeses)."""
    tags = prod.get("categories_tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split("|") if t.strip()]
    tag_str = " ".join(tags).lower()
    if any(t in tag_str for t in ["en:pastas", "en:noodles", "en:lasagna"]):
        return "pasta"
    if "en:chocolate" in tag_str:
        return "chocolate"
    if "en:cheese" in tag_str:
        return "cheeses"
    return None


def load_all_codes() -> dict:
    """Return {code: (category, product_dict)} for all cached codes."""
    out = {}
    for cache_dir in CACHE_DIRS:
        if not cache_dir.exists():
            continue
        for json_path in cache_dir.glob("*.json"):
            code = json_path.stem
            if code in out:
                continue
            try:
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            prod = data.get("product", {})
            if not prod:
                continue
            cat = infer_category(prod)
            if cat is None:
                continue
            out[code] = (cat, prod)
    return out


def _to_float(v):
    if v in (None, "", "nan"):
        return None
    try:
        return float(v)
    except Exception:
        return None


def build_truth_rows(codes: dict) -> list[dict]:
    rows = []
    for code, (cat, prod) in codes.items():
        nuts = prod.get("nutriments", {}) or {}
        proteins = _to_float(nuts.get("proteins_100g"))
        fat = _to_float(nuts.get("fat_100g"))

        # 1) nutri_score_grade — direct from OFF's own field
        nsg = (prod.get("nutriscore_grade") or "").strip().lower()
        if nsg and nsg in ("a", "b", "c", "d", "e"):
            rows.append({
                "category": cat, "code": code, "attr": "nutri_score_grade",
                "gold_value": nsg.upper(), "gold_is_null": False,
                "source": "off_field",
            })

        # 2) protein_class — only if proteins_100g present
        if proteins is not None:
            rows.append({
                "category": cat, "code": code, "attr": "protein_class",
                "gold_value": bucket_protein(proteins), "gold_is_null": False,
                "source": "computed_proteins_100g",
            })

        # 3) fat_class — only if fat_100g present (cheeses uses this attr)
        if fat is not None and cat == "cheeses":
            rows.append({
                "category": cat, "code": code, "attr": "fat_class",
                "gold_value": bucket_fat(fat), "gold_is_null": False,
                "source": "computed_fat_100g",
            })
    return rows


if __name__ == "__main__":
    print("Loading OFF cache JSONs from", [str(d) for d in CACHE_DIRS], "...")
    codes = load_all_codes()
    print(f"Total cached codes mapped to a domain: {len(codes):,}")
    by_cat = {}
    for code, (cat, _) in codes.items():
        by_cat.setdefault(cat, []).append(code)
    for cat, cs in by_cat.items():
        print(f"  {cat}: {len(cs):,}")

    rows = build_truth_rows(codes)
    df = pd.DataFrame(rows)
    df["code"] = df["code"].astype(str)
    print(f"\nTotal truth rows: {len(df):,}")
    print("Per category × attr:")
    print(df.groupby(["category", "attr"]).size().unstack(fill_value=0))

    out_path = Path("datasets/processed/off_derived_truth.parquet")
    df.to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path}")
