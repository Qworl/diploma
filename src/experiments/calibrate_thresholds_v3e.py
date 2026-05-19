"""Recalibrate per-attribute confidence thresholds for v3e on validation set.

Approach: scan thresholds 0.20→0.95 in 0.05 steps, pick the smallest threshold
where accuracy at that threshold is within 0.5 п.п. of max accuracy.

This recovers coverage that was lost because v3e's confidence distribution shifted
(trained on bigger noisier corpus) but old thresholds were calibrated for v3b/v3d.

Validation set: combination of
  - Clean Opus benchmark codes (blind_v2 + promptfix_v2_full) for non-derived
  - OFF-derived truth for derived attrs (nutri_score, protein_class, fat_class)
"""
import json, pickle, shutil, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
pd.set_option("display.width", 240)

from src.eval.cascade_predict import predict_cascade

# === Load validation truth ===
PROCESSED = Path("datasets/processed")

# Non-derived: blind_v2 + promptfix_v2_full Opus labels
truth_rows = []
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
                truth_rows.append({"category": cat, "code": str(code), "attr": attr,
                                   "gold_norm": "" if is_null else str(val).strip().lower(),
                                   "gold_is_null": is_null})
opus_truth = pd.DataFrame(truth_rows).drop_duplicates(subset=["category","code","attr"], keep="last")
opus_truth["code"] = opus_truth["code"].astype(str)

# Derived: OFF-derived truth
off = pd.read_parquet(PROCESSED / "off_derived_truth.parquet")
off["code"] = off["code"].astype(str)
off = off[~off["gold_is_null"]].copy()
off["gold_norm"] = off["gold_value"].astype(str).str.strip().str.lower()

DERIVED = {"nutri_score_grade", "protein_class", "fat_class"}

# Truth for non-derived = Opus, for derived = OFF
non_d = opus_truth[~opus_truth["attr"].isin(DERIVED) & ~opus_truth["gold_is_null"]]
der = off[off["attr"].isin(DERIVED)][["category","code","attr","gold_norm"]]
truth = pd.concat([non_d[["category","code","attr","gold_norm"]], der], ignore_index=True)
truth = truth.drop_duplicates(subset=["category","code","attr"])

# Limit OFF derived to sample (40k is too much, take 5k/cat-attr)
print(f"Validation truth: {len(truth):,} rows")
print(truth.groupby(["category","attr"]).size().unstack(fill_value=0))

# Sample OFF down to 1k/cat-attr-class to balance + speed
rng = np.random.RandomState(42)
truth_sampled_rows = []
for (cat, attr), grp in truth.groupby(["category","attr"]):
    if attr in DERIVED and len(grp) > 1000:
        # Stratify by class — take ~250 per class
        for cls, sub in grp.groupby("gold_norm"):
            take = min(250, len(sub))
            truth_sampled_rows.append(sub.sample(n=take, random_state=42))
    else:
        truth_sampled_rows.append(grp)
truth_s = pd.concat(truth_sampled_rows, ignore_index=True)
print(f"\nSampled validation: {len(truth_s):,} rows")
print(truth_s.groupby(["category","attr"]).size().unstack(fill_value=0))

val_codes = sorted(set(truth_s["code"]))
print(f"Unique codes: {len(val_codes):,}")

# === Run cascade with very low threshold (to get all predictions + their confidence) ===
all_preds_rows = []
for cat in ["pasta","chocolate","cheeses"]:
    s = pd.read_parquet(PROCESSED / f"{cat}_stratified_silver_standard.parquet")
    s["code"] = s["code"].astype(str)
    sub = s[s["code"].isin(val_codes)].copy()
    if sub.empty: continue
    # Override thresholds to 0.0 so cascade returns predictions for everything
    cat_attrs = sorted(set(truth_s[truth_s["category"] == cat]["attr"]))
    thr_zero = {a: 0.0 for a in cat_attrs}
    preds = predict_cascade(sub, category=f"{cat}_stratified",
                            use_hybrid=True, include_bayes=False, include_regex=True,
                            threshold_override=thr_zero)
    preds["category"] = cat
    all_preds_rows.append(preds)
preds_all = pd.concat(all_preds_rows, ignore_index=True)
preds_all["code"] = preds_all["code"].astype(str)
preds_all["pred_norm"] = preds_all["predicted"].astype(str).str.strip().str.lower()

# Merge with truth
m = truth_s.merge(preds_all, on=["category","code","attr"], how="inner")
m = m[m["pred_norm"].notna() & (m["pred_norm"] != "nan") & m["confidence"].notna()]
m["correct"] = (m["pred_norm"] == m["gold_norm"]).astype(int)
print(f"\nMatched (truth × pred with conf): {len(m):,}")

# === Per-attr threshold sweep ===
thresholds_grid = np.arange(0.20, 0.96, 0.05)
new_thresholds = {"pasta_stratified": {}, "chocolate_stratified": {}, "cheeses_stratified": {}}

print("\n=== Per-attr threshold sweep ===")
print(f"{'cat':10} {'attr':25} {'best_thr':>8} {'acc':>6} {'cov':>6} {'n':>5}")
for (cat, attr), grp in m.groupby(["category","attr"]):
    if grp["layer"].iloc[0] == "regex" and (grp["layer"]=="regex").all():
        continue  # regex doesn't have learnable threshold
    ml_grp = grp[grp["layer"] == "ml"].copy()
    if len(ml_grp) < 20:
        continue
    sweep = []
    for thr in thresholds_grid:
        kept = ml_grp[ml_grp["confidence"] >= thr]
        if len(kept) < 5: continue
        acc = kept["correct"].mean()
        cov = len(kept) / len(ml_grp)
        # Combined utility: prefer higher acc, but with reasonable coverage
        sweep.append({"thr": thr, "acc": acc, "cov": cov,
                      "utility": acc * (0.5 + 0.5 * cov)})
    if not sweep: continue
    sweep_df = pd.DataFrame(sweep)
    # Pick threshold maximizing utility (balance acc and cov)
    best_idx = sweep_df["utility"].idxmax()
    best = sweep_df.iloc[best_idx]
    new_thresholds[f"{cat}_stratified"][attr] = float(round(best["thr"], 2))
    print(f"{cat:10} {attr:25} {best['thr']:>8.2f} {best['acc']:>6.3f} {best['cov']:>6.3f} {len(ml_grp):>5}")

# Save new thresholds
MODELS_DIR = Path("models")
for cat_key, thr_dict in new_thresholds.items():
    out = MODELS_DIR / f"{cat_key}_thresholds.pkl"
    # Merge with existing (don't lose regex-attr defaults)
    existing = {}
    if out.exists():
        try: existing = pickle.load(open(out, "rb"))
        except Exception: pass
    existing.update(thr_dict)
    pickle.dump(existing, open(out, "wb"))
    print(f"\nSaved {out}: {len(existing)} attrs")
    for k, v in sorted(existing.items()):
        print(f"  {k}: {v:.3f}")
