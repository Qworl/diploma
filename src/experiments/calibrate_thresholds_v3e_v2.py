"""Recalibrate v3e thresholds на ЧЕСТНОМ ground truth (без Opus).

Sources:
1. OFF deterministic (nutri_score_grade, protein_class, fat_class) — точная функция nutriments
2. OFF labels_tags / categories_tags (is_organic, is_vegan, is_pdo, is_gluten_free,
   milk_source, country_of_origin, texture, grain_type) — где tag присутствует
3. Используем v3e_holdout модель (НЕ видела ~10.5k codes) для честного OOD threshold sweep.
"""
import json, pickle, shutil, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
pd.set_option("display.width", 240); pd.set_option("display.max_columns", None)

from src.eval.cascade_predict import predict_cascade

PROCESSED = Path("datasets/processed")
MODELS = Path("models")

# === 1. Build honest truth from OFF only (NO Opus) ===
# (a) derived attrs from off_derived_truth (deterministic from nutriments)
off = pd.read_parquet(PROCESSED / "off_derived_truth.parquet")
off["code"] = off["code"].astype(str)
off = off[~off["gold_is_null"]].copy()
off["gold_norm"] = off["gold_value"].astype(str).str.strip().str.lower()
derived_truth = off[["category","code","attr","gold_norm"]].copy()
print(f"OFF derived truth: {len(derived_truth):,} rows")

# (b) non-derived from silver_standard (which is OFF-tag-derived)
# Silver IS reliable where label exists (it's "OFF says X has tag" not "LLM thinks X")
tag_truth_rows = []
for cat in ["pasta","chocolate","cheeses"]:
    s = pd.read_parquet(PROCESSED / f"{cat}_stratified_silver_standard.parquet")
    s["code"] = s["code"].astype(str)
    # Identify tag-derived attrs (boolean + categorical from tags)
    tag_attrs = [c for c in s.columns if c not in (
        "code","product_name","brands","categories_tags","countries_tags","labels_tags",
        "ingredients_text","ingredients_analysis_tags","traces_tags","quantity",
        "fat_100g","sugars_100g","proteins_100g","carbohydrates_100g","alcohol_100g",
        "nutriscore_grade","nova_group",
        # exclude derived (handled above):
        "nutri_score_grade","protein_class","fat_class"
    )]
    for attr in tag_attrs:
        sub = s[["code", attr]].dropna(subset=[attr]).copy()
        sub["gold_norm"] = sub[attr].astype(str).str.strip().str.lower()
        sub = sub[~sub["gold_norm"].isin(["", "nan", "none"])]
        sub["category"] = cat
        sub["attr"] = attr
        tag_truth_rows.append(sub[["category","code","attr","gold_norm"]])
tag_truth = pd.concat(tag_truth_rows, ignore_index=True)
print(f"OFF tag-derived truth: {len(tag_truth):,} rows ({tag_truth['attr'].nunique()} attrs)")
print(tag_truth.groupby(["category","attr"]).size().unstack(fill_value=0))

truth = pd.concat([derived_truth, tag_truth], ignore_index=True)
truth = truth.drop_duplicates(subset=["category","code","attr"])
print(f"\nCombined truth: {len(truth):,} rows, {truth['code'].nunique():,} codes, "
      f"{truth['attr'].nunique()} attrs")

# === 2. Stratified hold-out 20% per (cat, attr, class) for OOD calibration ===
rng = np.random.RandomState(42)
holdout_rows = []
for (cat, attr), grp in truth.groupby(["category","attr"]):
    for cls, sub in grp.groupby("gold_norm"):
        n_take = max(5, int(len(sub) * 0.2))
        n_take = min(n_take, len(sub), 300)  # cap per-class
        holdout_rows.append(sub.sample(n=n_take, random_state=42))
holdout = pd.concat(holdout_rows, ignore_index=True)
holdout = holdout.drop_duplicates(subset=["category","code","attr"])
print(f"\nHoldout (calibration set): {len(holdout):,} rows, {holdout['code'].nunique():,} codes")

# === 3. Retrain v3e WITHOUT these codes → v3e_cal ===
v3e = pd.read_parquet(PROCESSED / "consensus_v3e.parquet")
v3e["code"] = v3e["code"].astype(str)
# Remove all gold rows for (code, attr) pairs in holdout
holdout["key"] = holdout["code"] + "::" + holdout["attr"]
v3e["key"] = v3e["code"] + "::" + v3e["attr"]
v3e_train = v3e[~v3e["key"].isin(set(holdout["key"]))].drop(columns=["key"]).copy()
print(f"v3e train: {len(v3e):,} → {len(v3e_train):,} (dropped {len(v3e)-len(v3e_train):,})")

WORK = Path("/tmp/v3e_calib"); WORK.mkdir(exist_ok=True)
train_path = WORK / "train.parquet"; v3e_train.to_parquet(train_path, index=False)
holdout_codes = sorted(set(holdout["code"]))
hcsv = WORK / "holdout_codes.csv"
pd.DataFrame({"code": holdout_codes}).to_csv(hcsv, index=False)

out_dir = WORK / "models_v3e_cal"; out_dir.mkdir(exist_ok=True)
log = WORK / "train.log"

import subprocess, os
cmd = [".venv/bin/python","-m","src.experiments.train_hybrid_cascade",
       "--gold-path", str(train_path),
       "--w-silver","1.0","--w-tier0","8.0","--w-tier1","6.0","--w-tier2","4.0","--w-tier3","2.0",
       "--output-dir", str(out_dir),
       "--holdout-codes", str(hcsv)]
print(f"\nRetraining v3e_cal (holdout {len(holdout_codes):,} codes) → {out_dir}")
with open(log, "w") as f:
    rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                        env={**os.environ, "OMP_NUM_THREADS":"1"}).returncode
print(f"  rc={rc}")

# === 4. Predict with thresholds=0 to get all predictions + their confidence ===
for p in MODELS.glob("*_xgb_hybrid.pkl"): p.unlink()
for p in MODELS.glob("*_le_hybrid.pkl"): p.unlink()
for p in out_dir.iterdir():
    if p.suffix == ".pkl": shutil.copy(p, MODELS / p.name)

held_codes = set(holdout["code"])
all_preds = []
for cat in ["pasta","chocolate","cheeses"]:
    s = pd.read_parquet(PROCESSED / f"{cat}_stratified_silver_standard.parquet")
    s["code"] = s["code"].astype(str)
    sub = s[s["code"].isin(held_codes)].copy()
    if sub.empty: continue
    cat_attrs = sorted(set(holdout[holdout["category"] == cat]["attr"]))
    thr_zero = {a: 0.0 for a in cat_attrs}
    preds = predict_cascade(sub, category=f"{cat}_stratified",
                            use_hybrid=True, include_bayes=False, include_regex=True,
                            threshold_override=thr_zero)
    preds["category"] = cat
    all_preds.append(preds)
preds_all = pd.concat(all_preds, ignore_index=True)
preds_all["code"] = preds_all["code"].astype(str)
preds_all["pred_norm"] = preds_all["predicted"].astype(str).str.strip().str.lower()

m = holdout.merge(preds_all, on=["category","code","attr"], how="inner")
m = m[m["pred_norm"].notna() & (m["pred_norm"]!="nan") & m["confidence"].notna()]
m["correct"] = (m["pred_norm"] == m["gold_norm"]).astype(int)
print(f"\nCalibration merged: {len(m):,} rows")

# === 5. Threshold sweep per attr — pick smallest threshold where precision >= 90% ===
print("\n=== Threshold sweep (honest OFF-based gold, OOD calibration) ===")
print(f"{'cat':10} {'attr':27} {'best_thr':>8} {'prec':>6} {'cov':>6} {'n_ml':>5}")

new_thresholds = {"pasta_stratified": {}, "chocolate_stratified": {}, "cheeses_stratified": {}}
sweeps = []
for (cat, attr), grp in m.groupby(["category","attr"]):
    ml = grp[grp["layer"] == "ml"]
    if len(ml) < 20: continue
    # Find smallest threshold where accuracy >= 0.90 (precision target)
    best = None
    for thr in np.arange(0.20, 0.96, 0.05):
        kept = ml[ml["confidence"] >= thr]
        if len(kept) < 10: continue
        acc = kept["correct"].mean()
        cov = len(kept) / len(ml)
        sweeps.append({"cat":cat, "attr":attr, "thr":thr, "acc":acc, "cov":cov, "n":len(kept)})
        if acc >= 0.90 and best is None:
            best = (thr, acc, cov)
    # Fallback: if no threshold reaches 0.90 acc, take the one with max acc
    if best is None:
        sw = pd.DataFrame([s for s in sweeps if s["cat"]==cat and s["attr"]==attr])
        if len(sw) == 0: continue
        idx = sw["acc"].idxmax()
        best = (sw.loc[idx,"thr"], sw.loc[idx,"acc"], sw.loc[idx,"cov"])
    new_thresholds[f"{cat}_stratified"][attr] = float(round(best[0], 2))
    print(f"{cat:10} {attr:27} {best[0]:>8.2f} {best[1]:>6.3f} {best[2]:>6.3f} {len(ml):>5}")

# Save new thresholds (merge with existing to keep attrs we didn't calibrate)
for cat_key, thr_dict in new_thresholds.items():
    out = MODELS / f"{cat_key}_thresholds.pkl"
    existing = {}
    if out.exists():
        try: existing = pickle.load(open(out, "rb"))
        except Exception: pass
    existing.update(thr_dict)
    pickle.dump(existing, open(out, "wb"))
    print(f"\nSaved {out}:")
    for k, v in sorted(existing.items()):
        print(f"  {k}: {v:.3f}")

# === 6. Swap production models back ===
for p in MODELS.glob("*_xgb_hybrid.pkl"): p.unlink()
for p in MODELS.glob("*_le_hybrid.pkl"): p.unlink()
for p in (MODELS / "v3e_off_tier0").iterdir():
    if p.suffix == ".pkl": shutil.copy(p, MODELS / p.name)
print("\nRestored production v3e models. New thresholds active.")

# Save sweep CSV for inspection
sweep_df = pd.DataFrame(sweeps)
sweep_df.to_csv("/tmp/v3e_calib/threshold_sweep.csv", index=False)
print(f"Sweep details → /tmp/v3e_calib/threshold_sweep.csv")
