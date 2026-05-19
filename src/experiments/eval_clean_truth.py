"""Eval v2/v3/v3b/v3c/v3d against cleanest truth: new Opus-promptfix labels for
the 151 holdout codes that have them. Both v3b and v3d had these codes excluded from training.
"""
import shutil
import sys
import warnings
from pathlib import Path
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(".").resolve()))
from src.eval.cascade_predict import predict_cascade

ROOT = Path("/Users/miafrolov/Desktop/stuff/ai_attributes/.worktrees/phase2-recomputes-on-v2-gold")
VARIANTS = {
    "v2":  ROOT / "models_backup/v2_clean",
    "v3":  ROOT / "models_backup/v3_clean",
    "v3b": ROOT / "models_backup/v3b_weighted",
    "v3c": ROOT / "models_backup/v3c_no_silver",
    "v3d": ROOT / "models_backup/v3d_promptfix",
}
MODELS = Path("models")
DERIVED = {"nutri_score_grade", "protein_class", "fat_class", "cocoa_percentage"}


def swap(src):
    for p in MODELS.glob("*_xgb_hybrid.pkl"): p.unlink()
    for p in MODELS.glob("*_le_hybrid.pkl"): p.unlink()
    for p in src.iterdir(): shutil.copy(p, MODELS / p.name)


def predict_on(codes):
    rows = []
    for cat in ["pasta","chocolate","cheeses"]:
        silver = pd.read_parquet(f"datasets/processed/{cat}_stratified_silver_standard.parquet")
        silver["code"] = silver["code"].astype(str)
        sub = silver[silver["code"].isin(codes)].copy()
        if sub.empty: continue
        preds = predict_cascade(sub, category=f"{cat}_stratified",
                                use_hybrid=True, include_bayes=False, include_regex=True)
        preds["category"] = cat
        rows.append(preds)
    out = pd.concat(rows, ignore_index=True)
    out["code"] = out["code"].astype(str)
    out["pred_norm"] = out["predicted"].astype(str).str.strip().str.lower()
    return out


def score(preds, truth):
    g = truth.copy()
    g["gold_norm"] = g["gold_value"].astype(str).str.strip().str.lower()
    g = g[g["gold_is_null"] != True]  # noqa
    m = g.merge(preds, on=["category","code","attr"], how="inner")
    m_eval = m[~m["layer"].isin(["abstain",None])]
    m_eval = m_eval[m_eval["pred_norm"].notna() & (m_eval["pred_norm"] != "nan")]
    acc = (m_eval.assign(c=(m_eval["pred_norm"]==m_eval["gold_norm"]).astype(int))
           .groupby(["category","attr"]).agg(n=("c","size"),acc=("c","mean")).reset_index())
    cov = (m.assign(c=(~m["layer"].isin(["abstain",None])).astype(int))
           .groupby(["category","attr"]).agg(coverage=("c","mean")).reset_index())
    return acc.merge(cov, on=["category","attr"])


truth = pd.read_parquet("datasets/processed/holdout534_opus_promptfix_truth.parquet")
truth["code"] = truth["code"].astype(str)
codes = sorted(truth["code"].unique())
print(f"Truth: {len(truth)} rows on {len(codes)} codes ({truth['attr'].nunique()} attrs)")

results = {}
for label, d in VARIANTS.items():
    print(f"\n=== {label} ===")
    swap(d)
    preds = predict_on(codes)
    results[label] = score(preds, truth)

cmp = results["v2"][["category","attr","n"]].rename(columns={"n":"n_v2"})
for label in VARIANTS:
    s = results[label].rename(columns={"acc":f"acc_{label}","coverage":f"cov_{label}","n":f"n_{label}"})
    cmp = cmp.merge(s[["category","attr",f"acc_{label}",f"cov_{label}"]], on=["category","attr"], how="left")

cmp["is_derived"] = cmp["attr"].isin(DERIVED)
print("\n========== PER-ATTR ==========")
print(cmp.drop(columns=["is_derived"]).to_string(index=False, float_format="%.3f"))

print("\n========== OVERALL ==========")
ov = {f"acc_{l}": cmp[f"acc_{l}"].mean() for l in VARIANTS}
ov.update({f"cov_{l}": cmp[f"cov_{l}"].mean() for l in VARIANTS})
print(pd.DataFrame([ov]).to_string(index=False, float_format="%.4f"))

print("\n========== DERIVED only ==========")
sub = cmp[cmp["is_derived"]]
print(f"  n_attrs={len(sub)}, total_obs={sub['n_v2'].sum()}")
od = {f"acc_{l}": sub[f"acc_{l}"].mean() for l in VARIANTS}
print(pd.DataFrame([od]).to_string(index=False, float_format="%.4f"))

print("\n========== NON-DERIVED only ==========")
sub = cmp[~cmp["is_derived"]]
print(f"  n_attrs={len(sub)}, total_obs={sub['n_v2'].sum()}")
ond = {f"acc_{l}": sub[f"acc_{l}"].mean() for l in VARIANTS}
print(pd.DataFrame([ond]).to_string(index=False, float_format="%.4f"))

cmp.to_parquet("datasets/processed/eval_5way_clean_opus_truth.parquet", index=False)
print("\nSaved eval_5way_clean_opus_truth.parquet")
