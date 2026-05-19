"""4-way comparison: v2 / v3 / v3b (weighted) / v3c (no silver).
Same 20% Tier1+2 holdout. Per-attr accuracy + coverage.
"""
import shutil
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(".").resolve()))

from src.eval.cascade_predict import predict_cascade

HOLDOUT = pd.read_parquet("datasets/processed/consensus_holdout.parquet")
HOLDOUT["code"] = HOLDOUT["code"].astype(str)
print(f"Holdout: {len(HOLDOUT)} rows, {HOLDOUT['code'].nunique()} codes")

VARIANTS = {
    "v2":  Path("/Users/miafrolov/Desktop/stuff/ai_attributes/.worktrees/phase2-recomputes-on-v2-gold/models_backup/v2_clean"),
    "v3":  Path("/Users/miafrolov/Desktop/stuff/ai_attributes/.worktrees/phase2-recomputes-on-v2-gold/models_backup/v3_clean"),
    "v3b": Path("/Users/miafrolov/Desktop/stuff/ai_attributes/.worktrees/phase2-recomputes-on-v2-gold/models_backup/v3b_weighted"),
    "v3c": Path("/Users/miafrolov/Desktop/stuff/ai_attributes/.worktrees/phase2-recomputes-on-v2-gold/models_backup/v3c_no_silver"),
}
MODELS = Path("models")


def swap(src_dir: Path):
    for p in MODELS.glob("*_xgb_hybrid.pkl"):
        p.unlink()
    for p in MODELS.glob("*_le_hybrid.pkl"):
        p.unlink()
    for p in src_dir.iterdir():
        shutil.copy(p, MODELS / p.name)


def run(label: str, src_dir: Path) -> pd.DataFrame:
    print(f"\n=== {label} ({src_dir.name}) ===")
    swap(src_dir)
    rows = []
    for cat in ["pasta", "chocolate", "cheeses"]:
        cat_codes = sorted(set(HOLDOUT[HOLDOUT["category"] == cat]["code"]))
        silver = pd.read_parquet(f"datasets/processed/{cat}_stratified_silver_standard.parquet")
        silver["code"] = silver["code"].astype(str)
        sub = silver[silver["code"].isin(cat_codes)].copy()
        preds = predict_cascade(sub, category=f"{cat}_stratified",
                                use_hybrid=True, include_bayes=False, include_regex=True)
        preds["category"] = cat
        rows.append(preds)
    out = pd.concat(rows, ignore_index=True)
    out["code"] = out["code"].astype(str)
    out["pred_norm"] = out["predicted"].astype(str).str.strip().str.lower()
    return out


def score(preds: pd.DataFrame) -> pd.DataFrame:
    gold = HOLDOUT.copy()
    gold["gold_norm"] = gold["gold_value"].astype(str).str.strip().str.lower()
    gold = gold[gold["gold_is_null"] != True]  # noqa: E712
    m = gold.merge(preds, on=["category", "code", "attr"], how="inner")
    m_eval = m[~m["layer"].isin(["abstain", None])]
    m_eval = m_eval[m_eval["pred_norm"].notna() & (m_eval["pred_norm"] != "nan")]
    acc_df = (m_eval.assign(correct=(m_eval["pred_norm"] == m_eval["gold_norm"]).astype(int))
              .groupby(["category", "attr"])
              .agg(n=("correct", "size"), acc=("correct", "mean"))
              .reset_index())
    cov_df = (m.assign(covered=(~m["layer"].isin(["abstain", None])).astype(int))
              .groupby(["category", "attr"])
              .agg(coverage=("covered", "mean"))
              .reset_index())
    return acc_df.merge(cov_df, on=["category", "attr"])


all_scores = {}
for label, d in VARIANTS.items():
    preds = run(label, d)
    s = score(preds)
    s = s.rename(columns={"acc": f"acc_{label}", "coverage": f"cov_{label}", "n": f"n_{label}"})
    all_scores[label] = s

cmp = all_scores["v2"]
for label in ["v3", "v3b", "v3c"]:
    cmp = cmp.merge(all_scores[label].drop(columns=[f"n_{label}"]),
                    on=["category", "attr"])

print("\n\n========== PER-ATTR (accuracy) ==========")
cols = ["category", "attr", "n_v2", "acc_v2", "acc_v3", "acc_v3b", "acc_v3c"]
print(cmp[cols].to_string(index=False, float_format="%.3f"))

print("\n========== PER-ATTR (coverage) ==========")
cols = ["category", "attr", "cov_v2", "cov_v3", "cov_v3b", "cov_v3c"]
print(cmp[cols].to_string(index=False, float_format="%.3f"))

print("\n========== PER-CAT MEAN ==========")
agg = cmp.groupby("category").agg({
    "acc_v2": "mean", "acc_v3": "mean", "acc_v3b": "mean", "acc_v3c": "mean",
    "cov_v2": "mean", "cov_v3": "mean", "cov_v3b": "mean", "cov_v3c": "mean",
}).reset_index()
print(agg.to_string(index=False, float_format="%.4f"))

print("\n========== OVERALL MEAN ==========")
ov = pd.DataFrame([{
    "acc_v2": cmp["acc_v2"].mean(),
    "acc_v3": cmp["acc_v3"].mean(),
    "acc_v3b": cmp["acc_v3b"].mean(),
    "acc_v3c": cmp["acc_v3c"].mean(),
    "cov_v2": cmp["cov_v2"].mean(),
    "cov_v3": cmp["cov_v3"].mean(),
    "cov_v3b": cmp["cov_v3b"].mean(),
    "cov_v3c": cmp["cov_v3c"].mean(),
}])
print(ov.to_string(index=False, float_format="%.4f"))

cmp.to_parquet("datasets/processed/holdout_eval_4way.parquet", index=False)
print("\nSaved to datasets/processed/holdout_eval_4way.parquet")
