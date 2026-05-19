"""Evaluate v2 hybrid vs v3 hybrid on 20% Tier1+2 holdout.
Runs predict_cascade twice (once with each .pkl set restored from /tmp/) on identical holdout.
Outputs per-attribute accuracy + macro delta.
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
print(f"Holdout: {len(HOLDOUT)} rows, {HOLDOUT['code'].nunique()} codes, "
      f"{HOLDOUT['attr'].nunique()} attrs")

V2_DIR = Path("/tmp/hybrid_v2_clean")
V3_DIR = Path("/tmp/hybrid_v3_clean")
MODELS = Path("models")


def swap(src_dir: Path):
    """Replace models/*_xgb_hybrid.pkl + *_le_hybrid.pkl with files from src_dir."""
    for p in MODELS.glob("*_xgb_hybrid.pkl"):
        p.unlink()
    for p in MODELS.glob("*_le_hybrid.pkl"):
        p.unlink()
    for p in src_dir.iterdir():
        shutil.copy(p, MODELS / p.name)


def run(label: str, src_dir: Path) -> pd.DataFrame:
    print(f"\n=== {label}: loading models from {src_dir} ===")
    swap(src_dir)
    rows = []
    for cat in ["pasta", "chocolate", "cheeses"]:
        cat_codes = sorted(set(HOLDOUT[HOLDOUT["category"] == cat]["code"]))
        silver = pd.read_parquet(f"datasets/processed/{cat}_stratified_silver_standard.parquet")
        silver["code"] = silver["code"].astype(str)
        sub = silver[silver["code"].isin(cat_codes)].copy()
        print(f"  [{cat}] {len(sub)}/{len(cat_codes)} codes have silver/text")
        preds = predict_cascade(sub, category=f"{cat}_stratified",
                                use_hybrid=True, include_bayes=False, include_regex=True)
        preds["category"] = cat
        rows.append(preds)
    out = pd.concat(rows, ignore_index=True)
    out["code"] = out["code"].astype(str)
    out["pred_norm"] = out["predicted"].astype(str).str.strip().str.lower()
    return out


def score(preds: pd.DataFrame, label: str) -> pd.DataFrame:
    gold = HOLDOUT.copy()
    gold["gold_norm"] = gold["gold_value"].astype(str).str.strip().str.lower()
    gold = gold[gold["gold_is_null"] != True]  # noqa: E712
    m = gold.merge(preds, on=["category", "code", "attr"], how="inner")
    # Drop abstained
    m_eval = m[~m["layer"].isin(["abstain", None])]
    m_eval = m_eval[m_eval["pred_norm"].notna() & (m_eval["pred_norm"] != "nan")]
    summary = (m_eval.assign(correct=(m_eval["pred_norm"] == m_eval["gold_norm"]).astype(int))
               .groupby(["category", "attr"])
               .agg(n=("correct", "size"), acc=("correct", "mean"))
               .reset_index())
    summary["label"] = label
    # Coverage (non-abstain rate)
    cov = (m.assign(covered=(~m["layer"].isin(["abstain", None])).astype(int))
           .groupby(["category", "attr"])
           .agg(coverage=("covered", "mean"))
           .reset_index())
    summary = summary.merge(cov, on=["category", "attr"])
    return summary


# Run both
v2_preds = run("v2", V2_DIR)
v3_preds = run("v3", V3_DIR)

v2_scores = score(v2_preds, "v2")
v3_scores = score(v3_preds, "v3")

# Merge for comparison
cmp = v2_scores.merge(v3_scores, on=["category", "attr"], suffixes=("_v2", "_v3"))
cmp["delta_acc"] = (cmp["acc_v3"] - cmp["acc_v2"]) * 100
cmp["delta_cov"] = (cmp["coverage_v3"] - cmp["coverage_v2"]) * 100

print("\n\n========== PER-ATTR COMPARISON (acc on holdout) ==========")
print(cmp[["category", "attr", "n_v2", "acc_v2", "acc_v3", "delta_acc", "coverage_v2", "coverage_v3"]]
      .to_string(index=False, float_format="%.3f"))

print("\n========== PER-CATEGORY MEAN ==========")
agg = cmp.groupby("category").agg(
    acc_v2=("acc_v2", "mean"),
    acc_v3=("acc_v3", "mean"),
    cov_v2=("coverage_v2", "mean"),
    cov_v3=("coverage_v3", "mean"),
    n_attrs=("attr", "count"),
).reset_index()
agg["delta_acc"] = (agg["acc_v3"] - agg["acc_v2"]) * 100
agg["delta_cov"] = (agg["cov_v3"] - agg["cov_v2"]) * 100
print(agg.to_string(index=False, float_format="%.4f"))

print("\n========== OVERALL ==========")
ovr = pd.DataFrame([{
    "acc_v2": cmp["acc_v2"].mean(),
    "acc_v3": cmp["acc_v3"].mean(),
    "delta_acc_pp": (cmp["acc_v3"].mean() - cmp["acc_v2"].mean()) * 100,
    "cov_v2": cmp["coverage_v2"].mean(),
    "cov_v3": cmp["coverage_v3"].mean(),
    "delta_cov_pp": (cmp["coverage_v3"].mean() - cmp["coverage_v2"].mean()) * 100,
}])
print(ovr.to_string(index=False, float_format="%.4f"))

cmp.to_parquet("datasets/processed/holdout_eval_v2_vs_v3.parquet", index=False)
print("\nSaved comparison to datasets/processed/holdout_eval_v2_vs_v3.parquet")
