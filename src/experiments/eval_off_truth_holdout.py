"""Eval v2/v3/v3b/v3c on OFF-derived truth, excluding training codes.

Compares cascade predictions for derived attrs (nutri_score_grade, protein_class,
fat_class) against deterministic OFF-computed truth on a holdout of codes that
were NOT seen during hybrid model training.

Output: per-(variant, attr) accuracy + coverage on the OFF-truth holdout.
"""
import shutil
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(".").resolve()))

from src.eval.cascade_predict import predict_cascade

TRUTH = pd.read_parquet("datasets/processed/off_derived_truth.parquet")
TRUTH["code"] = TRUTH["code"].astype(str)
TRAIN = pd.read_parquet("datasets/processed/consensus_v3_train.parquet")
TRAIN["code"] = TRAIN["code"].astype(str)
TRAIN_CODES = set(TRAIN["code"].unique())
print(f"Training codes (v3): {len(TRAIN_CODES):,}")
print(f"OFF-derived truth rows: {len(TRUTH):,} ({TRUTH['code'].nunique():,} codes)")

# Eval set: truth codes NOT in training
TRUTH_HOLDOUT = TRUTH[~TRUTH["code"].isin(TRAIN_CODES)].copy()
print(f"\nOFF-truth NOT in training: {len(TRUTH_HOLDOUT):,} rows "
      f"({TRUTH_HOLDOUT['code'].nunique():,} codes)")
print(TRUTH_HOLDOUT.groupby(["category", "attr"]).size().unstack(fill_value=0))

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
        eval_codes = sorted(set(TRUTH_HOLDOUT[TRUTH_HOLDOUT["category"] == cat]["code"]))
        if not eval_codes:
            continue
        silver = pd.read_parquet(f"datasets/processed/{cat}_stratified_silver_standard.parquet")
        silver["code"] = silver["code"].astype(str)
        sub = silver[silver["code"].isin(eval_codes)].copy()
        print(f"  [{cat}] {len(sub)}/{len(eval_codes)} eval codes in silver")
        if sub.empty:
            continue
        preds = predict_cascade(sub, category=f"{cat}_stratified",
                                use_hybrid=True, include_bayes=False, include_regex=True)
        preds["category"] = cat
        rows.append(preds)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out["code"] = out["code"].astype(str)
    out["pred_norm"] = out["predicted"].astype(str).str.strip().str.lower()
    return out


def score(preds: pd.DataFrame) -> pd.DataFrame:
    truth = TRUTH_HOLDOUT.copy()
    truth["gold_norm"] = truth["gold_value"].astype(str).str.strip().str.lower()
    m = truth.merge(preds, on=["category", "code", "attr"], how="inner")
    m_eval = m[~m["layer"].isin(["abstain", None])]
    m_eval = m_eval[m_eval["pred_norm"].notna() & (m_eval["pred_norm"] != "nan")]
    acc_df = (m_eval.assign(correct=(m_eval["pred_norm"] == m_eval["gold_norm"]).astype(int))
              .groupby(["category", "attr"])
              .agg(n=("correct", "size"), acc=("correct", "mean"))
              .reset_index())
    cov_df = (m.assign(covered=(~m["layer"].isin(["abstain", None])).astype(int))
              .groupby(["category", "attr"])
              .agg(n_total=("covered", "size"), coverage=("covered", "mean"))
              .reset_index())
    return acc_df.merge(cov_df, on=["category", "attr"])


all_scores = {}
for label, d in VARIANTS.items():
    preds = run(label, d)
    if preds.empty:
        print(f"  [{label}] no predictions")
        continue
    s = score(preds)
    s = s.rename(columns={"acc": f"acc_{label}", "coverage": f"cov_{label}",
                          "n": f"n_{label}"})
    all_scores[label] = s

if "v2" in all_scores:
    cmp = all_scores["v2"][["category", "attr", "n_total", "n_v2", "acc_v2", "cov_v2"]]
    for label in ["v3", "v3b", "v3c"]:
        if label in all_scores:
            cmp = cmp.merge(
                all_scores[label][["category", "attr", f"acc_{label}",
                                    f"cov_{label}"]],
                on=["category", "attr"], how="left",
            )

    print("\n\n========== PER-ATTR (OFF-truth holdout) ==========")
    print(cmp.to_string(index=False, float_format="%.3f"))

    print("\n========== OVERALL MEAN ==========")
    ov = {}
    for label in ["v2", "v3", "v3b", "v3c"]:
        if f"acc_{label}" in cmp.columns:
            ov[f"acc_{label}"] = cmp[f"acc_{label}"].mean()
            ov[f"cov_{label}"] = cmp[f"cov_{label}"].mean()
    print(pd.DataFrame([ov]).to_string(index=False, float_format="%.4f"))

    cmp.to_parquet("datasets/processed/eval_off_truth_4way.parquet", index=False)
    print("\nSaved to datasets/processed/eval_off_truth_4way.parquet")
