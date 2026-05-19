"""5-way eval v2 / v3 / v3b / v3c / v3d on:
 (a) Tier1+2 holdout (same 534 codes as before, OLD-prompt truth)
 (b) OFF-derived truth holdout (4500+ obs, deterministic)
Outputs single comparison parquet + console table.
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


def swap(src_dir: Path):
    for p in MODELS.glob("*_xgb_hybrid.pkl"):
        p.unlink()
    for p in MODELS.glob("*_le_hybrid.pkl"):
        p.unlink()
    for p in src_dir.iterdir():
        shutil.copy(p, MODELS / p.name)


def predict_on(codes: list[str]) -> pd.DataFrame:
    rows = []
    for cat in ["pasta", "chocolate", "cheeses"]:
        silver = pd.read_parquet(f"datasets/processed/{cat}_stratified_silver_standard.parquet")
        silver["code"] = silver["code"].astype(str)
        sub = silver[silver["code"].isin(codes)].copy()
        if sub.empty:
            continue
        preds = predict_cascade(sub, category=f"{cat}_stratified",
                                use_hybrid=True, include_bayes=False, include_regex=True)
        preds["category"] = cat
        rows.append(preds)
    out = pd.concat(rows, ignore_index=True)
    out["code"] = out["code"].astype(str)
    out["pred_norm"] = out["predicted"].astype(str).str.strip().str.lower()
    return out


def score(preds: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    g = truth.copy()
    g["gold_norm"] = g["gold_value"].astype(str).str.strip().str.lower()
    g = g[g["gold_is_null"] != True]  # noqa: E712
    m = g.merge(preds, on=["category", "code", "attr"], how="inner")
    m_eval = m[~m["layer"].isin(["abstain", None])]
    m_eval = m_eval[m_eval["pred_norm"].notna() & (m_eval["pred_norm"] != "nan")]
    acc = (m_eval.assign(c=(m_eval["pred_norm"] == m_eval["gold_norm"]).astype(int))
           .groupby(["category", "attr"]).agg(n=("c", "size"), acc=("c", "mean")).reset_index())
    cov = (m.assign(c=(~m["layer"].isin(["abstain", None])).astype(int))
           .groupby(["category", "attr"]).agg(coverage=("c", "mean")).reset_index())
    return acc.merge(cov, on=["category", "attr"])


# Eval set A: Tier1+2 holdout (534 codes)
holdout = pd.read_parquet("datasets/processed/consensus_holdout.parquet")
holdout["code"] = holdout["code"].astype(str)
holdout_codes = sorted(holdout["code"].unique())

# Eval set B: OFF-derived truth NOT in ANY model's training (proper fair eval)
truth = pd.read_parquet("datasets/processed/off_derived_truth.parquet")
truth["code"] = truth["code"].astype(str)
train_v3 = pd.read_parquet("datasets/processed/consensus_v3_train.parquet")
train_v3["code"] = train_v3["code"].astype(str)
train_v3d = pd.read_parquet("datasets/processed/consensus_v3d.parquet")
train_v3d["code"] = train_v3d["code"].astype(str)
all_train = set(train_v3["code"]) | set(train_v3d["code"])
truth_holdout = truth[~truth["code"].isin(all_train)]

print(f"Set A — Tier1+2 holdout: {len(holdout)} rows, {len(holdout_codes)} codes")
print(f"Set B — OFF-truth not in train: {len(truth_holdout)} rows, {truth_holdout['code'].nunique()} codes")

results_a, results_b = {}, {}
for label, d in VARIANTS.items():
    print(f"\n=== {label} ({d.name}) ===")
    swap(d)
    preds_a = predict_on(holdout_codes)
    s_a = score(preds_a, holdout)
    results_a[label] = s_a

    preds_b = predict_on(sorted(truth_holdout["code"].unique()))
    s_b = score(preds_b, truth_holdout)
    results_b[label] = s_b

# Combine Set A
cmp_a = results_a["v2"][["category", "attr", "n"]].rename(columns={"n": "n_v2"})
for label in VARIANTS:
    s = results_a[label].rename(columns={"acc": f"acc_{label}", "coverage": f"cov_{label}", "n": f"n_{label}"})
    cmp_a = cmp_a.merge(s[["category", "attr", f"acc_{label}", f"cov_{label}"]],
                        on=["category", "attr"], how="left")

print("\n\n========== SET A — Tier1+2 holdout (per-attr) ==========")
print(cmp_a.to_string(index=False, float_format="%.3f"))

print("\n========== SET A OVERALL ==========")
ov_a = {f"acc_{l}": cmp_a[f"acc_{l}"].mean() for l in VARIANTS}
ov_a.update({f"cov_{l}": cmp_a[f"cov_{l}"].mean() for l in VARIANTS})
print(pd.DataFrame([ov_a]).to_string(index=False, float_format="%.4f"))

# Combine Set B
cmp_b = results_b["v2"][["category", "attr", "n"]].rename(columns={"n": "n_v2"})
for label in VARIANTS:
    s = results_b[label].rename(columns={"acc": f"acc_{label}", "coverage": f"cov_{label}", "n": f"n_{label}"})
    cmp_b = cmp_b.merge(s[["category", "attr", f"acc_{label}", f"cov_{label}"]],
                        on=["category", "attr"], how="left")

print("\n\n========== SET B — OFF-derived truth (per-attr) ==========")
print(cmp_b.to_string(index=False, float_format="%.3f"))

print("\n========== SET B OVERALL ==========")
ov_b = {f"acc_{l}": cmp_b[f"acc_{l}"].mean() for l in VARIANTS}
ov_b.update({f"cov_{l}": cmp_b[f"cov_{l}"].mean() for l in VARIANTS})
print(pd.DataFrame([ov_b]).to_string(index=False, float_format="%.4f"))

cmp_a.to_parquet("datasets/processed/eval_5way_holdout534.parquet", index=False)
cmp_b.to_parquet("datasets/processed/eval_5way_offtruth.parquet", index=False)
print("\nSaved: eval_5way_holdout534.parquet, eval_5way_offtruth.parquet")
