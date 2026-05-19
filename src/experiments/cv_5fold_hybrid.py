"""5-fold cross-validation on Tier1+2 (Opus + gpt55) for hybrid v3b config.

Splits Tier1+2 codes (2666) into 5 stratified folds per category.
For each fold k:
  - train_codes = folds {0,1,2,3,4} \ {k}  (~2132 Tier1+2 codes)
  - holdout_codes = fold k                  (~534 codes)
  - Training data = train_codes Tier1+2 + ALL Tier3 (gemini)
  - Trains v3b config (--w-silver 1 --w-tier1 6 --w-tier2 4 --w-tier3 2)
    saves models to /tmp/cv_fold_k/
  - Evaluates on fold-k holdout, records per-fold accuracy
Final: mean ± std across 5 folds for tight CI.
"""
import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(".").resolve()))

from src.eval.cascade_predict import predict_cascade

K = 5
SEED = 42
TIER_WEIGHTS = ("1.0", "6.0", "4.0", "2.0")  # silver, t1, t2, t3 = v3b config

PROCESSED = Path("datasets/processed")
WORK = Path("/tmp/cv5fold")
WORK.mkdir(exist_ok=True)

v2 = pd.read_parquet(PROCESSED / "consensus_gold_v2_expanded.parquet")
v2["code"] = v2["code"].astype(str)
v2["tier"] = v2["opus_reasoning"].apply(lambda x: "tier1_opus" if x else "tier2_gpt55")

v3 = pd.read_parquet(PROCESSED / "consensus_hybrid_v3.parquet")
v3["code"] = v3["code"].astype(str)
tier3 = v3[v3["tier"] == "tier3_gemini"]

# Build folds per category (stratified by code)
rng = np.random.RandomState(SEED)
fold_assignment: dict[str, int] = {}
for cat, grp in v2.groupby("category"):
    codes = sorted(grp["code"].unique())
    shuf = rng.permutation(codes)
    fold_sizes = np.full(K, len(shuf) // K)
    fold_sizes[: len(shuf) % K] += 1  # distribute remainder
    idx = 0
    for k, size in enumerate(fold_sizes):
        for c in shuf[idx:idx + size]:
            fold_assignment[c] = k
        idx += size

print(f"Total Tier1+2 codes: {len(fold_assignment)}")
fold_counts = pd.Series(list(fold_assignment.values())).value_counts().sort_index()
print(f"Fold sizes: {fold_counts.tolist()}")

results = []
for k in range(K):
    print(f"\n========== FOLD {k} ==========")
    holdout_codes = {c for c, f in fold_assignment.items() if f == k}
    train_codes = {c for c, f in fold_assignment.items() if f != k}
    print(f"  train: {len(train_codes)}  holdout: {len(holdout_codes)}")

    # Build per-fold gold parquet
    train_v2 = v2[v2["code"].isin(train_codes)]
    train_full = pd.concat([train_v2, tier3], ignore_index=True)
    train_path = WORK / f"v3b_fold{k}_train.parquet"
    holdout_path = WORK / f"v3b_fold{k}_holdout.parquet"
    train_full.to_parquet(train_path, index=False)
    v2[v2["code"].isin(holdout_codes)].to_parquet(holdout_path, index=False)

    # Sidecar CSV of holdout codes for leakage-free training
    holdout_codes_csv = WORK / f"v3b_fold{k}_holdout_codes.csv"
    pd.DataFrame({"code": sorted(holdout_codes)}).to_csv(holdout_codes_csv, index=False)

    # Train
    model_dir = WORK / f"models_fold{k}"
    model_dir.mkdir(exist_ok=True)
    cmd = [".venv/bin/python", "-u", "-m", "src.experiments.train_hybrid_cascade",
           "--gold-path", str(train_path),
           "--w-silver", TIER_WEIGHTS[0],
           "--w-tier1", TIER_WEIGHTS[1],
           "--w-tier2", TIER_WEIGHTS[2],
           "--w-tier3", TIER_WEIGHTS[3],
           "--output-dir", str(model_dir),
           "--holdout-codes", str(holdout_codes_csv)]
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1"}
    log_path = WORK / f"fold{k}_train.log"
    print(f"  training → {model_dir} (log: {log_path})")
    with open(log_path, "w") as lf:
        rc = subprocess.run(cmd, env=env, stdout=lf, stderr=lf).returncode
    if rc != 0:
        print(f"  TRAIN FAILED rc={rc}, see {log_path}")
        continue

    # Evaluate: swap fold's models into models/ and run cascade
    models_live = Path("models")
    for p in models_live.glob("*_xgb_hybrid.pkl"):
        p.unlink()
    for p in models_live.glob("*_le_hybrid.pkl"):
        p.unlink()
    for p in model_dir.iterdir():
        shutil.copy(p, models_live / p.name)

    holdout = pd.read_parquet(holdout_path)
    holdout["code"] = holdout["code"].astype(str)
    pred_rows = []
    for cat in ["pasta", "chocolate", "cheeses"]:
        cat_codes = sorted(set(holdout[holdout["category"] == cat]["code"]))
        silver = pd.read_parquet(PROCESSED / f"{cat}_stratified_silver_standard.parquet")
        silver["code"] = silver["code"].astype(str)
        sub = silver[silver["code"].isin(cat_codes)].copy()
        if sub.empty:
            continue
        preds = predict_cascade(sub, category=f"{cat}_stratified",
                                use_hybrid=True, include_bayes=False, include_regex=True)
        preds["category"] = cat
        pred_rows.append(preds)
    preds = pd.concat(pred_rows, ignore_index=True)
    preds["code"] = preds["code"].astype(str)
    preds["pred_norm"] = preds["predicted"].astype(str).str.strip().str.lower()

    # Score
    gold = holdout.copy()
    gold["gold_norm"] = gold["gold_value"].astype(str).str.strip().str.lower()
    gold = gold[gold["gold_is_null"] != True]  # noqa: E712
    m = gold.merge(preds, on=["category", "code", "attr"], how="inner")
    m_eval = m[~m["layer"].isin(["abstain", None])]
    m_eval = m_eval[m_eval["pred_norm"].notna() & (m_eval["pred_norm"] != "nan")]
    acc = (m_eval["pred_norm"] == m_eval["gold_norm"]).mean()
    cov = (~m["layer"].isin(["abstain", None])).mean()
    n_holdout_obs = len(m)
    print(f"  fold {k}: acc={acc:.4f} cov={cov:.4f} (n_holdout={n_holdout_obs})")
    results.append({"fold": k, "acc": acc, "cov": cov,
                    "n_train": len(train_codes), "n_holdout": n_holdout_obs})

df = pd.DataFrame(results)
print("\n\n========== 5-FOLD CV RESULT (v3b config) ==========")
print(df.to_string(index=False, float_format="%.4f"))

print(f"\nAccuracy: {df['acc'].mean():.4f} ± {df['acc'].std(ddof=1):.4f}  "
      f"(95% CI ≈ ±{1.96 * df['acc'].std(ddof=1) / np.sqrt(K):.4f})")
print(f"Coverage: {df['cov'].mean():.4f} ± {df['cov'].std(ddof=1):.4f}")

# Bootstrap CI of mean accuracy
rng_bs = np.random.RandomState(SEED)
n_bs = 1000
bs = []
for _ in range(n_bs):
    sample = rng_bs.choice(df["acc"].values, K, replace=True)
    bs.append(sample.mean())
bs = np.array(sorted(bs))
print(f"Bootstrap 95% CI (1000 resamples): "
      f"[{bs[int(0.025 * n_bs)]:.4f}, {bs[int(0.975 * n_bs)]:.4f}]")

df.to_parquet(PROCESSED / "cv5fold_v3b.parquet", index=False)
print(f"\nSaved to {PROCESSED / 'cv5fold_v3b.parquet'}")
