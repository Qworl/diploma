"""Final OOD eval: cascade v3b vs v3d on 600 fresh OFF codes never seen in any training.
Truth = Opus 4.5 promptfix labels (best available).
"""
import json, shutil, sys, warnings
from pathlib import Path
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(".").resolve()))
from src.eval.cascade_predict import predict_cascade

ROOT = Path("/Users/miafrolov/Desktop/stuff/ai_attributes/.worktrees/phase2-recomputes-on-v2-gold")
VARIANTS = {
    "v3b": ROOT / "models_backup/v3b_weighted",
    "v3d": ROOT / "models_backup/v3d_promptfix",
}
MODELS = Path("models")

# Build truth from Opus fresh prod decisions
truth_rows = []
for cat in ["pasta","chocolate","cheeses"]:
    p = Path(f"datasets/manual_label/opus_batches/fresh_prod/{cat}_decisions.json")
    data = json.load(open(p, encoding="utf-8"))
    for code, attrs in data.items():
        if not isinstance(attrs, dict): continue
        for attr, payload in attrs.items():
            if not isinstance(payload, dict): continue
            val = payload.get("value")
            is_null = val is None or (isinstance(val, str) and val.strip().lower() in ("","null","none"))
            truth_rows.append({
                "category": cat, "code": str(code), "attr": attr,
                "gold_value": "" if is_null else str(val).strip(),
                "gold_is_null": is_null,
            })
truth = pd.DataFrame(truth_rows)
truth["code"] = truth["code"].astype(str)
codes = sorted(truth["code"].unique())
print(f"Truth: {len(truth)} rows, {len(codes)} codes (fresh OFF, never in training)")
print(f"Fill rate: {(~truth['gold_is_null']).mean():.3f}")

results = {}
for label, d in VARIANTS.items():
    print(f"\n=== {label} ===")
    for p in MODELS.glob("*_xgb_hybrid.pkl"): p.unlink()
    for p in MODELS.glob("*_le_hybrid.pkl"): p.unlink()
    for p in d.iterdir(): shutil.copy(p, MODELS / p.name)
    pred_rows = []
    for cat in ["pasta","chocolate","cheeses"]:
        cat_codes = [c for c in codes if c in set(truth[truth["category"]==cat]["code"])]
        s = pd.read_parquet(f"datasets/processed/{cat}_stratified_silver_standard.parquet")
        s["code"] = s["code"].astype(str)
        sub = s[s["code"].isin(cat_codes)].copy()
        if sub.empty: continue
        preds = predict_cascade(sub, category=f"{cat}_stratified", use_hybrid=True,
                                include_bayes=False, include_regex=True)
        preds["category"] = cat
        pred_rows.append(preds)
    out = pd.concat(pred_rows, ignore_index=True)
    out["code"] = out["code"].astype(str)
    out["pred_norm"] = out["predicted"].astype(str).str.strip().str.lower()
    results[label] = out

g = truth[truth["gold_is_null"] != True].copy()
g["gold_norm"] = g["gold_value"].astype(str).str.strip().str.lower()

DERIVED = {"nutri_score_grade", "protein_class", "fat_class", "cocoa_percentage"}
cmp_rows = []
for cat in ["pasta","chocolate","cheeses"]:
    for attr in g[g["category"]==cat]["attr"].unique():
        sub_truth = g[(g["category"]==cat) & (g["attr"]==attr)]
        row = {"category": cat, "attr": attr, "is_derived": attr in DERIVED, "n_truth": len(sub_truth)}
        for label, preds in results.items():
            sub_pred = preds[(preds["category"]==cat) & (preds["attr"]==attr)]
            m = sub_truth.merge(sub_pred, on=["category","code","attr"], how="inner")
            m_eval = m[~m["layer"].isin(["abstain", None])]
            m_eval = m_eval[m_eval["pred_norm"].notna() & (m_eval["pred_norm"] != "nan")]
            row[f"acc_{label}"] = (m_eval["pred_norm"] == m_eval["gold_norm"]).mean() if len(m_eval) else 0
            row[f"cov_{label}"] = (~m["layer"].isin(["abstain", None])).mean() if len(m) else 0
            row[f"n_{label}"] = len(m_eval)
        cmp_rows.append(row)

cmp = pd.DataFrame(cmp_rows)
print("\n========== PER-ATTR ==========")
print(cmp.drop(columns=["is_derived"]).to_string(index=False, float_format="%.3f"))
print("\n========== OVERALL ==========")
for label in ["v3b","v3d"]:
    print(f"  {label}: acc={cmp[f'acc_{label}'].mean():.4f} cov={cmp[f'cov_{label}'].mean():.4f} n_obs={cmp[f'n_{label}'].sum()}")
print("\n========== DERIVED only ==========")
sub = cmp[cmp["is_derived"]]
for label in ["v3b","v3d"]:
    print(f"  {label}: acc={sub[f'acc_{label}'].mean():.4f}  ({len(sub)} attrs, {sub[f'n_{label}'].sum()} obs)")
print("\n========== NON-DERIVED only ==========")
sub = cmp[~cmp["is_derived"]]
for label in ["v3b","v3d"]:
    print(f"  {label}: acc={sub[f'acc_{label}'].mean():.4f}  ({len(sub)} attrs, {sub[f'n_{label}'].sum()} obs)")

cmp.to_parquet("datasets/processed/eval_fresh_prod_v3b_vs_v3d.parquet", index=False)
print("\nSaved eval_fresh_prod_v3b_vs_v3d.parquet")
