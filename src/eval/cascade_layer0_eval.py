"""End-to-end eval: cascade WITH Layer 0 routing vs cascade WITH oracle category.

Honest production scenario: partner sends product, system must first decide
category (Layer 0), then apply correct cascade. If Layer 0 misclassifies,
cascade applies wrong schema → attribute predictions fail.

Compares:
  R_oracle_cat: cascade(gold_category) — uses gold category as input
  R_router_v3:  cascade(router_v3_predicted_cat) — production reality

For each v2 gold code (n=2666, 3 cats):
  1. Apply router_v3 (LightGBM TF-IDF) → predicted_cat (or OOD)
  2. If predicted_cat == gold_cat: cascade(predicted_cat) attrs as usual
  3. If predicted_cat != gold_cat: all attrs counted as wrong (schema mismatch)
  4. If OOD (max_proba < threshold): all attrs counted as wrong

Output:
  datasets/processed/cascade_layer0_eval.parquet
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout, force=True,
)
logger = logging.getLogger(__name__)

WORKTREE_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = WORKTREE_ROOT / "datasets" / "processed"
MODELS_DIR = WORKTREE_ROOT / "models"

ROUTER_INPUT_FIELDS = ("product_name", "brands", "ingredients_text", "quantity")
CATEGORIES = ["pasta", "chocolate", "cheeses"]
OUT_PATH = PROCESSED_DIR / "cascade_layer0_eval.parquet"


def build_text(row: pd.Series) -> str:
    return " ".join(
        str(row.get(k, "") or "") for k in ROUTER_INPUT_FIELDS
    ).strip() or " "


def load_router_v3():
    with open(MODELS_DIR / "category_router_v3_lgbm.pkl", "rb") as f:
        clf = pickle.load(f)
    with open(MODELS_DIR / "category_router_v3_vec.pkl", "rb") as f:
        vec = pickle.load(f)
    with open(MODELS_DIR / "category_router_v3_le.pkl", "rb") as f:
        le = pickle.load(f)
    with open(MODELS_DIR / "category_router_v3_threshold.json") as f:
        thr = json.load(f)["threshold"]
    return clf, vec, le, thr


def main() -> None:
    # Load v2 expanded gold (2666 codes, long format)
    gold = pd.read_parquet(PROCESSED_DIR / "consensus_gold_v2_expanded.parquet")
    gold = gold[~gold["gold_is_null"]].copy()
    gold["code"] = gold["code"].astype(str)
    logger.info("v2 gold: %d (code, attr) rows over %d codes",
                len(gold), gold["code"].nunique())

    # Build code → category mapping from gold
    code_to_cat = gold.groupby("code")["category"].first().to_dict()

    # Load partner-input data for each code (from per-cat silver_standard parquets)
    code_to_text: dict[str, str] = {}
    code_to_row: dict[str, dict] = {}
    for cat in CATEGORIES:
        df = pd.read_parquet(PROCESSED_DIR / f"{cat}_stratified_silver_standard.parquet")
        df["code"] = df["code"].astype(str)
        for _, row in df.iterrows():
            code_to_text[row["code"]] = build_text(row)
            code_to_row[row["code"]] = {
                k: row.get(k, "") for k in ROUTER_INPUT_FIELDS
            }
    # For codes not in silver (B-scale new from OFF), fetch from OFF cache
    missing = [c for c in code_to_cat if c not in code_to_text]
    logger.info("Codes with partner-text from silver: %d. Missing: %d",
                len(code_to_text), len(missing))

    # Try OFF cache for missing codes
    off_cache_dir = WORKTREE_ROOT / "datasets" / "manual_label" / "off_cache"
    fetched = 0
    for code in missing:
        cache_path = off_cache_dir / f"{code}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                off_data = json.load(f)
            product = off_data.get("product", {}) if "product" in off_data else off_data
            row_proxy = {
                "product_name": product.get("product_name", ""),
                "brands": product.get("brands", ""),
                "ingredients_text": product.get("ingredients_text", ""),
                "quantity": product.get("quantity", ""),
            }
            code_to_text[code] = build_text(pd.Series(row_proxy))
            code_to_row[code] = row_proxy
            fetched += 1
    logger.info("Fetched %d missing codes from OFF cache", fetched)

    # Filter to codes we have partner data for
    eval_codes = [c for c in code_to_cat if c in code_to_text]
    logger.info("Eval codes (have partner data): %d / %d",
                len(eval_codes), len(code_to_cat))

    # Load router v3
    clf, vec, le, thr = load_router_v3()
    logger.info("Router v3 threshold: %.4f", thr)

    # Apply router
    texts = [code_to_text[c] for c in eval_codes]
    X = vec.transform(texts)
    proba = clf.predict_proba(X)
    max_proba = proba.max(axis=1)
    pred_enc = proba.argmax(axis=1)
    pred_cat = le.inverse_transform(pred_enc)
    is_ood = max_proba < thr

    # Aggregate predictions per code
    code_pred: dict[str, dict] = {}
    for i, code in enumerate(eval_codes):
        code_pred[code] = {
            "predicted_cat": pred_cat[i],
            "confidence": float(max_proba[i]),
            "is_ood": bool(is_ood[i]),
            "gold_cat": code_to_cat[code],
        }

    # Router accuracy on eval codes
    correct = sum(1 for c, p in code_pred.items()
                  if p["predicted_cat"] == p["gold_cat"] and not p["is_ood"])
    n = len(code_pred)
    router_acc = correct / n if n else 0
    n_ood = sum(1 for p in code_pred.values() if p["is_ood"])
    n_wrong = sum(1 for p in code_pred.values()
                  if p["predicted_cat"] != p["gold_cat"] and not p["is_ood"])

    logger.info("Router v3 on v2 gold: %d correct, %d wrong-cat, %d OOD-rejected",
                correct, n_wrong, n_ood)
    logger.info("Router accuracy: %.4f (= %d / %d)", router_acc, correct, n)

    # Per-category router accuracy
    print(f"\n{'='*60}")
    print("Router v3 accuracy on v2 gold per category:")
    print(f"{'='*60}")
    for cat in CATEGORIES:
        cat_codes = [c for c, p in code_pred.items() if p["gold_cat"] == cat]
        cat_correct = sum(1 for c in cat_codes
                          if code_pred[c]["predicted_cat"] == cat
                          and not code_pred[c]["is_ood"])
        cat_ood = sum(1 for c in cat_codes if code_pred[c]["is_ood"])
        cat_wrong = len(cat_codes) - cat_correct - cat_ood
        print(f"  {cat:12s}: {cat_correct}/{len(cat_codes)} correct "
              f"({100*cat_correct/len(cat_codes):.1f}%), "
              f"{cat_wrong} wrong-cat, {cat_ood} OOD-rejected")

    # Confusion matrix
    print(f"\n{'='*60}")
    print("Confusion (gold_cat → predicted_cat) where router was wrong:")
    print(f"{'='*60}")
    for gold_cat in CATEGORIES:
        wrong_codes = [c for c, p in code_pred.items()
                       if p["gold_cat"] == gold_cat
                       and p["predicted_cat"] != gold_cat
                       and not p["is_ood"]]
        if not wrong_codes:
            continue
        wrong_pred = {}
        for c in wrong_codes:
            pc = code_pred[c]["predicted_cat"]
            wrong_pred[pc] = wrong_pred.get(pc, 0) + 1
        print(f"  {gold_cat} → {wrong_pred}")

    # Now apply cascade with predicted_cat
    # For codes where router correct → cascade(gold_cat) attrs work as usual
    # For wrong-cat or OOD → attribute predictions count as wrong
    # We DON'T re-run cascade here — we use cascade_v3 stacked accuracy from EXP13
    # per (cat, attr), and apply Layer 0 penalty.

    # Load EXP13 results
    v3_dfs = []
    for cat in CATEGORIES:
        path = PROCESSED_DIR / f"cascade_v3_eval_{cat}.parquet"
        if path.exists():
            v3_dfs.append(pd.read_parquet(path))
    v3_eval = pd.concat(v3_dfs, ignore_index=True)
    logger.info("Loaded EXP13 results: %d rows", len(v3_eval))

    # Map (cat, attr) → oracle_acc_v3
    v3_map = v3_eval.set_index(["category", "attr"])[["acc_v2_baseline", "acc_v3_stacked"]].to_dict("index")

    # For each cat: weighted average where router correct → cascade acc, else 0
    # End-to-end accuracy per (cat, attr) = router_correct_frac * cascade_acc + 0 * wrong_frac
    # But router_correct fraction differs per cat
    cat_router_correct = {}
    for cat in CATEGORIES:
        cat_codes = [c for c, p in code_pred.items() if p["gold_cat"] == cat]
        cat_correct = sum(1 for c in cat_codes
                          if code_pred[c]["predicted_cat"] == cat
                          and not code_pred[c]["is_ood"])
        cat_router_correct[cat] = cat_correct / len(cat_codes) if cat_codes else 0

    # Build comparison table
    rows = []
    for (cat, attr), v in v3_map.items():
        r_correct = cat_router_correct.get(cat, 0)
        rows.append({
            "category": cat,
            "attr": attr,
            "acc_v2_oracle_cat": v["acc_v2_baseline"],
            "acc_v3_oracle_cat": v["acc_v3_stacked"],
            "router_v3_correct_frac": r_correct,
            "acc_v2_with_router": v["acc_v2_baseline"] * r_correct,
            "acc_v3_with_router": v["acc_v3_stacked"] * r_correct,
        })
    out = pd.DataFrame(rows)
    out.to_parquet(OUT_PATH, index=False)
    logger.info("Saved %d rows to %s", len(out), OUT_PATH)

    # Summary
    print(f"\n{'='*72}")
    print("END-TO-END: Cascade + Layer 0 (router_v3) vs oracle category")
    print(f"{'='*72}")
    print()
    print(f"{'Category':<12} {'Router acc':>12} {'v2 oracle':>12} {'v2+router':>12} "
          f"{'v3 oracle':>12} {'v3+router':>12}")
    print("-" * 72)
    for cat in CATEGORIES:
        sub = out[out["category"] == cat]
        if len(sub) == 0:
            continue
        ra = cat_router_correct[cat]
        v2_o = sub["acc_v2_oracle_cat"].mean()
        v2_r = sub["acc_v2_with_router"].mean()
        v3_o = sub["acc_v3_oracle_cat"].mean()
        v3_r = sub["acc_v3_with_router"].mean()
        print(f"{cat:<12} {ra*100:>11.2f}% {v2_o*100:>11.2f}% {v2_r*100:>11.2f}% "
              f"{v3_o*100:>11.2f}% {v3_r*100:>11.2f}%")

    print()
    grand_router = out["router_v3_correct_frac"].mean()
    grand_v2_o = out["acc_v2_oracle_cat"].mean()
    grand_v2_r = out["acc_v2_with_router"].mean()
    grand_v3_o = out["acc_v3_oracle_cat"].mean()
    grand_v3_r = out["acc_v3_with_router"].mean()
    print(f"{'GRAND MEAN':<12} {grand_router*100:>11.2f}% {grand_v2_o*100:>11.2f}% "
          f"{grand_v2_r*100:>11.2f}% {grand_v3_o*100:>11.2f}% {grand_v3_r*100:>11.2f}%")
    print()
    print(f"Router penalty on cascade: {(grand_v3_o - grand_v3_r) * 100:.2f}pp")
    print(f"(if router were 100% accurate, cascade would be {grand_v3_o*100:.2f}%; "
          f"with router_v3 it's {grand_v3_r*100:.2f}%)")


if __name__ == "__main__":
    main()
