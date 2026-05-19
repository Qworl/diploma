"""E4: Bayes-validator demote precision/recall on v2-gold brand-disjoint test.

Methodology:
  For each cascade ML prediction (layer == 'ml'), call ValidatorService.validate_value
  with evidence = brand + all other cascade predictions for that code (minus self).
  Then compare validator flag against whether cascade was actually wrong vs gold.

  TP = flagged AND cascade wrong (validator caught real error)
  FP = flagged AND cascade right (validator wasted)
  TN = not flagged AND cascade right
  FN = not flagged AND cascade wrong (missed)

  demote_precision = TP / (TP + FP)
  demote_recall    = TP / (TP + FN)
  expected_delta_acc_if_demote = ((TP * llm_acc) + (FP * (llm_acc - 1))) / n_ml
                              (TP cells go from wrong→right with prob llm_acc;
                               FP cells go from right→wrong with prob 1-llm_acc)
  expected_delta_llm_cost      = (TP + FP) / n_ml

Output: datasets/processed/bayes_validator_demote_metric.parquet
Entry:  python -m src.eval.bayes_validator_demote_metric
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import pandas as pd

# Make demo/ml_service importable for ValidatorService
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_DEMO_DIR = os.path.join(_PROJECT_ROOT, "demo", "ml_service")
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)

from validator import ValidatorService  # noqa: E402


CATEGORIES = [
    ("pasta", "pasta_stratified"),
    ("chocolate", "chocolate_stratified"),
    ("cheeses", "cheeses_stratified"),
]


def load_brand_lookup(internal_cat: str) -> dict[str, str]:
    """Map code -> brand string from raw stratified parquet."""
    path = os.path.join("datasets", "processed", f"{internal_cat}_raw.parquet")
    df = pd.read_parquet(path, columns=["code", "brands"])
    df["code"] = df["code"].astype(str)
    df["brands"] = df["brands"].fillna("").astype(str)
    # Take first brand (cascade pattern uses first token usually; safe approximation)
    return dict(zip(df["code"], df["brands"]))


def load_gold(category: str) -> pd.DataFrame:
    g = pd.read_parquet("datasets/processed/consensus_gold_v2_expanded.parquet")
    g = g[(g["category"] == category) & (~g["gold_is_null"])].copy()
    g["code"] = g["code"].astype(str)
    # (code, attr) -> gold_value
    return g[["code", "attr", "gold_value"]]


def load_cascade(category: str) -> pd.DataFrame:
    p = f"datasets/processed/cascade_preds_{category}_v2_gold_hybrid_v3_fixed.parquet"
    df = pd.read_parquet(p)
    df["code"] = df["code"].astype(str)
    return df


def load_llm_acc_per_attr(internal_cat: str) -> dict[str, float]:
    """Mean LLM accuracy per attr on cells where both pred and gt are non-null."""
    cat_short = internal_cat.split("_")[0]  # pasta_stratified -> pasta
    path = f"datasets/processed/direct_llm_eval_{internal_cat}_sonnet45.parquet"
    df = pd.read_parquet(path)
    # correct_when_both_present is the right metric per spec
    mask = (df["predicted_non_null"] == 1) & (df["gt_non_null"] == 1)
    sub = df[mask]
    acc = sub.groupby("attr")["correct_when_both_present"].mean().to_dict()
    return acc


def _normalize_value(v: Any) -> Any:
    """Normalize a value for equality compare (handle bool/str/None)."""
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, str) and v.strip().lower() in {"", "none", "null", "nan"}:
        return None
    return v


def _eq(pred: Any, gold: Any) -> bool:
    p = _normalize_value(pred)
    g = _normalize_value(gold)
    if p is None or g is None:
        return False
    # Coerce both to lowercase strings if reasonable
    try:
        if isinstance(p, (bool, np.bool_)) or isinstance(g, (bool, np.bool_)):
            return str(p).lower() == str(g).lower()
        return str(p).strip().lower() == str(g).strip().lower()
    except Exception:
        return p == g


def evaluate_category(short_cat: str, internal_cat: str, validator: ValidatorService) -> pd.DataFrame:
    print(f"\n=== {short_cat} ({internal_cat}) ===", flush=True)
    cascade = load_cascade(short_cat)
    gold = load_gold(short_cat)
    brands = load_brand_lookup(internal_cat)
    llm_acc = load_llm_acc_per_attr(internal_cat)

    # Build per-code evidence: code -> {attr: pred_value} from ALL layers
    base_ev_by_code: dict[str, dict[str, Any]] = {}
    for code, grp in cascade.groupby("code"):
        ev: dict[str, Any] = {}
        if code in brands and brands[code]:
            ev["brand"] = brands[code]
        for _, row in grp.iterrows():
            if row["layer"] == "abstain":
                continue
            ev[row["attr"]] = row["predicted"]
        base_ev_by_code[code] = ev

    # Join cascade ML cells with gold
    cascade_ml = cascade[cascade["layer"] == "ml"].copy()
    merged = cascade_ml.merge(gold, on=["code", "attr"], how="inner")
    print(f"  cascade_ml rows: {len(cascade_ml)}, with gold: {len(merged)}", flush=True)

    rows: list[dict] = []
    flag_total = 0
    err_total = 0
    n_total = len(merged)
    flag_cells: list[dict] = []  # for sanity flag-rate

    for _, r in merged.iterrows():
        code = r["code"]
        attr = r["attr"]
        pred = r["predicted"]
        gold_v = r["gold_value"]

        ev = dict(base_ev_by_code.get(code, {}))
        ev.pop(attr, None)

        verdict = validator.validate_value(internal_cat, attr, pred, ev)
        if verdict is None:
            # No verdict — skip (validator can't bucketize or attr not in net)
            continue
        flagged = bool(verdict["flagged"])
        cascade_correct = _eq(pred, gold_v)
        flag_cells.append({"attr": attr, "flagged": flagged, "correct": cascade_correct})

    cells_df = pd.DataFrame(flag_cells)
    if cells_df.empty:
        return pd.DataFrame()

    # Aggregate per attr
    out_rows = []
    overall_flag = cells_df["flagged"].mean()
    print(f"  overall flag rate: {overall_flag:.4f} (target ~0.05)", flush=True)

    for attr, sub in cells_df.groupby("attr"):
        n_ml = len(sub)
        n_flag = int(sub["flagged"].sum())
        n_err = int((~sub["correct"]).sum())
        tp = int(((sub["flagged"]) & (~sub["correct"])).sum())
        fp = int(((sub["flagged"]) & (sub["correct"])).sum())
        tn = int(((~sub["flagged"]) & (sub["correct"])).sum())
        fn = int(((~sub["flagged"]) & (~sub["correct"])).sum())

        flag_rate = n_flag / n_ml if n_ml else 0.0
        demote_prec = tp / (tp + fp) if (tp + fp) else float("nan")
        demote_rec = tp / (tp + fn) if (tp + fn) else float("nan")
        cascade_acc = sub["correct"].mean() if n_ml else float("nan")
        random_baseline = 1.0 - cascade_acc  # P(cascade wrong) = random demote precision

        l_acc = llm_acc.get(attr, float("nan"))
        # Expected accuracy change if we demote ALL flagged:
        #   each TP cell: cascade_wrong -> LLM correct with prob l_acc => +l_acc
        #   each FP cell: cascade_right -> LLM correct with prob l_acc; was correct (1)
        #                so delta = l_acc - 1
        if not np.isnan(l_acc):
            expected_delta_acc = (tp * l_acc + fp * (l_acc - 1.0)) / n_ml if n_ml else 0.0
        else:
            expected_delta_acc = float("nan")
        expected_delta_cost = (tp + fp) / n_ml if n_ml else 0.0

        out_rows.append({
            "category": short_cat,
            "attr": attr,
            "n_ml_predictions": n_ml,
            "n_flagged": n_flag,
            "flag_rate": flag_rate,
            "n_flag_cascade_wrong": tp,
            "n_flag_cascade_right": fp,
            "tn": tn,
            "fn": fn,
            "cascade_acc_on_covered": cascade_acc,
            "random_demote_precision_baseline": random_baseline,
            "demote_precision": demote_prec,
            "demote_recall": demote_rec,
            "demote_precision_lift": (demote_prec - random_baseline) if not np.isnan(demote_prec) else float("nan"),
            "llm_acc_on_attr": l_acc,
            "expected_delta_acc_if_demote": expected_delta_acc,
            "expected_delta_llm_cost": expected_delta_cost,
        })

    return pd.DataFrame(out_rows)


def main() -> None:
    models_dir = os.path.join(_PROJECT_ROOT, "models")
    print("Loading ValidatorService...", flush=True)
    validator = ValidatorService(
        models_dir=models_dir,
        internal_categories=[ic for _, ic in CATEGORIES],
    )
    if not validator.ready():
        print("FATAL: validator not ready", flush=True)
        return

    all_rows: list[pd.DataFrame] = []
    for short, internal in CATEGORIES:
        df = evaluate_category(short, internal, validator)
        if df.empty:
            print(f"  WARN: no rows for {short}", flush=True)
            continue
        all_rows.append(df)

    if not all_rows:
        print("No data computed", flush=True)
        return

    full = pd.concat(all_rows, ignore_index=True)

    out_path = "datasets/processed/bayes_validator_demote_metric.parquet"
    full.to_parquet(out_path, index=False)
    print(f"\nSaved {len(full)} rows -> {out_path}", flush=True)

    # Pretty print per-attr
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))
    print("\n--- PER-ATTR TABLE ---")
    cols_show = [
        "category", "attr", "n_ml_predictions", "n_flagged", "flag_rate",
        "n_flag_cascade_wrong", "n_flag_cascade_right",
        "demote_precision", "random_demote_precision_baseline", "demote_precision_lift",
        "demote_recall", "llm_acc_on_attr",
        "expected_delta_acc_if_demote", "expected_delta_llm_cost",
    ]
    print(full[cols_show].to_string(index=False))

    # Per-category aggregate (micro)
    print("\n--- PER-CATEGORY AGGREGATE (micro) ---")
    agg_rows = []
    for cat, sub in full.groupby("category"):
        n_ml = sub["n_ml_predictions"].sum()
        n_flag = sub["n_flagged"].sum()
        tp = sub["n_flag_cascade_wrong"].sum()
        fp = sub["n_flag_cascade_right"].sum()
        fn = sub["fn"].sum()
        tn = sub["tn"].sum()
        cascade_acc = (tn + fp) / n_ml if n_ml else 0.0
        baseline = 1 - cascade_acc
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        flag_rate = n_flag / n_ml if n_ml else 0.0
        # Expected delta using attr-weighted (sum already weighted by n via expected_delta * n)
        # We sum (tp * l_acc + fp * (l_acc - 1)) over attrs then / total n_ml
        weighted = []
        cost = []
        for _, r in sub.iterrows():
            l = r["llm_acc_on_attr"]
            if not np.isnan(l):
                weighted.append(r["n_flag_cascade_wrong"] * l + r["n_flag_cascade_right"] * (l - 1.0))
                cost.append(r["n_flag_cascade_wrong"] + r["n_flag_cascade_right"])
        exp_dacc = sum(weighted) / n_ml if n_ml else float("nan")
        exp_dcost = sum(cost) / n_ml if n_ml else float("nan")
        agg_rows.append({
            "category": cat, "n_ml": int(n_ml), "n_flagged": int(n_flag),
            "flag_rate": flag_rate, "cascade_acc": cascade_acc,
            "demote_precision": prec, "random_baseline": baseline,
            "lift": (prec - baseline) if not np.isnan(prec) else float("nan"),
            "demote_recall": rec,
            "expected_delta_acc_if_demote": exp_dacc,
            "expected_delta_llm_cost": exp_dcost,
        })
    agg = pd.DataFrame(agg_rows)
    print(agg.to_string(index=False))

    # Grand summary
    print("\n--- GRAND SUMMARY ---")
    n_ml = full["n_ml_predictions"].sum()
    n_flag = full["n_flagged"].sum()
    tp = full["n_flag_cascade_wrong"].sum()
    fp = full["n_flag_cascade_right"].sum()
    fn = full["fn"].sum()
    tn = full["tn"].sum()
    cascade_acc = (tn + fp) / n_ml
    baseline = 1 - cascade_acc
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    print(f"  n_ml_predictions (with gold): {int(n_ml)}")
    print(f"  n_flagged: {int(n_flag)}  flag_rate: {n_flag / n_ml:.4f}  (target ~0.05)")
    print(f"  cascade_acc on covered: {cascade_acc:.4f}")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  demote_precision: {prec:.4f}")
    print(f"  random baseline (cascade error rate): {baseline:.4f}")
    print(f"  lift: {prec - baseline:+.4f}")
    print(f"  demote_recall: {rec:.4f}")


if __name__ == "__main__":
    main()
