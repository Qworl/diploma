"""E2 evaluation: compute pre/post cascade accuracy on holdout override∪manual_only cells.

Loads cascade_prefix_pasta.parquet (pre-fix models on full gold_250) and
cascade_postfix_pasta.parquet (post-fix models on full gold_250), restricts to
the holdout-brand subset (e2_holdout_audit_codes.json), and computes the
Δ accuracy on override∪manual_only cells.

Applies decision rule from docs/thesis/pre_registration_2026-Q2.md:
- Δ ≥ +35 п.п. → honest_no_overfit_keep_49.4
- +25 ≤ Δ < +35 → partial_overfit_report_holdout_number
- Δ < +25 → fix_overfits_major_revision
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import PROCESSED_DIR

OVERRIDE_STATUSES = {"override", "manual_only"}


def _norm(v):
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"", "nan", "none", "null"}:
            return None
        if s in {"true", "yes"}:
            return "true"
        if s in {"false", "no"}:
            return "false"
        return s
    if isinstance(v, (bool, np.bool_)):
        return "true" if bool(v) else "false"
    return str(v).strip().lower()


def measure(cascade_df: pd.DataFrame, holdout_codes: set[str]) -> dict:
    """Per-attribute and overall accuracy on override∪manual_only holdout cells.

    Denominator semantics: acc_on_audited (abstain counted as wrong).
    """
    df = cascade_df.copy()
    df["code"] = df["code"].astype(str)
    df = df[df["code"].isin(holdout_codes)]
    df = df[df["status"].isin(OVERRIDE_STATUSES)]
    # manual_value must be non-empty
    df = df[df["manual_value"].apply(lambda v: _norm(v) is not None)]
    df["pred_norm"] = df["cascade_pred"].apply(_norm)
    df["manual_norm"] = df["manual_value"].apply(_norm)
    df["correct"] = (
        df["pred_norm"].notna()
        & (df["pred_norm"] == df["manual_norm"])
    )
    n = int(len(df))
    n_correct = int(df["correct"].sum())
    acc = float(n_correct) / n if n else float("nan")

    by_attr = {}
    for attr, sub in df.groupby("attr"):
        n_a = int(len(sub))
        n_c = int(sub["correct"].sum())
        by_attr[str(attr)] = {
            "n": n_a,
            "n_correct": n_c,
            "accuracy": float(n_c) / n_a if n_a else float("nan"),
        }
    return {
        "n": n,
        "n_correct": n_correct,
        "accuracy": acc,
        "by_attr": by_attr,
    }


def main() -> None:
    with open(Path(PROCESSED_DIR) / "e2_holdout_audit_codes.json") as f:
        holdout_codes = set(json.load(f))

    prefix = pd.read_parquet(Path(PROCESSED_DIR) / "cascade_prefix_pasta.parquet")
    postfix = pd.read_parquet(Path(PROCESSED_DIR) / "cascade_postfix_pasta.parquet")

    pre = measure(prefix, holdout_codes)
    post = measure(postfix, holdout_codes)

    delta_pp = (post["accuracy"] - pre["accuracy"]) * 100 if (
        not np.isnan(pre["accuracy"]) and not np.isnan(post["accuracy"])
    ) else None

    if delta_pp is None:
        decision = "insufficient_data"
    elif delta_pp >= 35:
        decision = "honest_no_overfit_keep_49.4"
    elif delta_pp >= 25:
        decision = "partial_overfit_report_holdout_number"
    else:
        decision = "fix_overfits_major_revision"

    # pasta_shape specific
    ps_pre = pre["by_attr"].get("pasta_shape", {"n": 0, "accuracy": float("nan")})
    ps_post = post["by_attr"].get("pasta_shape", {"n": 0, "accuracy": float("nan")})
    if (ps_pre["n"] > 0 and ps_post["n"] > 0
            and not np.isnan(ps_pre["accuracy"]) and not np.isnan(ps_post["accuracy"])):
        delta_ps_pp = (ps_post["accuracy"] - ps_pre["accuracy"]) * 100
    else:
        delta_ps_pp = None

    summary = {
        "n_holdout_products": len(holdout_codes),
        "overall": {
            "n_evaluated_pre": pre["n"],
            "n_evaluated_post": post["n"],
            "accuracy_pre_fix": pre["accuracy"],
            "accuracy_post_fix": post["accuracy"],
            "delta_pp": delta_pp,
            "decision": decision,
        },
        "pasta_shape_specific": {
            "n_cells_pre": ps_pre["n"],
            "n_cells_post": ps_post["n"],
            "accuracy_pre_fix": ps_pre["accuracy"],
            "accuracy_post_fix": ps_post["accuracy"],
            "delta_pp": delta_ps_pp,
        },
        "by_attr_pre": pre["by_attr"],
        "by_attr_post": post["by_attr"],
        "decision_rule": {
            "honest_no_overfit_keep_49.4": "delta >= 35 пп",
            "partial_overfit_report_holdout_number": "25 <= delta < 35 пп",
            "fix_overfits_major_revision": "delta < 25 пп",
        },
    }

    print(f"Holdout products: {len(holdout_codes)}")
    print(f"Overall pre-fix accuracy: {pre['accuracy']:.4f} (n={pre['n']})")
    print(f"Overall post-fix accuracy: {post['accuracy']:.4f} (n={post['n']})")
    if delta_pp is not None:
        print(f"Δ accuracy overall: {delta_pp:+.2f} п.п.")
    print(f"Decision: {decision}")
    print()
    print(f"pasta_shape pre-fix: {ps_pre['accuracy']:.4f} (n={ps_pre['n']})")
    print(f"pasta_shape post-fix: {ps_post['accuracy']:.4f} (n={ps_post['n']})")
    if delta_ps_pp is not None:
        print(f"Δ pasta_shape: {delta_ps_pp:+.2f} п.п.")

    out = Path(PROCESSED_DIR) / "holdout_audit_pasta_eval.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
