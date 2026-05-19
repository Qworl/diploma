"""E1 circularity analysis: compare hybrid headline accuracy across Layer 4 backends.

Reads router_pareto_gold{,_<backend>}.parquet for baseline gpt-oss-120b and 4
alternative Layer 4 LLMs (3 consensus members + 1 control), reports per_attr_table
headline accuracy and delta vs baseline. Applies the pre-registered decision rule
from docs/thesis/pre_registration_2026-Q2.md.

NOTE on column names: the router_pareto_gold parquet uses columns
`strategy` (not `policy`) and `cost` (not `llm_cost`); `per_attr_table` rows
have a single point (single static decision, not a sweep). We pick the row with
the smallest |cost - target_budget|, which for per_attr_table is the only row.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import PROCESSED_DIR


def get_headline(pareto_df, target_budget=0.34):
    """Find per_attr_table point with LLM cost closest to target_budget.

    For per_attr_table there is exactly one row (single static decision), so
    the 'closest to target_budget' is trivially that row. We keep the API so
    downstream policies could be plugged in later.
    """
    sub = pareto_df[pareto_df["strategy"] == "per_attr_table"]
    if sub.empty:
        return None, None
    idx = (sub["cost"] - target_budget).abs().idxmin()
    row = sub.loc[idx]
    return float(row["accuracy"]), float(row["cost"])


BACKENDS = ["sonnet45", "gemini25flash", "gpt4o", "llama3b"]

# Baseline: gpt-oss-120b, но мы предпочитаем rebuilt-via-suffix версию,
# если она существует, для apples-to-apples сравнения по test row count
# (cached router_train.parquet имеет 1539 строк / только gpt-oss-join; альтернативные
# LLM имеют ~1502 строки — некоторые строки не пересекаются).
baseline_gptoss = Path(PROCESSED_DIR) / "router_pareto_gold_gptoss.parquet"
baseline_cached = Path(PROCESSED_DIR) / "router_pareto_gold.parquet"
if baseline_gptoss.exists():
    baseline_pareto = pd.read_parquet(baseline_gptoss)
    baseline_src = "router_pareto_gold_gptoss.parquet (rebuilt, n=1502)"
else:
    baseline_pareto = pd.read_parquet(baseline_cached)
    baseline_src = "router_pareto_gold.parquet (cached, n=1539)"
acc_baseline, budget_baseline = get_headline(baseline_pareto)
print(f"Baseline (gpt-oss-120b) [{baseline_src}]: acc={acc_baseline:.4f} @ budget={budget_baseline:.3f}")

results = []
for sfx in BACKENDS:
    p = Path(PROCESSED_DIR) / f"router_pareto_gold_{sfx}.parquet"
    if not p.exists():
        print(f"Missing: {p}")
        continue
    df = pd.read_parquet(p)
    acc, budget = get_headline(df)
    delta = (acc - acc_baseline) if (acc is not None and acc_baseline is not None) else None
    results.append({
        "backend": sfx,
        "accuracy": acc,
        "llm_cost": budget,
        "delta_vs_baseline_pp": (delta * 100) if delta is not None else None,
    })

summary_df = pd.DataFrame(results)
print(summary_df.to_string(index=False))

# Decision rule per E3 pre-registration:
# Δ < +1.5 → no_circularity_headline_valid
# +1.5 ≤ Δ < +3.0 → borderline_open_weight_caveat
# Δ ≥ +3.0 → circularity_confirmed_dual_headline
consensus_members = ["sonnet45", "gemini25flash", "gpt4o"]
control = ["llama3b"]

consensus_deltas = [r["delta_vs_baseline_pp"] for r in results
                    if r["backend"] in consensus_members
                    and r["delta_vs_baseline_pp"] is not None]
control_deltas = [r["delta_vs_baseline_pp"] for r in results
                  if r["backend"] in control
                  and r["delta_vs_baseline_pp"] is not None]

mean_consensus_delta = float(np.mean(consensus_deltas)) if consensus_deltas else None
mean_control_delta = float(np.mean(control_deltas)) if control_deltas else None

if mean_consensus_delta is None:
    decision = "insufficient_data"
elif mean_consensus_delta < 1.5:
    decision = "no_circularity_headline_valid"
elif mean_consensus_delta < 3.0:
    decision = "borderline_open_weight_caveat"
else:
    decision = "circularity_confirmed_dual_headline"

# --- Additional analysis: cost-matched delta ---
# Concern: per_attr_table point picks DIFFERENT LLM costs per backend. Compare at fixed
# LLM cost via interpolation along static_threshold Pareto curve for fair shared-prior eval.

def interp_accuracy_at_cost(pareto_df, target_cost=0.34, policy="static_threshold"):
    """Linearly interpolate accuracy at target LLM cost along given policy curve."""
    sub = pareto_df[pareto_df["strategy"] == policy].sort_values("cost")
    if sub.empty:
        return None
    if target_cost < sub["cost"].min() or target_cost > sub["cost"].max():
        return None  # outside Pareto range
    return float(np.interp(target_cost, sub["cost"], sub["accuracy"]))


print("\n--- Cost-matched analysis (interpolated to LLM cost = 0.34) ---")
TARGET_COST = 0.34
acc_baseline_matched = interp_accuracy_at_cost(baseline_pareto, TARGET_COST, "static_threshold")
print(f"Baseline (gpt-oss-120b) @ cost={TARGET_COST}: acc={acc_baseline_matched:.4f}")

cost_matched_results = []
for sfx in BACKENDS:
    p = Path(PROCESSED_DIR) / f"router_pareto_gold_{sfx}.parquet"
    if not p.exists():
        continue
    df = pd.read_parquet(p)
    acc_matched = interp_accuracy_at_cost(df, TARGET_COST, "static_threshold")
    delta_matched = ((acc_matched - acc_baseline_matched) * 100
                     if (acc_matched is not None and acc_baseline_matched is not None)
                     else None)
    cost_matched_results.append({
        "backend": sfx,
        "acc_at_cost_034": acc_matched,
        "delta_matched_pp": delta_matched,
    })

cm_df = pd.DataFrame(cost_matched_results)
print(cm_df.to_string(index=False))

consensus_matched_deltas = [
    r["delta_matched_pp"] for r in cost_matched_results
    if r["backend"] in consensus_members and r["delta_matched_pp"] is not None
]
control_matched_deltas = [
    r["delta_matched_pp"] for r in cost_matched_results
    if r["backend"] in control and r["delta_matched_pp"] is not None
]

mean_consensus_matched = float(np.mean(consensus_matched_deltas)) if consensus_matched_deltas else None
mean_control_matched = float(np.mean(control_matched_deltas)) if control_matched_deltas else None

print(f"\nmean_consensus_delta_matched_pp = {mean_consensus_matched}")
print(f"mean_control_delta_matched_pp = {mean_control_matched}")
print("(This is the SHARED-PRIOR isolated signal — same LLM cost, different Layer 4 model.)")

# Decision rule on cost-matched delta (same thresholds as headline)
if mean_consensus_matched is None:
    matched_decision = "insufficient_data_cost_matched"
elif mean_consensus_matched < 1.5:
    matched_decision = "no_circularity_at_matched_cost"
elif mean_consensus_matched < 3.0:
    matched_decision = "borderline_at_matched_cost"
else:
    matched_decision = "circularity_at_matched_cost"

# Extend the existing output dict
output = {
    "baseline_backend": "gpt-oss-120b",
    "baseline_source": baseline_src,
    "baseline_acc": acc_baseline,
    "baseline_budget": budget_baseline,
    "per_backend": results,
    "mean_consensus_delta_pp": mean_consensus_delta,
    "mean_control_delta_pp": mean_control_delta,
    "decision": decision,
    "cost_matched_analysis": {
        "target_llm_cost": TARGET_COST,
        "baseline_acc_at_cost": acc_baseline_matched,
        "per_backend_at_matched_cost": cost_matched_results,
        "mean_consensus_delta_matched_pp": mean_consensus_matched,
        "mean_control_delta_matched_pp": mean_control_matched,
        "decision_at_matched_cost": matched_decision,
        "comment": (
            "Cost-matched analysis isolates shared-prior signal from cost-Pareto movement. "
            "If matched-decision == headline-decision, circularity finding is robust. "
            "If matched < headline, the +Δ at default cost was largely Pareto-curve traversal."
        ),
    },
}

out = Path(PROCESSED_DIR) / "e1_circularity_summary.json"
with open(out, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nHeadline decision: {decision}")
print(f"Cost-matched decision: {matched_decision}")
print(f"Summary saved to {out}")
