"""E5: aggregate cold-start results across pasta/chocolate/cheeses, apply decision rule.

Reads `datasets/processed/cold_start_simulation_{cat}.json` for each domain,
applies the decision rule from `docs/thesis/pre_registration_2026-Q2.md`
(E5 boundary fix applied), and writes
`datasets/processed/e5_cold_start_cross_domain_summary.json`.
"""
import json
from pathlib import Path

import pandas as pd

from src.common import PROCESSED_DIR


DOMAINS = ["pasta", "chocolate", "cheeses"]
results = []
for cat in DOMAINS:
    p = Path(PROCESSED_DIR) / f"cold_start_simulation_{cat}.json"
    if not p.exists():
        print(f"Missing: {p}")
        continue
    with open(p) as f:
        data = json.load(f)
    results.append({
        "category": cat,
        "config": data.get("config"),
        "n_products": data.get("n_products") or data.get("n_total"),
        "fill_rate": data.get("fill_rate"),
        "accuracy_on_gold": data.get("accuracy_on_gold"),
    })

df = pd.DataFrame(results)
print(df.to_string(index=False))

# Decision rule per pre_registration_2026-Q2.md (post Task 1 fix):
# - Both >= 0.85 → cross_domain_replicated
# - At least one in [0.70, 0.85) → partial_replication_per_domain
# - Both < 0.70 → pasta_specific_finding
new_acc = [r["accuracy_on_gold"] for r in results
           if r["category"] in ("chocolate", "cheeses")
           and r["accuracy_on_gold"] is not None]

if len(new_acc) < 2:
    decision = "insufficient_data"
elif all(a >= 0.85 for a in new_acc):
    decision = "cross_domain_replicated"
elif any(0.70 <= a < 0.85 for a in new_acc):
    decision = "partial_replication_per_domain"
else:
    decision = "pasta_specific_finding"

summary = {
    "per_domain": results,
    "decision_rule": "from docs/thesis/pre_registration_2026-Q2.md (E5 boundary fix applied)",
    "decision": decision,
}
out = Path(PROCESSED_DIR) / "e5_cold_start_cross_domain_summary.json"
with open(out, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nDecision: {decision}")
print(f"Summary saved: {out}")
