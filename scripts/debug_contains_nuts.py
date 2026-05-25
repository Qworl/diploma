"""Debug contains_nuts rule predictions vs manual."""
import sys
from pathlib import Path

import pandas as pd

for root in ['/home/miafrolov/Desktop/diploma',
             '/Users/miafrolov/Desktop/stuff/ai_attributes']:
    if Path(root).exists():
        sys.path.insert(0, root)
        PROJECT_ROOT = Path(root)
        break

from src.pipeline.off_labels.rules import _type_e_regex
from scripts.build_gold_v4_wide import build_inputs_df
from scripts.eval_v4_manual import norm_value

manual = pd.read_parquet(PROJECT_ROOT / "datasets/processed/manual_eval_per_product.parquet")
manual["code"] = manual["code"].astype(str)
g = manual[(manual.category == "chocolate") & (manual.attr == "contains_nuts")].copy()
codes = set(g.code.unique())

off_dir = Path.home() / "off_work"
if not off_dir.exists():
    off_dir = PROJECT_ROOT / "datasets/raw"
inputs = build_inputs_df(off_dir / "chocolate_off_full.parquet", codes)
inputs["code"] = inputs["code"].astype(str)

merged = g.merge(inputs, on="code", how="left")
rows = []
for _, r in merged.iterrows():
    rule_val = _type_e_regex(r.to_dict(), "contains_nuts")
    manual_n = norm_value(r["manual"])
    rule_n = norm_value(str(rule_val) if rule_val is not None else None)
    correct = "✓" if (rule_n == manual_n) else "✗"
    pn = r["product_name"] if isinstance(r["product_name"], str) else ""
    ing = r["ingredients_text"] if isinstance(r["ingredients_text"], str) else ""
    rows.append({
        "code": r["code"], "product_name": pn[:45],
        "ing": ing[:90],
        "traces_tags": str(r.get("traces_tags", ""))[:50],
        "manual": manual_n, "rule": rule_n, "ok": correct,
    })

df = pd.DataFrame(rows)
print(f"Total: {len(df)}")
print(f"Correct: {(df.ok == '✓').sum()}")
print(f"\n=== WRONG cases ===")
wrong = df[df.ok == '✗']
for _, r in wrong.iterrows():
    print(f"  [{r.product_name:45s}] manual={r.manual} rule={r.rule}")
    print(f"        ing: {r.ing}")
    if r.traces_tags:
        print(f"        traces_tags: {r.traces_tags}")
