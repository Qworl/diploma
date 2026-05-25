"""Dump per-disagreement detail for weak v4 attrs."""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

for root in ['/home/miafrolov/Desktop/diploma',
             '/Users/miafrolov/Desktop/stuff/ai_attributes']:
    if Path(root).exists():
        sys.path.insert(0, root)
        PROJECT_ROOT = Path(root)
        break

from scripts.eval_v4_manual import predict_ml, predict_rules, norm_value
from scripts.build_gold_v4_wide import build_inputs_df


PROBLEM = {
    "chocolate": ["chocolate_extra", "contains_nuts", "chocolate_type", "protein_class"],
    "cheeses":   ["texture", "country_of_origin"],
}


def main():
    manual = pd.read_parquet(PROJECT_ROOT / "datasets/processed/manual_eval_per_product.parquet")
    manual["code"] = manual["code"].astype(str)
    off_dir = Path.home() / "off_work"
    if not off_dir.exists():
        off_dir = PROJECT_ROOT / "datasets/raw"

    for cat, attrs in PROBLEM.items():
        g = manual[manual.category == cat].copy()
        if cat == "cheeses":
            g["manual"] = g["manual"].replace({"semi_soft": "soft"})
        codes = set(g.code.unique())
        inputs = build_inputs_df(off_dir / f"{cat}_off_full.parquet", codes)
        rule_preds = predict_rules(inputs) if "protein_class" in attrs else pd.DataFrame()
        if not rule_preds.empty:
            rule_preds["m_key"] = rule_preds.code.astype(str) + "|" + rule_preds.attr
        ml_preds = predict_ml(inputs, f"{cat}_v4", prefix=f"{cat}_v4")
        ml_preds["m_key"] = ml_preds.code.astype(str) + "|" + ml_preds.attr
        g["m_key"] = g.code + "|" + g.attr
        for attr in attrs:
            print(f"\n### {cat}.{attr} ###")
            sub = g[g.attr == attr].copy()
            if attr == "protein_class":
                sub = sub.merge(
                    rule_preds[["m_key", "rule_pred"]].rename(columns={"rule_pred": "pred"}),
                    on="m_key", how="left",
                )
                sub["src"] = "rule"
                sub["ml_conf"] = float("nan")
            else:
                sub = sub.merge(
                    ml_preds[ml_preds.attr == attr][["m_key", "ml_pred", "ml_conf", "ml_fired"]],
                    on="m_key", how="left",
                )
                sub["pred"] = sub["ml_pred"]
                sub["src"] = "ml"
            sub["manual_n"] = sub["manual"].apply(norm_value)
            sub["pred_n"] = sub["pred"].apply(norm_value)
            valid = sub.dropna(subset=["manual_n", "pred_n"])
            wrong = valid[valid.manual_n != valid.pred_n]
            print(f"  n_eval={len(valid)} wrong={len(wrong)}")
            from collections import Counter
            conf = Counter()
            for _, r in wrong.iterrows():
                conf[f"{r.manual_n} -> {r.pred_n}"] += 1
            for pair, n in conf.most_common(10):
                print(f"    [{n:2d}] {pair}")
            print("  examples:")
            for _, r in wrong.head(15).iterrows():
                pn_row = inputs[inputs.code.astype(str) == r["code"]]
                pn = pn_row["product_name"].iloc[0] if len(pn_row) else "?"
                ing = pn_row["ingredients_text"].iloc[0] if len(pn_row) else ""
                conf_str = f" conf={r['ml_conf']:.2f}" if not pd.isna(r.get('ml_conf', float('nan'))) else ""
                print(f"    [{str(pn)[:45]:45s}]  manual={r.manual_n:14s} pred={r.pred_n:14s}{conf_str}")
                if cat == "chocolate" and attr == "contains_nuts" and ing:
                    print(f"        ing: {str(ing)[:120]}")


if __name__ == "__main__":
    main()
