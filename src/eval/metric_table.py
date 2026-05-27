"""Per-attribute metric table: macro-F1, micro-accuracy, balanced-accuracy + per-class breakdown.

Schema-aware: filters gold cells with deprecated classes (removed from schema after error
analysis 2026-05-25 — see thesis §3.3.X "Schema refactoring based on error analysis").

Usage: python -m src.eval.metric_table

Output: datasets/processed/v4_metric_table.json
"""
import sys, json, os
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

# Allow running from project root
_PROJECT_ROOT_GUESS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT_GUESS))

from scripts.eval_v4_manual import predict_ml, predict_rules, norm_value
from scripts.build_gold_v4_wide import build_inputs_df
from scripts.eval_v4_consensus import _process_struct
from scripts.eval_v4_consensus_clean import is_in_scope, CAT_VALID_TAGS

PROJECT_ROOT = Path(os.environ.get("DIPLOMA_ROOT", str(Path(__file__).resolve().parents[2])))
from sklearn.metrics import (f1_score, balanced_accuracy_score, precision_recall_fscore_support)

# Deprecated classes (post-refactor 2026-05-25). Gold cells with these values filtered before eval.
SCHEMA_EXCLUDE = {
    "chocolate_type": {"filled", "other"},  # filled→is_filled binary; other→unlearnable catch-all
    "chocolate_extra": {"filled", "other", "with_alcohol", "with_coffee"},  # n<10 each, recall=0
}


def process_cat(cat, off_dir, gold_df, gold_field="gold_value"):
    g = gold_df[gold_df.category == cat].copy()
    g["code"] = g["code"].astype(str)
    if cat == "cheeses":
        g[gold_field] = g[gold_field].replace({"semi_soft": "soft"})
    if "disputed" in g.columns and g.disputed.dtype == bool:
        g = g[~g.disputed]
    codes = set(g.code.unique())

    inputs = build_inputs_df(off_dir / f"{cat}_off_full.parquet", codes)
    missing = codes - set(inputs.code.astype(str))
    if missing:
        import duckdb
        miss_sql = ",".join(f"'{c}'" for c in missing)
        con = duckdb.connect()
        extra = con.execute(f"""
            SELECT code, product_name, brands, ingredients_text, quantity,
                   categories_tags, labels_tags, traces_tags, countries_tags,
                   ingredients_analysis_tags, nutriments
            FROM '{off_dir / 'food.parquet'}'
            WHERE CAST(code AS VARCHAR) IN ({miss_sql})
        """).fetchdf()
        if len(extra):
            extra = _process_struct(extra)
            inputs = pd.concat([inputs, extra], ignore_index=True)

    inputs["code"] = inputs["code"].astype(str)
    inputs["in_scope"] = inputs["categories_tags"].apply(
        lambda s: is_in_scope(s, CAT_VALID_TAGS[cat])
    )

    use_prefix = f"{cat}_v4_mpnet_tfidf_noleak"
    ml_preds = predict_ml(inputs, f"{cat}_v4", prefix=use_prefix)
    rule_preds = predict_rules(inputs)
    ml_preds["m_key"] = ml_preds.code.astype(str) + "|" + ml_preds.attr
    rule_preds["m_key"] = rule_preds.code.astype(str) + "|" + rule_preds.attr
    g["m_key"] = g.code + "|" + g.attr
    g = g.merge(inputs[["code", "in_scope"]], on="code", how="left")
    merged = g.merge(ml_preds[["m_key", "ml_pred", "ml_conf", "ml_fired"]], on="m_key", how="left")
    merged = merged.merge(rule_preds[["m_key", "rule_pred", "rule_tier"]], on="m_key", how="left")

    def _cascade(row):
        has_rule = (row["rule_pred"] is not None
                    and not (isinstance(row["rule_pred"], float) and pd.isna(row["rule_pred"])))
        tier = row.get("rule_tier")
        if has_rule and tier == "high":
            return ("rule_h", row["rule_pred"])
        if row["ml_fired"] is True:
            return ("ml", row["ml_pred"])
        if has_rule and tier == "low":
            return ("rule_l", row["rule_pred"])
        return ("fallback", None)
    res = merged.apply(_cascade, axis=1)
    merged["cascade_source"] = [c[0] for c in res]
    merged["cascade_pred"] = [c[1] for c in res]
    merged["gn"] = merged[gold_field].apply(norm_value)
    merged["pn"] = merged["cascade_pred"].apply(norm_value)

    # Apply schema filtering: remove gold cells with deprecated classes
    for attr_name, exclude_set in SCHEMA_EXCLUDE.items():
        cat_attr_mask = (merged.attr == attr_name) & merged.gn.isin(exclude_set)
        if cat_attr_mask.any():
            n_filtered = cat_attr_mask.sum()
            print(f"  [{cat}.{attr_name}] filtering {n_filtered} gold cells with deprecated classes: "
                  f"{merged[cat_attr_mask].gn.value_counts().to_dict()}")
        merged = merged[~cat_attr_mask]

    return merged[merged.in_scope == True], gold_field


def compute_attr_metrics(sub, gold_field):
    sub = sub.copy()
    valid = sub[sub.gn.notna() & sub.pn.notna()]
    if len(valid) == 0:
        return None
    y_true = valid.gn.values
    y_pred = valid.pn.values
    n = len(y_true)
    micro = (y_true == y_pred).mean()
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    balanced = balanced_accuracy_score(y_true, y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    per_class = {}
    for i, lbl in enumerate(labels):
        per_class[lbl] = {
            "n_true": int(((y_true == lbl).sum())),
            "n_pred": int(((y_pred == lbl).sum())),
            "prec": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
        }
    dead = [l for l, v in per_class.items() if v["n_true"] >= 3 and v["recall"] < 0.2]
    return {
        "n": n, "micro_acc": micro, "macro_f1": macro_f1, "weighted_f1": weighted_f1,
        "balanced_acc": balanced, "per_class": per_class,
        "dead_classes": dead,
    }


def main():
    off_dir = Path(os.environ.get("OFF_DATA_ROOT", "/home/miafrolov/off_work"))
    results = {}
    for gold_label, gold_path in [
        ("LLM-consensus", "datasets/processed/manual_gold_consensus.parquet"),
        ("HUMAN", "datasets/processed/manual_eval_per_product.parquet"),
    ]:
        gold = pd.read_parquet(gold_path)
        gold["code"] = gold["code"].astype(str)
        gold_field = "gold_value" if "gold_value" in gold.columns else "manual"
        if "category" not in gold.columns:
            continue
        print(f"\n{'#'*100}")
        print(f"# Gold: {gold_label}  field={gold_field}  total cells={len(gold)}")
        print(f"{'#'*100}")
        attrs_data = {}
        for cat in ["pasta", "chocolate", "cheeses"]:
            merged, gf = process_cat(cat, off_dir, gold, gold_field)
            for attr in sorted(merged.attr.unique()):
                sub = merged[merged.attr == attr]
                m = compute_attr_metrics(sub, gf)
                if m is None: continue
                attrs_data[(cat, attr)] = m

        print(f"\n{'attr':30s}  {'n':>5s}  {'micro':>7s}  {'macro_F1':>8s}  {'bal_acc':>7s}  {'dead':>15s}")
        print("-" * 100)
        for (cat, attr), m in attrs_data.items():
            dead_str = ",".join(m["dead_classes"]) if m["dead_classes"] else "-"
            print(f"{cat+'.'+attr:30s}  {m['n']:>5d}  {m['micro_acc']*100:>6.1f}%  "
                  f"{m['macro_f1']:>8.3f}  {m['balanced_acc']:>7.3f}  {dead_str[:15]:>15s}")

        ns = [m["n"] for m in attrs_data.values()]
        macros = [m["macro_f1"] for m in attrs_data.values()]
        micros = [m["micro_acc"] for m in attrs_data.values()]
        bals = [m["balanced_acc"] for m in attrs_data.values()]
        avg_macro = np.average(macros, weights=ns)
        avg_micro = np.average(micros, weights=ns)
        avg_bal = np.average(bals, weights=ns)
        print(f"{'CELLS-WEIGHTED AVG':30s}  {sum(ns):>5d}  {avg_micro*100:>6.1f}%  "
              f"{avg_macro:>8.3f}  {avg_bal:>7.3f}")
        avg_macro_uw = np.mean(macros)
        print(f"{'ATTR-UNWEIGHTED AVG':30s}    {'':5s}  {'':>7s}  {avg_macro_uw:>8.3f}")

        results[gold_label] = {f"{c}.{a}": m for (c, a), m in attrs_data.items()}

    out = {}
    for gold_label, attrs in results.items():
        out[gold_label] = attrs
    with open("datasets/processed/v4_metric_table_v2.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nFull table saved: datasets/processed/v4_metric_table_v2.json")


if __name__ == "__main__":
    main()
