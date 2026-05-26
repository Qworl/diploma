"""End-to-end pipeline evaluation: category router → cascade (rule_h → ML → rule_l → LLM fallback).

For each test cell (code × attribute):
  1. Layer 0 — category router v5: predict product category from partner-text fields
  2. If predicted_cat != true_cat → e2e answer = None (router misrouted, would invoke wrong models)
  3. If predicted_cat == true_cat → cascade prediction is the e2e answer
  4. If predicted_cat == 'ood' → system abstains (None)

Metrics:
  - Router accuracy (per-code)
  - Cascade-only accuracy (router NOT applied)
  - E2E accuracy with None=wrong (production-realistic)
  - E2E coverage (fraction of cells where router routed correctly)

Schema-aware: filters gold cells with deprecated classes (chocolate_type ∈ {filled, other},
chocolate_extra ∈ {filled, other, with_alcohol, with_coffee}) before computing accuracy.

Usage: python -m src.eval.end_to_end

Output: datasets/processed/v4_e2e_router_eval.json
"""
import sys, json, pickle, os
from pathlib import Path
import pandas as pd
import numpy as np

# Allow running from project root
_PROJECT_ROOT_GUESS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT_GUESS))

from scripts.eval_v4_manual import predict_ml, predict_rules, norm_value
from scripts.build_gold_v4_wide import build_inputs_df
from scripts.eval_v4_consensus import _process_struct
from scripts.eval_v4_consensus_clean import is_in_scope, CAT_VALID_TAGS
from src.common import build_text, EMBEDDING_MODEL

PROJECT_ROOT = Path(os.environ.get("DIPLOMA_ROOT", "/home/miafrolov/Desktop/diploma"))


def load_router():
    with open(PROJECT_ROOT / "models/category_router_v5.pkl", "rb") as f:
        clf = pickle.load(f)
    with open(PROJECT_ROOT / "models/category_router_v5_le.pkl", "rb") as f:
        le = pickle.load(f)
    return clf, le


def predict_router(inputs, clf, le):
    """Return predicted category per row (or 'ood')."""
    texts = build_text(inputs)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    X = model.encode(texts, show_progress_bar=False, batch_size=64).astype(np.float32)
    proba = clf.predict_proba(X)
    preds = proba.argmax(axis=1)
    return [str(le.classes_[p]) for p in preds]


def process_cat(cat, off_dir, gold_df, gold_field, router_clf, router_le):
    g = gold_df[gold_df.category == cat].copy()
    g["code"] = g["code"].astype(str)
    if cat == "cheeses":
        g[gold_field] = g[gold_field].replace({"semi_soft": "soft"})
    if "disputed" in g.columns:
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

    # Layer 0 routing
    inputs["router_pred"] = predict_router(inputs, router_clf, router_le)

    # Cascade
    use_prefix = f"{cat}_v4_mpnet_tfidf_noleak"
    ml_preds = predict_ml(inputs, f"{cat}_v4", prefix=use_prefix)
    rule_preds = predict_rules(inputs)
    ml_preds["m_key"] = ml_preds.code.astype(str) + "|" + ml_preds.attr
    rule_preds["m_key"] = rule_preds.code.astype(str) + "|" + rule_preds.attr
    g["m_key"] = g.code + "|" + g.attr
    g = g.merge(inputs[["code", "in_scope", "router_pred"]], on="code", how="left")
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

    # End-to-end: if router_pred != cat → answer is None (would route to wrong cascade)
    def _e2e(row):
        if row["router_pred"] != cat:
            return None  # router misrouted → wrong answer / abstain
        return row["cascade_pred"]
    merged["e2e_pred"] = merged.apply(_e2e, axis=1)
    return merged[merged.in_scope == True]


def main():
    router_clf, router_le = load_router()
    print("Router classes:", router_le.classes_)

    off_dir = Path("/home/miafrolov/off_work")
    results = {}
    for gold_label, gold_path in [
        ("LLM-consensus", "datasets/processed/manual_gold_consensus.parquet"),
        ("HUMAN", "datasets/processed/manual_eval_per_product.parquet"),
    ]:
        gold = pd.read_parquet(gold_path)
        gold["code"] = gold["code"].astype(str)
        gold_field = "gold_value" if "gold_value" in gold.columns else "manual"
        all_rows = []
        for cat in ["pasta", "chocolate", "cheeses"]:
            merged = process_cat(cat, off_dir, gold, gold_field, router_clf, router_le)
            all_rows.append(merged)
        df = pd.concat(all_rows, ignore_index=True)

        # Router accuracy
        df["router_correct"] = df["router_pred"] == df["category"]
        router_acc = df.drop_duplicates("code").router_correct.mean()
        router_n = df.drop_duplicates("code").shape[0]
        # Router per-cat
        per_cat_router = df.drop_duplicates("code").groupby("category").router_correct.agg(["mean", "count"])

        # Cascade-only acc
        df["gn"] = df[gold_field].apply(norm_value)
        df["cn"] = df["cascade_pred"].apply(norm_value)
        df["e2e_n"] = df["e2e_pred"].apply(norm_value)
        # Filter deprecated schema classes so they don't count as errors
        SCHEMA_EXCLUDE = {
            "chocolate_type": {"filled", "other"},
            "chocolate_extra": {"filled", "other", "with_alcohol", "with_coffee"},
        }
        schema_mask = pd.Series(True, index=df.index)
        for attr_name, excl_set in SCHEMA_EXCLUDE.items():
            schema_mask &= ~((df.attr == attr_name) & df.gn.isin(excl_set))
        df = df[schema_mask]
        valid_casc = df.gn.notna() & df.cn.notna()
        casc_acc = (df.gn == df.cn)[valid_casc].mean()
        # E2E (only counts cells where e2e_pred not None)
        valid_e2e = df.gn.notna() & df.e2e_n.notna()
        e2e_acc_strict = (df.gn == df.e2e_n)[valid_e2e].mean()
        # E2E coverage: how many cells did E2E answer
        coverage_e2e = valid_e2e.sum() / valid_casc.sum()
        # E2E acc on ALL valid cells (treating None as wrong)
        e2e_all = (df.gn == df.e2e_n) & valid_casc
        e2e_acc_strict_all = e2e_all.sum() / valid_casc.sum()

        print(f"\n{'='*80}")
        print(f"Gold: {gold_label}")
        print(f"{'='*80}")
        print(f"Router accuracy: {router_acc*100:.1f}% on {router_n} codes")
        print("Per-cat:")
        print(per_cat_router)
        print(f"\nCascade-only acc (router NOT applied): {casc_acc*100:.1f}% (n={valid_casc.sum()})")
        print(f"E2E coverage: {coverage_e2e*100:.1f}% (cells where router routed correctly)")
        print(f"E2E acc | router-correct: {e2e_acc_strict*100:.1f}%")
        print(f"E2E acc | None=wrong:    {e2e_acc_strict_all*100:.1f}%  ← production-realistic")
        # Per-layer cell counts (для §14.4 LLM fallback rate и
        # notebooks/03_evaluate.ipynb cell «layer contribution»).
        # Ключи: rule_h, ml, rule_l, fallback. Источник — cascade_source
        # из _cascade() функции выше. Считается ПОСЛЕ schema-фильтра,
        # но БЕЗ valid_casc-фильтра — иначе fallback-ячейки (где
        # cascade_pred=None) выпадают, и сумма не сходится с n_total.
        # Это полное распределение «куда уходит каждая gold-ячейка»,
        # совпадает с output scripts/llm_fallback_rate.py.
        layer_counts_raw = df.cascade_source.value_counts().to_dict()
        per_layer_counts = {
            k: int(layer_counts_raw.get(k, 0))
            for k in ("rule_h", "ml", "rule_l", "fallback")
        }
        n_for_layer = sum(per_layer_counts.values())
        per_layer_pct = {
            k: round(v / max(n_for_layer, 1) * 100, 2)
            for k, v in per_layer_counts.items()
        }
        results[gold_label] = {
            "router_acc": float(router_acc),
            "cascade_only_acc": float(casc_acc),
            "e2e_coverage": float(coverage_e2e),
            "e2e_acc_conditional": float(e2e_acc_strict),
            "e2e_acc_none_as_wrong": float(e2e_acc_strict_all),
            "n_valid_cells": int(valid_casc.sum()),
            "per_layer_counts": per_layer_counts,
            "per_layer_pct": per_layer_pct,
        }

    with open("datasets/processed/v4_e2e_router_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: datasets/processed/v4_e2e_router_eval.json")


if __name__ == "__main__":
    main()
