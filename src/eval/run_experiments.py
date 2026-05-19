"""
Comparative experiments: Regex vs ML vs Bayes vs LLM.

Measures per-attribute accuracy against silver standard ground truth.

Usage:
    python scripts/run_experiments.py --category pasta
    python scripts/run_experiments.py --category pasta --skip-llm
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from src.common import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    setup_logging,
)
from src.pipeline.schemas import (
    PASTA_SCHEMA, CHOCOLATE_SCHEMA, BEVERAGE_SCHEMA,
    COSMETICS_SCHEMA, CHEESES_SCHEMA, CEREALS_SCHEMA
)
from src.pipeline.regex.extractor import RegexExtractor

logger = logging.getLogger(__name__)

CATEGORY_CONFIG = {
    "pasta": {
        "silver_standard": "pasta_silver_standard.parquet",
        "embeddings_cache": "pasta_embeddings.npy",
        "bayesian_model": "pasta_bayesian.pkl",
        "schema": PASTA_SCHEMA,
        "ml_attrs": [
            "grain_type", "pasta_shape", "is_filled", "is_organic",
            "is_gluten_free", "is_vegan", "nutri_score_grade", "protein_class",
        ],
        "bayes_targets": [
            "grain_type", "pasta_shape", "is_filled", "is_organic",
            "is_gluten_free", "is_vegan", "nutri_score_grade", "protein_class",
        ],
        "regex_category": "pasta",
    },
    "pasta_stratified": {
        "silver_standard": "pasta_stratified_silver_standard.parquet",
        "embeddings_cache": "pasta_stratified_embeddings.npy",
        "bayesian_model": "pasta_stratified_bayesian.pkl",
        "schema": PASTA_SCHEMA,
        "ml_attrs": [
            "grain_type", "pasta_shape", "is_filled", "is_organic",
            "is_gluten_free", "is_vegan", "nutri_score_grade", "protein_class",
        ],
        "bayes_targets": [
            "grain_type", "pasta_shape", "is_filled", "is_organic",
            "is_gluten_free", "is_vegan", "nutri_score_grade", "protein_class",
        ],
        "regex_category": "pasta",
    },
    "chocolate": {
        "silver_standard": "chocolate_silver_standard.parquet",
        "embeddings_cache": "chocolate_embeddings.npy",
        "bayesian_model": "chocolate_bayesian.pkl",
        "schema": CHOCOLATE_SCHEMA,
        "ml_attrs": [
            "chocolate_type", "cocoa_percentage", "contains_nuts",
            "chocolate_extra", "is_organic", "nutri_score_grade",
            "protein_class",
        ],
        "bayes_targets": [
            "chocolate_type", "cocoa_percentage", "contains_nuts",
            "chocolate_extra", "is_organic", "nutri_score_grade",
            "protein_class",
        ],
        "regex_category": "chocolate",
    },
    "chocolate_stratified": {
        "silver_standard": "chocolate_stratified_silver_standard.parquet",
        "embeddings_cache": "chocolate_stratified_embeddings.npy",
        "bayesian_model": "chocolate_stratified_bayesian.pkl",
        "schema": CHOCOLATE_SCHEMA,
        "ml_attrs": [
            "chocolate_type", "cocoa_percentage", "contains_nuts",
            "chocolate_extra", "is_organic", "nutri_score_grade",
            "protein_class",
        ],
        "bayes_targets": [
            "chocolate_type", "cocoa_percentage", "contains_nuts",
            "chocolate_extra", "is_organic", "nutri_score_grade",
            "protein_class",
        ],
        "regex_category": "chocolate",
    },
    "beverages": {
        "silver_standard": "beverages_silver_standard.parquet",
        "embeddings_cache": "beverages_embeddings.npy",
        "bayesian_model": "beverages_bayesian.pkl",
        "schema": BEVERAGE_SCHEMA,
        "ml_attrs": [
            "beverage_type", "sugar_class", "is_organic", "is_carbonated",
            "nutri_score_grade", "nova_group", "protein_class",
        ],
        "bayes_targets": [
            "beverage_type", "sugar_class", "is_organic", "is_carbonated",
            "nutri_score_grade", "nova_group", "protein_class",
        ],
        "regex_category": "beverages",
    },
    "beverages_stratified": {
        "silver_standard": "beverages_stratified_silver_standard.parquet",
        "embeddings_cache": "beverages_stratified_embeddings.npy",
        "bayesian_model": "beverages_stratified_bayesian.pkl",
        "schema": BEVERAGE_SCHEMA,
        "ml_attrs": [
            "beverage_type", "sugar_class", "is_organic", "is_carbonated",
            "nutri_score_grade", "nova_group", "protein_class", "is_vegan",
        ],
        "bayes_targets": [
            "beverage_type", "sugar_class", "is_organic", "is_carbonated",
            "nutri_score_grade", "nova_group", "protein_class", "is_vegan",
        ],
        "regex_category": "beverages",
    },
    # baby_stratified removed 2026-05-11 — see train_classifiers.py для обоснования.
    "cosmetics_stratified": {
        "silver_standard": "cosmetics_stratified_silver_standard.parquet",
        "embeddings_cache": "cosmetics_stratified_embeddings.npy",
        "bayesian_model": "cosmetics_stratified_bayesian.pkl",
        "schema": COSMETICS_SCHEMA,
        "ml_attrs": [
            "product_type", "form_factor", "body_area",
            "has_sulfates", "has_silicones", "is_organic",
        ],
        "bayes_targets": [
            "product_type", "form_factor", "body_area",
            "has_sulfates", "has_silicones", "is_organic",
        ],
        # regex_category=None — для cosmetics regex_extractor пока не настроен,
        # используем только ML+Bayes layers; regex_layer вернёт {}.
        "regex_category": "cosmetics",
    },
    "cheeses_stratified": {
        "silver_standard": "cheeses_stratified_silver_standard.parquet",
        "embeddings_cache": "cheeses_stratified_embeddings.npy",
        "bayesian_model": "cheeses_stratified_bayesian.pkl",
        "schema": CHEESES_SCHEMA,
        "ml_attrs": [
            "milk_source", "texture", "country_of_origin", "fat_class",
            "is_pdo", "is_organic", "is_ultra_processed",
        ],
        "bayes_targets": [
            "milk_source", "texture", "country_of_origin", "fat_class",
            "is_pdo", "is_organic", "is_ultra_processed",
        ],
        "regex_category": "cheeses",
    },
    "cereals_stratified": {
        "silver_standard": "cereals_stratified_silver_standard.parquet",
        "embeddings_cache": "cereals_stratified_embeddings.npy",
        "bayesian_model": "cereals_stratified_bayesian.pkl",
        "schema": CEREALS_SCHEMA,
        "ml_attrs": [
            "cereal_type", "grain_type", "is_low_sugar", "is_high_fibre",
            "nova_class", "is_vegan", "is_whole_grain", "is_organic",
        ],
        "bayes_targets": [
            "cereal_type", "grain_type", "is_low_sugar", "is_high_fibre",
            "nova_class", "is_vegan", "is_whole_grain", "is_organic",
        ],
        "regex_category": "cereals",
    },
}


def load_thresholds(category: str) -> dict:
    path = os.path.join(MODELS_DIR, f"{category}_thresholds.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def load_ml_models(category: str, attrs: list[str]) -> dict:
    models = {}
    for attr in attrs:
        xgb_path = os.path.join(MODELS_DIR, f"{category}_{attr}_xgb.pkl")
        le_path = os.path.join(MODELS_DIR, f"{category}_{attr}_le.pkl")
        if os.path.exists(xgb_path):
            with open(xgb_path, "rb") as f:
                models[f"{attr}_xgb"] = pickle.load(f)
            if os.path.exists(le_path):
                with open(le_path, "rb") as f:
                    models[f"{attr}_le"] = pickle.load(f)
    return models


def load_bayesian(category: str):
    path = os.path.join(MODELS_DIR, f"{category}_bayesian.pkl")
    if not os.path.exists(path):
        return None, None
    with open(path, "rb") as f:
        model = pickle.load(f)
    from pgmpy.inference import VariableElimination
    return model, VariableElimination(model)


def regex_layer(row, rx: RegexExtractor, category: str) -> dict:
    name = str(row.get("product_name", ""))
    desc = str(row.get("generic_name", ""))
    qty = str(row.get("quantity", ""))
    results = rx.extract_all(name, desc, qty, category=category)
    return {k: (v.value, v.confidence) for k, v in results.items() if v.value is not None}


def ml_layer(row, models: dict, embeddings: np.ndarray, idx: int,
             attrs: list[str], thresholds: dict | None = None) -> dict:
    predictions = {}
    X = embeddings[idx:idx+1]
    for attr in attrs:
        xgb_key = f"{attr}_xgb"
        le_key = f"{attr}_le"
        if xgb_key not in models:
            continue
        clf = models[xgb_key]
        proba = clf.predict_proba(X)[0]
        max_idx = proba.argmax()
        confidence = float(proba[max_idx])
        threshold = (thresholds or {}).get(attr, DEFAULT_CONFIDENCE_THRESHOLD)
        if confidence < threshold:
            continue
        if le_key in models:
            value = models[le_key].inverse_transform([max_idx])[0]
        else:
            value = bool(max_idx)
        predictions[attr] = (value, confidence)
    return predictions


def bayes_layer(row, bayes_model, inference, targets: list[str],
                ml_predictions: dict | None = None,
                thresholds: dict | None = None) -> dict:
    """Bayesian inference using brand + ML predictions as evidence."""
    if not bayes_model or not inference:
        return {}
    predictions = {}
    evidence = {}
    model_nodes = set(bayes_model.nodes())

    if "brand" in model_nodes:
        brand_val = str(row.get("brands", "other"))
        cpd = bayes_model.get_cpds("brand")
        known = list(cpd.state_names["brand"])
        evidence["brand"] = brand_val if brand_val in known else "other"

    # Use ALL confident ML predictions as evidence (inter-attribute)
    if ml_predictions:
        for attr, (val, conf, _layer) in ml_predictions.items():
            if attr in model_nodes and conf >= 0.6:
                cpd = bayes_model.get_cpds(attr)
                known = list(cpd.state_names[attr])
                str_val = str(val)
                if str_val in known:
                    evidence[attr] = str_val

    for target in targets:
        if target in evidence:
            continue
        if target not in model_nodes:
            continue
        try:
            result = inference.query([target], evidence=evidence)
            probs = {str(s): float(result.values[i]) for i, s in enumerate(result.state_names[target])}
            best = max(probs, key=probs.get)
            conf = probs[best]
            threshold = (thresholds or {}).get(target, DEFAULT_CONFIDENCE_THRESHOLD)
            if conf >= threshold:
                predictions[target] = (best, conf)
        except Exception:
            pass
    return predictions


def run_pipeline(test_df, config_name, category, rx, ml_models, embeddings,
                 bayes_model, bayes_inference, test_indices, schema,
                 skip_llm=False, thresholds=None):
    cat_config = CATEGORY_CONFIG[category]
    results = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        extracted = {}
        emb_idx = test_indices[i]
        # OFF-tags-first configs ставят apply_off_labels первым приоритетом (~100% accuracy
        # там, где OFF знает) — остальные слои добивают непокрытые атрибуты.
        if config_name in ("off_tags_only", "off_ml", "off_ml_bayes", "full_hybrid"):
            from src.pipeline.off_labels import apply_off_labels
            off_attrs = apply_off_labels(row.to_dict(), schema)
            for attr, val in off_attrs.items():
                if val is None:
                    continue
                extracted[attr] = (val, 1.0, "off_tags")
        if config_name in ("regex_only", "regex_ml", "regex_ml_bayes",
                           "off_ml", "off_ml_bayes", "full_hybrid"):
            for attr, (val, conf) in regex_layer(row, rx, cat_config["regex_category"]).items():
                if attr not in extracted:
                    extracted[attr] = (val, conf, "regex")
        # TYPE_F regex на partner-only полях (product_type/form_factor по словам в
        # product_name + ingredients_text). В off_* configs уже отрабатывается через
        # apply_off_labels выше, тут только дополняем regex-side configs.
        if config_name in ("regex_only", "regex_ml", "regex_ml_bayes"):
            from src.pipeline.off_labels import apply_partner_type_f
            for attr, val in apply_partner_type_f(row.to_dict(), schema).items():
                if attr not in extracted:
                    extracted[attr] = (val, 1.0, "regex")
        if config_name in ("regex_ml", "regex_ml_bayes",
                           "off_ml", "off_ml_bayes", "full_hybrid"):
            for attr, (val, conf) in ml_layer(row, ml_models, embeddings, emb_idx,
                                               cat_config["ml_attrs"], thresholds).items():
                if attr not in extracted:
                    extracted[attr] = (val, conf, "ml")
        if config_name in ("regex_ml_bayes", "off_ml_bayes", "full_hybrid"):
            for attr, (val, conf) in bayes_layer(row, bayes_model, bayes_inference,
                                                  cat_config["bayes_targets"],
                                                  ml_predictions=extracted,
                                                  thresholds=thresholds).items():
                if attr not in extracted:
                    extracted[attr] = (val, conf, "bayes")
        if config_name in ("llm_only", "full_hybrid") and not skip_llm:
            all_attrs = set(cat_config["ml_attrs"])
            missing = all_attrs - set(extracted.keys())
            if config_name == "llm_only":
                missing = all_attrs
            if missing:
                from src.pipeline.llm_fallback import enrich_product
                llm_result = enrich_product(row.to_dict(), schema, backend="ollama", model="qwen2.5:7b")
                for attr, val in llm_result.items():
                    if attr in missing:
                        extracted[attr] = (val, 1.0, "llm")
        all_attrs = set(cat_config["ml_attrs"])
        covered = set(extracted.keys()) & all_attrs
        needs_llm_attrs = all_attrs - covered
        result_row = {"code": row.get("code"), "config": config_name}
        for attr in all_attrs:
            if attr in extracted:
                val, conf, layer = extracted[attr]
                result_row[f"{attr}_pred"] = val
                result_row[f"{attr}_conf"] = conf
                result_row[f"{attr}_layer"] = layer
            else:
                result_row[f"{attr}_pred"] = None
                result_row[f"{attr}_conf"] = 0.0
                result_row[f"{attr}_layer"] = "none"
        result_row["n_extracted"] = len(covered)
        result_row["n_needs_llm"] = len(needs_llm_attrs)
        result_row["needs_llm_attrs"] = ",".join(sorted(needs_llm_attrs))
        results.append(result_row)
    return pd.DataFrame(results)


def _normalize_for_compare(v):
    """Унифицирует формат для строкового сравнения pred vs gt.

    Float целые числа: 4.0 → '4'. Int: 4 → '4'. Str '4'/'4.0'/'4.000' → '4'.
    Bool/None/NaN остаются как есть. Это фикс для случая, когда разные слои
    возвращают разные типы (int от _type_d_direct, str от Bayes state names,
    float от silver storage), и astype(str) даёт несовместимые '4' vs '4.0'.
    """
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    if isinstance(v, bool):
        return str(v)  # 'True' / 'False'
    if isinstance(v, (int,)):
        return str(int(v))
    if isinstance(v, float):
        # 4.0 → '4', 70.5 → '70.5'
        return str(int(v)) if v.is_integer() else str(v)
    s = str(v).strip()
    # Try numeric coerce — if it's an integer-ish string, drop trailing .0
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except ValueError:
        return s


def evaluate(result_df, ground_truth, attrs, config_name):
    """Логирует результаты И возвращает per-attr rows для последующей сериализации."""
    logger.info("=" * 60)
    logger.info("  %s", config_name)
    logger.info("=" * 60)
    gt = ground_truth.set_index("code")
    pred = result_df.set_index("code")
    rows = []
    for attr in attrs:
        pred_col = f"{attr}_pred"
        if pred_col not in pred.columns or attr not in gt.columns:
            continue
        joined = pred[[pred_col]].join(gt[[attr]], how="inner").dropna()
        if len(joined) == 0:
            logger.info("  %s: no overlapping data", attr)
            rows.append({"config": config_name, "attr": attr, "n": 0,
                         "accuracy": None, "macro_f1": None, "coverage": 0.0})
            continue
        # Нормализуем оба к одному строковому формату — иначе int(4) vs float(4.0)
        # vs str('4') дают astype(str) → '4'/'4.0'/'4', и acc=0% даже когда всё верно.
        y_true = joined[attr].map(_normalize_for_compare).astype(str)
        y_pred = joined[pred_col].map(_normalize_for_compare).astype(str)
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        covered = (pred[pred_col].notna()).mean()
        # Layer breakdown: какой слой выдал предсказание для этого атрибута
        layer_col = f"{attr}_layer"
        layer_counts = {}
        layer_accuracy = {}  # per-layer accuracy на тех товарах, где этот слой стрелял
        if layer_col in pred.columns:
            layer_counts = pred[layer_col].value_counts().to_dict()
            joined_layers = pred[[pred_col, layer_col]].join(gt[[attr]], how="inner").dropna(subset=[attr])
            for layer in [l for l in layer_counts.keys() if l != "none"]:
                mask = joined_layers[layer_col] == layer
                if mask.sum() == 0:
                    continue
                lt = joined_layers.loc[mask, attr].map(_normalize_for_compare).astype(str)
                lp = joined_layers.loc[mask, pred_col].map(_normalize_for_compare).astype(str)
                layer_accuracy[layer] = {
                    "n": int(mask.sum()),
                    "accuracy": float((lt == lp).mean()),
                }
        rows.append({
            "config": config_name, "attr": attr,
            "n": int(len(joined)),
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "coverage": float(covered),
            "n_correct": int((y_true == y_pred).sum()),
            "layer_counts": json.dumps(layer_counts),
            "layer_accuracy": json.dumps(layer_accuracy),
        })
        logger.info("  %s: accuracy=%.3f, macro_f1=%.3f, coverage=%.1f%% (%d products)",
                     attr, acc, macro_f1, covered * 100, len(joined))
        if y_true.nunique() > 2:
            report = classification_report(y_true, y_pred, zero_division=0)
            for line in report.strip().split("\n"):
                logger.info("    %s", line)
    llm_rate = result_df["n_needs_llm"].mean() / len(attrs) if len(attrs) > 0 else 0
    avg_extracted = result_df["n_extracted"].mean()
    logger.info("  Avg attrs extracted: %.1f/%d", avg_extracted, len(attrs))
    logger.info("  Per-attribute LLM fallback rate: %.1f%%", llm_rate * 100)
    return rows


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, choices=list(CATEGORY_CONFIG.keys()))
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM calls")
    args = parser.parse_args()
    cat_config = CATEGORY_CONFIG[args.category]

    ss_path = os.path.join(PROCESSED_DIR, cat_config["silver_standard"])
    ss_df = pd.read_parquet(ss_path)
    logger.info("Silver standard: %d products", len(ss_df))

    if "fat_100g" in ss_df.columns:
        fat = pd.to_numeric(ss_df["fat_100g"], errors="coerce")
        ss_df["fat_bin"] = pd.cut(fat, bins=[0,1,2.5,5,10,100],
                                  labels=["very_low","low","medium","high","very_high"],
                                  include_lowest=True).astype(str)
    if "sugars_100g" in ss_df.columns:
        sugar = pd.to_numeric(ss_df["sugars_100g"], errors="coerce").fillna(0)
        ss_df["has_sugar"] = (sugar > 0.5).map({True: "yes", False: "no"})

    # Global split — aligned with train_classifiers.py
    _, test_idx = train_test_split(
        np.arange(len(ss_df)), test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    test_df = ss_df.iloc[test_idx]
    logger.info("Test set: %d products (aligned with training split)", len(test_df))

    emb_path = os.path.join(PROCESSED_DIR, cat_config["embeddings_cache"])
    if os.path.exists(emb_path):
        all_embeddings = np.load(emb_path)
        test_emb_indices = test_idx
    else:
        logger.warning("Embeddings not found at %s, ML layer will be skipped", emb_path)
        all_embeddings = np.zeros((len(ss_df), 384))
        test_emb_indices = np.arange(len(test_df))

    rx = RegexExtractor()
    ml_models = load_ml_models(args.category, cat_config["ml_attrs"])
    bayes_model, bayes_inference = load_bayesian(args.category)
    thresholds = load_thresholds(args.category)
    logger.info("ML models loaded: %s", list(ml_models.keys()))
    logger.info("Bayesian model: %s", "loaded" if bayes_model else "not found")
    logger.info("Thresholds: %s", thresholds or "default (0.7)")

    # Honest configs only — off_* убраны как методологически циркулярные
    # (silver построен из тех же OFF тегов, что использует apply_off_labels).
    configs = ["regex_only", "regex_ml", "regex_ml_bayes"]
    if not args.skip_llm:
        configs = ["llm_only"] + configs + ["full_hybrid"]

    results_cache = {}
    per_attr_rows = []
    for config_name in configs:
        start = time.time()
        result_df = run_pipeline(
            test_df, config_name, args.category,
            rx, ml_models, all_embeddings, bayes_model, bayes_inference,
            test_emb_indices, cat_config["schema"], skip_llm=args.skip_llm,
            thresholds=thresholds,
        )
        elapsed = time.time() - start
        ms_per = (elapsed / len(test_df)) * 1000
        results_cache[config_name] = {"df": result_df, "elapsed": elapsed, "ms_per": ms_per}
        attr_rows = evaluate(result_df, test_df, cat_config["ml_attrs"], config_name)
        per_attr_rows.extend(attr_rows)
        logger.info("  Time: %.2fs (%.1f ms/product)", elapsed, ms_per)

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("%-20s %14s %14s %12s", "Config", "LLM fallback%", "Attrs/product", "ms/product")
    logger.info("-" * 62)
    summary_rows = []
    for config_name, data in results_cache.items():
        rdf = data["df"]
        llm_rate = rdf["n_needs_llm"].mean() / len(cat_config["ml_attrs"])
        avg_ext = rdf["n_extracted"].mean()
        logger.info("%-20s %13.1f%% %13.1f %11.1f", config_name, llm_rate * 100, avg_ext, data["ms_per"])
        summary_rows.append({
            "category": args.category,
            "config": config_name,
            "llm_fallback_pct": float(llm_rate * 100),
            "attrs_per_product": float(avg_ext),
            "ms_per_product": float(data["ms_per"]),
            "n_test": int(len(test_df)),
            "elapsed_s": float(data["elapsed"]),
        })

    # Структурированный output для notebook'а — устраняет fragility log-парсинга.
    summary_path = os.path.join(PROCESSED_DIR, f"experiment_summary_{args.category}.parquet")
    per_attr_path = os.path.join(PROCESSED_DIR, f"experiment_per_attr_{args.category}.parquet")
    pd.DataFrame(summary_rows).to_parquet(summary_path, index=False)
    pd.DataFrame(per_attr_rows).to_parquet(per_attr_path, index=False)
    logger.info("Saved summary  → %s", summary_path)
    logger.info("Saved per-attr → %s", per_attr_path)

    # Per-product output (long format) — нужен для notebook'а: confusion matrices,
    # Bayes-vs-ML head-to-head, оценка LLM-fallback на хвосте конвейера.
    # Сохраняем строки сразу для двух конфигураций: off_ml_bayes (oracle) и
    # regex_ml_bayes (основная метрика без OFF). Из последней потом отбирается
    # хвост `layer='none'` для прогона Layer 4.
    long_rows = []
    gt_indexed = test_df.set_index("code")
    for cfg in ("off_ml_bayes", "regex_ml_bayes"):
        if cfg not in results_cache:
            continue
        rdf = results_cache[cfg]["df"]
        for _, r in rdf.iterrows():
            code = r["code"]
            for attr in cat_config["ml_attrs"]:
                if attr not in gt_indexed.columns:
                    continue
                gt_val = gt_indexed.loc[code, attr] if code in gt_indexed.index else None
                pred = r.get(f"{attr}_pred")
                conf = r.get(f"{attr}_conf")
                layer = r.get(f"{attr}_layer", "none")
                long_rows.append({
                    "config": cfg,
                    "code": str(code),
                    "attr": attr,
                    "gt": _normalize_for_compare(gt_val),
                    "pred": _normalize_for_compare(pred),
                    "conf": float(conf) if pd.notna(conf) else 0.0,
                    "layer": str(layer),
                })
    if long_rows:
        per_product_path = os.path.join(
            PROCESSED_DIR, f"experiment_per_product_{args.category}.parquet"
        )
        pd.DataFrame(long_rows).to_parquet(per_product_path, index=False)
        logger.info("Saved per-product → %s (configs=%s)",
                     per_product_path,
                     sorted({r['config'] for r in long_rows}))


if __name__ == "__main__":
    main()
