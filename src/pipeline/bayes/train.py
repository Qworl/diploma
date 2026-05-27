"""
Bayesian Network (Layer 3 of hybrid system).

Structure learned automatically from data via Hill Climb Search (BIC score).
Uses ML predictions (Layer 2) as evidence for inter-attribute inference.

Usage:
    python -m src.pipeline.bayes.train --category pasta
"""

import argparse
import logging
import os
import pickle
import random
import re
import warnings

import numpy as np
import pandas as pd

BAYES_TRAIN_SEED = 42

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator, HillClimbSearch
from pgmpy.inference import VariableElimination

from src.common import DEFAULT_CONFIDENCE_THRESHOLD, MODELS_DIR, PROCESSED_DIR, setup_logging

logger = logging.getLogger(__name__)


def extract_top(series: pd.Series, top_n: int, fill: str = "unknown") -> pd.Series:
    """Keep top N values, replace rest with 'other'."""
    s = series.fillna(fill)
    top = s.value_counts().head(top_n).index
    return s.where(s.isin(top), other="other")


_ORGANIC_MARKER_RE = re.compile(r"\b(bio|organic|organique|eco|ecol[oó]gico|ekol|öko)\b", re.IGNORECASE)


def normalize_brand(series: pd.Series) -> pd.Series:
    """Split OFF brand strings on comma, take first, lowercase + strip.

    OFF stores brands as comma-separated lists ('Carrefour BIO, Carrefour'). Without
    splitting these become distinct categorical states from 'Carrefour', diluting top-N
    coverage. We take the first brand only — typically the most specific (e.g. private
    label sub-brand).
    """
    s = series.fillna("unknown").astype(str)
    return s.str.split(",").str[0].str.strip().str.lower().replace("", "unknown")


def brand_organic_marker(series: pd.Series) -> pd.Series:
    """Detect organic markers in brand string, regardless of normalization.

    Operates on RAW brand string (not normalized) so 'Carrefour BIO, Carrefour' still
    triggers True even if we drop the BIO suffix during normalization. This becomes a
    separate Bayesian node — clean binary signal independent of brand cardinality.
    """
    return series.fillna("").astype(str).apply(
        lambda s: "True" if _ORGANIC_MARKER_RE.search(s) else "False"
    )


def prepare_pasta_data(df: pd.DataFrame) -> pd.DataFrame:
    data = pd.DataFrame()
    data["brand"] = extract_top(normalize_brand(df["brands"]), top_n=20)
    data["brand_has_organic_marker"] = brand_organic_marker(df["brands"])
    data["grain_type"] = extract_top(df["grain_type"], top_n=7, fill="other")
    data["pasta_shape"] = extract_top(df["pasta_shape"], top_n=12, fill="unknown")
    if "is_filled" in df.columns:
        data["is_filled"] = df["is_filled"].fillna(False).astype(str)
    data["is_gluten_free"] = df["is_gluten_free"].fillna("False").astype(str)
    data["is_organic"] = df["is_organic"].fillna(False).astype(str)
    if "is_vegan" in df.columns:
        data["is_vegan"] = df["is_vegan"].fillna(False).astype(str)
    if "nutri_score_grade" in df.columns:
        data["nutri_score_grade"] = extract_top(df["nutri_score_grade"], top_n=5, fill="unknown")
    if "protein_class" in df.columns:
        data["protein_class"] = extract_top(df["protein_class"], top_n=12, fill="unknown")
    return data.dropna()


def prepare_chocolate_data(df: pd.DataFrame) -> pd.DataFrame:
    data = pd.DataFrame()
    data["brand"] = extract_top(normalize_brand(df["brands"]), top_n=20)
    data["brand_has_organic_marker"] = brand_organic_marker(df["brands"])
    data["chocolate_type"] = extract_top(df["chocolate_type"], top_n=5, fill="other")
    if "cocoa_percentage" in df.columns:
        data["cocoa_percentage"] = extract_top(df["cocoa_percentage"], top_n=5, fill="unknown")
    if "contains_nuts" in df.columns:
        data["contains_nuts"] = df["contains_nuts"].fillna(False).astype(str)
    if "chocolate_extra" in df.columns:
        data["chocolate_extra"] = extract_top(df["chocolate_extra"], top_n=9, fill="other")
    if "is_organic" in df.columns:
        data["is_organic"] = df["is_organic"].fillna(False).astype(str)
    if "nutri_score_grade" in df.columns:
        data["nutri_score_grade"] = extract_top(df["nutri_score_grade"], top_n=5, fill="unknown")
    if "protein_class" in df.columns:
        data["protein_class"] = extract_top(df["protein_class"], top_n=10, fill="unknown")
    return data.dropna()


def prepare_electronics_data(df: pd.DataFrame) -> pd.DataFrame:
    """Phones: brand уже в schema формате (10 values), normalize не нужен.
    Все 7 атрибутов (price_tier missing) идут в один граф — Hill Climb сам
    найдёт ожидаемые edges (brand→os, brand→form_factor, ram→storage, etc).
    """
    data = pd.DataFrame()
    data["brand"] = df["brand"].fillna("Other").astype(str)
    for col in ["os", "form_factor", "screen_size_class", "ram_class",
                "storage_class", "release_year_class"]:
        if col in df.columns:
            data[col] = df[col].fillna("unknown").astype(str)
    return data.dropna()


def prepare_baby_data(df: pd.DataFrame) -> pd.DataFrame:
    """Baby food: 6 атрибутов + brand. Главные causal chains:
    milk_type → is_lactose_free, milk_type → minimal_age, milk_type → flavour."""
    data = pd.DataFrame()
    data["brand"] = extract_top(normalize_brand(df["brands"]), top_n=20)
    data["brand_has_organic_marker"] = brand_organic_marker(df["brands"])
    if "milk_type" in df.columns:
        data["milk_type"] = extract_top(df["milk_type"], top_n=8, fill="other")
    if "minimal_age" in df.columns:
        data["minimal_age"] = extract_top(df["minimal_age"], top_n=5, fill="unknown")
    if "feeding_purpose" in df.columns:
        data["feeding_purpose"] = extract_top(df["feeding_purpose"], top_n=7, fill="other")
    if "format" in df.columns:
        data["format"] = extract_top(df["format"], top_n=3, fill="powder")
    if "is_organic" in df.columns:
        data["is_organic"] = df["is_organic"].fillna(False).astype(str)
    if "is_lactose_free" in df.columns:
        data["is_lactose_free"] = df["is_lactose_free"].fillna(False).astype(str)
    if "is_gluten_free" in df.columns:
        data["is_gluten_free"] = df["is_gluten_free"].fillna(False).astype(str)
    return data.dropna()


def prepare_cosmetics_data(df: pd.DataFrame) -> pd.DataFrame:
    """OBF cosmetics: 6 атрибутов, brand_has_organic_marker как для food."""
    data = pd.DataFrame()
    data["brand"] = extract_top(normalize_brand(df["brands"]), top_n=20)
    data["brand_has_organic_marker"] = brand_organic_marker(df["brands"])
    data["product_type"] = extract_top(df["product_type"], top_n=12, fill="other")
    if "form_factor" in df.columns:
        data["form_factor"] = extract_top(df["form_factor"], top_n=8, fill="unknown")
    if "body_area" in df.columns:
        data["body_area"] = extract_top(df["body_area"], top_n=10, fill="other")
    if "has_sulfates" in df.columns:
        # 3-state: NaN remains NaN, dropna убирает unsignal'd rows
        data["has_sulfates"] = df["has_sulfates"].astype(object).where(
            df["has_sulfates"].notna(), None)
        data["has_sulfates"] = data["has_sulfates"].map(
            lambda x: str(x) if x is not None else None)
    if "has_silicones" in df.columns:
        data["has_silicones"] = df["has_silicones"].astype(object).where(
            df["has_silicones"].notna(), None)
        data["has_silicones"] = data["has_silicones"].map(
            lambda x: str(x) if x is not None else None)
    if "is_organic" in df.columns:
        data["is_organic"] = df["is_organic"].fillna(False).astype(str)
    return data.dropna()


def prepare_cheeses_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cheeses: 6 атрибутов с явными causal chains
    (milk_source → texture, country → milk_source, texture → fat_class)."""
    data = pd.DataFrame()
    data["brand"] = extract_top(normalize_brand(df["brands"]), top_n=20)
    data["brand_has_organic_marker"] = brand_organic_marker(df["brands"])
    if "milk_source" in df.columns:
        data["milk_source"] = extract_top(df["milk_source"], top_n=6, fill="other")
    if "texture" in df.columns:
        data["texture"] = extract_top(df["texture"], top_n=7, fill="other")
    if "country_of_origin" in df.columns:
        data["country_of_origin"] = extract_top(df["country_of_origin"], top_n=9, fill="other")
    if "fat_class" in df.columns:
        data["fat_class"] = extract_top(df["fat_class"], top_n=4, fill="medium")
    if "is_pdo" in df.columns:
        data["is_pdo"] = df["is_pdo"].fillna(False).astype(str)
    if "is_organic" in df.columns:
        data["is_organic"] = df["is_organic"].fillna(False).astype(str)
    if "is_ultra_processed" in df.columns:
        data["is_ultra_processed"] = df["is_ultra_processed"].fillna(False).astype(str)
    return data.dropna()


def prepare_cereals_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cereals: 8 атрибутов. Ожидаемые причинные цепи (Hill Climb + BIC):
    cereal_type → nova_class (chocolate_cereal → ultra_processed),
    cereal_type → is_vegan (chocolate_cereal с молочным шоколадом → False),
    nova_class → is_low_sugar, is_organic → is_vegan."""
    data = pd.DataFrame()
    data["brand"] = extract_top(normalize_brand(df["brands"]), top_n=20)
    data["brand_has_organic_marker"] = brand_organic_marker(df["brands"])
    if "cereal_type" in df.columns:
        data["cereal_type"] = extract_top(df["cereal_type"], top_n=8, fill="other")
    if "grain_type" in df.columns:
        data["grain_type"] = extract_top(df["grain_type"], top_n=6, fill="other")
    if "nova_class" in df.columns:
        data["nova_class"] = extract_top(df["nova_class"], top_n=3, fill="ultra_processed")
    if "is_low_sugar" in df.columns:
        data["is_low_sugar"] = df["is_low_sugar"].fillna(False).astype(str)
    if "is_high_fibre" in df.columns:
        data["is_high_fibre"] = df["is_high_fibre"].fillna(False).astype(str)
    if "is_vegan" in df.columns:
        # NaN остаётся NaN — dropna уберёт; для cereals 38% NaN, Bayes учится на 62%
        data["is_vegan"] = df["is_vegan"].astype(object).where(df["is_vegan"].notna(), None)
        data["is_vegan"] = data["is_vegan"].map(
            lambda x: str(x) if x is not None else None)
    if "is_whole_grain" in df.columns:
        data["is_whole_grain"] = df["is_whole_grain"].fillna(False).astype(str)
    if "is_organic" in df.columns:
        data["is_organic"] = df["is_organic"].fillna(False).astype(str)
    return data.dropna()


def prepare_beverages_data(df: pd.DataFrame) -> pd.DataFrame:
    data = pd.DataFrame()
    data["brand"] = extract_top(normalize_brand(df["brands"]), top_n=20)
    data["brand_has_organic_marker"] = brand_organic_marker(df["brands"])
    data["beverage_type"] = extract_top(df["beverage_type"], top_n=8, fill="other")
    if "sugar_class" in df.columns:
        data["sugar_class"] = extract_top(df["sugar_class"], top_n=4, fill="unknown")
    if "is_organic" in df.columns:
        data["is_organic"] = df["is_organic"].fillna(False).astype(str)
    if "is_carbonated" in df.columns:
        data["is_carbonated"] = df["is_carbonated"].fillna(False).astype(str)
    if "nutri_score_grade" in df.columns:
        data["nutri_score_grade"] = extract_top(df["nutri_score_grade"], top_n=5, fill="unknown")
    if "nova_group" in df.columns:
        nv = pd.to_numeric(df["nova_group"], errors="coerce").fillna(0).astype(int)
        data["nova_group"] = nv.astype(str)
    if "protein_class" in df.columns:
        data["protein_class"] = extract_top(df["protein_class"], top_n=10, fill="unknown")
    if "is_vegan" in df.columns:
        # NaN остаётся NaN — dropna уберёт; beverages ~50% None
        data["is_vegan"] = df["is_vegan"].astype(object).where(df["is_vegan"].notna(), None)
        data["is_vegan"] = data["is_vegan"].map(
            lambda x: str(x) if x is not None else None)
    return data.dropna()




def learn_and_build(data: pd.DataFrame, prefix: str):
    """Learn structure via Hill Climb Search (BIC), fit parameters, save model."""
    if len(data) < 50:
        logger.warning("Not enough data (%d rows). Skipping.", len(data))
        return

    logger.info("Building %s Bayesian network with %d rows", prefix, len(data))
    for col in data.columns:
        logger.info("  %s: %d unique values", col, data[col].nunique())

    logger.info("Learning structure (Hill Climb + BIC, seed=%d)...", BAYES_TRAIN_SEED)
    # HillClimbSearch использует np.random внутри для tie-breaking при равных
    # BIC-скорах между соседями. Без явного seed граф меняется между запусками
    # → меняется P_B(y|e) → меняется порог θ_a → меняются headline-числа.
    # См. docs/po/critique/2026-05-27-2231.md, Находка 1.
    random.seed(BAYES_TRAIN_SEED)
    np.random.seed(BAYES_TRAIN_SEED)
    hc = HillClimbSearch(data)
    best_model = hc.estimate(scoring_method="bic-d", max_indegree=3)
    edges = list(best_model.edges())
    logger.info("Discovered %d edges:", len(edges))

    for src, dst in edges:
        logger.info("  %s -> %s", src, dst)

    if not edges:
        logger.warning("No edges discovered. Skipping.")
        return

    model = BayesianNetwork(edges)
    est = BayesianEstimator(model, data)
    for node in model.nodes():
        cpd = est.estimate_cpd(node, prior_type="BDeu", equivalent_sample_size=10)
        model.add_cpds(cpd)
    model.check_model()

    evidence_only = {"brand", "brand_has_organic_marker"}
    query_targets = [n for n in model.nodes() if n not in evidence_only]
    save_and_test(model, data, prefix, query_targets=query_targets)


def save_and_test(model, data, prefix, query_targets):
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{prefix}_bayesian.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved to %s", model_path)

    inference = VariableElimination(model)

    logger.info("=== Sample Inference (%s) ===", prefix)
    sample_rows = data.sample(min(5, len(data)), random_state=42)
    evidence_nodes = [n for n in model.nodes() if n not in query_targets]

    for _, row in sample_rows.iterrows():
        evidence = {n: row[n] for n in evidence_nodes if n in row.index}
        for target in query_targets:
            if target in evidence:
                continue
            try:
                result = inference.query([target], evidence=evidence)
                probs = {
                    str(s): float(result.values[i])
                    for i, s in enumerate(result.state_names[target])
                }
                best = max(probs, key=probs.get)
                conf = probs[best]
                action = "USE" if conf >= DEFAULT_CONFIDENCE_THRESHOLD else "LLM"
                logger.info("  P(%s | %s): %s (%.2f) [%s]", target, evidence, best, conf, action)
            except Exception as e:
                logger.warning("  P(%s | %s): ERROR %s", target, evidence, e)


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True,
                        choices=["pasta", "pasta_stratified",
                                 "chocolate", "chocolate_stratified",
                                 "beverages", "beverages_stratified",
                                 "electronics",
                                 "cosmetics_stratified",
                                 "cheeses_stratified", "cereals_stratified"])
    args = parser.parse_args()

    if args.category in ("pasta", "pasta_stratified"):
        ss_name = ("pasta_stratified_silver_standard.parquet"
                   if args.category == "pasta_stratified"
                   else "pasta_silver_standard.parquet")
        ss_path = os.path.join(PROCESSED_DIR, ss_name)
        if not os.path.exists(ss_path):
            logger.error("Silver standard not found: %s", ss_path)
            return
        df = pd.read_parquet(ss_path)
        logger.info("Loaded %d products", len(df))
        data = prepare_pasta_data(df)
        learn_and_build(data, args.category)
    elif args.category in ("chocolate", "chocolate_stratified"):
        ss_name = ("chocolate_stratified_silver_standard.parquet"
                   if args.category == "chocolate_stratified"
                   else "chocolate_silver_standard.parquet")
        ss_path = os.path.join(PROCESSED_DIR, ss_name)
        if not os.path.exists(ss_path):
            logger.error("Silver standard not found: %s", ss_path)
            return
        df = pd.read_parquet(ss_path)
        logger.info("Loaded %d products", len(df))
        data = prepare_chocolate_data(df)
        learn_and_build(data, args.category)
    elif args.category in ("beverages", "beverages_stratified"):
        ss_name = ("beverages_stratified_silver_standard.parquet"
                   if args.category == "beverages_stratified"
                   else "beverages_silver_standard.parquet")
        ss_path = os.path.join(PROCESSED_DIR, ss_name)
        if not os.path.exists(ss_path):
            logger.error("Silver standard not found: %s", ss_path)
            return
        df = pd.read_parquet(ss_path)
        logger.info("Loaded %d products", len(df))
        data = prepare_beverages_data(df)
        learn_and_build(data, args.category)
    elif args.category == "electronics":
        ss_path = os.path.join(PROCESSED_DIR, "electronics_silver_standard.parquet")
        if not os.path.exists(ss_path):
            logger.error("Silver standard not found: %s", ss_path)
            return
        df = pd.read_parquet(ss_path)
        logger.info("Loaded %d products", len(df))
        data = prepare_electronics_data(df)
        learn_and_build(data, "electronics")
    elif args.category == "cosmetics_stratified":
        ss_path = os.path.join(PROCESSED_DIR, "cosmetics_stratified_silver_standard.parquet")
        if not os.path.exists(ss_path):
            logger.error("Silver standard not found: %s", ss_path)
            return
        df = pd.read_parquet(ss_path)
        logger.info("Loaded %d products", len(df))
        data = prepare_cosmetics_data(df)
        learn_and_build(data, args.category)
    elif args.category == "cheeses_stratified":
        ss_path = os.path.join(PROCESSED_DIR, "cheeses_stratified_silver_standard.parquet")
        if not os.path.exists(ss_path):
            logger.error("Silver standard not found: %s", ss_path)
            return
        df = pd.read_parquet(ss_path)
        logger.info("Loaded %d products", len(df))
        data = prepare_cheeses_data(df)
        learn_and_build(data, args.category)
    elif args.category == "cereals_stratified":
        ss_path = os.path.join(PROCESSED_DIR, "cereals_stratified_silver_standard.parquet")
        if not os.path.exists(ss_path):
            logger.error("Silver standard not found: %s", ss_path)
            return
        df = pd.read_parquet(ss_path)
        logger.info("Loaded %d products", len(df))
        data = prepare_cereals_data(df)
        learn_and_build(data, args.category)


if __name__ == "__main__":
    main()
