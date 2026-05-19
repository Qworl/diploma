"""
Bayesian Network inference (Layer 3 of hybrid system).

Loads pickled DiscreteBayesianNetwork model and performs variable elimination
with brand + ML predictions as evidence.
"""

import logging
import os
import pickle

from pgmpy.inference import VariableElimination

from src.common import DEFAULT_CONFIDENCE_THRESHOLD, MODELS_DIR

logger = logging.getLogger(__name__)


def load_network(category: str):
    """Load pickled DiscreteBayesianNetwork from MODELS_DIR/{category}_bayesian.pkl.

    Args:
        category: Category name (e.g., 'pasta', 'pasta_stratified', 'chocolate', etc.)

    Returns:
        Tuple[model, inference] where model is DiscreteBayesianNetwork and
        inference is VariableElimination instance. Returns (None, None) if not found.
    """
    path = os.path.join(MODELS_DIR, f"{category}_bayesian.pkl")
    if not os.path.exists(path):
        logger.warning("Bayesian model not found: %s", path)
        return None, None

    with open(path, "rb") as f:
        model = pickle.load(f)

    inference = VariableElimination(model)
    return model, inference


def query_bayes(bayes_model, inference, targets: list[str],
                evidence: dict, thresholds: dict | None = None) -> dict:
    """Query Bayesian network with given evidence.

    Args:
        bayes_model: DiscreteBayesianNetwork instance
        inference: VariableElimination inference engine
        targets: List of target variables to predict
        evidence: Dict of {variable: value} for conditioning
        thresholds: Dict of {target: confidence_threshold} (default DEFAULT_CONFIDENCE_THRESHOLD)

    Returns:
        Dict of {target: (best_value, confidence)} for predictions exceeding threshold.
    """
    if not bayes_model or not inference:
        return {}

    predictions = {}
    model_nodes = set(bayes_model.nodes())

    for target in targets:
        if target in evidence:
            continue
        if target not in model_nodes:
            continue

        try:
            result = inference.query([target], evidence=evidence)
            # Extract probability distribution
            probs = {
                str(s): float(result.values[i])
                for i, s in enumerate(result.state_names[target])
            }
            best = max(probs, key=probs.get)
            conf = probs[best]

            # Apply threshold
            threshold = (thresholds or {}).get(target, DEFAULT_CONFIDENCE_THRESHOLD)
            if conf >= threshold:
                predictions[target] = (best, conf)
        except Exception as e:
            logger.debug("Bayesian query failed for %s with evidence %s: %s", target, evidence, e)

    return predictions


def bayes_layer(row, bayes_model, inference, targets: list[str],
                ml_predictions: dict | None = None,
                thresholds: dict | None = None) -> dict:
    """Bayesian inference using brand + ML predictions as evidence.

    This is the full bayes_layer function from run_experiments.py adapted for
    the modular pipeline. It assembles evidence from brand and ML layer, then
    queries the Bayesian network.

    Args:
        row: DataFrame row with product data (must contain 'brands' key)
        bayes_model: DiscreteBayesianNetwork instance
        inference: VariableElimination inference engine
        targets: List of target attributes to predict
        ml_predictions: Dict from ML layer of {attr: (value, confidence, source)}
        thresholds: Per-attribute confidence thresholds

    Returns:
        Dict of {target: (best_value, confidence)} for confident predictions.
    """
    if not bayes_model or not inference:
        return {}

    predictions = {}
    evidence = {}
    model_nodes = set(bayes_model.nodes())

    # Add brand evidence if available in model
    if "brand" in model_nodes:
        brand_val = str(row.get("brands", "other"))
        try:
            cpd = bayes_model.get_cpds("brand")
            known = list(cpd.state_names["brand"])
            evidence["brand"] = brand_val if brand_val in known else "other"
        except Exception as e:
            logger.debug("Failed to set brand evidence: %s", e)

    # Use all confident ML predictions as inter-attribute evidence
    if ml_predictions:
        for attr, (val, conf, _source) in ml_predictions.items():
            if attr in model_nodes and conf >= 0.6:
                try:
                    cpd = bayes_model.get_cpds(attr)
                    known = list(cpd.state_names[attr])
                    str_val = str(val)
                    if str_val in known:
                        evidence[attr] = str_val
                except Exception as e:
                    logger.debug("Failed to set evidence for %s: %s", attr, e)

    # Query targets with assembled evidence
    return query_bayes(bayes_model, inference, targets, evidence, thresholds)
