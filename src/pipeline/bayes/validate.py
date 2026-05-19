"""Core Bayes-validation primitives.

This module reuses already-trained DiscreteBayesianNetwork models
(`models/{cat}_stratified_bayesian.pkl`) in a validation role:
given an observed value for an attribute and a set of evidence values
for other attributes, compute the conditional density P(attr=v | evidence)
and decide whether the value is plausible (above a per-attribute threshold).
"""
from __future__ import annotations

import itertools
import math
import random
from math import factorial
from typing import Literal

from src.pipeline.bayes.bucketize import bucketize


def _clean_evidence(evidence: dict, bayes_model) -> dict:
    """Drop keys not in the network; project remaining values via bucketize.

    Brand values outside the trained brand support collapse to "other"
    if "other" is a state (matches existing _bayes_layer behaviour).
    """
    cleaned: dict[str, str] = {}
    nodes = set(bayes_model.nodes())
    for k, v in evidence.items():
        if k not in nodes:
            continue
        bucketed = bucketize(k, v, bayes_model)
        if bucketed is None and k == "brand":
            # Brand fallback: collapse unknown brand to "other" if present.
            states = [str(s) for s in bayes_model.get_cpds("brand").state_names["brand"]]
            if "other" in states:
                cleaned["brand"] = "other"
            continue
        if bucketed is None:
            continue
        cleaned[k] = bucketed
    return cleaned


def attribute_likelihood(
    attr: str,
    value,
    evidence: dict,
    bayes_model,
    inference,
) -> float | None:
    """Compute P(attr=value | evidence) under bayes_model.

    Returns None in three situations:
      • attr is not a node of bayes_model;
      • value cannot be projected to any state of attr (out-of-domain input);
      • bucketize returns None.

    None means "no verdict" (UI shows no badge). 0.0 means "valid form but
    Bayes assigns zero probability".
    """
    if attr not in bayes_model.nodes():
        return None
    state = bucketize(attr, value, bayes_model)
    if state is None:
        return None

    cleaned_evidence = _clean_evidence(evidence, bayes_model)
    # attr itself must not appear in evidence — drop it defensively.
    cleaned_evidence.pop(attr, None)

    result = inference.query([attr], evidence=cleaned_evidence, show_progress=False)
    states = [str(s) for s in result.state_names[attr]]
    return float(result.values[states.index(state)])


def top_contributors_pmi(
    attr: str,
    value,
    evidence: dict,
    bayes_model,
    inference,
    k: int = 2,
) -> list[dict]:
    """Per-evidence pointwise mutual information; return k most negative.

    PMI(attr=v ; e_i | rest) = log[P(v | rest, e_i) / P(v | rest)].
    Strongly negative PMI means the evidence variable pushes the
    probability of `value` down — exactly what we want to surface as
    "conflict" in the UI.

    Returns a list of dicts: {"attr": str, "value": original, "pmi": float}.
    Evidence values are reported as the *original* user-provided forms
    (not the bucketed state-names) — that's what UI/users recognize.
    """
    if not evidence:
        return []

    contribs: list[dict] = []
    for ev_attr, ev_val in evidence.items():
        evidence_without = {a: v for a, v in evidence.items() if a != ev_attr}
        p_full = attribute_likelihood(attr, value, evidence, bayes_model, inference)
        p_without = attribute_likelihood(
            attr, value, evidence_without, bayes_model, inference
        )
        if p_full is None or p_without is None:
            continue
        if p_full <= 0 and p_without <= 0:
            pmi = 0.0
        elif p_full <= 0:
            pmi = float("-inf")
        elif p_without <= 0:
            pmi = float("inf")
        else:
            pmi = math.log(p_full / p_without)
        contribs.append({"attr": ev_attr, "value": ev_val, "pmi": pmi})

    contribs.sort(key=lambda c: c["pmi"])
    return contribs[:k]


def _log_p(attr, value, evidence_subset, bayes_model, inference) -> float:
    """Log of P(attr=value | evidence_subset); -inf if zero."""
    p = attribute_likelihood(
        attr, value, evidence_subset, bayes_model, inference
    )
    if p is None or p <= 0:
        return float("-inf")
    return math.log(p)


def shapley_attribution(
    attr: str,
    value,
    evidence: dict,
    bayes_model,
    inference,
    monte_carlo_samples: int | None = None,
) -> dict:
    """Shapley values of evidence variables for log P(attr=value | S).

    Exact mode (monte_carlo_samples=None) — enumerates all 2^n subsets.
    Sampled mode — averages over k random permutations.

    Returns:
      {
        "attribution": [{"attr", "value", "shapley"}, ...],
        "p_full": float,
        "p_marginal": float,
        "log_likelihood_diff": log p_full - log p_marginal,
        "sum_shapley": sum of attributions,
        "efficiency_residual": sum_shapley - log_likelihood_diff,
      }
    """
    ev_items = list(evidence.items())
    n = len(ev_items)
    ev_attrs = [a for a, _ in ev_items]

    if n == 0:
        p0 = attribute_likelihood(attr, value, {}, bayes_model, inference)
        return {
            "attribution": [],
            "p_full": float(p0) if p0 is not None else 0.0,
            "p_marginal": float(p0) if p0 is not None else 0.0,
            "log_likelihood_diff": 0.0,
            "sum_shapley": 0.0,
            "efficiency_residual": 0.0,
        }

    p_full_val = attribute_likelihood(attr, value, evidence, bayes_model, inference)
    p_marginal_val = attribute_likelihood(attr, value, {}, bayes_model, inference)
    if p_full_val is None or p_marginal_val is None:
        raise ValueError("Cannot compute Shapley: full or marginal P is None")

    shap = {a: 0.0 for a in ev_attrs}

    if monte_carlo_samples is None:
        # Exact enumeration of subsets.
        for r in range(n):
            for subset in itertools.combinations(ev_attrs, r):
                S = dict((a, evidence[a]) for a in subset)
                weight = factorial(r) * factorial(n - r - 1) / factorial(n)
                log_S = _log_p(attr, value, S, bayes_model, inference)
                for i in ev_attrs:
                    if i in subset:
                        continue
                    S_with = dict(S)
                    S_with[i] = evidence[i]
                    log_S_with = _log_p(attr, value, S_with, bayes_model, inference)
                    shap[i] += weight * (log_S_with - log_S)
    else:
        # Monte-Carlo over random permutations.
        for _ in range(monte_carlo_samples):
            perm = ev_attrs[:]
            random.shuffle(perm)
            S: dict[str, object] = {}
            log_prev = _log_p(attr, value, S, bayes_model, inference)
            for i in perm:
                S[i] = evidence[i]
                log_curr = _log_p(attr, value, S, bayes_model, inference)
                shap[i] += (log_curr - log_prev) / monte_carlo_samples
                log_prev = log_curr

    attribution = [
        {"attr": a, "value": evidence[a], "shapley": shap[a]} for a in ev_attrs
    ]
    sum_shapley = sum(shap.values())
    log_diff = math.log(p_full_val) - math.log(p_marginal_val)

    return {
        "attribution": attribution,
        "p_full": float(p_full_val),
        "p_marginal": float(p_marginal_val),
        "log_likelihood_diff": float(log_diff),
        "sum_shapley": float(sum_shapley),
        "efficiency_residual": float(sum_shapley - log_diff),
    }


def brand_status(brand: str, bayes_model) -> Literal["known", "ood", "n/a"]:
    """Return 'known' if brand is in trained support, 'ood' if outside it,
    'n/a' if the network has no 'brand' node (validator cannot judge)."""
    if "brand" not in bayes_model.nodes():
        return "n/a"
    cpd = bayes_model.get_cpds("brand")
    known_brands = set(str(s) for s in cpd.state_names["brand"])
    return "known" if str(brand) in known_brands else "ood"


def calibrate_thresholds(
    bayes_model,
    train_df,
    inference,
    q: float = 0.05,
) -> dict[str, float]:
    """Per-attribute q-percentile of P(true_value | evidence) on train data.

    For each network node (excluding 'brand'), iterate train rows, compute
    P(attr=row[attr] | other attrs of the same row), collect into a list,
    take q-percentile. Result is the threshold below which the validator
    flags a value.
    """
    import numpy as np  # local import to keep top of module light

    thresholds: dict[str, float] = {}
    for attr in bayes_model.nodes():
        if attr == "brand":
            continue
        ps: list[float] = []
        for _, row in train_df.iterrows():
            true_val = row.get(attr)
            if true_val is None or (isinstance(true_val, float) and np.isnan(true_val)):
                continue
            evidence = {
                a: row[a]
                for a in bayes_model.nodes()
                if a != attr and a in row.index and row[a] is not None
                and not (isinstance(row[a], float) and np.isnan(row[a]))
            }
            p = attribute_likelihood(
                attr, true_val, evidence, bayes_model, inference
            )
            if p is not None:
                ps.append(p)
        if not ps:
            thresholds[attr] = 0.0
            continue
        thresholds[attr] = float(np.percentile(ps, q * 100))
    return thresholds
