"""Demo-side wrapper around src.pipeline.bayes.validate.

Holds per-category BayesianNetwork + inference + thresholds, exposes a
single `validate_predictions(predictions, expected, brand) -> dict` for
the cascade to call after regex + ML.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import sys
from typing import Any

# Ensure project root on sys.path so `src.pipeline.bayes.*` is importable when
# the demo service is launched from `demo/ml_service`.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.pipeline.bayes.validate import (  # noqa: E402
    attribute_likelihood,
    top_contributors_pmi,
    brand_status as _brand_status,
)
from src.pipeline.bayes.bucketize import bucketize  # noqa: E402

logger = logging.getLogger(__name__)


class ValidatorService:
    def __init__(self, models_dir: str, internal_categories: list[str]):
        from pgmpy.inference import VariableElimination

        self.models: dict[str, Any] = {}
        self.inferences: dict[str, Any] = {}
        self.thresholds: dict[str, dict[str, float]] = {}

        for cat in internal_categories:
            bayes_path = os.path.join(models_dir, f"{cat}_bayesian.pkl")
            thr_path = os.path.join(models_dir, f"{cat}_validation_thresholds.json")
            if not (os.path.exists(bayes_path) and os.path.exists(thr_path)):
                logger.warning("Validator artifacts missing for %s", cat)
                continue
            with open(bayes_path, "rb") as f:
                bayes = pickle.load(f)
            with open(thr_path) as f:
                thr_doc = json.load(f)
            self.models[cat] = bayes
            self.inferences[cat] = VariableElimination(bayes)
            self.thresholds[cat] = thr_doc["thresholds"]

    def ready(self) -> bool:
        return bool(self.models)

    def validate_value(
        self, internal_category: str, attr: str, value, evidence: dict
    ) -> dict | None:
        """Return validation dict for a single (attr, value), or None if no verdict.

        Shape:
            {
              "flagged": bool, "p": float, "marginal_p": float,
              "threshold": float, "contributors": [{"attr","value","pmi"}, ...]
            }
        """
        bayes = self.models.get(internal_category)
        if bayes is None or attr not in bayes.nodes():
            return None
        inference = self.inferences[internal_category]
        thr = self.thresholds[internal_category].get(attr)
        if thr is None:
            return None
        p = attribute_likelihood(attr, value, evidence, bayes, inference)
        if p is None:
            return None
        marginal = attribute_likelihood(attr, value, {}, bayes, inference)
        contribs = top_contributors_pmi(
            attr, value, evidence, bayes, inference, k=2
        )
        return {
            "flagged": p < thr,
            "p": float(p),
            "marginal_p": float(marginal) if marginal is not None else 0.0,
            "threshold": float(thr),
            "contributors": contribs,
        }

    def brand_status(self, internal_category: str, brand: str) -> str:
        bayes = self.models.get(internal_category)
        if bayes is None:
            return "n/a"
        return _brand_status(brand, bayes)

    def bucketize_value(self, internal_category: str, attr: str, value) -> str | None:
        bayes = self.models.get(internal_category)
        if bayes is None or attr not in bayes.nodes():
            return None
        return bucketize(attr, value, bayes)
