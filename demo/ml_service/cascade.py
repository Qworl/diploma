"""
Обёртка существующего каскада regex → ML → validator для одного товара.

Используется FastAPI-сервисом в demo/ml_service/main.py.

Per spec §6.13.1: Bayes больше не используется как классификатор. Вместо
этого ValidatorService применяет ту же сеть как валидатор низко-вероятных
(attr, value) пар поверх предсказаний regex + ML.
"""
from __future__ import annotations

import logging
import os
import pickle
import re
import sys
import warnings
from typing import Any

# Layer −1: дешёвая проверка product_name до SBERT/роутера.
# Ловит то, что не должно даже попадать на эмбеддинг: пусто, шум, набор символов,
# повторяющиеся паттерны (фывфыв = "фыв"×2, asdasd, aaaaaa), пробег по клавиатурной
# строке (фывапролд, qwertyuiop, 12345).
_LETTER_RE = re.compile(r"[^\W\d_]", flags=re.UNICODE)
_MIN_NAME_LEN = 3
_MIN_LETTER_RATIO = 0.3

_KEYBOARD_ROWS = (
    "qwertyuiop", "asdfghjkl", "zxcvbnm",
    "йцукенгшщзхъ", "фывапролджэ", "ячсмитьбю",
    "1234567890",
)


def _is_repeated_pattern(s: str) -> bool:
    """True if s is essentially N≥2 copies of a 1–4 char prefix, optionally
    with a trailing partial copy.

    Catches both exact repetitions (``фывфыв`` = ``фыв`` × 2) and
    typing-overshoot tails (``фывфывф`` = ``фыв`` × 2 + ``ф``).
    """
    s = s.lower().replace(" ", "")
    n = len(s)
    if n < 6:
        return False
    for plen in range(1, 5):
        if plen * 2 > n:
            break
        pattern = s[:plen]
        n_full = n // plen
        if n_full < 2:
            continue
        rebuilt = pattern * n_full + pattern[: n - n_full * plen]
        if rebuilt == s:
            return True
    return False


def _is_keyboard_row(s: str) -> bool:
    """True if s is a contiguous substring of a keyboard row (or its reverse)."""
    s = s.lower().replace(" ", "")
    if len(s) < 4:
        return False
    for row in _KEYBOARD_ROWS:
        if s in row or s in row[::-1]:
            return True
    return False

warnings.filterwarnings("ignore", category=FutureWarning)

# Project root on sys.path so `src.*` packages are importable when uvicorn
# is launched from `demo/ml_service`.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from src.common import EMBEDDING_MODEL, MODELS_DIR, DEFAULT_CONFIDENCE_THRESHOLD
from src.pipeline.regex.extractor import RegexExtractor
from src.pipeline.schemas import (
    CHEESES_SCHEMA, CHOCOLATE_SCHEMA, PASTA_SCHEMA,
)
from src.pipeline.off_labels.apply import apply_off_labels
from src.pipeline.category_router.constants import DEMO_SUPPORTED_CATEGORIES as _CAT_SET
from src.pipeline.category_router.infer import CategoryRouter
from src.pipeline.category_router.mahalanobis_infer import MahalanobisOOD

DEMO_SUPPORTED_CATEGORIES = set(_CAT_SET)

logger = logging.getLogger(__name__)


# Маппинг публичной категории (как её видит пользователь) -> внутренняя
# категория, на которой обучены модели. Scope ВКР: 3 audit-grade категории
# (pasta_v4/chocolate_v4/cheeses_v4) после rebuild 2026-05-25 на gold v4 +
# MPNet эмбеддингах + TF-IDF.
PUBLIC_TO_INTERNAL = {
    "pasta": "pasta_v4",
    "chocolate": "chocolate_v4",
    "cheeses": "cheeses_v4",
}

# TYPE_C attrs (nutri_score_grade, protein_class, cocoa_percentage, fat_class)
# исключены из ML: они deterministic from nutriments через src/pipeline/off_labels/
# rules.py:TYPE_C_RULES. ML обучается только на semantic atrs.
CATEGORY_CONFIG = {
    "pasta_v4": {
        "schema": PASTA_SCHEMA,
        "ml_attrs": ["grain_type", "pasta_shape", "is_filled", "is_organic",
                      "is_gluten_free", "is_vegan", "cuisine_origin"],
        "regex_category": "pasta",
    },
    "chocolate_v4": {
        "schema": CHOCOLATE_SCHEMA,
        "ml_attrs": ["chocolate_type", "contains_nuts", "chocolate_extra",
                      "is_organic", "flavor_profile"],
        "regex_category": "chocolate",
    },
    "cheeses_v4": {
        "schema": CHEESES_SCHEMA,
        "ml_attrs": ["milk_source", "texture", "country_of_origin",
                      "is_pdo", "is_organic", "is_ultra_processed", "aging"],
        "regex_category": "cheeses",  # no regex rules; cascade falls through to ML
    },
}

INPUT_FIELDS = ("product_name", "brands", "ingredients_text", "quantity")


def _validate_input(product: dict) -> dict | None:
    """Layer −1 cheap input validator (runs before SBERT + router).

    Returns None when name looks like a plausible product label; otherwise a
    dict {reason, message} so the caller can short-circuit with a friendly
    rejection. Catches empty / too-short / non-letter-dominated inputs that
    the router would otherwise misroute with high confidence.
    """
    name = str(product.get("product_name") or "").strip()
    if not name:
        return {"reason": "empty", "message": "Название товара пустое"}
    if len(name) < _MIN_NAME_LEN:
        return {
            "reason": "too_short",
            "message": f"Название слишком короткое (< {_MIN_NAME_LEN} символов)",
        }
    total = len(name.replace(" ", ""))
    letters = len(_LETTER_RE.findall(name))
    if total > 0 and letters / total < _MIN_LETTER_RATIO:
        return {
            "reason": "no_letters",
            "message": "Название не содержит распознаваемых букв",
        }
    if _is_repeated_pattern(name):
        return {
            "reason": "repeated_pattern",
            "message": "Название — повторение короткой последовательности",
        }
    if _is_keyboard_row(name):
        return {
            "reason": "keyboard_row",
            "message": "Название — пробег по клавиатуре",
        }
    return None


class CascadePipeline:
    """Loaded once at service start; processes products one-at-a-time."""

    def __init__(self, lazy_embedder: bool = False):
        logger.info("Init CascadePipeline")
        self.rx = RegexExtractor()

        self._embedder = None

        self.thresholds: dict[str, dict[str, float]] = {}
        self.ml_models: dict[str, dict] = {}

        for cat in CATEGORY_CONFIG:
            self.thresholds[cat] = self._load_thresholds(cat)
            self.ml_models[cat] = self._load_ml_models(cat)

        from validator import ValidatorService
        self.validator = ValidatorService(
            models_dir=MODELS_DIR,
            internal_categories=list(CATEGORY_CONFIG.keys()),
        )

        if not lazy_embedder:
            self._init_embedder()

        try:
            self.router = CategoryRouter.load(MODELS_DIR, embedder=self._embedder)
            logger.info("CategoryRouter loaded")
        except FileNotFoundError as e:
            logger.warning("CategoryRouter artefacts not found: %s — auto mode disabled", e)
            self.router = None

        try:
            self.mahalanobis = MahalanobisOOD.load()
            logger.info(
                "MahalanobisOOD loaded (threshold=%.3f)", self.mahalanobis.threshold
            )
        except FileNotFoundError as e:
            logger.warning(
                "MahalanobisOOD artefact missing: %s — semantic OOD disabled", e
            )
            self.mahalanobis = None

        logger.info("CascadePipeline ready (validator_ready=%s)", self.validator.ready())

    def _init_embedder(self):
        from sentence_transformers import SentenceTransformer
        logger.info("Loading SBERT %s ...", EMBEDDING_MODEL)
        self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("SBERT loaded")

    def _load_thresholds(self, category: str) -> dict:
        path = os.path.join(MODELS_DIR, f"{category}_thresholds.pkl")
        if not os.path.exists(path):
            return {}
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_ml_models(self, category: str) -> dict:
        models = {}
        attrs = CATEGORY_CONFIG[category]["ml_attrs"]
        for attr in attrs:
            # Production models are *_xgb_hybrid.pkl (v3e snapshot, n=384 SBERT input).
            # Legacy fallback to *_xgb.pkl for compatibility with old training runs.
            xgb_hybrid_path = os.path.join(MODELS_DIR, f"{category}_{attr}_xgb_hybrid.pkl")
            le_hybrid_path = os.path.join(MODELS_DIR, f"{category}_{attr}_le_hybrid.pkl")
            xgb_legacy_path = os.path.join(MODELS_DIR, f"{category}_{attr}_xgb.pkl")
            le_legacy_path = os.path.join(MODELS_DIR, f"{category}_{attr}_le.pkl")
            xgb_path = xgb_hybrid_path if os.path.exists(xgb_hybrid_path) else xgb_legacy_path
            le_path = le_hybrid_path if os.path.exists(le_hybrid_path) else le_legacy_path
            if not os.path.exists(xgb_path):
                continue
            with open(xgb_path, "rb") as f:
                models[f"{attr}_xgb"] = pickle.load(f)
            if os.path.exists(le_path):
                with open(le_path, "rb") as f:
                    models[f"{attr}_le"] = pickle.load(f)
        return models

    def _embed(self, product: dict) -> np.ndarray:
        if self._embedder is None:
            self._init_embedder()
        text = " ".join(
            str(product.get(k, "") or "") for k in INPUT_FIELDS
        ).strip() or " "
        return self._embedder.encode([text], show_progress_bar=False)

    def _regex_layer(self, product: dict, regex_category: str) -> dict:
        results = self.rx.extract_all(
            str(product.get("product_name", "")),
            str(product.get("generic_name", "")),
            str(product.get("quantity", "")),
            category=regex_category,
        )
        return {k: (v.value, float(v.confidence)) for k, v in results.items()
                if v.value is not None}

    def _ml_layer(self, embedding: np.ndarray, category: str) -> dict:
        models = self.ml_models[category]
        thresholds = self.thresholds[category]
        attrs = CATEGORY_CONFIG[category]["ml_attrs"]
        predictions = {}
        for attr in attrs:
            xgb_key = f"{attr}_xgb"
            le_key = f"{attr}_le"
            if xgb_key not in models:
                continue
            clf = models[xgb_key]
            proba = clf.predict_proba(embedding)[0]
            max_idx = int(proba.argmax())
            confidence = float(proba[max_idx])
            threshold = thresholds.get(attr, DEFAULT_CONFIDENCE_THRESHOLD)
            if confidence < threshold:
                continue
            if le_key in models:
                value = models[le_key].inverse_transform([max_idx])[0]
                value = value.item() if hasattr(value, "item") else value
            else:
                value = bool(max_idx)
            predictions[attr] = (value, confidence)
        return predictions

    def _validate_layer(
        self,
        internal_category: str,
        brand: str,
        predictions: dict,
        expected: dict,
    ) -> tuple[dict, dict, str]:
        """Compute validation blocks for predictions and expected; return brand status."""
        # Build a base evidence set: brand + all predicted attrs + all expected attrs.
        base_evidence: dict[str, Any] = {}
        if brand:
            base_evidence["brand"] = brand
        for attr, (val, _conf, _layer) in predictions.items():
            base_evidence[attr] = val
        for attr, val in expected.items():
            # Expected wins in the base evidence for cross-validation purposes.
            base_evidence[attr] = val

        # Валидация имеет смысл только для вероятностных слоёв. Regex и
        # off_tags — детерминированные правила: если паттерн матчится,
        # значение правильное by design (например, «white» в названии →
        # chocolate_type=white). Статистическая валидация на них даёт
        # false-positive на редких, но корректных значениях.
        prediction_validation: dict[str, dict | None] = {}
        for attr, (val, _conf, layer) in predictions.items():
            if layer in ("regex", "off_tags"):
                prediction_validation[attr] = None
                continue
            evidence_minus_self = {a: v for a, v in base_evidence.items() if a != attr}
            prediction_validation[attr] = self.validator.validate_value(
                internal_category, attr, val, evidence_minus_self
            )

        expected_validation: dict[str, dict | None] = {}
        for attr, val in expected.items():
            evidence_minus_self = {a: v for a, v in base_evidence.items() if a != attr}
            expected_validation[attr] = self.validator.validate_value(
                internal_category, attr, val, evidence_minus_self
            )

        bstatus = self.validator.brand_status(internal_category, brand or "")
        return prediction_validation, expected_validation, bstatus

    def predict(
        self,
        public_category: str | None,
        product: dict,
        use_off_layer: bool = False,
        validate_mode: str = "warn",
        expected: dict | None = None,
        fallback_on_ood: bool = False,
    ) -> dict:
        """Return cascade output + validation blocks.

        Response shape matches spec §4.1.
        """
        invalid = _validate_input(product)
        if invalid is not None:
            return self._build_invalid_input_response(invalid)

        # Layer 0.5: семантический OOD по Махаланобису. Дёшев (одно умножение
        # на матрицу 384×384), независим от softmax-уверенности роутера.
        # Срабатывает только в auto-режиме — при ручной категории доверяем
        # пользователю.
        mahalanobis_score = None
        if public_category is None and getattr(self, "mahalanobis", None) is not None:
            vec = self._embed(product).reshape(-1)
            mahalanobis_score = self.mahalanobis.score(vec)
            if mahalanobis_score["is_ood"]:
                return self._build_semantic_ood_response(mahalanobis_score)

        router_out = None
        if public_category is None:
            if getattr(self, "router", None) is None:
                raise ValueError("router unavailable; specify category explicitly")
            router_out = self.router.predict(product)
            if router_out["is_ood"]:
                return self._build_ood_response(router_out, fallback_on_ood)
            if router_out["predicted"] not in DEMO_SUPPORTED_CATEGORIES:
                return self._build_unsupported_response(router_out)
            public_category = router_out["predicted"]

        if public_category not in PUBLIC_TO_INTERNAL:
            raise ValueError(f"Unknown category: {public_category}")
        category = PUBLIC_TO_INTERNAL[public_category]
        cfg = CATEGORY_CONFIG[category]
        schema = cfg["schema"]
        all_attrs = list(cfg["ml_attrs"])
        expected = expected or {}

        extracted: dict[str, tuple[Any, float, str]] = {}

        if use_off_layer:
            off = apply_off_labels(product, schema)
            for attr, val in off.items():
                if val is None:
                    continue
                if attr in all_attrs:
                    extracted[attr] = (val, 1.0, "off_tags")

        for attr, (val, conf) in self._regex_layer(product, cfg["regex_category"]).items():
            if attr in all_attrs and attr not in extracted:
                extracted[attr] = (val, conf, "regex")

        embedding = self._embed(product)
        for attr, (val, conf) in self._ml_layer(embedding, category).items():
            if attr in all_attrs and attr not in extracted:
                extracted[attr] = (val, conf, "ml")

        # Validate predictions and expected.
        brand = str(product.get("brands", "") or "")
        pred_val, exp_val, bstatus = self._validate_layer(
            internal_category=category,
            brand=brand,
            predictions=extracted,
            expected=expected,
        )

        # Apply demote-mode reaction.
        if validate_mode == "demote":
            for attr in list(extracted.keys()):
                v = pred_val.get(attr)
                if v is None or not v.get("flagged"):
                    continue
                extracted[attr] = (None, 0.0, "rejected_by_validator")

        # Assemble response predictions block.
        predictions: dict[str, dict] = {}
        for attr in all_attrs:
            if attr in extracted:
                val, conf, layer = extracted[attr]
                predictions[attr] = {
                    "value": _serialize(val),
                    "layer": layer,
                    "confidence": float(conf),
                    "validation": pred_val.get(attr),
                }
            else:
                predictions[attr] = {
                    "value": None,
                    "layer": "llm_fallback",
                    "confidence": 0.0,
                    "validation": None,
                }

        # Assemble expected block (only for attributes user provided).
        expected_out: dict[str, dict] = {}
        for attr, val in expected.items():
            if attr not in all_attrs:
                continue
            v_block = exp_val.get(attr)
            predicted_state = self.validator.bucketize_value(
                category, attr,
                extracted[attr][0] if attr in extracted else None,
            ) if attr in extracted else None
            expected_state = self.validator.bucketize_value(category, attr, val)
            if attr not in extracted:
                agrees = None
            else:
                agrees = (predicted_state == expected_state)
            expected_out[attr] = {
                "value": _serialize(val),
                "bucketized_to": expected_state,
                "validation": v_block,
                "agrees_with_predicted": agrees,
            }

        n_covered = sum(1 for p in predictions.values() if p["value"] is not None)
        n_flagged_pred = sum(
            1 for v in pred_val.values() if v and v.get("flagged")
        )
        n_flagged_exp = sum(
            1 for v in exp_val.values() if v and v.get("flagged")
        )

        return {
            "category": public_category,
            "internal_category": category,
            "n_attrs_total": len(all_attrs),
            "n_covered": n_covered,
            "n_llm_fallback": len(all_attrs) - n_covered,
            "predictions": predictions,
            "expected": expected_out,
            "validation_summary": {
                "n_flagged_predictions": n_flagged_pred,
                "n_flagged_expected": n_flagged_exp,
                "brand_status": bstatus,
                "mode": validate_mode,
            },
            "category_inference": router_out,
            "is_ood": False,
            "is_known_but_unsupported": False,
            "pending_llm_fallback": False,
            "is_invalid_input": False,
            "invalid_input": None,
            "semantic_ood": None,
        }


    def _build_ood_response(self, router_out, fallback_on_ood):
        return {
            "category": None,
            "internal_category": None,
            "n_attrs_total": 0,
            "n_covered": 0,
            "n_llm_fallback": 0,
            "predictions": {},
            "expected": {},
            "validation_summary": {
                "n_flagged_predictions": 0, "n_flagged_expected": 0,
                "brand_status": "n/a", "mode": "warn",
            },
            "category_inference": router_out,
            "is_ood": True,
            "is_known_but_unsupported": False,
            "pending_llm_fallback": bool(fallback_on_ood),
            "is_invalid_input": False,
            "invalid_input": None,
            "semantic_ood": None,
        }

    def _build_semantic_ood_response(self, score):
        return {
            "category": None,
            "internal_category": None,
            "n_attrs_total": 0,
            "n_covered": 0,
            "n_llm_fallback": 0,
            "predictions": {},
            "expected": {},
            "validation_summary": {
                "n_flagged_predictions": 0, "n_flagged_expected": 0,
                "brand_status": "n/a", "mode": "warn",
            },
            "category_inference": None,
            "is_ood": True,
            "is_known_but_unsupported": False,
            "pending_llm_fallback": False,
            "is_invalid_input": False,
            "invalid_input": None,
            "semantic_ood": dict(score),
        }

    def _build_invalid_input_response(self, invalid):
        return {
            "category": None,
            "internal_category": None,
            "n_attrs_total": 0,
            "n_covered": 0,
            "n_llm_fallback": 0,
            "predictions": {},
            "expected": {},
            "validation_summary": {
                "n_flagged_predictions": 0, "n_flagged_expected": 0,
                "brand_status": "n/a", "mode": "warn",
            },
            "category_inference": None,
            "is_ood": False,
            "is_known_but_unsupported": False,
            "pending_llm_fallback": False,
            "is_invalid_input": True,
            "invalid_input": invalid,
            "semantic_ood": None,
        }

    def _build_unsupported_response(self, router_out):
        return {
            "category": router_out["predicted"],
            "internal_category": None,
            "n_attrs_total": 0,
            "n_covered": 0,
            "n_llm_fallback": 0,
            "predictions": {},
            "expected": {},
            "validation_summary": {
                "n_flagged_predictions": 0, "n_flagged_expected": 0,
                "brand_status": "n/a", "mode": "warn",
            },
            "category_inference": router_out,
            "is_ood": False,
            "is_known_but_unsupported": True,
            "pending_llm_fallback": False,
            "is_invalid_input": False,
            "invalid_input": None,
            "semantic_ood": None,
        }


def _serialize(value):
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)
