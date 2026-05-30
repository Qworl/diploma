"""
FastAPI-сервис каскада. Принимает товар, возвращает предсказания по слоям.

Endpoints:
- GET  /health        — проверка живости + validator_ready
- GET  /categories    — список доступных категорий и атрибутов
- POST /api/enrich    — обогатить один товар (с валидацией)
- POST /api/explain   — Shapley/PMI разбор для одной (attr, value) пары

Запуск:
    cd demo/ml_service
    uvicorn main:app --host 127.0.0.1 --port 8001 --reload
"""
from __future__ import annotations

import logging
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

try:
    from demo.ml_service.cascade import CATEGORY_CONFIG, PUBLIC_TO_INTERNAL, CascadePipeline
except ModuleNotFoundError:
    from cascade import CATEGORY_CONFIG, PUBLIC_TO_INTERNAL, CascadePipeline  # type: ignore[no-redef]
from src.pipeline.bayes.validate import shapley_attribution, top_contributors_pmi
from src.pipeline.category_router.constants import DEMO_SUPPORTED_CATEGORIES

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("ml_service")

app = FastAPI(title="AI Attributes — ML Service",
              description="Каскад regex → ML → Bayes-валидатор для одного товара")

# Разрешаем запросы от Go-gateway и для локальной отладки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class EnrichRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    category: Optional[Literal["pasta", "chocolate", "cheeses"]] = None
    product_name: str = Field(min_length=0, max_length=500)
    brands: str = Field(default="", max_length=300)
    ingredients_text: str = Field(default="", max_length=5000)
    quantity: str = Field(default="", max_length=100)
    validate_mode: Literal["warn", "demote"] = Field(default="warn", alias="validate")
    expected: dict[str, Any] = Field(default_factory=dict)
    # Operator-confirmed values: short-circuit cascade за оператора;
    # отображаются на UI как layer=«operator».
    confirmed: dict[str, Any] = Field(default_factory=dict)
    fallback_on_ood: bool = False


class ValidationBlock(BaseModel):
    flagged: bool
    p: float
    marginal_p: float
    threshold: float
    contributors: list[dict[str, Any]]


class PredictionBlock(BaseModel):
    value: Any
    layer: str
    confidence: float
    validation: Optional[ValidationBlock]


class ExpectedBlock(BaseModel):
    value: Any
    bucketized_to: Optional[str]
    validation: Optional[ValidationBlock]
    agrees_with_predicted: Optional[bool]


class ValidationSummary(BaseModel):
    n_flagged_predictions: int
    n_flagged_expected: int
    brand_status: Literal["known", "ood", "n/a"]
    mode: Literal["warn", "demote"]


class EnrichResponse(BaseModel):
    category: Optional[str]
    internal_category: Optional[str]
    n_attrs_total: int
    n_covered: int
    n_llm_fallback: int
    predictions: dict[str, PredictionBlock]
    expected: dict[str, ExpectedBlock]
    validation_summary: ValidationSummary
    category_inference: Optional[dict[str, Any]] = None
    is_ood: bool = False
    is_known_but_unsupported: bool = False
    pending_llm_fallback: bool = False
    is_invalid_input: bool = False
    invalid_input: Optional[dict[str, Any]] = None
    semantic_ood: Optional[dict[str, Any]] = None


class ExplainRequest(BaseModel):
    category: Literal["pasta", "chocolate", "cheeses"]
    brands: str = Field(default="", max_length=300)
    attribute: str
    value: Any
    evidence: dict[str, Any] = Field(default_factory=dict)
    shapley_mode: Literal["exact", "sampled"] = "sampled"
    shapley_samples: int = 100


# Глобальный экземпляр каскада — загружается один раз при старте процесса
pipeline: Optional[CascadePipeline] = None


@app.on_event("startup")
def startup_event():
    global pipeline
    logger.info("Старт сервиса: инициализация каскада ...")
    pipeline = CascadePipeline()
    logger.info("Сервис готов")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "pipeline_ready": pipeline is not None,
        "validator_ready": pipeline.validator.ready() if pipeline is not None else False,
        "router_ready": getattr(pipeline, "router", None) is not None if pipeline is not None else False,
    }


import re as _re

_NUMERIC_BIN_RE = _re.compile(r"^(?:\d+(?:\.\d+)?-\d+(?:\.\d+)?|\d+(?:\.\d+)?\+|<\d+(?:\.\d+)?)$")


def _classify_states(states: list[str]) -> str:
    """Return 'bool' | 'numeric_bins' | 'enum'."""
    s = {str(x) for x in states}
    if s == {"True", "False"}:
        return "bool"
    if all(_NUMERIC_BIN_RE.match(str(x)) for x in states):
        return "numeric_bins"
    return "enum"


@app.get("/categories")
def list_categories():
    out = []
    for public, internal in PUBLIC_TO_INTERNAL.items():
        cfg = CATEGORY_CONFIG[internal]
        schema = cfg["schema"]
        attrs_info: list[dict] = []
        bayes = (
            pipeline.validator.models.get(internal) if pipeline is not None else None
        )
        for attr in cfg["ml_attrs"]:
            entry: dict = {"name": attr, "states": None, "kind": None}
            # Источник states по приоритету: Bayes CPD → schema. Bayes даёт
            # рантайм-знание о реальных вакансиях в обученных моделях; schema
            # — статический список, нужен когда validator не загружен.
            if bayes is not None and attr in bayes.nodes():
                states = [str(s) for s in bayes.get_cpds(attr).state_names[attr]]
                entry["states"] = states
                entry["kind"] = _classify_states(states)
            elif attr in schema:
                info = schema[attr]
                t = info.get("type")
                # schema использует "bool" (короткая форма); "boolean" покрываем
                # на всякий случай.
                if t in ("bool", "boolean"):
                    entry["states"] = ["True", "False"]
                    entry["kind"] = "bool"
                elif t == "enum" and info.get("values"):
                    entry["states"] = [str(v) for v in info["values"]]
                    entry["kind"] = "enum"
            attrs_info.append(entry)
        out.append({
            "category": public,
            "internal": internal,
            "attrs": [a["name"] for a in attrs_info],
            "attr_info": attrs_info,
            "demo_cascade_supported": public in DEMO_SUPPORTED_CATEGORIES,
        })
    return out


@app.post("/api/enrich", response_model=EnrichResponse)
def enrich(req: EnrichRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Каскад ещё инициализируется")
    out = pipeline.predict(
        public_category=req.category,
        product={
            "product_name": req.product_name,
            "brands": req.brands,
            "ingredients_text": req.ingredients_text,
            "quantity": req.quantity,
        },
        validate_mode=req.validate_mode,
        expected=req.expected,
        confirmed=req.confirmed,
        fallback_on_ood=req.fallback_on_ood,
    )
    return out


@app.post("/api/explain")
def explain(req: ExplainRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Каскад ещё инициализируется")

    internal = PUBLIC_TO_INTERNAL.get(req.category)
    if internal is None:
        raise HTTPException(status_code=400, detail=f"unknown category {req.category}")
    bayes = pipeline.validator.models.get(internal)
    if bayes is None:
        raise HTTPException(status_code=503, detail="validator unavailable")
    if req.attribute not in bayes.nodes():
        raise HTTPException(
            status_code=400,
            detail=f"attribute '{req.attribute}' not in validator network for {req.category}",
        )
    inference = pipeline.validator.inferences[internal]
    evidence = dict(req.evidence)
    if req.brands:
        evidence.setdefault("brand", req.brands)

    samples = None if req.shapley_mode == "exact" else req.shapley_samples
    try:
        shap = shapley_attribution(
            attr=req.attribute,
            value=req.value,
            evidence=evidence,
            bayes_model=bayes,
            inference=inference,
            monte_carlo_samples=samples,
        )
        pmi_list = top_contributors_pmi(
            attr=req.attribute,
            value=req.value,
            evidence=evidence,
            bayes_model=bayes,
            inference=inference,
            k=len(evidence),
        )
    except ValueError as exc:
        logger.exception("explain: shapley/pmi failed")
        raise HTTPException(status_code=422, detail=str(exc))

    pmi_by_attr = {p["attr"]: p for p in pmi_list}
    attribution = []
    for a in shap["attribution"]:
        merged = {
            "evidence": f"{a['attr']}={a['value']}",
            "pmi": pmi_by_attr.get(a["attr"], {}).get("pmi"),
            "shapley": a["shapley"],
        }
        attribution.append(merged)

    return {
        "attribute": req.attribute,
        "value": req.value,
        "p_full": shap["p_full"],
        "p_marginal": shap["p_marginal"],
        "log_likelihood_diff": shap["log_likelihood_diff"],
        "attribution": attribution,
        "shapley_efficiency_check": {
            "sum_shapley": shap["sum_shapley"],
            "log_likelihood_diff": shap["log_likelihood_diff"],
            "residual": shap["efficiency_residual"],
        },
    }
