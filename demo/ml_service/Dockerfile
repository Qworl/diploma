# ML-сервис: FastAPI поверх каскада regex → ML(XGBoost+SBERT) → Bayes.
# Build context — корень репо (нужен доступ к src/ + demo/ + requirements.txt).
#
#   docker build -f demo/ml_service/Dockerfile -t ai-attrs-ml .
#
# Sentence-Transformers (~1 GB) и обученные модели прибиваются volume'ами,
# не вшиваются в image. См. demo/docker-compose.yml.

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/hf-cache

# libgomp1 для xgboost; curl для healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Корневой requirements.txt + demo deps. Слой кэшируется до изменения lock'ов.
COPY requirements.txt /app/requirements.txt
COPY demo/ml_service/requirements.txt /app/demo/ml_service/requirements.txt
RUN pip install -r /app/requirements.txt \
    && pip install -r /app/demo/ml_service/requirements.txt

# Исходники проекта. src/ нужен потому что demo/ml_service/cascade.py его импортирует.
COPY src /app/src
COPY demo/ml_service /app/demo/ml_service

# /app/models и /hf-cache подмонтированы извне (см. docker-compose.yml).
VOLUME ["/app/models", "/hf-cache"]

EXPOSE 8001

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD curl -f http://127.0.0.1:8001/health || exit 1

WORKDIR /app/demo/ml_service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
