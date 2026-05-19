"""Single source of truth for category-router file paths and class set."""
from __future__ import annotations

import os

from src.common import MODELS_DIR, PROCESSED_DIR, PARTNER_TEXT_FIELDS

# 7 known классов — детерминированный порядок (используется в LabelEncoder).
ROUTER_CLASSES: tuple[str, ...] = (
    "pasta", "chocolate", "beverages",
    "cheeses", "cereals", "cosmetics", "electronics",
)

# Категории, для которых в demo/ml_service/cascade.py подключён полный
# per-category каскад. Manual override через API доступен только для них.
# Scope ВКР после §6.20: 3 audit-grade категории.
DEMO_SUPPORTED_CATEGORIES: frozenset[str] = frozenset(
    {"pasta", "chocolate", "cheeses"}
)

# Re-export as immutable tuple under the router-namespace name.
ROUTER_INPUT_FIELDS: tuple[str, ...] = tuple(PARTNER_TEXT_FIELDS)

ARTIFACT_XGB = os.path.join(MODELS_DIR, "category_router_xgb.pkl")
ARTIFACT_LE = os.path.join(MODELS_DIR, "category_router_le.pkl")
ARTIFACT_THRESHOLD = os.path.join(MODELS_DIR, "category_router_threshold.json")
ARTIFACT_META = os.path.join(MODELS_DIR, "category_router_meta.json")
ARTIFACT_LOCO = os.path.join(MODELS_DIR, "category_router_loco.parquet")

TRAIN_PARQUET = os.path.join(PROCESSED_DIR, "category_router_train.parquet")
TEST_PARQUET = os.path.join(PROCESSED_DIR, "category_router_test.parquet")
EMBEDDINGS_NPY = os.path.join(PROCESSED_DIR, "category_router_embeddings.npy")
