"""Shared utilities for all pipeline scripts."""

import logging
import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROCESSED_DIR = os.path.join(PROJECT_DIR, "datasets", "processed")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
RAW_DIR = os.path.join(PROJECT_DIR, "datasets", "raw")

# Train/test split params — shared across train_classifiers and run_experiments
TEST_SIZE = 0.2
RANDOM_STATE = 42
DEFAULT_CONFIDENCE_THRESHOLD = 0.7

EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
PARTNER_TEXT_FIELDS = ["product_name", "brands", "ingredients_text", "quantity"]

# 3 main categories для текущей итерации fix-cycle.
# Остальные 4 (beverages, cereals, cosmetics, electronics) — следующая итерация.
MAIN_CATEGORIES = ["pasta", "chocolate", "cheeses"]
ALL_CATEGORIES = ["pasta", "chocolate", "beverages", "cheeses", "cereals", "cosmetics"]


def wilson_ci(n_correct: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval для биномиальной пропорции (95% по умолчанию).

    Лучше чем normal approx на малых n. Возвращает (lower, upper) ∈ [0, 1].
    Формула: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    """
    if n_total == 0:
        return (0.0, 1.0)
    p = n_correct / n_total
    z2 = z * z
    denom = 1.0 + z2 / n_total
    centre = (p + z2 / (2.0 * n_total)) / denom
    margin = (z * np.sqrt(p * (1.0 - p) / n_total + z2 / (4.0 * n_total ** 2))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def build_text(df: pd.DataFrame) -> list[str]:
    """Combine partner-available text fields into strings for embedding.

    Only uses fields that would come from a partner:
    product_name, brands, ingredients_text, quantity.
    Does NOT use categories_tags, labels_tags (those are enrichment targets).
    """
    texts = []
    for _, row in df.iterrows():
        parts = []
        for col in PARTNER_TEXT_FIELDS:
            val = row.get(col)
            if pd.notna(val):
                parts.append(str(val))
        texts.append(" ".join(parts))
    return texts


def get_embeddings(texts: list[str], cache_path: str | None = None) -> np.ndarray:
    """Compute or load cached multilingual embeddings."""
    logger = logging.getLogger(__name__)
    if cache_path and os.path.exists(cache_path):
        logger.info("Loading cached embeddings from %s", cache_path)
        return np.load(cache_path)

    logger.info("Computing embeddings for %d texts...", len(texts))
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, embeddings)
        logger.info("Cached embeddings to %s", cache_path)

    return embeddings
