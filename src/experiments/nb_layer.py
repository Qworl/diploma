"""NaiveBayes Layer 1.5 — data-driven token→value associations.

Trains one MultinomialNB per (cat, attr) on 80% gold codes (same split as
eval_v2_expanded.py, seed=42). Input features: bag-of-words over
product_name + " " + ingredients_text + " " + brands.

Persists:
  models/{cat}_stratified_{attr}_nb.pkl     — fitted NB classifier
  models/{cat}_stratified_{attr}_nb_vec.pkl — fitted TfidfVectorizer

Usage:
    python -m src.experiments.nb_layer [--tau 0.85]
"""
from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from src.common import MODELS_DIR, PROCESSED_DIR, setup_logging

logger = logging.getLogger(__name__)

CATEGORIES = ["pasta", "chocolate", "cheeses"]
SEED = 42
TEST_SIZE = 0.2

GOLD_PATH = Path(PROCESSED_DIR) / "consensus_gold_v2_expanded.parquet"


def _build_text(row: pd.Series) -> str:
    """Build text feature from partner-available fields."""
    parts = []
    for col in ["product_name", "ingredients_text", "brands", "quantity"]:
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    return " ".join(parts)


def train_nb_for_attr(
    cat: str,
    attr: str,
    gold: pd.DataFrame,
    silver: pd.DataFrame,
    train_codes: list[str],
) -> tuple[MultinomialNB | None, TfidfVectorizer | None]:
    """Train NB classifier for (cat, attr) on train_codes gold subset.

    Returns (None, None) if not enough data or <2 classes.
    """
    # Filter gold to train codes, non-null
    train_set = set(str(c) for c in train_codes)
    attr_gold = gold[
        (gold["category"] == cat)
        & (gold["attr"] == attr)
        & ~gold["gold_is_null"]
        & gold["code"].astype(str).isin(train_set)
    ].copy()
    attr_gold["code"] = attr_gold["code"].astype(str)

    if len(attr_gold) < 10:
        logger.debug("[%s/%s] only %d gold train rows — skip NB", cat, attr, len(attr_gold))
        return None, None

    n_classes = attr_gold["gold_value"].nunique()
    if n_classes < 2:
        logger.debug("[%s/%s] only %d unique classes — skip NB", cat, attr, n_classes)
        return None, None

    # Join product text from silver
    silver_sub = silver[["code", "product_name", "ingredients_text", "brands", "quantity"]].copy()
    silver_sub["code"] = silver_sub["code"].astype(str)
    merged = attr_gold.merge(silver_sub, on="code", how="left")

    if merged["product_name"].isna().all():
        logger.warning("[%s/%s] no product text found in silver — skip NB", cat, attr)
        return None, None

    merged["_text"] = merged.apply(_build_text, axis=1)
    X_raw = merged["_text"].fillna("").tolist()
    y = merged["gold_value"].astype(str).tolist()

    # TF-IDF with unigrams + bigrams, sublinear scaling
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=20_000,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    X = vec.fit_transform(X_raw)

    # MultinomialNB requires non-negative features — TF-IDF is always >= 0 so OK
    nb = MultinomialNB(alpha=0.5)
    nb.fit(X, y)
    logger.debug("[%s/%s] NB trained on %d samples, %d classes", cat, attr, len(y), n_classes)
    return nb, vec


def save_nb_model(nb: MultinomialNB, vec: TfidfVectorizer, cat: str, attr: str) -> None:
    """Persist NB + vectorizer to models/."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    base = os.path.join(MODELS_DIR, f"{cat}_stratified_{attr}")
    with open(base + "_nb.pkl", "wb") as f:
        pickle.dump(nb, f)
    with open(base + "_nb_vec.pkl", "wb") as f:
        pickle.dump(vec, f)
    logger.debug("[%s/%s] NB model saved", cat, attr)


def load_nb_model(
    cat: str, attr: str
) -> tuple[MultinomialNB | None, TfidfVectorizer | None]:
    """Load NB + vectorizer from models/. Returns (None, None) if not found."""
    base = os.path.join(MODELS_DIR, f"{cat}_stratified_{attr}")
    nb_path = base + "_nb.pkl"
    vec_path = base + "_nb_vec.pkl"
    if not (os.path.exists(nb_path) and os.path.exists(vec_path)):
        return None, None
    with open(nb_path, "rb") as f:
        nb = pickle.load(f)
    with open(vec_path, "rb") as f:
        vec = pickle.load(f)
    return nb, vec


def predict_nb(
    nb: MultinomialNB,
    vec: TfidfVectorizer,
    texts: list[str],
    tau: float = 0.85,
) -> list[tuple[str | None, float]]:
    """Predict class with probability. Returns (label, proba) per text.

    If max proba < tau → returns (None, proba) meaning NB abstains.
    """
    X = vec.transform(texts)
    proba_matrix = nb.predict_proba(X)
    results = []
    for row in proba_matrix:
        top_idx = row.argmax()
        top_proba = float(row[top_idx])
        if top_proba >= tau:
            label = nb.classes_[top_idx]
            results.append((str(label), top_proba))
        else:
            results.append((None, top_proba))
    return results


def train_all(gold_path: Path = GOLD_PATH) -> None:
    """Train and save NB models for all (cat, attr) pairs."""
    setup_logging()
    gold = pd.read_parquet(gold_path)
    gold["code"] = gold["code"].astype(str)

    for cat in CATEGORIES:
        logger.info("=== Category: %s ===", cat)
        silver = pd.read_parquet(
            Path(PROCESSED_DIR) / f"{cat}_stratified_silver_standard.parquet"
        )
        silver["code"] = silver["code"].astype(str)

        cat_gold = gold[gold["category"] == cat].copy()
        unique_codes = sorted(cat_gold["code"].unique().tolist())

        # Same 80/20 split as regex_ablation and eval_v2_expanded
        train_codes, test_codes = train_test_split(
            unique_codes, test_size=TEST_SIZE, random_state=SEED
        )
        logger.info("  %s: %d train codes, %d test codes (NB trained on train only)",
                    cat, len(train_codes), len(test_codes))

        attrs = sorted(cat_gold["attr"].unique().tolist())
        for attr in attrs:
            nb, vec = train_nb_for_attr(cat, attr, cat_gold, silver, train_codes)
            if nb is not None and vec is not None:
                save_nb_model(nb, vec, cat, attr)
                logger.info("  [%s] NB saved", attr)
            else:
                logger.info("  [%s] NB skipped (insufficient data)", attr)


def print_top_tokens(
    cat: str,
    attr: str,
    n_top: int = 5,
) -> dict[str, list[tuple[str, float]]]:
    """Return top n_top tokens per class for a (cat, attr) NB model.

    Used for interpretability comparison with hand-crafted regex.
    """
    nb, vec = load_nb_model(cat, attr)
    if nb is None:
        return {}

    feature_names = vec.get_feature_names_out()
    results: dict[str, list[tuple[str, float]]] = {}

    for i, class_label in enumerate(nb.classes_):
        # log_prob for this class per feature; higher = more discriminative
        log_probs = nb.feature_log_prob_[i]
        top_indices = log_probs.argsort()[::-1][:n_top]
        results[str(class_label)] = [
            (feature_names[idx], float(log_probs[idx]))
            for idx in top_indices
        ]

    return results


if __name__ == "__main__":
    train_all()
