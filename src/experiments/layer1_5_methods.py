"""Layer 1.5 alternative methods: DT, Centroid, LogReg.

Each method provides a train_* and predict_* function.
Interface is intentionally symmetric with nb_layer.py so the
orchestrator (layer1_5_full_comparison.py) can treat all methods uniformly.

  train_dt(X_train_texts, y_train) -> (vec, clf)
  predict_dt(vec, clf, texts, tau) -> list[(label|None, proba)]

  train_centroid(X_train_emb, y_train) -> dict[str, np.ndarray]  (class -> centroid)
  predict_centroid(centroids, emb_rows, margin_tau) -> list[(label|None, float)]

  train_logreg(X_train_texts, y_train) -> (vec, clf)
  predict_logreg(vec, clf, texts, tau) -> list[(label|None, proba)]
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_tfidf(ngram_range=(1, 2)) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        min_df=1,
        max_features=20_000,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )


def _is_trainable(y: list[str]) -> bool:
    """Return True iff training set has >=10 samples and >=2 classes."""
    if len(y) < 10:
        return False
    if len(set(y)) < 2:
        return False
    return True


# ---------------------------------------------------------------------------
# Decision Tree
# ---------------------------------------------------------------------------

def train_dt(
    X_train_texts: list[str],
    y_train: list[str],
) -> tuple[Optional[TfidfVectorizer], Optional[DecisionTreeClassifier]]:
    """Train Decision Tree on TF-IDF features.

    Returns (None, None) if training set is too small or has <2 classes.
    """
    if not _is_trainable(y_train):
        return None, None

    vec = _build_tfidf(ngram_range=(1, 2))
    X = vec.fit_transform(X_train_texts)

    # Limit depth to avoid overfitting on small datasets
    n_classes = len(set(y_train))
    max_depth = min(8, max(3, n_classes + 2))
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X, y_train)
    logger.debug("DT trained on %d samples, %d classes, depth=%d",
                 len(y_train), n_classes, max_depth)
    return vec, clf


def predict_dt(
    vec: TfidfVectorizer,
    clf: DecisionTreeClassifier,
    texts: list[str],
    tau: float = 0.80,
) -> list[tuple[Optional[str], float]]:
    """Predict with DT; abstain when top-class proba < tau.

    Returns list of (label|None, proba).
    """
    X = vec.transform(texts)
    proba_matrix = clf.predict_proba(X)
    results = []
    for row in proba_matrix:
        top_idx = row.argmax()
        top_proba = float(row[top_idx])
        if top_proba >= tau:
            label = clf.classes_[top_idx]
            results.append((str(label), top_proba))
        else:
            results.append((None, top_proba))
    return results


# ---------------------------------------------------------------------------
# Nearest-Centroid in embedding space
# ---------------------------------------------------------------------------

def train_centroid(
    X_train_emb: np.ndarray,
    y_train: list[str],
) -> Optional[dict[str, np.ndarray]]:
    """Compute L2-normalised mean centroid per class.

    Returns dict[class_label -> unit_vector] or None if not trainable.
    """
    if not _is_trainable(y_train):
        return None

    y_arr = np.array(y_train)
    classes = sorted(set(y_train))
    centroids: dict[str, np.ndarray] = {}

    for c in classes:
        mask = y_arr == c
        vecs = X_train_emb[mask]
        mean_vec = vecs.mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 1e-9:
            mean_vec = mean_vec / norm
        centroids[c] = mean_vec

    logger.debug("Centroid trained: %d classes", len(classes))
    return centroids


def predict_centroid(
    centroids: dict[str, np.ndarray],
    emb_rows: np.ndarray,
    margin_tau: float = 0.1,
) -> list[tuple[Optional[str], float]]:
    """Predict nearest centroid (cosine sim); abstain if margin < margin_tau.

    margin = top1_sim - top2_sim (for single-class datasets always fires).
    Returns list of (label|None, top_sim).
    """
    classes = sorted(centroids.keys())
    centroid_mat = np.stack([centroids[c] for c in classes], axis=0)  # (K, D)

    # Normalise query vectors
    norms = np.linalg.norm(emb_rows, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    normed = emb_rows / norms  # (N, D)

    sims = normed @ centroid_mat.T  # (N, K)

    results = []
    for row_sims in sims:
        sorted_idx = np.argsort(row_sims)[::-1]
        top1_idx = sorted_idx[0]
        top1_sim = float(row_sims[top1_idx])
        label = classes[top1_idx]

        if len(classes) >= 2:
            top2_sim = float(row_sims[sorted_idx[1]])
            margin = top1_sim - top2_sim
        else:
            margin = 1.0  # single class — always fire

        if margin >= margin_tau:
            results.append((str(label), top1_sim))
        else:
            results.append((None, top1_sim))
    return results


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

def train_logreg(
    X_train_texts: list[str],
    y_train: list[str],
) -> tuple[Optional[TfidfVectorizer], Optional[LogisticRegression]]:
    """Train Logistic Regression on TF-IDF features.

    Returns (None, None) if training set is too small or has <2 classes.
    """
    if not _is_trainable(y_train):
        return None, None

    vec = _build_tfidf(ngram_range=(1, 2))
    X = vec.fit_transform(X_train_texts)

    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X, y_train)
    logger.debug("LogReg trained on %d samples, %d classes",
                 len(y_train), len(set(y_train)))
    return vec, clf


def predict_logreg(
    vec: TfidfVectorizer,
    clf: LogisticRegression,
    texts: list[str],
    tau: float = 0.95,
) -> list[tuple[Optional[str], float]]:
    """Predict with LogReg; abstain when top-class proba < tau.

    Returns list of (label|None, proba).
    """
    X = vec.transform(texts)
    proba_matrix = clf.predict_proba(X)
    results = []
    for row in proba_matrix:
        top_idx = row.argmax()
        top_proba = float(row[top_idx])
        if top_proba >= tau:
            label = clf.classes_[top_idx]
            results.append((str(label), top_proba))
        else:
            results.append((None, top_proba))
    return results
