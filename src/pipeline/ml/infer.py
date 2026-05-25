"""ML-layer inference: load pickled XGBoost + label encoder, predict on embeddings.

Core logic for inference based on run_experiments.py ml_layer() function.

Гибридные модели: если в MODELS_DIR присутствует {category}_tfidf.pkl, считается,
что соответствующие XGB-классификаторы обучены на признаковом пространстве
[SBERT, sparse TF-IDF] (см. флаг --with-tfidf в train.py). В этом случае
predict_batch ожидает дополнительный аргумент texts и расширяет вход hstack-ом
SBERT-эмбеддингов и TF-IDF-векторов.
"""

import os
import pickle

import numpy as np
from scipy.sparse import csr_matrix, hstack

from src.common import DEFAULT_CONFIDENCE_THRESHOLD, MODELS_DIR


def load_classifier(category: str, attribute: str):
    """Load (clf, label_encoder) for a (category, attribute) pair.

    Files expected in MODELS_DIR:
    - {category}_{attribute}_xgb.pkl
    - {category}_{attribute}_le.pkl (optional, only for multiclass)

    Returns:
        tuple: (clf, le) where le is None for binary classifiers
    """
    clf_path = os.path.join(MODELS_DIR, f"{category}_{attribute}_xgb.pkl")
    le_path = os.path.join(MODELS_DIR, f"{category}_{attribute}_le.pkl")

    with open(clf_path, "rb") as f:
        clf = pickle.load(f)

    le = None
    if os.path.exists(le_path):
        with open(le_path, "rb") as f:
            le = pickle.load(f)

    return clf, le


def load_tfidf(category: str):
    """Загружает TF-IDF vectorizer + SVD reducer для категории, если есть.

    Возвращает (vectorizer, svd_or_None, True) если файлы найдены; иначе (None, None, False).
    SVD-редьюсер опционален: если присутствует {prefix}_tfidf_svd.pkl, TF-IDF сжимается
    до dense 128-dim и конкатенируется к SBERT плотно. Иначе — старая sparse hstack-схема.
    """
    path = os.path.join(MODELS_DIR, f"{category}_tfidf.pkl")
    svd_path = os.path.join(MODELS_DIR, f"{category}_tfidf_svd.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            vec = pickle.load(f)
        svd = None
        if os.path.exists(svd_path):
            with open(svd_path, "rb") as f:
                svd = pickle.load(f)
        return vec, svd, True
    return None, None, False


def load_thresholds(category: str) -> dict:
    """Load per-attribute confidence thresholds.

    Returns:
        dict: {attribute_name: threshold_value}
    """
    path = os.path.join(MODELS_DIR, f"{category}_thresholds.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def predict_with_threshold(clf, le, embedding: np.ndarray, threshold: float):
    """Predict label only if max probability >= threshold; else return (None, conf).

    Args:
        clf: XGBoost classifier
        le: LabelEncoder (None for binary classifiers)
        embedding: 1D embedding vector
        threshold: Confidence threshold

    Returns:
        tuple: (label, confidence) or (None, confidence) if below threshold
    """
    proba = clf.predict_proba(embedding.reshape(1, -1))[0]
    max_idx = int(np.argmax(proba))
    confidence = float(proba[max_idx])

    if confidence < threshold:
        return None, confidence

    if le is not None:
        # Multiclass: use label encoder
        label = le.inverse_transform([max_idx])[0]
    else:
        # Binary: convert to boolean
        label = bool(max_idx)

    return label, confidence


def predict_batch(category: str, attribute: str, embeddings: np.ndarray,
                  threshold: float | None = None,
                  texts: list[str] | None = None) -> list:
    """Predict on batch of embeddings.

    Args:
        category: Category name (e.g., 'pasta')
        attribute: Attribute name (e.g., 'grain_type')
        embeddings: (N, 384) array of embeddings
        threshold: Confidence threshold. If None, use default or per-category threshold
        texts: Список текстов длины N. Обязателен для hybrid-моделей
            (если найден {category}_tfidf.pkl). Игнорируется для SBERT-only моделей.

    Returns:
        list: [(label, confidence), ...] or (None, confidence) for low confidence
    """
    clf, le = load_classifier(category, attribute)
    vectorizer, svd, has_tfidf = load_tfidf(category)

    if threshold is None:
        thresholds = load_thresholds(category)
        threshold = thresholds.get(attribute, DEFAULT_CONFIDENCE_THRESHOLD)

    if has_tfidf:
        if texts is None:
            raise ValueError(
                f"Категория '{category}' использует hybrid-модель (TF-IDF+SBERT); "
                "predict_batch требует аргумент texts длины embeddings."
            )
        if len(texts) != len(embeddings):
            raise ValueError(
                f"len(texts)={len(texts)} != len(embeddings)={len(embeddings)}"
            )
        X_tfidf_sparse = vectorizer.transform(texts)
        if svd is not None:
            X_tfidf_dense = svd.transform(X_tfidf_sparse).astype(np.float32)
            X_combined = np.hstack([embeddings, X_tfidf_dense])
        else:
            X_combined = hstack([csr_matrix(embeddings), X_tfidf_sparse]).tocsr()
        proba = clf.predict_proba(X_combined)
        results = []
        for row in proba:
            max_idx = int(np.argmax(row))
            conf = float(row[max_idx])
            if conf < threshold:
                results.append((None, conf))
            elif le is not None:
                results.append((le.inverse_transform([max_idx])[0], conf))
            else:
                results.append((bool(max_idx), conf))
        return results

    results = []
    for emb in embeddings:
        label, conf = predict_with_threshold(clf, le, emb, threshold)
        results.append((label, conf))

    return results
