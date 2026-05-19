"""Hybrid features experiment for chocolate ML layer.

Probe (TF-IDF + LogReg) опережает каскад на парах
`chocolate / chocolate_type`, `chocolate / chocolate_extra`, `chocolate / contains_nuts`
(см. §3.3.7.2 ВКР, артефакт `off_leakage_probe.parquet`). Гипотеза §5.5 п.2:
причина в недоиспользовании лексических n-грамм текущим Layer 2 (SBERT+XGBoost).

Скрипт сравнивает четыре конфигурации Layer 2 на тех же train/test (brand-disjoint,
test_codes из `cascade_preds_chocolate_v2_gold_hybrid_v3_fixed.parquet`):

  A. baseline: XGBoost на SBERT-эмбеддингах (текущая конфигурация Layer 2).
  B. hybrid: XGBoost на [SBERT, sparse TF-IDF (1,2-граммы, top-5000)].
  C. probe: LogisticRegression на TF-IDF (реплицирует off_leakage_probe).
  D. xgb_tfidf: XGBoost на TF-IDF без SBERT (контроль).

Каждая конфигурация оценивается дважды:
  - vs silver labels на test_codes (сравнимо с probe из off_leakage_probe.parquet)
  - vs gold labels на test_codes (consensus_gold_v2_expanded.parquet) — реальная правильность

Результаты сохраняются в datasets/processed/chocolate_hybrid_features.parquet.

Usage:
    OMP_NUM_THREADS=1 python -m src.diagnostics.ml.chocolate_hybrid_features
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.common import PARTNER_TEXT_FIELDS, PROCESSED_DIR, setup_logging

logger = logging.getLogger("chocolate_hybrid_features")

CATEGORY = "chocolate"
ATTRS = ["chocolate_type", "chocolate_extra", "contains_nuts"]
SILVER_PATH = os.path.join(PROCESSED_DIR, f"{CATEGORY}_stratified_silver_standard.parquet")
EMBED_PATH = os.path.join(PROCESSED_DIR, f"{CATEGORY}_stratified_embeddings.npy")
CASCADE_PATH = os.path.join(
    PROCESSED_DIR, f"cascade_preds_{CATEGORY}_v2_gold_hybrid_v3_fixed.parquet"
)
GOLD_PATH = os.path.join(PROCESSED_DIR, "consensus_gold_v2_expanded.parquet")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "chocolate_hybrid_features.parquet")

TFIDF_KWARGS = dict(max_features=5000, ngram_range=(1, 2), lowercase=True)


def _build_text(df: pd.DataFrame) -> list[str]:
    parts = []
    for col in PARTNER_TEXT_FIELDS:
        if col in df.columns:
            parts.append(df[col].astype("string").fillna(""))
        else:
            parts.append(pd.Series([""] * len(df), index=df.index))
    return parts[0].str.cat(parts[1:], sep=" ", na_rep="").fillna("").tolist()


def _xgb_multiclass(X_train, y_train, n_classes: int) -> XGBClassifier:
    clf = XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="mlogloss", n_jobs=1,
    )
    sw = compute_sample_weight("balanced", y_train)
    clf.fit(X_train, y_train, sample_weight=sw, verbose=False)
    return clf


def _xgb_binary(X_train, y_train) -> XGBClassifier:
    n_pos = max(int((y_train == 1).sum()), 1)
    n_neg = int((y_train == 0).sum())
    clf = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="logloss", n_jobs=1,
        scale_pos_weight=n_neg / n_pos,
    )
    clf.fit(X_train, y_train, verbose=False)
    return clf


def _train_and_eval(
    attr: str,
    is_binary: bool,
    silver_train: pd.DataFrame,
    silver_test: pd.DataFrame,
    gold_test_map: dict,
    emb_train: np.ndarray,
    emb_test: np.ndarray,
) -> list[dict]:
    """Train four configurations and report accuracy on silver and gold test."""
    train_text = _build_text(silver_train)
    test_text = _build_text(silver_test)

    vec = TfidfVectorizer(**TFIDF_KWARGS)
    X_tfidf_train = vec.fit_transform(train_text)
    X_tfidf_test = vec.transform(test_text)
    logger.info("  TF-IDF: train shape=%s, test shape=%s, vocab=%d",
                X_tfidf_train.shape, X_tfidf_test.shape, len(vec.vocabulary_))

    X_sbert_train_sp = csr_matrix(emb_train)
    X_sbert_test_sp = csr_matrix(emb_test)

    X_hybrid_train = hstack([X_sbert_train_sp, X_tfidf_train]).tocsr()
    X_hybrid_test = hstack([X_sbert_test_sp, X_tfidf_test]).tocsr()

    y_train_raw = silver_train[attr].astype(str).to_numpy()
    y_test_silver_raw = silver_test[attr].astype(str).to_numpy()
    codes_test = silver_test["code"].astype(str).to_numpy()

    le = LabelEncoder()
    le.fit(y_train_raw)
    y_train = le.transform(y_train_raw)
    train_classes = set(le.classes_.tolist())

    rows = []

    def _evaluate(name: str, preds_raw: np.ndarray):
        # vs silver
        sil_seen = np.array([y in train_classes for y in y_test_silver_raw])
        sil_correct = int(((preds_raw == y_test_silver_raw) & sil_seen).sum())
        sil_n = int(sil_seen.sum())
        sil_acc = sil_correct / sil_n if sil_n else float("nan")

        # vs gold
        gold_pairs = [(p, gold_test_map.get(c)) for p, c in zip(preds_raw, codes_test)]
        gold_pairs = [(p, g) for p, g in gold_pairs if g is not None]
        # gold уже как str; для contains_nuts может быть True/False — приводим
        gold_pairs = [(str(p).lower(), str(g).lower()) for p, g in gold_pairs]
        gold_n = len(gold_pairs)
        gold_correct = sum(1 for p, g in gold_pairs if p == g)
        gold_acc = gold_correct / gold_n if gold_n else float("nan")

        rows.append({
            "attr": attr,
            "config": name,
            "n_test_silver": sil_n,
            "silver_acc": sil_acc,
            "n_test_gold": gold_n,
            "gold_acc": gold_acc,
        })
        logger.info(
            "    [%s] silver_acc=%.4f (n=%d), gold_acc=%.4f (n=%d)",
            name, sil_acc, sil_n, gold_acc, gold_n,
        )

    # A. XGBoost + SBERT (baseline)
    if is_binary:
        clf = _xgb_binary(emb_train, y_train)
        preds_a = le.inverse_transform(clf.predict(emb_test).astype(int))
    else:
        clf = _xgb_multiclass(emb_train, y_train, n_classes=len(le.classes_))
        preds_a = le.inverse_transform(clf.predict(emb_test))
    _evaluate("xgb_sbert", preds_a)

    # B. XGBoost + SBERT + TF-IDF (hybrid)
    if is_binary:
        clf = _xgb_binary(X_hybrid_train, y_train)
        preds_b = le.inverse_transform(clf.predict(X_hybrid_test).astype(int))
    else:
        clf = _xgb_multiclass(X_hybrid_train, y_train, n_classes=len(le.classes_))
        preds_b = le.inverse_transform(clf.predict(X_hybrid_test))
    _evaluate("xgb_sbert_tfidf", preds_b)

    # C. LogReg + TF-IDF (probe replica)
    clf = LogisticRegression(max_iter=1000, n_jobs=1)
    clf.fit(X_tfidf_train, y_train)
    preds_c = le.inverse_transform(clf.predict(X_tfidf_test))
    _evaluate("logreg_tfidf", preds_c)

    # D. XGBoost + TF-IDF only (control)
    if is_binary:
        clf = _xgb_binary(X_tfidf_train, y_train)
        preds_d = le.inverse_transform(clf.predict(X_tfidf_test).astype(int))
    else:
        clf = _xgb_multiclass(X_tfidf_train, y_train, n_classes=len(le.classes_))
        preds_d = le.inverse_transform(clf.predict(X_tfidf_test))
    _evaluate("xgb_tfidf", preds_d)

    return rows


def main() -> None:
    setup_logging()

    silver = pd.read_parquet(SILVER_PATH)
    silver["code"] = silver["code"].astype(str)
    emb = np.load(EMBED_PATH)
    assert len(silver) == len(emb), "silver/embeddings misaligned"
    logger.info("Silver: %d rows, embeddings: %s", len(silver), emb.shape)

    cascade = pd.read_parquet(CASCADE_PATH)
    cascade["code"] = cascade["code"].astype(str)
    test_codes = set(cascade["code"].unique())
    logger.info("Test codes (brand-disjoint): %d", len(test_codes))

    gold = pd.read_parquet(GOLD_PATH)
    gold["code"] = gold["code"].astype(str)
    gold = gold[(gold.category == CATEGORY) & (~gold.gold_is_null)]

    silver["_pos"] = np.arange(len(silver))
    is_test_mask = silver["code"].isin(test_codes)
    silver_train_all = silver[~is_test_mask]
    silver_test_all = silver[is_test_mask]
    logger.info("Train: %d, Test: %d", len(silver_train_all), len(silver_test_all))

    all_rows = []
    for attr in ATTRS:
        if attr not in silver.columns:
            logger.warning("attr=%s missing in silver, skipping", attr)
            continue
        train_mask = silver_train_all[attr].notna()
        test_mask = silver_test_all[attr].notna()
        s_train = silver_train_all[train_mask].copy()
        s_test = silver_test_all[test_mask].copy()
        emb_train = emb[s_train["_pos"].values]
        emb_test = emb[s_test["_pos"].values]

        gold_attr = gold[gold.attr == attr]
        gold_map = dict(zip(gold_attr["code"], gold_attr["gold_value"].astype(str)))

        logger.info("=== %s ===  n_train=%d  n_test_silver=%d  gold_codes=%d",
                    attr, len(s_train), len(s_test), len(gold_map))
        is_binary = attr == "contains_nuts"
        rows = _train_and_eval(
            attr, is_binary, s_train, s_test, gold_map, emb_train, emb_test,
        )
        all_rows.extend(rows)

    out = pd.DataFrame(all_rows)
    out.to_parquet(OUTPUT_PATH, index=False)
    logger.info("Saved %d rows to %s", len(out), OUTPUT_PATH)

    print()
    print("=" * 78)
    print("HYBRID FEATURES SUMMARY — chocolate Layer 2 reconfigurations")
    print("=" * 78)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
