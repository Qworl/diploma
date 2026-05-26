"""Empirical comparison of Layer 2 classifiers (XGBoost vs alternatives).

Backing experiment for advisor comments #1-2 (`docs/po/tickets/2026-05-26-advisor-comment-rf.md`)
— why XGBoost was chosen over Random Forest / Logistic Regression / MLP.

Data sources (note: project pivoted from `*_gold_v4_wide.parquet` to silver_standard
+ consensus_hybrid_v3 ground truth after v4 wide artifacts were dropped in cleanup
commit 07fcd04. We use the closest available analog):

- Inputs (product_name, brands, ingredients_text, quantity):
    datasets/processed/{cat}_stratified_silver_standard.parquet
- Train/val/test brand-disjoint code-level split:
    datasets/processed/{cat}_gold_split.parquet (60/20/20)
- Labels (training and eval):
    datasets/processed/consensus_hybrid_v3.parquet — 3-LLM consensus
    (qwen3.7-max + deepseek-r1 + mistral-large-2411). This is the same
    "hybrid v3" mentioned in commit message bf4e37f (60k+ cells, $40 LLM cost,
    91.7% accuracy vs silver 88.6%). Codes split by `*_gold_split.parquet`:
        train+val codes → training labels
        test codes      → consensus eval labels
- Conservative eval (human gold): datasets/processed/manual_eval_per_product.parquet
    — human Opus labels, ~50 products per attr-cat. Codes excluded from training.

Feature pipeline = production noleak config:
    MPNet (768d, paraphrase-multilingual-mpnet-base-v2)
    ⊕ TF-IDF(max_features=5000, ngram=(1,2)) → TruncatedSVD(128)
    = 896-dim concatenated vector.

Classifiers compared (all on identical features and splits):
    1. XGBoost (production baseline) — isotonic calibration
    2. RandomForest — Platt calibration
    3. LogisticRegression — Platt calibration (built into CalibratedClassifierCV)
    4. MLPClassifier — native softmax probabilities

Metrics per (category, attribute, classifier, eval_source):
    - micro_accuracy
    - macro_f1
    - balanced_accuracy
    - ECE (10-bin, top-1 confidence vs correctness)
    - runtime_seconds (fit + predict_proba on eval)

Output:
    datasets/processed/method_comparison_results.parquet

Run:
    OMP_NUM_THREADS=1 python scripts/method_comparison.py            # all
    OMP_NUM_THREADS=1 python scripts/method_comparison.py --smoke    # 1 attr only
    OMP_NUM_THREADS=1 python scripts/method_comparison.py --category pasta
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Force OMP_NUM_THREADS=1 (libomp/torch segfault risk on macOS)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
RESULTS_PATH = PROCESSED_DIR / "method_comparison_results.parquet"
LOG_PATH = PROCESSED_DIR / "method_comparison.log"

# Line-buffered file log for incremental tail -f
file_handler = logging.FileHandler(LOG_PATH, mode="a")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
)
file_handler.setLevel(logging.INFO)
try:
    file_handler.stream.reconfigure(line_buffering=True)
except Exception:
    pass
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
)
logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler],
                    force=True)
logger = logging.getLogger("method_comparison")

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Schema: per category, which attrs are ML-eligible (deterministic TYPE_C skipped)
# ---------------------------------------------------------------------------
ATTR_CONFIG = {
    "pasta": {
        "grain_type": "multiclass",
        "pasta_shape": "multiclass",
        "is_filled": "binary",
        "is_gluten_free": "binary",
        "is_organic": "binary",
        "is_vegan": "binary",
    },
    "chocolate": {
        "chocolate_type": "multiclass",
        "chocolate_extra": "multiclass",
        "contains_nuts": "binary",
        "is_organic": "binary",
    },
    "cheeses": {
        "milk_source": "multiclass",
        "texture": "multiclass",
        "country_of_origin": "multiclass",
        "is_pdo": "binary",
        "is_organic": "binary",
        "is_ultra_processed": "binary",
    },
}

# Classes that should be excluded from train+eval (dead/catch-all, §5.2 methodology)
SCHEMA_EXCLUDE = {
    "chocolate_extra": {"filled", "other", "with_alcohol", "with_coffee"},
    "chocolate_type": {"filled", "other"},
    "texture": {"other"},
}

ALL_METHODS = ["xgboost", "random_forest", "logreg", "mlp"]


# ---------------------------------------------------------------------------
# Text + features
# ---------------------------------------------------------------------------
PARTNER_TEXT_FIELDS = ["product_name", "brands", "ingredients_text", "quantity"]


def build_text(df: pd.DataFrame) -> list[str]:
    """Combine partner-available fields → text per row."""
    parts_list = []
    for col in PARTNER_TEXT_FIELDS:
        if col in df.columns:
            parts_list.append(df[col].fillna("").astype(str).values)
        else:
            parts_list.append(np.array([""] * len(df)))
    n = len(df)
    return [" ".join(p[i] for p in parts_list).strip() for i in range(n)]


def get_or_compute_mpnet(texts: list[str], cache_path: Path) -> np.ndarray:
    """Load cached MPNet embeddings or compute with MPS/CPU."""
    if cache_path.exists():
        emb = np.load(cache_path)
        if emb.shape[0] == len(texts):
            logger.info("Loaded cached MPNet embeddings from %s (shape=%s)",
                        cache_path.name, emb.shape)
            return emb
        logger.warning("Cache size mismatch (%d vs %d) — recomputing",
                       emb.shape[0], len(texts))

    logger.info("Computing MPNet embeddings for %d texts (this may take a while)...",
                len(texts))
    from sentence_transformers import SentenceTransformer
    import torch

    device = "mps" if torch.backends.mps.is_available() else \
             ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("  using device: %s", device)
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device=device)
    emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                       convert_to_numpy=True)
    emb = emb.astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, emb)
    logger.info("  cached MPNet embeddings → %s (shape=%s)", cache_path.name, emb.shape)
    return emb


def compute_tfidf_svd(texts_train: list[str], texts_eval: list[str],
                      n_components: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Fit TF-IDF + TruncatedSVD on train texts; transform train + eval."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tf = vec.fit_transform(texts_train)
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    X_train_svd = svd.fit_transform(X_train_tf)

    X_eval_tf = vec.transform(texts_eval)
    X_eval_svd = svd.transform(X_eval_tf)
    return X_train_svd.astype(np.float32), X_eval_svd.astype(np.float32)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_cat_data(cat: str) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Returns: silver_df (with split), mpnet_embeddings, consensus_gold, human_gold."""
    silver_path = PROCESSED_DIR / f"{cat}_stratified_silver_standard.parquet"
    split_path = PROCESSED_DIR / f"{cat}_gold_split.parquet"

    df = pd.read_parquet(silver_path)
    df["code"] = df["code"].astype(str)
    df = df.reset_index(drop=True)

    split_df = pd.read_parquet(split_path)
    split_df["code"] = split_df["code"].astype(str)
    df = df.merge(split_df, on="code", how="left")
    # Some silver rows may not be in split (rare drift) — drop them
    n_no_split = df["split"].isna().sum()
    if n_no_split > 0:
        logger.warning("%s: %d rows without split assignment — dropping", cat, n_no_split)
        df = df[df["split"].notna()].reset_index(drop=True)

    # MPNet embeddings — cache per category
    emb_cache = PROCESSED_DIR / f"{cat}_v4_embeddings.npy"
    texts = build_text(df)
    emb = get_or_compute_mpnet(texts, emb_cache)
    assert len(emb) == len(df), f"emb {emb.shape} != df {len(df)}"

    # Consensus gold
    gold = pd.read_parquet(PROCESSED_DIR / "consensus_hybrid_v3.parquet")
    gold["code"] = gold["code"].astype(str)
    gold_cat = gold[gold["category"] == cat].copy()

    # Human gold (manual_eval_per_product)
    human = pd.read_parquet(PROCESSED_DIR / "manual_eval_per_product.parquet")
    human["code"] = human["code"].astype(str)
    human_cat = human[human["category"] == cat].copy()

    return df, emb, gold_cat, human_cat


# ---------------------------------------------------------------------------
# Classifier factory
# ---------------------------------------------------------------------------
def make_classifier(method: str, n_classes: int, attr_type: str):
    """Return a fresh classifier + whether/how it needs calibration."""
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    if method == "xgboost":
        # Production-style parameters; binary uses different (smaller) config.
        # early_stopping_rounds drastically reduces fit time on easier attrs
        # while keeping the same architecture comparison (other methods fit to
        # convergence too).
        if attr_type == "binary":
            clf = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
                early_stopping_rounds=20,
                n_jobs=int(os.environ.get("XGB_N_JOBS", "2")),
                random_state=RANDOM_STATE, eval_metric="logloss",
            )
        else:
            clf = XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
                early_stopping_rounds=20,
                n_jobs=int(os.environ.get("XGB_N_JOBS", "2")),
                random_state=RANDOM_STATE, eval_metric="mlogloss",
            )
        # XGB recommends isotonic for ≥1000 samples, but we play safe with sigmoid
        # (Platt) since some attrs have small train sets after filtering.
        return clf, "sigmoid"

    if method == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_split=4,
            n_jobs=2, class_weight="balanced", random_state=RANDOM_STATE,
        )
        return clf, "sigmoid"

    if method == "logreg":
        clf = LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=2,
        )
        return clf, "sigmoid"

    if method == "mlp":
        # Smaller MLP if many classes (memory) — fixed for fairness
        clf = MLPClassifier(
            hidden_layer_sizes=(256, 128), max_iter=100, early_stopping=True,
            validation_fraction=0.15, random_state=RANDOM_STATE,
        )
        return clf, None  # MLP outputs softmax natively, skip calibration

    raise ValueError(f"unknown method: {method}")


def calibrate(clf, X_calib, y_calib, method: str | None):
    """Wrap a fitted classifier in CalibratedClassifierCV using held-out data."""
    if method is None or len(y_calib) < 30:
        return clf, False
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    unique, counts = np.unique(y_calib, return_counts=True)
    cv = min(3, int(counts.min()))
    if cv < 2:
        return clf, False
    try:
        cal = CalibratedClassifierCV(FrozenEstimator(clf), method=method, cv=cv)
        cal.fit(X_calib, y_calib)
        return cal, True
    except (ValueError, IndexError) as e:
        logger.warning("  calibration failed (%s) — returning raw clf", str(e)[:80])
        return clf, False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_ece(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (max-prob style)."""
    confidences = proba.max(axis=1)
    preds = proba.argmax(axis=1)
    correct = (preds == y_true).astype(float)
    n = len(y_true)
    if n == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        bin_acc = float(correct[mask].mean())
        bin_conf = float(confidences[mask].mean())
        ece += (cnt / n) * abs(bin_acc - bin_conf)
    return float(ece)


def metrics_block(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> dict:
    """All metrics from probas + true labels (encoded)."""
    from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score)
    pred = proba.argmax(axis=1)
    return dict(
        micro_acc=float(accuracy_score(y_true, pred)),
        macro_f1=float(f1_score(y_true, pred, average="macro", zero_division=0)),
        balanced_acc=float(balanced_accuracy_score(y_true, pred)),
        ece=compute_ece(y_true, proba),
    )


# ---------------------------------------------------------------------------
# Per-attribute experiment runner
# ---------------------------------------------------------------------------
def _filter_min_class(values: pd.Series, min_count: int = 5) -> set:
    counts = values.value_counts()
    return set(counts[counts >= min_count].index)


def _norm_label(v, attr_type: str):
    if pd.isna(v):
        return None
    if attr_type == "binary":
        if v in (True, "True", "true", 1, "1"):
            return "True"
        if v in (False, "False", "false", 0, "0"):
            return "False"
        # fallback: str() then map
        s = str(v).strip().lower()
        if s in ("true", "1", "yes"):
            return "True"
        if s in ("false", "0", "no"):
            return "False"
        return None
    return str(v).strip()


def build_train_eval(cat: str, attr: str, attr_type: str,
                     silver_df: pd.DataFrame, emb: np.ndarray,
                     gold_cat: pd.DataFrame, human_cat: pd.DataFrame
                     ) -> dict | None:
    """Assemble train and eval matrices for one (cat, attr).

    Labels: from consensus_hybrid_v3 (the LLM-consensus). Split: by brand-disjoint
    {cat}_gold_split.parquet — train+val codes go to training, test codes go to
    consensus eval. Human gold codes are EXCLUDED from training to ensure
    code-disjoint test-time evaluation on manual_eval_per_product.
    """
    code_to_idx = {c: i for i, c in enumerate(silver_df["code"].values)}

    # ------- Consensus labels (training source + consensus eval) ------------
    gold_attr = gold_cat[gold_cat["attr"] == attr].copy()
    # Drop explicit null markers
    gold_attr = gold_attr[gold_attr["gold_value"].notna() &
                          (~gold_attr["gold_is_null"].fillna(False))]
    gold_attr["label"] = gold_attr["gold_value"].apply(lambda v: _norm_label(v, attr_type))
    gold_attr = gold_attr[gold_attr["label"].notna()]
    exclude = SCHEMA_EXCLUDE.get(attr, set())
    if exclude:
        gold_attr = gold_attr[~gold_attr["label"].isin(exclude)]
    # Restrict to codes for which we have inputs/embeddings
    gold_attr = gold_attr[gold_attr["code"].isin(code_to_idx.keys())].copy()
    if len(gold_attr) < 50:
        logger.warning("  %s/%s: only %d gold rows, skip", cat, attr, len(gold_attr))
        return None

    # Attach split info
    split_map = dict(zip(silver_df["code"].values, silver_df["split"].values))
    gold_attr["split"] = gold_attr["code"].map(split_map)
    gold_attr = gold_attr[gold_attr["split"].notna()]

    # Determine valid classes by TRAIN count (≥5 in train split)
    train_counts = gold_attr.loc[gold_attr["split"] == "train", "label"].value_counts()
    valid_classes = sorted([c for c, n in train_counts.items() if n >= 5])
    if attr_type == "binary":
        # Binary needs both classes
        if "True" not in valid_classes or "False" not in valid_classes:
            logger.warning("  %s/%s: binary missing class — valid=%s, skip",
                           cat, attr, valid_classes)
            return None
    if len(valid_classes) < 2:
        logger.warning("  %s/%s: only %d valid classes, skip", cat, attr, len(valid_classes))
        return None
    gold_attr = gold_attr[gold_attr["label"].isin(valid_classes)].reset_index(drop=True)

    # ------- Human gold codes (must be excluded from train) -----------------
    human_attr = human_cat[human_cat["attr"] == attr].copy()
    human_attr = human_attr[human_attr["manual"].notna()]
    human_attr["label"] = human_attr["manual"].apply(lambda v: _norm_label(v, attr_type))
    human_attr = human_attr[human_attr["label"].notna()]
    if exclude:
        human_attr = human_attr[~human_attr["label"].isin(exclude)]
    human_attr = human_attr[human_attr["label"].isin(valid_classes)]
    human_attr = human_attr[human_attr["code"].isin(code_to_idx.keys())]
    human_codes_set = set(human_attr["code"])

    # ------- Build train, consensus eval, human eval ------------------------
    train_pool = gold_attr[gold_attr["split"].isin(["train", "val"]) &
                           ~gold_attr["code"].isin(human_codes_set)].copy()
    test_pool = gold_attr[gold_attr["split"] == "test"].copy()
    if len(train_pool) < 50 or len(test_pool) < 5:
        logger.warning("  %s/%s: small pools train=%d test=%d, skip",
                       cat, attr, len(train_pool), len(test_pool))
        return None

    train_pool["_emb_idx"] = train_pool["code"].map(code_to_idx).astype(int)
    test_pool["_emb_idx"] = test_pool["code"].map(code_to_idx).astype(int)

    train_idx = train_pool["_emb_idx"].values
    train_labels = train_pool["label"].values
    consensus_idx = test_pool["_emb_idx"].values
    consensus_labels = test_pool["label"].values

    if len(human_attr) >= 5:
        human_attr = human_attr.copy()
        human_attr["_emb_idx"] = human_attr["code"].map(code_to_idx).astype(int)
        human_idx = human_attr["_emb_idx"].values
        human_labels = human_attr["label"].values
    else:
        human_idx = np.array([], dtype=int)
        human_labels = np.array([])

    return dict(
        train_idx=train_idx, train_labels=train_labels,
        consensus_idx=consensus_idx, consensus_labels=consensus_labels,
        human_idx=human_idx, human_labels=human_labels,
        valid_classes=valid_classes,
        n_train=len(train_idx),
        n_consensus=len(consensus_idx),
        n_human=len(human_idx),
    )


def run_method(method: str, attr_type: str,
               X_train: np.ndarray, y_train_str: np.ndarray,
               X_eval_dict: dict, y_eval_dict: dict,
               classes: list[str]) -> dict:
    """Train one classifier and evaluate on each eval source.

    X_eval_dict: {source_name: X_array}
    y_eval_dict: {source_name: y_array_str}
    """
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    t0 = time.time()
    le = LabelEncoder()
    le.fit(classes)
    y_train = le.transform(y_train_str)
    n_classes = len(classes)

    # Hold out 10% for calibration
    try:
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X_train, y_train, test_size=0.10, random_state=RANDOM_STATE,
            stratify=y_train if min(np.bincount(y_train)) >= 2 else None,
        )
    except ValueError:
        X_fit, X_cal, y_fit, y_cal = X_train, X_train[:0], y_train, y_train[:0]

    clf, calib_method = make_classifier(method, n_classes, attr_type)

    # Sample weights for class balance (helps RF/LogReg are class_weight=balanced already;
    # for XGB we apply manually)
    if method == "xgboost":
        class_counts = np.bincount(y_fit, minlength=n_classes)
        weights = len(y_fit) / (n_classes * np.maximum(class_counts, 1))
        sw_fit = weights[y_fit]
        # carve internal eval split for early stopping
        try:
            X_inner, X_es, y_inner, y_es, sw_inner, _ = train_test_split(
                X_fit, y_fit, sw_fit, test_size=0.15,
                random_state=RANDOM_STATE,
                stratify=y_fit if min(np.bincount(y_fit)) >= 2 else None,
            )
            try:
                clf.fit(X_inner, y_inner, sample_weight=sw_inner,
                        eval_set=[(X_es, y_es)], verbose=False)
            except Exception as e:
                # mlogloss requires all classes present in val — fallback no-es
                logger.warning("    xgb early-stop val incompatible (%s) — refitting w/o ES",
                               str(e)[:80])
                # rebuild clf without early_stopping_rounds
                params = clf.get_xgb_params()
                params.pop("early_stopping_rounds", None)
                from xgboost import XGBClassifier as XGBC
                clf = XGBC(**params)
                clf.fit(X_fit, y_fit, sample_weight=sw_fit, verbose=False)
        except ValueError:
            # train_test_split failed (rare class) — fit on full
            clf.fit(X_fit, y_fit, sample_weight=sw_fit)
    else:
        clf.fit(X_fit, y_fit)

    # Calibrate if asked
    if calib_method is not None and len(y_cal) >= 30:
        cal_clf, ok = calibrate(clf, X_cal, y_cal, calib_method)
        if ok:
            clf = cal_clf

    fit_runtime = time.time() - t0

    # Evaluate on each source
    out_rows = []
    for source, X_eval in X_eval_dict.items():
        y_eval_str = y_eval_dict[source]
        if len(y_eval_str) == 0 or X_eval is None:
            continue
        # Filter eval rows to classes seen in train (LE.transform will fail otherwise)
        mask_known = np.isin(y_eval_str, classes)
        if mask_known.sum() == 0:
            continue
        X_eval_m = X_eval[mask_known]
        y_eval_m = le.transform(y_eval_str[mask_known])

        proba = clf.predict_proba(X_eval_m)
        # Ensure proba shape (n, n_classes) — if clf saw fewer classes, pad
        if proba.shape[1] != n_classes:
            full = np.zeros((proba.shape[0], n_classes), dtype=proba.dtype)
            for j, cls in enumerate(getattr(clf, "classes_", np.arange(proba.shape[1]))):
                full[:, int(cls)] = proba[:, j]
            proba = full

        m = metrics_block(y_eval_m, proba, np.array(classes))
        m["eval_source"] = source
        m["n_eval"] = int(mask_known.sum())
        m["n_eval_dropped_unknown_class"] = int((~mask_known).sum())
        out_rows.append(m)

    return dict(rows=out_rows, runtime_seconds=fit_runtime)


def already_done(df_existing: pd.DataFrame | None, cat: str, attr: str,
                 method: str) -> bool:
    if df_existing is None:
        return False
    sel = (
        (df_existing["category"] == cat)
        & (df_existing["attribute"] == attr)
        & (df_existing["classifier"] == method)
    )
    return sel.any()


def append_results(rows: list[dict]):
    """Idempotent append to RESULTS_PATH."""
    if not rows:
        return
    new = pd.DataFrame(rows)
    if RESULTS_PATH.exists():
        existing = pd.read_parquet(RESULTS_PATH)
        combined = pd.concat([existing, new], ignore_index=True)
        # Dedup: keep latest per (cat, attr, method, eval_source)
        combined = combined.drop_duplicates(
            subset=["category", "attribute", "classifier", "eval_source"], keep="last"
        )
    else:
        combined = new
    combined.to_parquet(RESULTS_PATH, index=False)


def run_category(cat: str, methods: list[str], smoke: bool, force: bool):
    """Run all methods for all attrs in one category."""
    logger.info("=== Category: %s ===", cat)
    silver_df, emb, gold_cat, human_cat = load_cat_data(cat)

    df_existing = pd.read_parquet(RESULTS_PATH) if RESULTS_PATH.exists() else None

    # Pre-compute TF-IDF SVD ONCE per category (fit on full train pool union → re-fit per attr would
    # be wasteful; alternative is to refit per attr but that's >18 fits. Single global TF-IDF on
    # silver(train+val) is what production "noleak" does too)
    # We'll fit TF-IDF on train+val rows (silver) and transform full silver embeddings positions.
    # For per-attr filtering, just use the precomputed full TF-IDF SVD.
    train_val_mask = silver_df["split"].isin(["train", "val"]).values
    train_val_texts = build_text(silver_df[train_val_mask].reset_index(drop=True))
    all_texts = build_text(silver_df)
    logger.info("  fitting TF-IDF + SVD on %d train+val rows...", len(train_val_texts))
    t0 = time.time()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    vec.fit(train_val_texts)
    X_tf_all = vec.transform(all_texts)
    svd = TruncatedSVD(n_components=128, random_state=RANDOM_STATE)
    svd.fit(X_tf_all[train_val_mask])
    X_svd_all = svd.transform(X_tf_all).astype(np.float32)
    logger.info("  TF-IDF+SVD ready in %.1fs (shape=%s)", time.time() - t0, X_svd_all.shape)

    # Concatenate per-row: MPNet (n × 768) ⊕ TF-IDF SVD (n × 128) = n × 896
    X_full = np.concatenate([emb, X_svd_all], axis=1).astype(np.float32)
    logger.info("  Concat features: %s (MPNet %s + TF-IDF SVD %s)",
                X_full.shape, emb.shape, X_svd_all.shape)

    attrs = list(ATTR_CONFIG[cat].keys())
    if smoke:
        attrs = attrs[:1]
        methods = ALL_METHODS

    for attr in attrs:
        attr_type = ATTR_CONFIG[cat][attr]
        logger.info("  --- %s.%s (%s) ---", cat, attr, attr_type)
        data = build_train_eval(cat, attr, attr_type, silver_df, emb,
                                gold_cat, human_cat)
        if data is None:
            continue

        X_train = X_full[data["train_idx"]]
        y_train_str = data["train_labels"]
        X_consensus = X_full[data["consensus_idx"]] if len(data["consensus_idx"]) > 0 else None
        X_human = X_full[data["human_idx"]] if len(data["human_idx"]) > 0 else None

        logger.info("    n_train=%d, n_consensus=%d, n_human=%d, classes=%d",
                    data["n_train"], data["n_consensus"], data["n_human"],
                    len(data["valid_classes"]))

        for method in methods:
            if not force and already_done(df_existing, cat, attr, method):
                logger.info("    [%s] already done — skip", method)
                continue
            logger.info("    [%s] fitting...", method)
            try:
                res = run_method(
                    method, attr_type, X_train, y_train_str,
                    X_eval_dict={"consensus": X_consensus, "human": X_human},
                    y_eval_dict={"consensus": data["consensus_labels"],
                                 "human": data["human_labels"]},
                    classes=data["valid_classes"],
                )
            except Exception as e:
                logger.exception("    [%s] FAILED: %s", method, e)
                continue

            rows_out = []
            for r in res["rows"]:
                rows_out.append({
                    "category": cat,
                    "attribute": attr,
                    "attribute_type": attr_type,
                    "classifier": method,
                    "n_train": int(data["n_train"]),
                    "n_eval": int(r["n_eval"]),
                    "n_eval_dropped_unknown_class": int(r["n_eval_dropped_unknown_class"]),
                    "n_classes": int(len(data["valid_classes"])),
                    "eval_source": r["eval_source"],
                    "micro_acc": float(r["micro_acc"]),
                    "macro_f1": float(r["macro_f1"]),
                    "balanced_acc": float(r["balanced_acc"]),
                    "ece": float(r["ece"]),
                    "runtime_seconds": float(res["runtime_seconds"]),
                })
                logger.info("      %s: acc=%.3f, F1=%.3f, bal_acc=%.3f, ECE=%.3f (n=%d) %.1fs",
                            r["eval_source"], r["micro_acc"], r["macro_f1"],
                            r["balanced_acc"], r["ece"], r["n_eval"],
                            res["runtime_seconds"])
            append_results(rows_out)
            df_existing = pd.read_parquet(RESULTS_PATH)  # reload for next-method idempotency

    logger.info("=== Done: %s ===", cat)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--category", choices=list(ATTR_CONFIG.keys()) + ["all"], default="all")
    p.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=ALL_METHODS)
    p.add_argument("--smoke", action="store_true", help="1 attr × 4 methods per cat")
    p.add_argument("--force", action="store_true", help="ignore existing results")
    args = p.parse_args()

    cats = list(ATTR_CONFIG.keys()) if args.category == "all" else [args.category]
    logger.info("=== method_comparison start ===")
    logger.info("Categories: %s | Methods: %s | Smoke: %s | Force: %s",
                cats, args.methods, args.smoke, args.force)

    overall_t0 = time.time()
    for cat in cats:
        run_category(cat, args.methods, args.smoke, args.force)

    logger.info("=== method_comparison done in %.1f min ===",
                (time.time() - overall_t0) / 60)
    if RESULTS_PATH.exists():
        df = pd.read_parquet(RESULTS_PATH)
        logger.info("Final results: %d rows in %s", len(df), RESULTS_PATH)


if __name__ == "__main__":
    sys.exit(main() or 0)
