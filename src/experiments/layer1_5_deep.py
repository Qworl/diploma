"""P2-EXP5: Layer 1.5 deep models — fastText, DistilBERT, LightGBM.

Variants K/L/M on 80/20 split (seed=42) identical to EXP3.

  K  fasttext_ml  — fastText char-ngrams + hybrid_ml fallback
  L  distilbert_ml — multilingual DistilBERT fine-tune + hybrid_ml fallback
  M  lightgbm_ml  — LightGBM on TF-IDF + hybrid_ml fallback

Output:
  datasets/processed/layer1_5_deep.parquet
  docs/layer1_5_deep_comparison.md
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.common import PROCESSED_DIR, setup_logging
from src.experiments.layer1_5_honest_comparison import (
    _build_regex_preds,
    _train_fresh_hybrid,
)
from src.pipeline.regex.extractor import RegexExtractor

logger = logging.getLogger(__name__)

CATEGORIES = ["pasta", "chocolate", "cheeses"]
SEED = 42
TEST_SIZE = 0.2
MIN_GOLD = 20
FASTTEXT_TAU = 0.85
DISTILBERT_TAU = 0.85
LGBM_TAU = 0.85

GOLD_PATH = Path(PROCESSED_DIR) / "consensus_gold_v2_expanded.parquet"
OUT_PATH = Path(PROCESSED_DIR) / "layer1_5_deep.parquet"
DOC_PATH = Path("docs") / "layer1_5_deep_comparison.md"

# ---------------------------------------------------------------------------
# Availability flags (set by try-import below)
# ---------------------------------------------------------------------------
FASTTEXT_AVAILABLE = False
DISTILBERT_AVAILABLE = False
LGBM_AVAILABLE = False

try:
    import fasttext  # noqa: F401
    FASTTEXT_AVAILABLE = True
    logger.debug("fasttext available")
except ImportError:
    logger.warning("fasttext not installed — variant K will be skipped")

try:
    import lightgbm  # noqa: F401
    LGBM_AVAILABLE = True
    logger.debug("LightGBM available")
except ImportError:
    logger.warning("lightgbm not installed — variant M will be skipped")

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    DISTILBERT_AVAILABLE = True
    logger.debug("transformers/torch available")
except ImportError:
    logger.warning("transformers/torch not installed — variant L will be skipped")


# ---------------------------------------------------------------------------
# Shared text builder (same as honest_comparison.py)
# ---------------------------------------------------------------------------

def _build_text(row: pd.Series) -> str:
    parts = []
    for col in ["product_name", "ingredients_text", "brands", "quantity"]:
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    return " ".join(parts)


# ---------------------------------------------------------------------------
# K. fastText methods
# ---------------------------------------------------------------------------

def train_fasttext(train_texts: list[str], y_train: list[str]):
    """Train fastText supervised model. Returns model or None."""
    if not FASTTEXT_AVAILABLE:
        return None
    if len(y_train) < 10 or len(set(y_train)) < 2:
        return None

    import fasttext

    # Write training file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        tmp_path = f.name
        for text, label in zip(train_texts, y_train):
            clean = text.replace("\n", " ").strip() or "unknown"
            f.write(f"__label__{label} {clean}\n")

    try:
        model = fasttext.train_supervised(
            input=tmp_path,
            epoch=25,
            lr=0.5,
            wordNgrams=2,
            minn=3,
            maxn=6,
            dim=100,
            verbose=0,
        )
    finally:
        os.unlink(tmp_path)

    return model


def predict_fasttext(model, texts: list[str], tau: float = 0.85):
    """Predict with fastText model. Returns list of (label|None, proba)."""
    results = []
    for text in texts:
        clean = text.replace("\n", " ").strip() or "unknown"
        labels, probas = model.predict(clean, k=1)
        label_raw = labels[0]  # "__label__VALUE"
        # fasttext-wheel returns numpy array — use float() to handle numpy compat
        proba = float(probas[0]) if hasattr(probas[0], '__float__') else float(list(probas)[0])
        label = label_raw.replace("__label__", "")
        if proba >= tau:
            results.append((label, proba))
        else:
            results.append((None, proba))
    return results


# ---------------------------------------------------------------------------
# L. DistilBERT fine-tune (per-attr, per-cat)
# ---------------------------------------------------------------------------

DISTILBERT_MODEL_NAME = "distilbert-base-multilingual-cased"
DISTILBERT_MAX_LEN = 128
DISTILBERT_EPOCHS = 3
DISTILBERT_BATCH = 16
DISTILBERT_LR = 2e-5


def train_distilbert(train_texts: list[str], y_train: list[str]):
    """Fine-tune DistilBERT for sequence classification. Returns (model, tokenizer, le) or None."""
    if not DISTILBERT_AVAILABLE:
        return None
    if len(y_train) < 10 or len(set(y_train)) < 2:
        return None

    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    classes = sorted(set(y_train))
    le = LabelEncoder()
    le.fit(classes)
    y_enc = le.transform(y_train)

    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_NAME)

    class TextDataset(Dataset):
        def __init__(self, texts, labels):
            self.texts = texts
            self.labels = labels

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = tokenizer(
                self.texts[idx],
                max_length=DISTILBERT_MAX_LEN,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(self.labels[idx], dtype=torch.long),
            }

    dataset = TextDataset(train_texts, y_enc.tolist())
    loader = DataLoader(dataset, batch_size=DISTILBERT_BATCH, shuffle=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    n_labels = len(classes)

    model = AutoModelForSequenceClassification.from_pretrained(
        DISTILBERT_MODEL_NAME, num_labels=n_labels
    )
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=DISTILBERT_LR)

    for epoch in range(DISTILBERT_EPOCHS):
        total_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logger.debug("DistilBERT epoch %d/%d loss=%.4f", epoch + 1, DISTILBERT_EPOCHS, total_loss / max(len(loader), 1))

    model.eval()
    return model, tokenizer, le, device


def predict_distilbert(model_tuple, texts: list[str], tau: float = 0.85):
    """Predict with fine-tuned DistilBERT. Returns list of (label|None, proba)."""
    if model_tuple is None:
        return [(None, 0.0)] * len(texts)

    import torch
    import torch.nn.functional as F

    model, tokenizer, le, device = model_tuple
    results = []

    model.eval()
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                max_length=DISTILBERT_MAX_LEN,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probas = F.softmax(outputs.logits, dim=-1).squeeze(0)
            top_idx = int(probas.argmax().item())
            top_proba = float(probas[top_idx].item())
            label = le.inverse_transform([top_idx])[0]
            if top_proba >= tau:
                results.append((str(label), top_proba))
            else:
                results.append((None, top_proba))

    return results


# ---------------------------------------------------------------------------
# M. LightGBM on TF-IDF
# ---------------------------------------------------------------------------

def train_lgbm(train_texts: list[str], y_train: list[str]):
    """Train LightGBM on TF-IDF features. Returns (vec, clf, le) or None."""
    if not LGBM_AVAILABLE:
        return None
    if len(y_train) < 10 or len(set(y_train)) < 2:
        return None

    from lightgbm import LGBMClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=20_000,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    X = vec.fit_transform(train_texts)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    n_classes = len(le.classes_)
    objective = "binary" if n_classes == 2 else "multiclass"
    num_class_kwarg = {"num_class": n_classes} if n_classes > 2 else {}

    clf = LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        objective=objective,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
        **num_class_kwarg,
    )
    clf.fit(X, y_enc)
    return vec, clf, le


def predict_lgbm(model_tuple, texts: list[str], tau: float = 0.85):
    """Predict with LightGBM. Returns list of (label|None, proba)."""
    if model_tuple is None:
        return [(None, 0.0)] * len(texts)

    vec, clf, le = model_tuple
    X = vec.transform(texts)
    proba_matrix = clf.predict_proba(X)
    results = []
    for row in proba_matrix:
        top_idx = int(row.argmax())
        top_proba = float(row[top_idx])
        label = le.inverse_transform([top_idx])[0]
        if top_proba >= tau:
            results.append((str(label), top_proba))
        else:
            results.append((None, top_proba))
    return results


# ---------------------------------------------------------------------------
# Per-(cat, attr) computation for EXP5 variants
# ---------------------------------------------------------------------------

def run_one_attr_deep(
    cat: str,
    attr: str,
    gold: pd.DataFrame,
    silver: pd.DataFrame,
    emb_all: np.ndarray,
    code_to_idx: dict[str, int],
    train_codes_set: set[str],
    test_codes_set: set[str],
    regex_preds: dict[str, dict[str, str]],
    test_products: pd.DataFrame,
) -> list[dict]:
    """Run K/L/M variants for one (cat, attr) pair. Returns list of result rows."""
    # Filter gold
    cat_gold = gold[(gold["category"] == cat) & (gold["attr"] == attr) & ~gold["gold_is_null"]].copy()
    cat_gold["code"] = cat_gold["code"].astype(str)
    cat_gold = cat_gold[cat_gold["code"].isin(code_to_idx)]

    if len(cat_gold) < MIN_GOLD:
        logger.info("[%s/%s] only %d non-null gold cells, skipping", cat, attr, len(cat_gold))
        return []

    train_gold = cat_gold[cat_gold["code"].isin(train_codes_set)]
    test_gold = cat_gold[cat_gold["code"].isin(test_codes_set)]

    if len(train_gold) < 10 or len(test_gold) < 5:
        logger.info("[%s/%s] insufficient train/test split, skipping", cat, attr)
        return []

    # Build arrays
    train_idx = np.array([code_to_idx[c] for c in train_gold["code"]])
    test_idx = np.array([code_to_idx[c] for c in test_gold["code"]])

    X_gold_train = emb_all[train_idx]
    y_gold_train = train_gold["gold_value"].astype(str).values
    X_test_emb = emb_all[test_idx]
    y_test = test_gold["gold_value"].astype(str).values
    test_codes_list = test_gold["code"].tolist()

    # Build silver training data (exclude test codes)
    X_silver: np.ndarray = np.empty((0, emb_all.shape[1]))
    y_silver: np.ndarray = np.array([], dtype=str)

    if attr in silver.columns:
        silver_attr = silver[silver[attr].notna()].copy()
        silver_attr["code"] = silver_attr["code"].astype(str)
        silver_attr = silver_attr[~silver_attr["code"].isin(test_codes_set)]
        silver_attr = silver_attr[~silver_attr["code"].isin(train_codes_set)]
        silver_attr = silver_attr[silver_attr["code"].isin(code_to_idx)]

        silver_idx = np.array([code_to_idx[c] for c in silver_attr["code"]])
        if len(silver_idx) > 0:
            X_silver = emb_all[silver_idx]
            y_silver = silver_attr[attr].astype(str).values

    # Train fresh hybrid ML (baseline fallback)
    if len(X_silver) > 0:
        clf, le = _train_fresh_hybrid(X_silver, y_silver, X_gold_train, y_gold_train)
    else:
        all_classes = sorted(set(y_gold_train.tolist()))
        if len(all_classes) < 2:
            return []
        le = LabelEncoder()
        le.fit(all_classes)
        y_enc = le.transform(y_gold_train)
        n_classes = len(all_classes)
        kwargs = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8,
                      reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
                      tree_method="hist", verbosity=0)
        if n_classes == 2:
            pos = int((y_enc == 1).sum())
            neg = int((y_enc == 0).sum())
            kwargs["scale_pos_weight"] = max(neg / max(pos, 1), 0.5)
            clf = xgb.XGBClassifier(**kwargs)
        else:
            clf = xgb.XGBClassifier(objective="multi:softmax", num_class=n_classes, **kwargs)
        clf.fit(X_gold_train, y_enc)

    if clf is None or le is None:
        return []

    # Hybrid ML fallback predictions
    enc_preds = clf.predict(X_test_emb)
    ml_labels_all = le.inverse_transform(enc_preds).tolist()

    # Build text features
    train_gold_with_text = train_gold.merge(
        silver[["code", "product_name", "ingredients_text", "brands", "quantity"]],
        on="code", how="left",
    )
    train_texts = [_build_text(row) for _, row in train_gold_with_text.iterrows()]
    y_train_texts = train_gold_with_text["gold_value"].astype(str).tolist()

    test_products_idx = test_products.set_index("code")
    test_texts_list = []
    for c in test_codes_list:
        if c in test_products_idx.index:
            test_texts_list.append(_build_text(test_products_idx.loc[c]))
        else:
            test_texts_list.append("")

    n = len(y_test)

    # ---------------------------------------------------------------
    # K. fastText
    # ---------------------------------------------------------------
    preds_k = list(ml_labels_all)  # default = ML fallback
    if FASTTEXT_AVAILABLE:
        ft_model = train_fasttext(train_texts, y_train_texts)
        if ft_model is not None:
            ft_results = predict_fasttext(ft_model, test_texts_list, tau=FASTTEXT_TAU)
            preds_k = []
            for i, (label, _) in enumerate(ft_results):
                if label is not None:
                    preds_k.append(str(label))
                else:
                    preds_k.append(ml_labels_all[i])

    # ---------------------------------------------------------------
    # L. DistilBERT
    # ---------------------------------------------------------------
    preds_l = list(ml_labels_all)
    if DISTILBERT_AVAILABLE:
        db_model = train_distilbert(train_texts, y_train_texts)
        if db_model is not None:
            db_results = predict_distilbert(db_model, test_texts_list, tau=DISTILBERT_TAU)
            preds_l = []
            for i, (label, _) in enumerate(db_results):
                if label is not None:
                    preds_l.append(str(label))
                else:
                    preds_l.append(ml_labels_all[i])
            # Free GPU memory
            try:
                import torch
                del db_model
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass

    # ---------------------------------------------------------------
    # M. LightGBM
    # ---------------------------------------------------------------
    preds_m = list(ml_labels_all)
    if LGBM_AVAILABLE:
        lgbm_model = train_lgbm(train_texts, y_train_texts)
        if lgbm_model is not None:
            lgbm_results = predict_lgbm(lgbm_model, test_texts_list, tau=LGBM_TAU)
            preds_m = []
            for i, (label, _) in enumerate(lgbm_results):
                if label is not None:
                    preds_m.append(str(label))
                else:
                    preds_m.append(ml_labels_all[i])

    # ---------------------------------------------------------------
    # Compute accuracies
    # ---------------------------------------------------------------
    variant_preds = {}
    if FASTTEXT_AVAILABLE:
        variant_preds["K_fasttext_ml"] = preds_k
    if DISTILBERT_AVAILABLE:
        variant_preds["L_distilbert_ml"] = preds_l
    if LGBM_AVAILABLE:
        variant_preds["M_lightgbm_ml"] = preds_m

    rows = []
    for vname, preds in variant_preds.items():
        correct = sum(1 for p, g in zip(preds, y_test) if p == g)
        acc = correct / n if n > 0 else float("nan")
        rows.append({
            "category": cat,
            "attr": attr,
            "variant": vname,
            "accuracy": acc,
            "n_test": n,
            "n_train_gold": len(train_gold),
            "n_silver": len(y_silver),
        })

    log_parts = " ".join(
        f"{v.split('_')[0]}={r['accuracy']:.3f}"
        for v, r in zip(variant_preds.keys(), rows)
    )
    logger.info("[%s/%s] n_test=%d | %s", cat, attr, n, log_parts)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()

    logger.info("P2-EXP5: Layer 1.5 Deep Models")
    logger.info(
        "Available: fastText=%s, DistilBERT=%s, LightGBM=%s",
        FASTTEXT_AVAILABLE, DISTILBERT_AVAILABLE, LGBM_AVAILABLE,
    )

    gold = pd.read_parquet(GOLD_PATH)
    gold["code"] = gold["code"].astype(str)
    logger.info("Loaded gold: %d rows, %d codes", len(gold), gold["code"].nunique())

    all_rows: list[dict] = []

    for cat in CATEGORIES:
        logger.info("=== Category: %s ===", cat)

        silver = pd.read_parquet(
            Path(PROCESSED_DIR) / f"{cat}_stratified_silver_standard.parquet"
        )
        silver["code"] = silver["code"].astype(str)

        emb_all = np.load(Path(PROCESSED_DIR) / f"{cat}_stratified_embeddings.npy")
        code_to_idx: dict[str, int] = {c: i for i, c in enumerate(silver["code"].tolist())}

        cat_gold = gold[gold["category"] == cat].copy()
        unique_codes = sorted(cat_gold["code"].unique().tolist())

        train_codes, test_codes = train_test_split(
            unique_codes, test_size=TEST_SIZE, random_state=SEED
        )
        train_codes_set = set(train_codes)
        test_codes_set = set(test_codes)
        logger.info("  Split: %d train codes, %d test codes", len(train_codes), len(test_codes))

        test_products = silver[silver["code"].isin(test_codes_set)].copy()

        logger.info("  Building regex predictions for test products...")
        regex_preds = _build_regex_preds(test_products, cat)

        attrs = sorted(cat_gold["attr"].unique().tolist())
        logger.info("  Attrs: %s", attrs)

        for attr in attrs:
            rows = run_one_attr_deep(
                cat=cat,
                attr=attr,
                gold=cat_gold,
                silver=silver,
                emb_all=emb_all,
                code_to_idx=code_to_idx,
                train_codes_set=train_codes_set,
                test_codes_set=test_codes_set,
                regex_preds=regex_preds,
                test_products=test_products,
            )
            all_rows.extend(rows)

    result = pd.DataFrame(all_rows)
    result.to_parquet(OUT_PATH, index=False)
    logger.info("Wrote %d rows to %s", len(result), OUT_PATH)

    _print_summary(result)
    _write_doc(result)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

# EXP3/EXP4 baselines for comparison
BASELINES = {
    "A_regex_ml": 0.8358,
    "B_ml_only": 0.8220,
    "D_dt_ml": 0.8298,
    "H_regex_dt_ml": 0.8378,  # EXP4 best
}

VARIANT_ORDER_DEEP = ["K_fasttext_ml", "L_distilbert_ml", "M_lightgbm_ml"]


def _print_summary(result: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("P2-EXP5: Deep Layer 1.5 — Grand Means")
    print("=" * 70)
    grand = result.groupby("variant")["accuracy"].mean()
    for v in VARIANT_ORDER_DEEP:
        if v in grand.index:
            print(f"  {v:25s}: {grand[v] * 100:.2f}%")

    print("\nComparison to EXP3/EXP4 baselines:")
    for v in VARIANT_ORDER_DEEP:
        if v in grand.index:
            beat_regex = grand[v] - BASELINES["A_regex_ml"]
            print(f"  {v:25s} vs regex_ml: {beat_regex * 100:+.2f} pp")

    print("\n" + "=" * 70)
    print("Per-category mean accuracy")
    print("=" * 70)
    pivot_cat = result.pivot_table(
        index="category", columns="variant", values="accuracy", aggfunc="mean"
    )
    cols = [v for v in VARIANT_ORDER_DEEP if v in pivot_cat.columns]
    if cols:
        print(pivot_cat[cols].round(4).to_string())

    print("\n" + "=" * 70)
    print("Per-(category, attr) accuracy")
    print("=" * 70)
    pivot_attr = result.pivot_table(
        index=["category", "attr"], columns="variant", values="accuracy", aggfunc="mean"
    )
    cols = [v for v in VARIANT_ORDER_DEEP if v in pivot_attr.columns]
    if cols:
        pivot_attr = pivot_attr[cols]
        pivot_attr["winner"] = pivot_attr.idxmax(axis=1)
        print(pivot_attr.round(4).to_string())


# ---------------------------------------------------------------------------
# Markdown doc writer
# ---------------------------------------------------------------------------

def _write_doc(result: pd.DataFrame) -> None:
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)

    grand = result.groupby("variant")["accuracy"].mean()

    lines: list[str] = []
    lines.append("# P2-EXP5: Layer 1.5 Deep Models — fastText / DistilBERT / LightGBM")
    lines.append("")
    lines.append("**Methodology:** Same 80/20 split (seed=42) as EXP3. Each deep model is trained")
    lines.append("on 80% gold text only, applied with τ=0.85 confidence threshold, with fresh")
    lines.append("hybrid_ml (XGBoost on silver+gold) as fallback. 2666 gold codes across pasta/")
    lines.append("chocolate/cheeses × 22 attributes.")
    lines.append("")

    # Installation status
    lines.append("## Installation Status")
    lines.append("")
    lines.append("| Variant | Package | Status |")
    lines.append("|---------|---------|--------|")
    lines.append(f"| K fastText | fasttext-wheel | {'OK' if FASTTEXT_AVAILABLE else 'FAILED'} |")
    lines.append(f"| L DistilBERT | transformers+torch | {'OK' if DISTILBERT_AVAILABLE else 'FAILED'} |")
    lines.append(f"| M LightGBM | lightgbm | {'OK' if LGBM_AVAILABLE else 'FAILED'} |")
    lines.append("")

    # Grand mean comparison table
    lines.append("## Grand Mean Accuracy — EXP5 vs EXP3/EXP4 Baselines")
    lines.append("")
    lines.append("| Variant | Accuracy | vs A_regex_ml (pp) | vs H_regex_dt_ml (pp) |")
    lines.append("|---------|----------|--------------------|-----------------------|")

    # EXP3/EXP4 baselines
    for bname, bacc in [
        ("A_regex_ml (EXP3 best)", BASELINES["A_regex_ml"]),
        ("B_ml_only (EXP3)", BASELINES["B_ml_only"]),
        ("D_dt_ml (EXP3)", BASELINES["D_dt_ml"]),
        ("H_regex_dt_ml (EXP4)", BASELINES["H_regex_dt_ml"]),
    ]:
        vs_a = (bacc - BASELINES["A_regex_ml"]) * 100
        vs_h = (bacc - BASELINES["H_regex_dt_ml"]) * 100
        lines.append(f"| {bname} | {bacc*100:.2f}% | {vs_a:+.2f} | {vs_h:+.2f} |")

    # EXP5 results
    for v in VARIANT_ORDER_DEEP:
        if v in grand.index:
            acc = grand[v]
            vs_a = (acc - BASELINES["A_regex_ml"]) * 100
            vs_h = (acc - BASELINES["H_regex_dt_ml"]) * 100
            lines.append(f"| {v} | {acc*100:.2f}% | {vs_a:+.2f} | {vs_h:+.2f} |")
        else:
            lines.append(f"| {v} | SKIPPED | — | — |")
    lines.append("")

    # Per-category
    pivot_cat = result.pivot_table(
        index="category", columns="variant", values="accuracy", aggfunc="mean"
    )
    cols = [v for v in VARIANT_ORDER_DEEP if v in pivot_cat.columns]

    if cols:
        lines.append("## Per-Category Mean Accuracy")
        lines.append("")
        header = "| Category | " + " | ".join(cols) + " |"
        sep = "|----------|" + "---------|" * len(cols)
        lines.append(header)
        lines.append(sep)
        for cat, row in pivot_cat[cols].round(4).iterrows():
            vals = " | ".join(f"{v*100:.2f}%" for v in row)
            lines.append(f"| {cat} | {vals} |")
        lines.append("")

    # Per-(cat, attr)
    pivot_attr = result.pivot_table(
        index=["category", "attr"], columns="variant", values="accuracy", aggfunc="mean"
    )
    cols = [v for v in VARIANT_ORDER_DEEP if v in pivot_attr.columns]

    if cols:
        pivot_attr = pivot_attr[cols].round(4)
        pivot_attr["winner"] = pivot_attr.idxmax(axis=1)
        winner_counts = pivot_attr["winner"].value_counts()

        lines.append("## Per-(Category, Attr) Accuracy")
        lines.append("")
        header2 = "| Category | Attr | " + " | ".join(cols) + " | Winner |"
        sep2 = "|----------|------|" + "---------|" * len(cols) + "--------|"
        lines.append(header2)
        lines.append(sep2)
        for (cat, attr), row in pivot_attr.iterrows():
            acc_vals = " | ".join(f"{v*100:.2f}%" for v in row[cols])
            winner = row["winner"]
            lines.append(f"| {cat} | {attr} | {acc_vals} | {winner} |")
        lines.append("")

        lines.append("## Winner Counts")
        lines.append("")
        lines.append("| Variant | # Attrs Won |")
        lines.append("|---------|-------------|")
        for v in VARIANT_ORDER_DEEP:
            cnt = winner_counts.get(v, 0)
            lines.append(f"| {v} | {cnt} |")
        lines.append("")

    # Verdict
    lines.append("## Verdict: Does Deep Layer 1.5 Beat Regex?")
    lines.append("")
    if len(grand) == 0:
        lines.append("No variants ran successfully.")
    else:
        best_variant = grand.idxmax()
        best_acc = grand[best_variant]
        beat_regex = best_acc > BASELINES["A_regex_ml"]
        beat_h = best_acc > BASELINES["H_regex_dt_ml"]
        lines.append(f"**Best EXP5 variant:** `{best_variant}` at {best_acc*100:.2f}%")
        lines.append("")
        if beat_h:
            lines.append(
                f"**YES** — beats H_regex_dt_ml ({BASELINES['H_regex_dt_ml']*100:.2f}%) by "
                f"{(best_acc - BASELINES['H_regex_dt_ml'])*100:+.2f} pp. "
                "Deep Layer 1.5 is viable on 2666 gold codes."
            )
        elif beat_regex:
            lines.append(
                f"**PARTIAL** — beats A_regex_ml ({BASELINES['A_regex_ml']*100:.2f}%) by "
                f"{(best_acc - BASELINES['A_regex_ml'])*100:+.2f} pp "
                f"but not H_regex_dt_ml. Incremental gain."
            )
        else:
            lines.append(
                f"**NO** — best EXP5 ({best_acc*100:.2f}%) still below regex baseline "
                f"({BASELINES['A_regex_ml']*100:.2f}%). "
                "More training data needed for deep models to dominate on this dataset size."
            )
        lines.append("")
        lines.append("**Analysis:**")
        lines.append(f"- fastText: char-ngram supervised, fast, multilingual-friendly")
        lines.append(f"- DistilBERT: contextual fine-tune, 135M params, but only ~700 train/attr")
        lines.append(f"- LightGBM: gradient boosting on TF-IDF, low bias vs DT")
        lines.append("")
        lines.append("With only ~700 samples per attr, DistilBERT likely overfits unless")
        lines.append("the task is simple. LightGBM should be a strict improvement over DT.")
        lines.append("fastText benefits from char-ngrams (handles multilingual morphology).")

    doc_text = "\n".join(lines) + "\n"
    DOC_PATH.write_text(doc_text, encoding="utf-8")
    logger.info("Wrote doc to %s", DOC_PATH)


if __name__ == "__main__":
    main()
