"""
Feature engineering for cost-aware router.

8 base features per (product, attr) row. All available at inference time
(no silver_gt, no llm_pred). One-hot encoding expands to ~60 columns total.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ATTR_TYPE: taxonomy from §3.1 of notebook. Mirror exactly to keep consistency.
ATTR_TYPE: dict[tuple[str, str], str] = {
    # pasta
    ("pasta", "is_whole_grain"): "binary",
    ("pasta", "is_organic"): "binary",
    ("pasta", "is_gluten_free"): "binary",
    ("pasta", "is_vegan"): "binary",
    ("pasta", "is_filled"): "binary",
    ("pasta", "grain_type"): "multiclass-high",
    ("pasta", "pasta_shape"): "multiclass-high",
    ("pasta", "nutri_score_grade"): "numeric-bucketed",
    ("pasta", "protein_class"): "numeric-bucketed",
    # chocolate
    ("chocolate", "chocolate_type"): "multiclass-low",
    ("chocolate", "chocolate_extra"): "multiclass-high",
    ("chocolate", "contains_nuts"): "binary",
    ("chocolate", "is_organic"): "binary",
    ("chocolate", "cocoa_percentage"): "numeric-bucketed",
    ("chocolate", "nutri_score_grade"): "numeric-bucketed",
    ("chocolate", "protein_class"): "numeric-bucketed",
    # beverages
    ("beverages", "beverage_type"): "multiclass-high",
    ("beverages", "is_carbonated"): "binary",
    ("beverages", "is_organic"): "binary",
    ("beverages", "is_vegan"): "binary",
    ("beverages", "is_no_added_sugar"): "binary",
    ("beverages", "sugar_class"): "numeric-bucketed",
    ("beverages", "nutri_score_grade"): "numeric-bucketed",
    ("beverages", "nova_group"): "numeric-bucketed",
    ("beverages", "protein_class"): "numeric-bucketed",
    # cheeses
    ("cheeses", "milk_source"): "multiclass-low",
    ("cheeses", "texture"): "multiclass-low",
    ("cheeses", "country_of_origin"): "multiclass-high",
    ("cheeses", "is_pdo"): "binary",
    ("cheeses", "is_organic"): "binary",
    ("cheeses", "is_ultra_processed"): "binary",
    ("cheeses", "fat_class"): "numeric-bucketed",
    # cereals
    ("cereals", "cereal_type"): "multiclass-low",
    ("cereals", "grain_type"): "multiclass-low",
    ("cereals", "is_low_sugar"): "binary",
    ("cereals", "is_high_fibre"): "binary",
    ("cereals", "is_whole_grain"): "binary",
    ("cereals", "is_vegan"): "binary",
    ("cereals", "is_organic"): "binary",
    ("cereals", "nova_class"): "multiclass-low",
    # cosmetics
    ("cosmetics", "product_type"): "multiclass-high",
    ("cosmetics", "form_factor"): "multiclass-low",
    ("cosmetics", "body_area"): "multiclass-low",
    ("cosmetics", "has_sulfates"): "binary",
    ("cosmetics", "has_silicones"): "binary",
    ("cosmetics", "is_organic"): "binary",
}

ATTR_TYPE_VALUES = ["binary", "multiclass-low", "multiclass-high", "numeric-bucketed"]
CASCADE_LAYERS = ["regex", "ml", "bayes", "none"]
NAME_LANGS = ["fr", "en", "es", "de", "it", "unknown"]
CATEGORIES_FIXED = ["pasta", "chocolate", "beverages", "cheeses", "cereals", "cosmetics"]


_LANG_MARKERS = {
    "fr": {"de", "le", "la", "les", "à", "et", "au", "aux", "pour", "sans"},
    "en": {"with", "the", "and", "for", "of", "no", "added"},
    "es": {"el", "la", "los", "las", "y", "de", "con", "sin", "para"},
    "de": {"mit", "ohne", "und", "für", "der", "die", "das"},
    "it": {"di", "con", "senza", "per", "il", "la", "le", "al"},
}


def detect_lang(text: str | None) -> str:
    """Crude language detection by stopword overlap. Returns one of NAME_LANGS."""
    if not text or not isinstance(text, str):
        return "unknown"
    tokens = set(re.findall(r"\b\w+\b", text.lower()))
    best, best_score = "unknown", 0
    for lang, markers in _LANG_MARKERS.items():
        score = len(tokens & markers)
        if score > best_score:
            best, best_score = lang, score
    return best if best_score > 0 else "unknown"


def build_brand_set(df: pd.DataFrame, brand_col: str = "brands") -> set[str]:
    """Lowercase, split on comma, dedupe."""
    if brand_col not in df.columns:
        return set()
    out: set[str] = set()
    for b in df[brand_col].dropna().astype(str):
        for part in b.split(","):
            part = part.strip().lower()
            if part:
                out.add(part)
    return out


def _brand_known(brands_value, brand_set: set[str]) -> int:
    if not isinstance(brands_value, str) or not brands_value:
        return 0
    for part in brands_value.split(","):
        if part.strip().lower() in brand_set:
            return 1
    return 0


def _attr_type(category: str, attr: str) -> str:
    return ATTR_TYPE.get((category, attr), "binary")


FEATURE_COLUMNS: list[str] = []


def build_class_freq_table(
    train_df: pd.DataFrame,
) -> dict[tuple[str, str, str], float]:
    """For each (category, attr, silver_gt_value) — relative frequency in train.

    Used as `cascade_pred_class_freq` feature: when cascade predicts a class
    that's rare in train silver, that's a risk signal.
    """
    out: dict[tuple[str, str, str], float] = {}
    for (cat, attr), grp in train_df.groupby(["category", "attr"]):
        total = len(grp)
        if total == 0:
            continue
        for cls, count in grp["silver_gt"].value_counts().items():
            out[(cat, attr, str(cls))] = count / total
    return out


def build_brand_attr_acc_table(
    train_df: pd.DataFrame,
) -> dict[tuple[str, str, str], float]:
    """For each (brand, category, attr) — observed cascade_correct rate in train.

    Used as `brand_attr_train_acc` feature: brands × attrs with low train accuracy
    are risky.
    """
    if "brands" not in train_df.columns or "cascade_correct" not in train_df.columns:
        return {}
    out: dict[tuple[str, str, str], float] = {}
    # Use first brand part lowercased as key
    keyed = train_df.copy()
    keyed["_brand_key"] = keyed["brands"].apply(_first_brand)
    for (brand, cat, attr), grp in keyed.groupby(["_brand_key", "category", "attr"]):
        if not brand or len(grp) < 3:  # require at least 3 observations
            continue
        out[(brand, cat, attr)] = float(grp["cascade_correct"].mean())
    return out


_NUM_RE = re.compile(r"\d")


def _name_has_digits(name) -> int:
    if not isinstance(name, str):
        return 0
    return 1 if _NUM_RE.search(name) else 0


def _name_token_count(name) -> int:
    if not isinstance(name, str):
        return 0
    return len(re.findall(r"\b\w+\b", name))


def _first_brand(b) -> str:
    if not isinstance(b, str):
        return ""
    return b.split(",")[0].strip().lower()


def featurize(
    df: pd.DataFrame,
    brand_set: set[str] | None = None,
    drop_category: bool = False,
    class_freq_table: dict | None = None,
    brand_attr_acc_table: dict | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Convert rows to feature matrix with fixed-vocab one-hot."""
    if brand_set is None:
        brand_set = set()

    n = len(df)
    cols: list[str] = []

    # Category one-hot
    if not drop_category:
        cat_oh = np.zeros((n, len(CATEGORIES_FIXED)), dtype=np.float32)
        for i, c in enumerate(CATEGORIES_FIXED):
            cat_oh[:, i] = (df["category"].values == c).astype(np.float32)
            cols.append(f"cat_{c}")
    else:
        cat_oh = np.zeros((n, 0), dtype=np.float32)

    # Attr one-hot
    attr_vocab = sorted({a for (_, a) in ATTR_TYPE.keys()})
    attr_oh = np.zeros((n, len(attr_vocab)), dtype=np.float32)
    for i, a in enumerate(attr_vocab):
        attr_oh[:, i] = (df["attr"].values == a).astype(np.float32)
        cols.append(f"attr_{a}")

    # attr_type one-hot
    attr_types = [_attr_type(c, a) for c, a in zip(df["category"], df["attr"])]
    attype_oh = np.zeros((n, len(ATTR_TYPE_VALUES)), dtype=np.float32)
    for i, t in enumerate(ATTR_TYPE_VALUES):
        attype_oh[:, i] = (np.array(attr_types) == t).astype(np.float32)
        cols.append(f"attr_type_{t}")

    # cascade_layer one-hot
    layer_oh = np.zeros((n, len(CASCADE_LAYERS)), dtype=np.float32)
    for i, lay in enumerate(CASCADE_LAYERS):
        layer_oh[:, i] = (df["cascade_layer"].values == lay).astype(np.float32)
        cols.append(f"cascade_layer_{lay}")

    # cascade_conf
    conf = df["cascade_conf"].fillna(0.0).astype(np.float32).values.reshape(-1, 1)
    cols.append("cascade_conf")

    # product_name_length
    name_len = (
        df["product_name"].fillna("").astype(str).str.len().astype(np.float32).values.reshape(-1, 1)
    )
    cols.append("product_name_length")

    # brand_known
    if "brands" in df.columns:
        bk = np.array(
            [_brand_known(b, brand_set) for b in df["brands"].values],
            dtype=np.float32,
        ).reshape(-1, 1)
    else:
        bk = np.zeros((n, 1), dtype=np.float32)
    cols.append("brand_known")

    # name_lang one-hot
    langs = [detect_lang(t) for t in df["product_name"].fillna("").astype(str)]
    lang_oh = np.zeros((n, len(NAME_LANGS)), dtype=np.float32)
    for i, lg in enumerate(NAME_LANGS):
        lang_oh[:, i] = (np.array(langs) == lg).astype(np.float32)
        cols.append(f"lang_{lg}")

    # cascade_pred_class_freq
    if class_freq_table is not None and "cascade_pred" in df.columns:
        freqs = np.array([
            class_freq_table.get((c, a, str(p)), 0.0)
            for c, a, p in zip(df["category"].values, df["attr"].values, df["cascade_pred"].values)
        ], dtype=np.float32).reshape(-1, 1)
    else:
        freqs = np.zeros((n, 1), dtype=np.float32)
    cols.append("cascade_pred_class_freq")

    # is_rare_class (< 5% in train)
    is_rare = (freqs < 0.05).astype(np.float32)
    cols.append("is_rare_class")

    # brand_attr_train_acc
    if brand_attr_acc_table is not None and "brands" in df.columns:
        brand_acc = np.array([
            brand_attr_acc_table.get((_first_brand(b), c, a), 0.5)  # neutral default
            for b, c, a in zip(df["brands"].values, df["category"].values, df["attr"].values)
        ], dtype=np.float32).reshape(-1, 1)
    else:
        brand_acc = np.full((n, 1), 0.5, dtype=np.float32)
    cols.append("brand_attr_train_acc")

    # name_has_digits
    has_digits = np.array([_name_has_digits(t) for t in df["product_name"].values],
                          dtype=np.float32).reshape(-1, 1)
    cols.append("name_has_digits")

    # name_token_count
    tok_count = np.array([_name_token_count(t) for t in df["product_name"].values],
                          dtype=np.float32).reshape(-1, 1)
    cols.append("name_token_count")

    X = np.concatenate([cat_oh, attr_oh, attype_oh, layer_oh, conf, name_len, bk, lang_oh,
                         freqs, is_rare, brand_acc, has_digits, tok_count], axis=1)
    return X, cols
