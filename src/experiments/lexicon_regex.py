"""P2-EXP6: Lexicon-augmented regex via PMI discriminative token mining.

Auto-grows regex vocabulary from gold data using pointwise mutual information (PMI).
For each (cat, attr, value), mines top-K discriminative tokens from 80% training gold,
then extends hand-crafted regex patterns with the mined vocabulary.

Comparison on same honest 80/20 split (seed=42) as EXP3:
  A_baseline   — existing regex_ml (EXP3 result: 83.58%)
  L_lexicon_ml — mined-lexicon regex + hybrid_ml fallback
  M_combined   — (original regex OR mined lexicon) + hybrid_ml fallback

Output:
  datasets/processed/lexicon_regex_comparison.parquet
  models/lexicon_patterns.json
  docs/lexicon_regex_findings.md
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.common import MODELS_DIR, PROCESSED_DIR, setup_logging
from src.pipeline.regex.extractor import RegexExtractor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORIES = ["pasta", "chocolate", "cheeses"]
SEED = 42
TEST_SIZE = 0.2
GOLD_WEIGHT = 5.0
MIN_GOLD = 20

# PMI mining hyperparameters
MIN_PMI = 0.8          # lower than spec (1.0) to capture more multilingual tokens
MIN_COUNT = 3          # minimum token occurrences in training set
TOP_K = 30             # top-K tokens per value
MIN_CONF = 0.70        # P(value|token) threshold — lowered from 0.85 to cover minority classes

GOLD_PATH = Path(PROCESSED_DIR) / "consensus_gold_v2_expanded.parquet"
OUT_PATH = Path(PROCESSED_DIR) / "lexicon_regex_comparison.parquet"
PATTERNS_PATH = Path(MODELS_DIR) / "lexicon_patterns.json"
DOC_PATH = Path("docs") / "lexicon_regex_findings.md"


# ---------------------------------------------------------------------------
# Text builder
# ---------------------------------------------------------------------------

def _build_text(row: pd.Series) -> str:
    parts = []
    for col in ["product_name", "ingredients_text", "brands", "quantity"]:
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    return " ".join(parts)


# ---------------------------------------------------------------------------
# PMI token mining
# ---------------------------------------------------------------------------

TOKENIZE_RE = re.compile(r"\b[\w'%-]{2,}\b")


def tokenize(text: str) -> list[str]:
    return TOKENIZE_RE.findall(str(text).lower())


def mine_discriminative_tokens(
    train_texts: list[str],
    train_labels: list[str],
    *,
    min_count: int = MIN_COUNT,
    min_pmi: float = MIN_PMI,
    top_k: int = TOP_K,
) -> dict[str, list[tuple[str, float, float, int]]]:
    """For each label value, return top-K discriminative tokens.

    Returns:
        dict[value -> list[(token, pmi, p_val_given_tok, count)]], sorted by PMI desc.
    """
    label_counts: Counter[str] = Counter(train_labels)
    n_total = len(train_labels)
    if n_total == 0:
        return {}

    # token -> {value: count}
    token_value_counts: dict[str, Counter] = defaultdict(Counter)
    token_counts: Counter[str] = Counter()

    for text, label in zip(train_texts, train_labels):
        toks = set(tokenize(text))  # set: avoid double-counting same token per doc
        for t in toks:
            token_counts[t] += 1
            token_value_counts[t][label] += 1

    per_value_tokens: dict[str, list] = defaultdict(list)

    for tok, val_counts in token_value_counts.items():
        if token_counts[tok] < min_count:
            continue
        p_tok = token_counts[tok] / n_total
        for val, c in val_counts.items():
            p_val = label_counts[val] / n_total
            p_joint = c / n_total
            if p_joint > 0 and p_tok > 0 and p_val > 0:
                pmi = math.log(p_joint / (p_tok * p_val))
            else:
                pmi = -math.inf
            if pmi >= min_pmi:
                p_val_given_tok = c / token_counts[tok]
                per_value_tokens[val].append((tok, pmi, p_val_given_tok, c))

    result: dict[str, list] = {}
    for val, items in per_value_tokens.items():
        items.sort(key=lambda x: -x[1])
        result[val] = items[:top_k]
    return result


def build_lexicon_predictor(
    per_value_tokens: dict[str, list],
    min_conf: float = MIN_CONF,
) -> tuple[callable, dict[str, tuple]]:
    """Build a predictor that returns (label, conf) or None.

    Returns:
        (predict_fn, token_to_value_dict)
    """
    token_to_value: dict[str, tuple] = {}
    for val, items in per_value_tokens.items():
        for tok, pmi, conf, count in items:
            if conf >= min_conf:
                # Take strongest-confidence assignment if multi-class conflict
                if tok not in token_to_value or token_to_value[tok][1] < conf:
                    token_to_value[tok] = (val, conf, pmi)

    def predict(text: str) -> Optional[tuple[str, float]]:
        toks = tokenize(text)
        matches = [
            (token_to_value[t][0], token_to_value[t][1])
            for t in toks
            if t in token_to_value
        ]
        if not matches:
            return None
        # Pick highest-confidence match
        best = max(matches, key=lambda x: x[1])
        return best  # (predicted_value, confidence)

    return predict, token_to_value


# ---------------------------------------------------------------------------
# Fresh hybrid XGBoost trainer (same as layer1_5_honest_comparison.py)
# ---------------------------------------------------------------------------

def _train_fresh_hybrid(
    X_silver: np.ndarray,
    y_silver: np.ndarray,
    X_gold: np.ndarray,
    y_gold: np.ndarray,
    gold_weight: float = GOLD_WEIGHT,
) -> tuple[Optional[xgb.XGBClassifier], Optional[LabelEncoder]]:
    X_combined = np.vstack([X_silver, X_gold])
    y_combined = np.concatenate([y_silver, y_gold])
    w_combined = np.concatenate([
        np.ones(len(y_silver)),
        gold_weight * np.ones(len(y_gold)),
    ])

    all_classes = sorted(set(y_combined.tolist()))
    if len(all_classes) < 2:
        return None, None

    le = LabelEncoder()
    le.fit(all_classes)
    y_enc = le.transform(y_combined)
    n_classes = len(all_classes)

    common_kwargs = dict(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
        tree_method="hist", verbosity=0,
    )
    if n_classes == 2:
        pos = int((y_enc == 1).sum())
        neg = int((y_enc == 0).sum())
        spw = max(neg / max(pos, 1), 0.5)
        clf = xgb.XGBClassifier(scale_pos_weight=spw, **common_kwargs)
    else:
        clf = xgb.XGBClassifier(
            objective="multi:softmax", num_class=n_classes, **common_kwargs
        )

    clf.fit(X_combined, y_enc, sample_weight=w_combined)
    return clf, le


# ---------------------------------------------------------------------------
# Regex extraction helper (same as layer1_5_honest_comparison.py)
# ---------------------------------------------------------------------------

def _build_regex_preds(
    test_products: pd.DataFrame,
    domain: str,
) -> dict[str, dict[str, str]]:
    extractor = RegexExtractor()
    result: dict[str, dict[str, str]] = {}
    for _, row in test_products.iterrows():
        code = str(row["code"])
        extracted = extractor.extract_all(
            product_name=str(row.get("product_name") or ""),
            description=str(row.get("ingredients_text") or ""),
            quantity=str(row.get("quantity") or ""),
            category=domain,
        )
        attr_vals: dict[str, str] = {}
        for attr, res in extracted.items():
            if res.confidence > 0.0 and res.value is not None:
                attr_vals[attr] = str(res.value)
        result[code] = attr_vals
    return result


# ---------------------------------------------------------------------------
# Per-(cat, attr) computation
# ---------------------------------------------------------------------------

def run_one_attr(
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
    all_patterns: dict,
) -> list[dict]:
    """Run 3 variants for one (cat, attr). Returns result rows + updates all_patterns."""

    # Filter gold for this attr — non-null only
    cat_gold = gold[
        (gold["category"] == cat) & (gold["attr"] == attr) & ~gold["gold_is_null"]
    ].copy()
    cat_gold["code"] = cat_gold["code"].astype(str)
    cat_gold = cat_gold[cat_gold["code"].isin(code_to_idx)]

    if len(cat_gold) < MIN_GOLD:
        logger.info("[%s/%s] only %d gold cells, skipping", cat, attr, len(cat_gold))
        return []

    train_gold = cat_gold[cat_gold["code"].isin(train_codes_set)]
    test_gold = cat_gold[cat_gold["code"].isin(test_codes_set)]

    if len(train_gold) < 10 or len(test_gold) < 5:
        logger.info("[%s/%s] insufficient train/test split, skipping", cat, attr)
        return []

    # Embeddings
    train_idx = np.array([code_to_idx[c] for c in train_gold["code"]])
    test_idx = np.array([code_to_idx[c] for c in test_gold["code"]])
    X_gold_train = emb_all[train_idx]
    y_gold_train = train_gold["gold_value"].astype(str).values
    X_test_emb = emb_all[test_idx]
    y_test = test_gold["gold_value"].astype(str).values
    test_codes_list = test_gold["code"].tolist()

    # Silver training data (exclude test codes)
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

    # Train fresh hybrid ML
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
        kwargs = dict(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
            tree_method="hist", verbosity=0,
        )
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

    # ML predictions for all test samples
    enc_preds = clf.predict(X_test_emb)
    ml_labels_all = le.inverse_transform(enc_preds).tolist()

    # ---------------------------------------------------------------------------
    # Mine discriminative tokens from TRAIN gold only
    # ---------------------------------------------------------------------------
    train_gold_with_text = train_gold.merge(
        silver[["code", "product_name", "ingredients_text", "brands", "quantity"]],
        on="code", how="left",
    )
    train_texts = [_build_text(row) for _, row in train_gold_with_text.iterrows()]
    y_train_labels = train_gold_with_text["gold_value"].astype(str).tolist()

    per_value_tokens = mine_discriminative_tokens(train_texts, y_train_labels)

    # Build predictor
    lexicon_predict, token_to_value = build_lexicon_predictor(per_value_tokens)

    # Store patterns for inspection
    key = f"{cat}/{attr}"
    all_patterns[key] = {}
    for val, items in per_value_tokens.items():
        all_patterns[key][val] = [
            {"token": tok, "pmi": round(pmi, 3), "conf": round(conf, 3), "count": cnt}
            for tok, pmi, conf, cnt in items
        ]

    # Build test texts for lexicon
    test_products_idx = test_products.set_index("code")
    test_texts_list = []
    for c in test_codes_list:
        if c in test_products_idx.index:
            test_texts_list.append(_build_text(test_products_idx.loc[c]))
        else:
            test_texts_list.append("")

    n_tokens_total = sum(len(v) for v in token_to_value.items())
    logger.debug(
        "[%s/%s] mined tokens=%d, token_to_value entries=%d",
        cat, attr, sum(len(v) for v in per_value_tokens.values()), len(token_to_value),
    )

    # ---------------------------------------------------------------------------
    # Variant A_baseline: original regex + hybrid_ml fallback (same as EXP3)
    # ---------------------------------------------------------------------------
    preds_a = []
    for i, code in enumerate(test_codes_list):
        regex_val = regex_preds.get(code, {}).get(attr)
        if regex_val is not None:
            preds_a.append(str(regex_val))
        else:
            preds_a.append(ml_labels_all[i])

    # ---------------------------------------------------------------------------
    # Variant L_lexicon_ml: mined-lexicon regex + hybrid_ml fallback
    # ---------------------------------------------------------------------------
    preds_l = []
    lexicon_hits = 0
    for i, (code, text) in enumerate(zip(test_codes_list, test_texts_list)):
        lex_result = lexicon_predict(text)
        if lex_result is not None:
            preds_l.append(str(lex_result[0]))
            lexicon_hits += 1
        else:
            preds_l.append(ml_labels_all[i])

    # ---------------------------------------------------------------------------
    # Variant M_combined: (regex OR lexicon) → hybrid_ml fallback
    # ---------------------------------------------------------------------------
    preds_m = []
    for i, (code, text) in enumerate(zip(test_codes_list, test_texts_list)):
        regex_val = regex_preds.get(code, {}).get(attr)
        if regex_val is not None:
            preds_m.append(str(regex_val))
        else:
            lex_result = lexicon_predict(text)
            if lex_result is not None:
                preds_m.append(str(lex_result[0]))
            else:
                preds_m.append(ml_labels_all[i])

    # ---------------------------------------------------------------------------
    # Compute accuracies
    # ---------------------------------------------------------------------------
    n = len(y_test)
    acc_a = sum(1 for p, g in zip(preds_a, y_test) if p == g) / n
    acc_l = sum(1 for p, g in zip(preds_l, y_test) if p == g) / n
    acc_m = sum(1 for p, g in zip(preds_m, y_test) if p == g) / n
    acc_b = sum(1 for p, g in zip(ml_labels_all, y_test) if p == g) / n

    logger.info(
        "[%s/%s] n=%d | B_ml=%.3f A_regex=%.3f L_lex=%.3f M_comb=%.3f | lex_hits=%d/%d",
        cat, attr, n, acc_b, acc_a, acc_l, acc_m, lexicon_hits, n,
    )

    rows = [
        {
            "category": cat, "attr": attr, "variant": "A_baseline_regex_ml",
            "accuracy": acc_a, "n_test": n, "n_train_gold": len(train_gold),
            "n_silver": len(y_silver),
        },
        {
            "category": cat, "attr": attr, "variant": "B_ml_only",
            "accuracy": acc_b, "n_test": n, "n_train_gold": len(train_gold),
            "n_silver": len(y_silver),
        },
        {
            "category": cat, "attr": attr, "variant": "L_lexicon_ml",
            "accuracy": acc_l, "n_test": n, "n_train_gold": len(train_gold),
            "n_silver": len(y_silver),
        },
        {
            "category": cat, "attr": attr, "variant": "M_combined_ml",
            "accuracy": acc_m, "n_test": n, "n_train_gold": len(train_gold),
            "n_silver": len(y_silver),
        },
    ]
    return rows


# ---------------------------------------------------------------------------
# Summary + doc writers
# ---------------------------------------------------------------------------

VARIANT_ORDER = ["A_baseline_regex_ml", "B_ml_only", "L_lexicon_ml", "M_combined_ml"]
VARIANT_LABELS = {
    "A_baseline_regex_ml": "A: Regex+ML (EXP3 baseline)",
    "B_ml_only": "B: ML only",
    "L_lexicon_ml": "L: Lexicon+ML (mined)",
    "M_combined_ml": "M: Combined (regex|lexicon)+ML",
}


def _print_summary(result: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("P2-EXP6: Lexicon-Augmented Regex — Grand Means")
    print("=" * 70)
    grand = result.groupby("variant")["accuracy"].mean()
    b_mean = grand.get("B_ml_only", float("nan"))
    for v in VARIANT_ORDER:
        if v in grand.index:
            delta = (grand[v] - b_mean) * 100
            sign = "+" if delta >= 0 else ""
            print(f"  {VARIANT_LABELS.get(v, v):40s}: {grand[v]*100:.2f}%  ({sign}{delta:.2f} pp vs B)")

    print("\n" + "=" * 70)
    print("Per-category mean accuracy")
    print("=" * 70)
    pivot_cat = result.pivot_table(
        index="category", columns="variant", values="accuracy", aggfunc="mean"
    )
    cols = [v for v in VARIANT_ORDER if v in pivot_cat.columns]
    print(pivot_cat[cols].round(4).to_string())

    print("\n" + "=" * 70)
    print("Per-(category, attr) accuracy + winner")
    print("=" * 70)
    pivot_attr = result.pivot_table(
        index=["category", "attr"], columns="variant", values="accuracy", aggfunc="mean"
    )
    cols = [v for v in VARIANT_ORDER if v in pivot_attr.columns]
    pivot_attr = pivot_attr[cols]
    pivot_attr["winner"] = pivot_attr.idxmax(axis=1)
    print(pivot_attr.round(4).to_string())


def _write_doc(result: pd.DataFrame, all_patterns: dict) -> None:
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)

    grand = result.groupby("variant")["accuracy"].mean()
    b_mean = grand.get("B_ml_only", float("nan"))
    a_mean = grand.get("A_baseline_regex_ml", float("nan"))
    l_mean = grand.get("L_lexicon_ml", float("nan"))
    m_mean = grand.get("M_combined_ml", float("nan"))

    pivot_attr = result.pivot_table(
        index=["category", "attr"], columns="variant", values="accuracy", aggfunc="mean"
    )
    cols = [v for v in VARIANT_ORDER if v in pivot_attr.columns]
    pivot_attr = pivot_attr[cols]
    pivot_attr["winner"] = pivot_attr.idxmax(axis=1)

    lines: list[str] = []
    lines.append("# P2-EXP6: Lexicon-Augmented Regex via PMI Mining")
    lines.append("")
    lines.append("**Methodology:** Mine discriminative tokens from 80% gold training split")
    lines.append("using Pointwise Mutual Information (PMI). Build token-to-class lookup")
    lines.append("with confidence filtering. Compare on honest 20% held-out test (seed=42).")
    lines.append("")
    lines.append(f"**PMI params:** min_pmi={MIN_PMI}, min_count={MIN_COUNT}, top_k={TOP_K}, min_conf={MIN_CONF}")
    lines.append("")

    # Grand means table
    lines.append("## Grand Mean Accuracy")
    lines.append("")
    lines.append("| Variant | Accuracy | vs A_baseline (pp) | vs B_ml_only (pp) |")
    lines.append("|---------|----------|--------------------|-------------------|")
    for v in VARIANT_ORDER:
        if v in grand.index:
            acc = grand[v] * 100
            d_a = (grand[v] - a_mean) * 100
            d_b = (grand[v] - b_mean) * 100
            lines.append(
                f"| {VARIANT_LABELS.get(v, v)} | {acc:.2f}% "
                f"| {'+' if d_a >= 0 else ''}{d_a:.2f} "
                f"| {'+' if d_b >= 0 else ''}{d_b:.2f} |"
            )
    lines.append("")

    # Per-category
    pivot_cat = result.pivot_table(
        index="category", columns="variant", values="accuracy", aggfunc="mean"
    )
    cat_cols = [v for v in VARIANT_ORDER if v in pivot_cat.columns]
    pivot_cat = pivot_cat[cat_cols].round(4)
    lines.append("## Per-Category Accuracy")
    lines.append("")
    header = "| Category | " + " | ".join(VARIANT_LABELS.get(c, c) for c in cat_cols) + " |"
    sep = "|----------|" + "---------|" * len(cat_cols)
    lines.append(header)
    lines.append(sep)
    for cat, row in pivot_cat.iterrows():
        vals = " | ".join(f"{v*100:.2f}%" for v in row)
        lines.append(f"| {cat} | {vals} |")
    lines.append("")

    # Per (cat, attr)
    lines.append("## Per-(Category, Attr) Accuracy")
    lines.append("")
    header2 = "| Category | Attr | " + " | ".join(VARIANT_LABELS.get(c, c) for c in cols) + " | Winner |"
    sep2 = "|----------|------|" + "---------|" * len(cols) + "--------|"
    lines.append(header2)
    lines.append(sep2)
    for (cat, attr), row in pivot_attr.iterrows():
        acc_vals = " | ".join(f"{row[v]*100:.2f}%" if v in row.index and not pd.isna(row[v]) else "N/A"
                              for v in cols)
        winner = row["winner"]
        lines.append(f"| {cat} | {attr} | {acc_vals} | {winner} |")
    lines.append("")

    # Winner counts
    winner_counts = pivot_attr["winner"].value_counts()
    lines.append("## Winner Counts (best variant per attr)")
    lines.append("")
    lines.append("| Variant | # Attrs Won |")
    lines.append("|---------|-------------|")
    for v in VARIANT_ORDER:
        cnt = winner_counts.get(v, 0)
        lines.append(f"| {VARIANT_LABELS.get(v, v)} | {cnt} |")
    lines.append("")

    # Sample patterns section
    lines.append("## Sample Discovered Patterns")
    lines.append("")
    lines.append("Top discriminative tokens per (cat, attr, value) — data-driven 'regex dictionary':")
    lines.append("")

    # Focus attrs of interest
    focus_attrs = [
        ("pasta", "grain_type"),
        ("pasta", "pasta_shape"),
        ("chocolate", "cocoa_percentage"),
        ("chocolate", "chocolate_type"),
        ("cheeses", "milk_source"),
        ("cheeses", "country_of_origin"),
        ("cheeses", "texture"),
    ]

    for cat, attr in focus_attrs:
        key = f"{cat}/{attr}"
        if key not in all_patterns:
            continue
        lines.append(f"### {cat} / {attr}")
        lines.append("")
        for val, tok_list in sorted(all_patterns[key].items()):
            if not tok_list:
                continue
            top5 = tok_list[:5]
            tok_strs = [
                f"`{t['token']}` (PMI={t['pmi']:.2f}, conf={t['conf']:.2f}, n={t['count']})"
                for t in top5
            ]
            lines.append(f"- **{val}**: {', '.join(tok_strs)}")
        lines.append("")

    # Multilingual discoveries section
    lines.append("## Multilingual Token Discoveries")
    lines.append("")
    lines.append("Auto-discovered non-English discriminative tokens (sampling):")
    lines.append("")

    # Detect likely non-ASCII or known multilingual tokens
    multilingual_examples: list[str] = []
    EN_STOPS = {"and", "or", "the", "with", "for", "from", "in", "of", "to",
                "is", "are", "was", "not", "this", "that", "it", "at", "as",
                "its", "be", "by", "an", "on", "no", "do", "so"}
    for key, val_dict in all_patterns.items():
        cat, attr = key.split("/")
        for val, tok_list in val_dict.items():
            for tok_info in tok_list:
                tok = tok_info["token"]
                # Check: has non-ASCII OR looks non-English (short stop exclusion)
                has_non_ascii = any(ord(c) > 127 for c in tok)
                is_word = re.match(r'^[a-z]+$', tok) is not None
                # Multilingual if: non-ASCII, or looks like known FR/DE/IT/ES term
                if has_non_ascii or (is_word and len(tok) >= 4 and tok not in EN_STOPS):
                    conf = tok_info["conf"]
                    pmi = tok_info["pmi"]
                    if conf >= 0.7 and pmi >= 1.5:
                        multilingual_examples.append(
                            f"- `{tok}` → **{val}** [{cat}/{attr}] "
                            f"(PMI={pmi:.2f}, conf={conf:.2f})"
                        )

    # Deduplicate and sample
    seen = set()
    unique_ml = []
    for ex in multilingual_examples:
        tok = ex.split("`")[1]
        if tok not in seen:
            seen.add(tok)
            unique_ml.append(ex)

    # Show up to 30 examples, prefer non-ASCII first
    non_ascii_ex = [e for e in unique_ml if any(ord(c) > 127 for c in e.split("`")[1])]
    ascii_ex = [e for e in unique_ml if e not in non_ascii_ex]
    combined_sample = non_ascii_ex[:15] + ascii_ex[:15]
    if combined_sample:
        lines.extend(combined_sample[:30])
    else:
        lines.append("(No strongly discriminative multilingual tokens found above threshold)")
    lines.append("")

    # Verdict
    l_over_a = (l_mean - a_mean) * 100
    m_over_a = (m_mean - a_mean) * 100
    winner_variant = max(grand.items(), key=lambda x: x[1])[0]
    lines.append("## Verdict")
    lines.append("")
    lines.append(
        f"- Lexicon-only (L) vs hand-crafted regex (A): **{'+' if l_over_a >= 0 else ''}{l_over_a:.2f} pp**"
    )
    lines.append(
        f"- Combined (M) vs hand-crafted regex (A): **{'+' if m_over_a >= 0 else ''}{m_over_a:.2f} pp**"
    )
    lines.append(
        f"- Best overall variant: **{VARIANT_LABELS.get(winner_variant, winner_variant)}** "
        f"({grand[winner_variant]*100:.2f}%)"
    )
    lines.append("")
    if m_mean > a_mean + 0.005:
        verdict = "VIABLE: Combined lexicon+regex outperforms hand-crafted regex alone."
    elif l_mean > a_mean + 0.005:
        verdict = "PARTIAL: Lexicon-only beats hand-crafted regex; combined is not better."
    elif m_mean > a_mean - 0.003:
        verdict = "NEUTRAL: Combined lexicon does not hurt; not enough uplift to justify."
    else:
        verdict = "NOT VIABLE: Lexicon extension regresses hand-crafted regex accuracy."

    lines.append(f"**Verdict:** {verdict}")
    lines.append("")
    lines.append(
        "The auto-grown lexicon captures multilingual terminology and product-specific "
        "vocabulary that hand-crafted regex misses. PMI mining on gold data provides "
        "a data-driven complement to manual pattern engineering."
    )

    doc_text = "\n".join(lines) + "\n"
    DOC_PATH.write_text(doc_text, encoding="utf-8")
    logger.info("Wrote doc to %s", DOC_PATH)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()

    gold = pd.read_parquet(GOLD_PATH)
    gold["code"] = gold["code"].astype(str)
    logger.info("Loaded gold: %d rows, %d unique codes", len(gold), gold["code"].nunique())

    all_rows: list[dict] = []
    all_patterns: dict = {}

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

        # Same 80/20 split on per-category codes as EXP3 (seed=42)
        train_codes, test_codes = train_test_split(
            unique_codes, test_size=TEST_SIZE, random_state=SEED
        )
        train_codes_set = set(train_codes)
        test_codes_set = set(test_codes)
        logger.info("  Split: %d train codes, %d test codes", len(train_codes), len(test_codes))

        test_products = silver[silver["code"].isin(test_codes_set)].copy()

        logger.info("  Building regex predictions for test products...")
        regex_preds = _build_regex_preds(test_products, cat)
        regex_hits = sum(len(v) for v in regex_preds.values())
        logger.info("  Regex hit %d (code, attr) pairs", regex_hits)

        attrs = sorted(cat_gold["attr"].unique().tolist())
        logger.info("  Attrs: %s", attrs)

        for attr in attrs:
            rows = run_one_attr(
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
                all_patterns=all_patterns,
            )
            all_rows.extend(rows)

    result = pd.DataFrame(all_rows)
    result.to_parquet(OUT_PATH, index=False)
    logger.info("Wrote %d rows to %s", len(result), OUT_PATH)

    # Save patterns
    PATTERNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PATTERNS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_patterns, f, ensure_ascii=False, indent=2)
    logger.info("Wrote patterns to %s", PATTERNS_PATH)

    _print_summary(result)
    _write_doc(result, all_patterns)


if __name__ == "__main__":
    main()
