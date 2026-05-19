"""E2: OFF↔OFF circularity probe.

Train a trivial TF-IDF + LogisticRegression baseline on partner-input text only
(product_name + brands + ingredients_text + quantity) and compare its
accuracy to the cascade's reported accuracy (acc_oracle_cat) on the SAME
brand-disjoint test set.

If probe_acc ≈ cascade_acc → cascade's accuracy is largely explainable by
trivial text patterns / class skew → eval is measuring task structure, not
the cascade's representations. Tier per attribute:
    safe       — leakage_ratio < 0.70
    moderate   — 0.70 ≤ leakage_ratio < 0.85
    suspect    — leakage_ratio ≥ 0.85

This is NOT a train/test leakage check (no codes overlap). It's a "how much of
cascade accuracy is structural?" check for §4.2 of the thesis.

Usage:
    OMP_NUM_THREADS=1 python -m src.diagnostics.silver.leakage_probe
"""

from __future__ import annotations

import logging
import os
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from src.common import PARTNER_TEXT_FIELDS, PROCESSED_DIR, setup_logging

CATEGORIES = ("pasta", "chocolate", "cheeses")
SILVER_TMPL = os.path.join(PROCESSED_DIR, "{cat}_stratified_silver_standard.parquet")
CASCADE_TMPL = os.path.join(
    PROCESSED_DIR, "cascade_preds_{cat}_v2_gold_hybrid_v3_fixed.parquet"
)
HEADLINE_PATH = os.path.join(PROCESSED_DIR, "headline_v3e_final.parquet")
SIGNAL_PATH = os.path.join(PROCESSED_DIR, "attribute_signal_taxonomy.parquet")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "off_leakage_probe.parquet")

logger = logging.getLogger("leakage_probe")


def _build_text_vectorized(df: pd.DataFrame) -> list[str]:
    """Concatenate partner text fields, NaN-safe. Vectorized for speed."""
    parts = []
    for col in PARTNER_TEXT_FIELDS:
        if col in df.columns:
            parts.append(df[col].astype("string").fillna(""))
        else:
            parts.append(pd.Series([""] * len(df), index=df.index))
    out = parts[0].str.cat(parts[1:], sep=" ", na_rep="")
    return out.fillna("").tolist()


def _classify_tier(ratio: float) -> str:
    if not np.isfinite(ratio):
        return "n/a"
    if ratio < 0.70:
        return "safe"
    if ratio < 0.85:
        return "moderate"
    return "suspect"


def _run_attr(
    silver: pd.DataFrame,
    test_codes: set[str],
    attr: str,
) -> dict | None:
    """Train TF-IDF + LR probe and evaluate on brand-disjoint test set."""
    if attr not in silver.columns:
        logger.warning("attr=%s missing in silver, skipping", attr)
        return None

    # Restrict to non-null gold (mirrors the oracle eval which only scores
    # cells with non-null silver label).
    have_label = silver[silver[attr].notna()].copy()
    if have_label.empty:
        return None

    is_test = have_label["code"].isin(test_codes)
    train_df = have_label[~is_test]
    test_df = have_label[is_test]

    n_train = len(train_df)
    n_test = len(test_df)
    if n_train < 10 or n_test < 5:
        logger.warning(
            "attr=%s too small (n_train=%d, n_test=%d), skipping", attr, n_train, n_test
        )
        return None

    # Targets — cast to string for label encoder (handles bools/numerics uniformly).
    y_train_raw = train_df[attr].astype(str).to_numpy()
    y_test_raw = test_df[attr].astype(str).to_numpy()

    # Encode labels. Unseen test labels get filtered out of accuracy denominator
    # to remain comparable to cascade (which can't predict unseen classes either).
    le = LabelEncoder()
    le.fit(y_train_raw)
    train_classes = set(le.classes_.tolist())

    seen_test_mask = np.array([y in train_classes for y in y_test_raw])
    n_test_seen = int(seen_test_mask.sum())
    n_test_unseen = int((~seen_test_mask).sum())
    if n_test_seen < 5:
        logger.warning(
            "attr=%s only %d test rows with seen labels, skipping", attr, n_test_seen
        )
        return None

    # Text features
    train_text = _build_text_vectorized(train_df)
    test_text = _build_text_vectorized(test_df)

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), lowercase=True)
    X_train = vec.fit_transform(train_text)
    X_test = vec.transform(test_text)

    y_train = le.transform(y_train_raw)

    # If only one class in train, predict majority — LogReg requires ≥2.
    if len(train_classes) < 2:
        majority = le.classes_[0]
        preds = np.array([majority] * n_test)
        correct = int(((preds == y_test_raw) & seen_test_mask).sum())
        probe_acc = correct / n_test_seen if n_test_seen else float("nan")
        logger.info(
            "attr=%s SINGLE-CLASS train → majority baseline acc=%.4f",
            attr,
            probe_acc,
        )
        return {
            "n_train": n_train,
            "n_test": n_test_seen,
            "n_test_unseen_labels": n_test_unseen,
            "probe_acc": probe_acc,
            "majority_baseline_only": True,
        }

    # Class-balance-aware logistic regression.
    clf = LogisticRegression(max_iter=1000, n_jobs=1)
    clf.fit(X_train, y_train)
    pred_idx = clf.predict(X_test)
    preds = le.inverse_transform(pred_idx)

    # Score only on test rows whose true label is seen in train (apples-to-apples
    # with cascade, which couldn't have predicted unseen classes either).
    correct = int(((preds == y_test_raw) & seen_test_mask).sum())
    probe_acc = correct / n_test_seen

    # Majority baseline (sanity for skew-driven attrs).
    maj_class = pd.Series(y_train_raw).value_counts().index[0]
    maj_correct = int(((y_test_raw == maj_class) & seen_test_mask).sum())
    maj_acc = maj_correct / n_test_seen

    return {
        "n_train": n_train,
        "n_test": n_test_seen,
        "n_test_unseen_labels": n_test_unseen,
        "probe_acc": probe_acc,
        "majority_acc": maj_acc,
        "majority_baseline_only": False,
    }


def main() -> None:
    setup_logging()
    headline = pd.read_parquet(HEADLINE_PATH)
    signal = pd.read_parquet(SIGNAL_PATH)[["category", "attr", "signal_type"]]

    rows: list[dict] = []

    for cat in CATEGORIES:
        logger.info("=== category=%s ===", cat)
        silver_path = SILVER_TMPL.format(cat=cat)
        cascade_path = CASCADE_TMPL.format(cat=cat)
        silver = pd.read_parquet(silver_path)
        silver["code"] = silver["code"].astype(str)
        cascade = pd.read_parquet(cascade_path)
        cascade["code"] = cascade["code"].astype(str)
        test_codes = set(cascade["code"].unique())

        cat_attrs: Iterable[str] = headline.loc[
            headline["category"] == cat, "attr"
        ].tolist()

        for attr in cat_attrs:
            cascade_acc_row = headline[
                (headline["category"] == cat) & (headline["attr"] == attr)
            ]
            if cascade_acc_row.empty:
                logger.warning("no headline row for %s/%s", cat, attr)
                continue
            cascade_acc = float(cascade_acc_row["acc_oracle_cat"].iloc[0])

            res = _run_attr(silver, test_codes, attr)
            if res is None:
                continue

            ratio = res["probe_acc"] / cascade_acc if cascade_acc > 0 else float("nan")
            tier = _classify_tier(ratio)

            sig_row = signal[(signal["category"] == cat) & (signal["attr"] == attr)]
            signal_type = (
                str(sig_row["signal_type"].iloc[0]) if not sig_row.empty else "unknown"
            )

            log_msg = (
                f"  {attr:>22s}  probe={res['probe_acc']:.3f}  "
                f"casc={cascade_acc:.3f}  ratio={ratio:.3f}  tier={tier:<8s}"
                f"  n_tr={res['n_train']}  n_te={res['n_test']}"
                f"  sig={signal_type}"
            )
            if not res["majority_baseline_only"]:
                log_msg += f"  maj={res['majority_acc']:.3f}"
            logger.info(log_msg)

            rows.append(
                {
                    "category": cat,
                    "attr": attr,
                    "n_train": res["n_train"],
                    "n_test": res["n_test"],
                    "n_test_unseen_labels": res["n_test_unseen_labels"],
                    "probe_acc": res["probe_acc"],
                    "majority_acc": res.get("majority_acc", float("nan")),
                    "cascade_acc": cascade_acc,
                    "leakage_ratio": ratio,
                    "tier": tier,
                    "signal_type": signal_type,
                }
            )

    out = pd.DataFrame(rows)
    out.to_parquet(OUTPUT_PATH, index=False)
    logger.info("Saved %d rows to %s", len(out), OUTPUT_PATH)

    # --- Summary ---
    print()
    print("=" * 78)
    print("LEAKAGE PROBE SUMMARY")
    print("=" * 78)

    tier_counts = out["tier"].value_counts().to_dict()
    print("\nPer-tier counts:")
    for tier in ("safe", "moderate", "suspect", "n/a"):
        print(f"  {tier:<10s}: {tier_counts.get(tier, 0)}")

    print("\nPer-attr table (sorted by leakage_ratio desc):")
    sorted_out = out.sort_values("leakage_ratio", ascending=False)
    print(
        sorted_out[
            [
                "category",
                "attr",
                "n_train",
                "n_test",
                "probe_acc",
                "majority_acc",
                "cascade_acc",
                "leakage_ratio",
                "tier",
                "signal_type",
            ]
        ].to_string(index=False)
    )

    print("\nSuspect attrs (ratio ≥ 0.85): probe alone explains most of cascade acc.")
    suspect = sorted_out[sorted_out["tier"] == "suspect"]
    if suspect.empty:
        print("  (none)")
    else:
        for _, r in suspect.iterrows():
            print(
                f"  {r['category']:<10s}/{r['attr']:<22s}  "
                f"probe={r['probe_acc']:.3f}  casc={r['cascade_acc']:.3f}  "
                f"ratio={r['leakage_ratio']:.3f}  maj={r['majority_acc']:.3f}  "
                f"sig={r['signal_type']}"
            )

    print("\nNarrative — 'harder than they look' (low probe, high cascade → real ML lift):")
    real_lift = sorted_out[sorted_out["tier"] == "safe"]
    for _, r in real_lift.iterrows():
        lift = r["cascade_acc"] - r["probe_acc"]
        print(
            f"  {r['category']:<10s}/{r['attr']:<22s}  "
            f"probe={r['probe_acc']:.3f}  casc={r['cascade_acc']:.3f}  "
            f"lift=+{lift:.3f}  sig={r['signal_type']}"
        )

    print("\nNarrative — 'easy by structure' (high probe ≈ high cascade):")
    easy = sorted_out[sorted_out["tier"].isin(["suspect", "moderate"])]
    for _, r in easy.iterrows():
        print(
            f"  {r['category']:<10s}/{r['attr']:<22s}  "
            f"probe={r['probe_acc']:.3f}  casc={r['cascade_acc']:.3f}  "
            f"ratio={r['leakage_ratio']:.3f}  maj={r['majority_acc']:.3f}  "
            f"sig={r['signal_type']}"
        )


if __name__ == "__main__":
    main()
