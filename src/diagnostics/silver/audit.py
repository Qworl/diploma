"""
Pre-training data quality audit for silver standard.

For each attribute in a category's schema, computes:
- Coverage (% non-null)
- Class distribution (n classes, top class %, min class size)
- Recommendation: ML-ready / Bayesian-only / drop

Hard rules for "drop from ML":
- Coverage < 40% → not enough signal for ML
- Min class < 10 samples → rare classes hopeless
- Top class > 70% → trivial baseline, ML adds nothing

Usage:
    python -m src.diagnostics.silver.audit --category pasta
    python -m src.diagnostics.silver.audit --category chocolate --json
"""

import argparse
import json
import logging
import os
import sys

import pandas as pd

from src.common import PROCESSED_DIR, setup_logging
from src.pipeline.schemas import PASTA_SCHEMA

# Try-import schemas that may not exist yet
try:
    from src.pipeline.schemas import CHOCOLATE_SCHEMA
except ImportError:
    CHOCOLATE_SCHEMA = None
try:
    from src.pipeline.schemas import BEVERAGE_SCHEMA
except ImportError:
    BEVERAGE_SCHEMA = None
try:
    from src.pipeline.schemas import ELECTRONICS_SCHEMA
except ImportError:
    ELECTRONICS_SCHEMA = None

logger = logging.getLogger(__name__)

CATEGORY_CONFIG = {
    "pasta": {
        "silver_standard": "pasta_silver_standard.parquet",
        "schema": PASTA_SCHEMA,
    },
}
if CHOCOLATE_SCHEMA is not None:
    CATEGORY_CONFIG["chocolate"] = {
        "silver_standard": "chocolate_silver_standard.parquet",
        "schema": CHOCOLATE_SCHEMA,
    }
if BEVERAGE_SCHEMA is not None:
    CATEGORY_CONFIG["beverages"] = {
        "silver_standard": "beverages_silver_standard.parquet",
        "schema": BEVERAGE_SCHEMA,
    }
if ELECTRONICS_SCHEMA is not None:
    CATEGORY_CONFIG["electronics"] = {
        "silver_standard": "electronics_silver_standard.parquet",
        "schema": ELECTRONICS_SCHEMA,
    }

# Hard rules (matched against decisions in train_classifiers)
COVERAGE_MIN = 0.40
MIN_CLASS_SIZE = 10
TOP_CLASS_MAX = 0.70


def audit_attr(series: pd.Series, schema_spec: dict) -> dict:
    """Audit a single attribute column. Returns decision dict."""
    n_total = len(series)
    n_non_null = series.notna().sum()
    coverage = n_non_null / n_total if n_total else 0.0

    if n_non_null == 0:
        return {
            "coverage": 0.0,
            "n_classes": 0,
            "decision": "drop",
            "reason": "0% coverage — attribute is entirely None",
        }

    counts = series.dropna().value_counts()
    n_classes = len(counts)
    top_class = counts.index[0]
    top_share = counts.iloc[0] / n_non_null
    min_class_size = int(counts.iloc[-1])

    # Decision logic
    reasons = []
    if coverage < COVERAGE_MIN:
        reasons.append(f"coverage {coverage:.1%} < {COVERAGE_MIN:.0%}")
    if min_class_size < MIN_CLASS_SIZE:
        reasons.append(f"min class {min_class_size} < {MIN_CLASS_SIZE}")
    if top_share > TOP_CLASS_MAX:
        reasons.append(f"top class '{top_class}' = {top_share:.0%} > {TOP_CLASS_MAX:.0%}")

    if not reasons:
        decision = "ml_ready"
        reason = "all gates pass"
    elif coverage < COVERAGE_MIN:
        decision = "drop"
        reason = "; ".join(reasons)
    elif min_class_size < MIN_CLASS_SIZE and n_classes <= 2:
        # binary attr with too-few positives
        decision = "drop"
        reason = "; ".join(reasons) + " — binary, can't train"
    elif min_class_size < MIN_CLASS_SIZE:
        # multiclass with rare class — drop the rare class but keep training
        decision = "ml_ready_with_caveat"
        reason = "; ".join(reasons) + " — consider merging rare classes into 'other'"
    else:
        # imbalanced but coverage OK
        decision = "bayesian_only"
        reason = "; ".join(reasons) + " — too imbalanced for ML, try Bayesian via parents"

    return {
        "coverage": float(coverage),
        "n_total": int(n_total),
        "n_non_null": int(n_non_null),
        "n_classes": int(n_classes),
        "top_class": str(top_class),
        "top_share": float(top_share),
        "min_class_size": int(min_class_size),
        "class_distribution": {str(k): int(v) for k, v in counts.head(10).items()},
        "decision": decision,
        "reason": reason,
    }


def audit_category(category: str) -> dict:
    """Audit all attributes in a category's silver standard."""
    cfg = CATEGORY_CONFIG[category]
    path = os.path.join(PROCESSED_DIR, cfg["silver_standard"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Silver standard missing: {path}")

    df = pd.read_parquet(path)
    schema = cfg["schema"]

    results = {
        "category": category,
        "n_products": len(df),
        "n_columns": len(df.columns),
        "attributes": {},
    }

    for attr in schema:
        if attr not in df.columns:
            results["attributes"][attr] = {
                "decision": "drop",
                "reason": "column missing from silver standard",
            }
            continue
        results["attributes"][attr] = audit_attr(df[attr], schema[attr])

    # Summary
    decisions = [info["decision"] for info in results["attributes"].values()]
    results["summary"] = {
        "ml_ready": sum(1 for d in decisions if d == "ml_ready"),
        "ml_ready_with_caveat": sum(1 for d in decisions if d == "ml_ready_with_caveat"),
        "bayesian_only": sum(1 for d in decisions if d == "bayesian_only"),
        "drop": sum(1 for d in decisions if d == "drop"),
    }
    return results


def log_report(results: dict) -> None:
    logger.info("=" * 78)
    logger.info("DATA QUALITY AUDIT — %s", results["category"])
    logger.info("=" * 78)
    logger.info("Silver standard: %d products, %d columns",
                results["n_products"], results["n_columns"])
    logger.info("%-22s %6s %5s %6s %6s  Decision",
                "Attribute", "Cov", "NCls", "Top", "MinSz")
    logger.info("-" * 78)
    for attr, info in results["attributes"].items():
        if info["decision"] == "drop" and "column missing" in info.get("reason", ""):
            logger.info("%-22s %6s %5s %6s %6s  DROP (col missing)",
                        attr, "—", "—", "—", "—")
            continue
        # 0% coverage rows не имеют top_share / min_class_size
        if "top_share" not in info:
            logger.info("%-22s %6s %5d %6s %6s  DROP (0%% coverage)",
                        attr, f"{info.get('coverage', 0):.1%}", info.get("n_classes", 0),
                        "—", "—")
            continue
        logger.info(
            "%-22s %6s %5d %6s %6d  %s",
            attr, f"{info['coverage']:.1%}", info["n_classes"],
            f"{info['top_share']:.0%}", info["min_class_size"],
            info["decision"].upper(),
        )
        if info["decision"] != "ml_ready":
            logger.info("  → %s", info["reason"])

    s = results["summary"]
    logger.info(
        "Summary: %d ml-ready, %d caveats, %d bayesian-only, %d drop",
        s["ml_ready"], s["ml_ready_with_caveat"], s["bayesian_only"], s["drop"],
    )


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, choices=list(CATEGORY_CONFIG.keys()))
    parser.add_argument("--json", action="store_true", help="Output JSON instead of table")
    args = parser.parse_args()

    results = audit_category(args.category)

    if args.json:
        # JSON mode: write to stdout for piping; logger goes to stderr.
        sys.stdout.write(json.dumps(results, indent=2, default=str) + "\n")
    else:
        log_report(results)


if __name__ == "__main__":
    main()
