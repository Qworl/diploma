"""
Ablation hard-thresholds в audit_silver: COVERAGE_MIN, MIN_CLASS_SIZE, TOP_CLASS_MAX.

Без обоснования значения 0.40 / 10 / 0.70 выглядят произвольно. Этот скрипт
прогоняет grid и показывает, насколько меняется decisions distribution.
Полезно для дипломной — превращает «hard rules» в видимый sensitivity analysis.

Usage:
    python -m src.diagnostics.silver.audit_ablation
    python -m src.diagnostics.silver.audit_ablation --json > ablation.json
"""

import argparse
import itertools
import json
import logging
import os
import sys

import pandas as pd

from src.common import setup_logging

from src.diagnostics.silver import audit as audit_silver
from src.diagnostics.silver.audit import audit_attr, CATEGORY_CONFIG, PROCESSED_DIR

logger = logging.getLogger(__name__)

# Diapason для каждого порога — узкая решётка вокруг текущих значений
COVERAGE_MIN_GRID = [0.20, 0.30, 0.40, 0.50, 0.60]
MIN_CLASS_SIZE_GRID = [5, 10, 15, 20]
TOP_CLASS_MAX_GRID = [0.60, 0.70, 0.80, 0.90]


def run_one(df: pd.DataFrame, schema: dict,
            cov_min: float, min_class: int, top_max: float) -> dict:
    """Прогнать audit_attr с monkey-patched порогами и собрать decision counts."""
    orig = (audit_silver.COVERAGE_MIN, audit_silver.MIN_CLASS_SIZE, audit_silver.TOP_CLASS_MAX)
    audit_silver.COVERAGE_MIN = cov_min
    audit_silver.MIN_CLASS_SIZE = min_class
    audit_silver.TOP_CLASS_MAX = top_max
    try:
        decisions = []
        for attr in schema:
            if attr not in df.columns:
                continue
            info = audit_attr(df[attr], schema[attr])
            decisions.append(info["decision"])
    finally:
        audit_silver.COVERAGE_MIN, audit_silver.MIN_CLASS_SIZE, audit_silver.TOP_CLASS_MAX = orig
    counts = {d: 0 for d in ("ml_ready", "ml_ready_with_caveat", "bayesian_only", "drop")}
    for d in decisions:
        counts[d] = counts.get(d, 0) + 1
    return counts


def ablate_category(category: str) -> pd.DataFrame:
    cfg = CATEGORY_CONFIG[category]
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, cfg["silver_standard"]))
    rows = []
    for cov, mc, tc in itertools.product(COVERAGE_MIN_GRID, MIN_CLASS_SIZE_GRID, TOP_CLASS_MAX_GRID):
        counts = run_one(df, cfg["schema"], cov, mc, tc)
        rows.append({
            "category": category,
            "coverage_min": cov,
            "min_class_size": mc,
            "top_class_max": tc,
            **counts,
        })
    return pd.DataFrame(rows)


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true",
                        help="Output JSON to stdout (для notebook)")
    parser.add_argument("--out", default=None,
                        help="Save merged ablation parquet (default: PROCESSED_DIR/audit_ablation.parquet)")
    args = parser.parse_args()

    all_rows = []
    for cat in CATEGORY_CONFIG:
        all_rows.append(ablate_category(cat))
    df = pd.concat(all_rows, ignore_index=True)

    out = args.out or os.path.join(PROCESSED_DIR, "audit_ablation.parquet")
    df.to_parquet(out, index=False)
    logger.info("Saved ablation grid -> %s (%d rows)", out, len(df))

    if args.json:
        sys.stdout.write(df.to_json(orient="records", indent=2))
        sys.stdout.write("\n")
        return

    # Текущие значения = 0.40 / 10 / 0.70 — печатаем sensitivity отдельно
    logger.info("=" * 70)
    logger.info("Sensitivity вокруг текущих значений (0.40 / 10 / 0.70)")
    logger.info("=" * 70)
    for cat in CATEGORY_CONFIG:
        sub = df[df.category == cat].copy()
        baseline_mask = (
            (sub.coverage_min == 0.40)
            & (sub.min_class_size == 10)
            & (sub.top_class_max == 0.70)
        )
        if not baseline_mask.any():
            continue
        baseline = sub[baseline_mask].iloc[0]
        logger.info("\n%s — baseline:  ml_ready=%d, caveat=%d, bayes=%d, drop=%d",
                    cat, int(baseline["ml_ready"]), int(baseline["ml_ready_with_caveat"]),
                    int(baseline["bayesian_only"]), int(baseline["drop"]))
        for col, lower, upper in [
            ("coverage_min", 0.30, 0.50),
            ("min_class_size", 5, 20),
            ("top_class_max", 0.60, 0.80),
        ]:
            for v in (lower, upper):
                base_others = (
                    (sub.coverage_min == (0.40 if col != "coverage_min" else v))
                    & (sub.min_class_size == (10 if col != "min_class_size" else v))
                    & (sub.top_class_max == (0.70 if col != "top_class_max" else v))
                )
                if not base_others.any():
                    continue
                row = sub[base_others].iloc[0]
                logger.info("  %s=%s: ml=%d caveat=%d bayes=%d drop=%d",
                            col, v, int(row["ml_ready"]), int(row["ml_ready_with_caveat"]),
                            int(row["bayesian_only"]), int(row["drop"]))


if __name__ == "__main__":
    main()
