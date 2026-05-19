"""
DAG structure bootstrap — насколько Hill Climb граф воспроизводим.

Hill Climb + BIC чувствителен к sample. Запускаем структурное обучение N раз на
bootstrap subsamples (с возвратом). Edge стабилен, если воспроизводится в >X% запусков.

Это критично для дипломной: если защитник скажет «вы получили этот граф
случайно», bootstrap-frequency ответ: «edge brand→os воспроизводится в 100%
запусков (стабильный сигнал), edge storage→brand — только в 30% (артефакт sample)».

Usage:
    python -m src.diagnostics.ml.dag_bootstrap --category electronics --n-bootstrap 30
    python -m src.diagnostics.ml.dag_bootstrap --category beverages --n-bootstrap 30
"""

import argparse
import collections
import logging
import os
import sys
import warnings

import pandas as pd
import numpy as np

from src.common import PROCESSED_DIR, setup_logging

warnings.filterwarnings("ignore", category=FutureWarning)

from pgmpy.estimators import HillClimbSearch  # noqa: E402

logger = logging.getLogger(__name__)


CATEGORY_PREP = {
    "electronics": {
        "silver": "electronics_silver_standard.parquet",
        "cols": ["brand", "os", "form_factor", "screen_size_class",
                 "ram_class", "storage_class", "release_year_class"],
    },
    "beverages": {
        "silver": "beverages_stratified_silver_standard.parquet",
        "cols": ["beverage_type", "sugar_class", "is_organic", "is_carbonated",
                 "nutri_score_grade", "nova_group", "protein_class", "is_vegan"],
    },
    "chocolate": {
        "silver": "chocolate_stratified_silver_standard.parquet",
        "cols": ["chocolate_type", "cocoa_percentage", "contains_nuts",
                 "chocolate_extra", "is_organic", "nutri_score_grade", "protein_class"],
    },
    "pasta": {
        "silver": "pasta_stratified_silver_standard.parquet",
        "cols": ["grain_type", "pasta_shape", "is_organic", "is_filled",
                 "is_gluten_free", "is_vegan", "nutri_score_grade", "protein_class"],
    },
    "cheeses": {
        "silver": "cheeses_stratified_silver_standard.parquet",
        "cols": ["milk_source", "texture", "country_of_origin", "fat_class",
                 "is_pdo", "is_organic", "is_ultra_processed"],
    },
    "cereals": {
        "silver": "cereals_stratified_silver_standard.parquet",
        "cols": ["cereal_type", "grain_type", "is_low_sugar", "is_high_fibre",
                 "nova_class", "is_vegan", "is_whole_grain", "is_organic"],
    },
    "cosmetics": {
        "silver": "cosmetics_stratified_silver_standard.parquet",
        "cols": ["product_type", "form_factor", "body_area",
                 "has_sulfates", "has_silicones", "is_organic"],
    },
}


def prepare(category: str) -> pd.DataFrame:
    cfg = CATEGORY_PREP[category]
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, cfg["silver"]))
    cols = [c for c in cfg["cols"] if c in df.columns]
    out = df[cols].copy()
    # Bin continuous categories: keep top-N to avoid blowup
    for col in out.columns:
        out[col] = out[col].fillna("unknown").astype(str)
        # cap cardinality at 12 — top values + 'other'
        vc = out[col].value_counts()
        if len(vc) > 12:
            keep = set(vc.head(12).index)
            out[col] = out[col].where(out[col].isin(keep), "other")
    return out.dropna()


def learn_one(data: pd.DataFrame, max_indegree: int = 3) -> set:
    """Returns set of edges (src, dst)."""
    hc = HillClimbSearch(data)
    best = hc.estimate(scoring_method="bic-d", max_indegree=max_indegree, show_progress=False)
    return set(best.edges())


def bootstrap_dag(category: str, n_bootstrap: int = 30, seed_base: int = 42):
    data = prepare(category)
    logger.info("Loaded %s — %d rows, %d cols", category, len(data), len(data.columns))

    # Reference graph on full data
    logger.info("Reference graph on full data...")
    ref_edges = learn_one(data)
    logger.info("  %d edges", len(ref_edges))

    edge_counter = collections.Counter()
    n_edges_per_run = []
    rng = np.random.default_rng(seed_base)

    for i in range(n_bootstrap):
        # Bootstrap (sample with replacement) — same size as original
        idx = rng.integers(0, len(data), size=len(data))
        sub = data.iloc[idx].reset_index(drop=True)
        edges = learn_one(sub)
        edge_counter.update(edges)
        n_edges_per_run.append(len(edges))
        logger.info("  bootstrap %d/%d: %d edges", i + 1, n_bootstrap, len(edges))

    # Stability per edge
    rows = []
    all_edges = set(edge_counter.keys()) | ref_edges
    for src, dst in sorted(all_edges):
        freq = edge_counter[(src, dst)] / n_bootstrap
        in_ref = (src, dst) in ref_edges
        # Inverse direction frequency (does Hill Climb sometimes flip?)
        rev_freq = edge_counter[(dst, src)] / n_bootstrap
        rows.append({
            "category": category,
            "edge": f"{src} → {dst}",
            "src": src, "dst": dst,
            "in_reference": in_ref,
            "bootstrap_freq": float(freq),
            "reverse_freq": float(rev_freq),
            "interpretation": (
                "STABLE" if freq >= 0.80 else
                "SOMETIMES" if freq >= 0.50 else
                "RARE"
            ),
        })

    df_stab = pd.DataFrame(rows).sort_values("bootstrap_freq", ascending=False)
    logger.info("\n" + "=" * 78)
    logger.info("DAG STRUCTURE STABILITY — %s (n_bootstrap=%d)", category, n_bootstrap)
    logger.info("=" * 78)
    logger.info("Mean edges per bootstrap run: %.1f (ref: %d)",
                np.mean(n_edges_per_run), len(ref_edges))
    logger.info("")
    logger.info("Edge stability:")
    logger.info("  %-40s %3s %8s %8s  %s", "edge", "ref", "freq", "rev_freq", "interp")
    for _, r in df_stab.iterrows():
        if r.bootstrap_freq < 0.05 and r.reverse_freq < 0.05:
            continue  # skip near-zero
        ref_marker = " ★ " if r.in_reference else "   "
        logger.info("  %-40s %3s %7.0f%% %7.0f%%  %s",
                    r.edge, ref_marker, r.bootstrap_freq * 100,
                    r.reverse_freq * 100, r.interpretation)
    logger.info("")
    logger.info("★ = edge присутствует в reference graph (на полных данных)")
    return df_stab


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True, choices=list(CATEGORY_PREP.keys()))
    p.add_argument("--n-bootstrap", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    df = bootstrap_dag(args.category, n_bootstrap=args.n_bootstrap, seed_base=args.seed)
    out = args.out or os.path.join(PROCESSED_DIR, f"dag_stability_{args.category}.parquet")
    df.to_parquet(out, index=False)
    logger.info("Saved -> %s", out)


if __name__ == "__main__":
    main()
