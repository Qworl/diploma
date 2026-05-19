"""Brand-clustered bootstrap CI для 11 рабочих точек cost-quality scatter (cell 16).

Центральные значения точно совпадают со scatter (`cascade_plus_llm4_summary.parquet`,
колонки `grand_acc_*`). Бутстрэп БРЕНДОВ с возвращением (1000 итераций):
- варьирует cascade portion (coverage и acc_on_covered) на v2-gold;
- per-attr LLM accuracy фиксирована (взята из `cascade_plus_llm4_hybrid.parquet`,
  где `llm_acc_on_attr` оценен по силверу из 250 кодов на категорию).

CI отражает brand-level uncertainty в поведении каскада при фиксированной LLM-оценке.

Output: datasets/processed/cost_quality_ci.parquet
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
PROCESSED = ROOT / "datasets" / "processed"
OUT = PROCESSED / "cost_quality_ci.parquet"
CATEGORIES = ["pasta", "chocolate", "cheeses"]
LLM_MODELS = ["sonnet45", "gpt4o", "gemini25flash", "gptoss", "llama3b"]
N_BOOTSTRAP = 1000
RNG_SEED = 42
ROUTER_ACC = {"pasta": 0.962880, "chocolate": 0.985377, "cheeses": 0.977477}


def normalize(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().lower()
    if s in ("none", "null", "nan", ""):
        return None
    return s


def load_per_cell() -> pd.DataFrame:
    """Per-cell long table: gold + cascade pred + brand."""
    gold = pd.read_parquet(PROCESSED / "consensus_gold_v2_expanded.parquet")
    gold = gold[~gold.gold_is_null].copy()
    gold["gold_norm"] = gold.gold_value.map(normalize)
    gold["code"] = gold["code"].astype(str)

    parts = []
    for cat in CATEGORIES:
        casc = pd.read_parquet(PROCESSED / f"cascade_preds_{cat}_v2_gold_hybrid_v3_fixed.parquet")
        casc["code"] = casc["code"].astype(str)
        casc["cascade_pred_norm"] = casc.predicted.map(normalize)
        casc["cascade_abstain"] = (casc["layer"] == "abstain")
        brands = pd.read_parquet(PROCESSED / f"{cat}_stratified_silver_standard.parquet",
                                 columns=["code", "brands"])
        brands["code"] = brands["code"].astype(str)
        brands = brands.drop_duplicates("code")
        g_cat = gold[gold.category == cat][["code", "attr", "gold_norm"]]
        m = (casc[["code", "attr", "cascade_pred_norm", "cascade_abstain"]]
             .merge(g_cat, on=["code", "attr"], how="inner")
             .merge(brands, on="code", how="left"))
        m["brand"] = m["brands"].fillna("__nobrand__")
        m["category"] = cat
        m = m.drop(columns=["brands"])
        parts.append(m)
    df = pd.concat(parts, ignore_index=True)
    logger.info("Long table: %d rows, %d brands", len(df), df.brand.nunique())
    return df


def load_llm_per_attr() -> dict[str, dict[tuple[str, str], float]]:
    """LLM accuracy per (cat, attr) per model — фиксированные значения из summary-pipeline."""
    h = pd.read_parquet(PROCESSED / "cascade_plus_llm4_hybrid.parquet")
    out: dict[str, dict[tuple[str, str], float]] = {m: {} for m in LLM_MODELS}
    for _, r in h.iterrows():
        out[r.llm_model][(r.category, r.attr)] = float(r.llm_acc_on_attr)
    return out


def compute_grand(df: pd.DataFrame, llm_per_attr: dict) -> dict[str, float]:
    """Compute 11 configurations on the given df (либо центральная, либо бутстрэп-ресэмпл)."""
    out: dict[str, float] = {}
    correct_e2e = (~df.cascade_abstain) & (df.cascade_pred_norm == df.gold_norm)
    out["cascade_only"] = float(correct_e2e.mean())

    # per (cat, attr) cascade statistics
    grp = df.groupby(["category", "attr"], sort=False)
    casc_stats = []
    for (cat, attr), sub in grp:
        n = len(sub)
        cov_mask = ~sub.cascade_abstain
        n_cov = int(cov_mask.sum())
        coverage = n_cov / n
        if n_cov:
            acc_cov = float((sub.loc[cov_mask, "cascade_pred_norm"] == sub.loc[cov_mask, "gold_norm"]).mean())
        else:
            acc_cov = 0.0
        casc_stats.append((cat, attr, n, coverage, acc_cov))

    n_total = sum(s[2] for s in casc_stats)

    for model in LLM_MODELS:
        per_attr = llm_per_attr[model]
        # all_llm: weighted average of per-(cat,attr) llm_acc by n_test (на v2-gold)
        num = denom = 0.0
        for cat, attr, n, _, _ in casc_stats:
            a = per_attr.get((cat, attr), 0.0)
            num += a * n
            denom += n
        out[f"all_llm_{model}"] = float(num / denom) if denom else 0.0

        # cascade_plus_<m>: proxy * router_acc, weighted by n_test
        num = denom = 0.0
        for cat, attr, n, cov, acc_cov in casc_stats:
            llm_a = per_attr.get((cat, attr), 0.0)
            proxy = cov * acc_cov + (1 - cov) * llm_a
            proxy *= ROUTER_ACC.get(cat, 1.0)
            num += proxy * n
            denom += n
        out[f"cascade_plus_{model}"] = float(num / denom) if denom else 0.0

    return out


def main():
    df = load_per_cell()
    llm_per_attr = load_llm_per_attr()

    central = compute_grand(df, llm_per_attr)
    logger.info("Central values computed.")

    brands = df.brand.unique()
    by_brand = {b: df[df.brand == b].reset_index(drop=True) for b in brands}
    rng = np.random.default_rng(RNG_SEED)
    config_names = list(central.keys())
    boot_acc: dict[str, list[float]] = {k: [] for k in config_names}

    for i in range(N_BOOTSTRAP):
        sampled = rng.choice(brands, size=len(brands), replace=True)
        parts = [by_brand[b] for b in sampled]
        boot = pd.concat(parts, ignore_index=True)
        accs = compute_grand(boot, llm_per_attr)
        for k, v in accs.items():
            boot_acc[k].append(v)
        if (i + 1) % 100 == 0:
            logger.info("  bootstrap %d/%d", i + 1, N_BOOTSTRAP)

    rows = []
    for k in config_names:
        arr = np.array(boot_acc[k])
        rows.append({"config": k, "acc": central[k],
                     "ci_lo": float(np.percentile(arr, 2.5)),
                     "ci_hi": float(np.percentile(arr, 97.5)),
                     "n_cells": len(df), "n_brands": len(brands)})

    out = pd.DataFrame(rows)
    out.to_parquet(OUT, index=False)
    logger.info("Saved %d rows → %s", len(out), OUT)

    print("\n" + "=" * 80)
    print(f"{'config':<28} {'acc':>10} {'95% CI':>22}")
    print("-" * 80)
    for _, r in out.iterrows():
        print(f"{r.config:<28} {r.acc*100:>8.2f}%  [{r.ci_lo*100:>5.2f}, {r.ci_hi*100:>5.2f}]")
    print("=" * 80)


if __name__ == "__main__":
    main()
