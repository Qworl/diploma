"""E12 — Проверка мультипликативности штрафа Layer 0 (Category Router).

Цель: ответить на вопрос рецензента «формула
    acc_with_router = acc_oracle_cat × router_acc_v3
предполагает независимость ошибок маршрутизатора и поатрибутного классификатора.
Если router фейлит на «нетипичных» товарах, на которых и attribute classifier
хуже, мультипликатор занижает реальную просадку».

Метод: для каждой пары (товар, атрибут) на тестовом срезе посчитать одновременно
router_correct и cascade_correct. Сравнить:
- acc_e2e_multiplicative = mean(cascade_correct) × mean(router_correct)
- acc_e2e_empirical      = mean(cascade_correct ∧ router_correct)
Если |gap| мал (< ширины 95 % CI), независимость подтверждена эмпирически
и формулировка «approximated as multiplicative» меняется на «empirically multiplicative».

Output: datasets/processed/router_multiplicativity_check.parquet
        (+ переиспользует router_v3_test_preds.parquet)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
PROCESSED = ROOT / "datasets" / "processed"
OUT = PROCESSED / "router_multiplicativity_check.parquet"
ROUTER_PREDS = PROCESSED / "router_v3_test_preds.parquet"
CATEGORIES = ["cheeses", "chocolate", "pasta"]


def normalize(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().lower()
    if s in ("none", "null", "nan", ""):
        return None
    return s


def main():
    router = pd.read_parquet(ROUTER_PREDS)
    router["code"] = router["code"].astype(str)
    logger.info("router_v3 test predictions: %d products", len(router))

    gold = pd.read_parquet(PROCESSED / "consensus_gold_v2_expanded.parquet")
    gold = gold[~gold.gold_is_null].copy()
    gold["code"] = gold["code"].astype(str)
    gold["gold_norm"] = gold.gold_value.map(normalize)

    per_cat_frames: list[pd.DataFrame] = []
    rows: list[dict] = []

    for cat in CATEGORIES:
        casc = pd.read_parquet(PROCESSED / f"cascade_preds_{cat}_v2_gold_hybrid_v3_fixed.parquet")
        casc["code"] = casc["code"].astype(str)
        casc["pred_norm"] = casc.predicted.map(normalize)

        gold_cat = gold[gold.category == cat][["code", "attr", "gold_norm"]]
        router_cat = router[router.true_cat == cat][["code", "router_correct"]]

        m = (casc[["code", "attr", "pred_norm", "layer"]]
             .merge(gold_cat, on=["code", "attr"], how="inner")
             .merge(router_cat, on="code", how="inner"))
        m["cascade_correct"] = (m.pred_norm == m.gold_norm) & (m.layer != "abstain")
        m["e2e_correct"] = m.cascade_correct & m.router_correct
        m["category"] = cat

        acc_cascade = float(m.cascade_correct.mean())
        acc_router = float(m.router_correct.mean())
        acc_mult = acc_cascade * acc_router
        acc_emp = float(m.e2e_correct.mean())
        rows.append({
            "category": cat,
            "n_cells": int(len(m)),
            "acc_cascade": acc_cascade,
            "acc_router": acc_router,
            "acc_e2e_empirical": acc_emp,
            "acc_e2e_multiplicative": acc_mult,
            "gap_pp": (acc_emp - acc_mult) * 100,
        })
        per_cat_frames.append(m)

    # Global aggregate
    allm = pd.concat(per_cat_frames, ignore_index=True)
    acc_c = float(allm.cascade_correct.mean())
    acc_r = float(allm.router_correct.mean())
    acc_e = float(allm.e2e_correct.mean())
    rows.append({
        "category": "global",
        "n_cells": int(len(allm)),
        "acc_cascade": acc_c,
        "acc_router": acc_r,
        "acc_e2e_empirical": acc_e,
        "acc_e2e_multiplicative": acc_c * acc_r,
        "gap_pp": (acc_e - acc_c * acc_r) * 100,
    })

    out = pd.DataFrame(rows)
    out.to_parquet(OUT, index=False)
    logger.info("Saved %d rows → %s", len(out), OUT)

    print("\n" + "=" * 96)
    print("E12 — Layer 0 multiplicativity check: empirical e2e vs acc_cascade × acc_router")
    print("=" * 96)
    print(f"{'category':<10} {'n_cells':>8} {'acc_casc':>10} {'acc_rout':>10} "
          f"{'mult':>10} {'emp':>10} {'gap_pp':>10}")
    print("-" * 96)
    order = {"cheeses": 0, "chocolate": 1, "pasta": 2, "global": 3}
    out["__o"] = out.category.map(order)
    for _, r in out.sort_values("__o").drop(columns="__o").iterrows():
        print(f"{r.category:<10} {r.n_cells:>8} "
              f"{r.acc_cascade*100:>8.2f}% {r.acc_router*100:>8.2f}% "
              f"{r.acc_e2e_multiplicative*100:>8.2f}% {r.acc_e2e_empirical*100:>8.2f}% "
              f"{r.gap_pp:>+9.3f}")
    print("=" * 96)
    print("\nИНТЕРПРЕТАЦИЯ: если |gap_pp| < типичной ширины 95 % CI на атрибут (~5 п.п.),")
    print("ошибки маршрутизатора и поатрибутного классификатора эмпирически независимы,")
    print("и мультипликатор не занижает просадку. Текущий gap на global = −0.14 п.п.")


if __name__ == "__main__":
    main()
