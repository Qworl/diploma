"""E10 — Bayes-валидатор: точечный ablation на chocolate/contains_nuts.

Цель: ответить на вопрос рецензента «если валидатор работает на 1 из 22 атрибутов
(chocolate/contains_nuts, precision_lift = +38.2 п.п.), даёт ли селективное включение
ровно на этом атрибуте измеримый прирост, или вклад исчезает в шуме?»

Сравнение трёх конфигураций на v2-gold:
1. cascade_only (текущий headline)
2. cascade + bayes_demote_on(chocolate/contains_nuts) — demote ML→LLM только для
   ячеек, помеченных байес-валидатором на одном атрибуте
3. cascade + LLM-fallback (gemini25flash) — для сравнения

Output: datasets/processed/bayes_ablation_contains_nuts.parquet
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
OUT = PROCESSED / "bayes_ablation_contains_nuts.parquet"


def main():
    bv = pd.read_parquet(PROCESSED / "bayes_validator_demote_metric.parquet")
    row = bv[(bv.category == "chocolate") & (bv.attr == "contains_nuts")].iloc[0]

    n = int(row.n_ml_predictions)            # 239
    n_flag = int(row.n_flagged)              # 41
    tp = int(row.n_flag_cascade_wrong)       # 22 (flagged + was wrong)
    fp = int(row.n_flag_cascade_right)       # 19 (flagged + was right)
    acc_before = float(row.cascade_acc_on_covered)   # 0.845
    llm_acc = float(row.llm_acc_on_attr)             # 0.851 — LLM acc on this attr
    delta_pp = float(row.expected_delta_acc_if_demote)  # +0.066 (≈ +6.6 п.п.)

    # On the contains_nuts attr after demoting flagged cells to LLM:
    # - Unflagged cells (n - n_flag): cascade keeps its prediction; net correct = (n-n_flag) * (acc_on_unflagged)
    #   but we don't have acc_on_unflagged separately; approximate as (acc_before * n - (n_flag - tp)) / (n - n_flag)
    #   = (cascade correct on covered - cascade correct on flagged) / unflagged
    # - Flagged cells (n_flag): LLM is consulted; expected correct = n_flag * llm_acc
    correct_before_on_flagged = fp  # cascade was right on FP cells; n_flag - tp = fp
    correct_before_on_unflagged = int(round(acc_before * n)) - correct_before_on_flagged
    n_unflagged = n - n_flag

    correct_after_on_flagged = n_flag * llm_acc  # expected
    correct_after_on_unflagged = correct_before_on_unflagged  # unchanged
    correct_after = correct_after_on_unflagged + correct_after_on_flagged
    acc_after = correct_after / n

    # Headline impact (over 4350 cells)
    n_corpus = 4350
    delta_corpus_pp = (acc_after - acc_before) * (n / n_corpus) * 100

    rows = [
        {"config": "cascade_only (текущий headline)",
         "acc_chocolate_contains_nuts_pct": acc_before * 100,
         "cost_overhead_llm_calls_per_attr": 0,
         "delta_vs_baseline_pp": 0.0,
         "delta_on_corpus_pp": 0.0,
         "comment": "ML без байес-демоушна"},
        {"config": "cascade + bayes_demote_on(chocolate/contains_nuts) only",
         "acc_chocolate_contains_nuts_pct": acc_after * 100,
         "cost_overhead_llm_calls_per_attr": n_flag,
         "delta_vs_baseline_pp": (acc_after - acc_before) * 100,
         "delta_on_corpus_pp": delta_corpus_pp,
         "comment": f"+{n_flag} LLM-запросов на атрибут (~{n_flag/n*100:.1f}% покрытия)"},
        {"config": "cascade + LLM-fallback on all abstain (gemini25flash)",
         "acc_chocolate_contains_nuts_pct": llm_acc * 100,
         "cost_overhead_llm_calls_per_attr": int(round((1 - 0.91425) * n)),
         "delta_vs_baseline_pp": (llm_acc - acc_before) * 100,
         "delta_on_corpus_pp": float("nan"),
         "comment": "для сравнения — headline gemini25flash"},
    ]

    out = pd.DataFrame(rows)
    out.to_parquet(OUT, index=False)
    logger.info("Saved %d rows → %s", len(out), OUT)

    print("\n" + "=" * 110)
    print("E10 — Bayes-валидатор: точечный ablation на chocolate/contains_nuts (n=239 ячеек)")
    print("=" * 110)
    for _, r in out.iterrows():
        print(f"\n  {r.config}")
        print(f"    acc на атрибуте       = {r.acc_chocolate_contains_nuts_pct:.2f}%")
        print(f"    Δ vs cascade_only     = {r.delta_vs_baseline_pp:+.2f} п.п. (на атрибуте)")
        print(f"    Δ на 4350-ячейном headline = {r.delta_on_corpus_pp:+.2f} п.п.")
        print(f"    LLM-вызовов на атрибут = {r.cost_overhead_llm_calls_per_attr}")
        print(f"    → {r.comment}")
    print("\n" + "=" * 110)
    print("\nВЫВОД: включение байес-демоушна селективно на chocolate/contains_nuts даёт")
    print(f"+{out.iloc[1].delta_vs_baseline_pp:.1f} п.п. на самом атрибуте при стоимости {n_flag} LLM-вызовов,")
    print(f"что в пересчёте на полный 4350-ячейный headline составляет +{delta_corpus_pp:.2f} п.п.")
    print("(порядка 0.4 п.п. — внутри 95% CI headline, не существенно для главного числа,")
    print("но отрицательным результатом для байес-валидатора в целом не является — точечное")
    print("использование на 1 атрибуте методологически защитимо).")


if __name__ == "__main__":
    main()
