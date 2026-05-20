"""Изоляция вклада Bayes-validator vs ML threshold tuning.

Прогоняем 4 конфигурации на production-эмуляции:
  • baseline   — cascade_after_fix + LLM-fallback (без Bayes, без новых ML thr)
  • E_only     — новые ML thr (deployed), без Bayes-demote
  • A_only     — старые ML thr, но с Bayes-demote (current deployed validator)
  • combined   — обе механики (текущий prod)

Для A_only временно загружаем старые ML thresholds из backup'а.

Запуск:
  OMP_NUM_THREADS=1 python -m src.experiments.bayes_isolated_contribution
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("demo/ml_service")))
from validator import ValidatorService

PROCESSED = Path("datasets/processed")
MODELS = Path("models")

CATS = [
    ("pasta", "pasta_stratified"),
    ("chocolate", "chocolate_stratified"),
    ("cheeses", "cheeses_stratified"),
]


def _normalize(v):
    if v is None: return None
    if isinstance(v, float) and np.isnan(v): return None
    if isinstance(v, str) and v.strip().lower() in {"", "none", "null", "nan"}: return None
    return v


def _eq(a, b):
    a, b = _normalize(a), _normalize(b)
    if a is None or b is None: return False
    return str(a).strip().lower() == str(b).strip().lower()


def llm_acc(internal):
    df = pd.read_parquet(PROCESSED / f"direct_llm_eval_{internal}_sonnet45.parquet")
    m = (df.predicted_non_null == 1) & (df.gt_non_null == 1)
    return df[m].groupby("attr").correct_when_both_present.mean().to_dict()


def run_config(use_bayes: bool, cascade_files: dict) -> dict:
    """Применить cascade + (optional) Bayes-validator + LLM-fallback, вернуть e2e."""
    if use_bayes:
        validator = ValidatorService(models_dir=str(MODELS),
                                      internal_categories=[ic for _, ic in CATS])
    else:
        validator = None

    gold = pd.read_parquet(PROCESSED / "consensus_gold_v2_expanded.parquet")
    gold = gold[~gold.gold_is_null].copy()
    gold["code"] = gold["code"].astype(str)
    gold["gold_norm"] = gold.gold_value.astype(str).str.lower()

    total_n = 0; correct_sum = 0.0
    n_abstain = n_flag = n_tp = n_fp = 0

    for short, internal in CATS:
        cpath = cascade_files[short]
        cascade = pd.read_parquet(cpath)
        cascade["code"] = cascade["code"].astype(str)
        raw = pd.read_parquet(PROCESSED / f"{internal}_raw.parquet", columns=["code","brands"])
        raw["code"] = raw["code"].astype(str); raw["brands"] = raw["brands"].fillna("").astype(str)
        brands_lk = dict(zip(raw["code"], raw["brands"]))
        base_ev = {}
        for code, grp in cascade.groupby("code"):
            ev = {}
            if code in brands_lk and brands_lk[code]: ev["brand"] = brands_lk[code]
            for _, r in grp.iterrows():
                if r["layer"] == "abstain": continue
                ev[r["attr"]] = r["predicted"]
            base_ev[code] = ev

        g = gold[gold.category == short][["code","attr","gold_norm"]]
        m = cascade.merge(g, on=["code","attr"], how="inner")
        m["pred_norm"] = m.predicted.astype(str).str.lower()
        la = llm_acc(internal)

        for _, r in m.iterrows():
            attr = r["attr"]; layer = r["layer"]
            ok_cascade = (r["pred_norm"] == r["gold_norm"]) and (layer != "abstain")
            l = la.get(attr, 0.70)
            if layer == "abstain":
                correct_sum += l; n_abstain += 1
            elif layer == "ml" and validator is not None:
                ev = dict(base_ev.get(r["code"], {})); ev.pop(attr, None)
                v = validator.validate_value(internal, attr, r["predicted"], ev)
                if v is not None and v["flagged"]:
                    n_flag += 1
                    if ok_cascade: n_fp += 1
                    else: n_tp += 1
                    correct_sum += l; n_abstain += 1
                else:
                    correct_sum += 1 if ok_cascade else 0
            else:
                correct_sum += 1 if ok_cascade else 0
            total_n += 1

    return {
        "n": total_n,
        "headline": correct_sum / total_n,
        "n_abstain": n_abstain,
        "n_flag": n_flag, "n_tp": n_tp, "n_fp": n_fp,
    }


def main():
    # Cascade files: current (deployed E) and "old E" (cascade before ML threshold update)
    # Нет старого cascade — после deploy он был перезаписан. Воссоздать его не дёшево.
    # Альтернатива: для A_only используем backup (scenario_c_backup) thresholds через
    # временную подмену файлов. Но это хрупкое. Вместо этого: сравнение деплоированной
    # конфигурации vs скрипт с прямой ML-predict при разных thr.

    # Простая декомпозиция через парные сравнения:
    # baseline_no_bayes  = cascade_after_fix (deployed E thresholds) без validator
    # combined_with_bayes= cascade_after_fix + validator (current prod)
    # → разница = Bayes-эффект В КОНТЕКСТЕ новых thr

    cascade_cur = {short: PROCESSED / f"cascade_preds_{short}_after_fix.parquet" for short, _ in CATS}

    print("=" * 80)
    print("CONFIG 1: cascade (deployed E thresholds) + LLM-fallback, NO Bayes")
    print("=" * 80)
    res_no_bayes = run_config(use_bayes=False, cascade_files=cascade_cur)
    print(f"  Headline: {res_no_bayes['headline']*100:.3f}% on {res_no_bayes['n']} cells")
    print(f"  LLM calls: {res_no_bayes['n_abstain']} ({res_no_bayes['n_abstain']/res_no_bayes['n']*100:.2f}%)")

    print("\n" + "=" * 80)
    print("CONFIG 2: cascade (deployed E thresholds) + Bayes + LLM-fallback (CURRENT PROD)")
    print("=" * 80)
    res_combined = run_config(use_bayes=True, cascade_files=cascade_cur)
    print(f"  Headline: {res_combined['headline']*100:.3f}% on {res_combined['n']} cells")
    print(f"  LLM calls: {res_combined['n_abstain']} ({res_combined['n_abstain']/res_combined['n']*100:.2f}%)")
    print(f"  Bayes flags: {res_combined['n_flag']} (TP={res_combined['n_tp']}, FP={res_combined['n_fp']})")
    print(f"  Bayes precision: {res_combined['n_tp']/max(res_combined['n_flag'],1)*100:.1f}%")

    # Δ from Bayes
    delta_bayes = (res_combined['headline'] - res_no_bayes['headline']) * 100
    extra_llm = res_combined['n_abstain'] - res_no_bayes['n_abstain']
    print(f"\nBayes-isolated contribution (на текущих ML thr):")
    print(f"  Δheadline: {delta_bayes:+.3f} пп")
    print(f"  Extra LLM calls: {extra_llm} ({extra_llm/res_no_bayes['n']*100:+.2f}%)")
    print(f"  Cost/quality: {delta_bayes / max(extra_llm/res_no_bayes['n']*100, 0.01):.2f} пп / 1 % LLM")

    # E-only contribution: сравним с pre-deploy cascade. Прежний after_fix есть только
    # в виде backup'а кода (cascade_preds_*_v2_gold_hybrid_v3_fixed.parquet — ещё более старый).
    # Возьмём это как нижнюю границу — baseline (perpre-fix).
    print("\n" + "=" * 80)
    print("CONFIG 0 (FOR REFERENCE): pre-fix cascade (v3_fixed), без Bayes")
    print("  — это очень старый baseline, до сегодняшних improvements")
    print("=" * 80)
    cascade_old = {short: PROCESSED / f"cascade_preds_{short}_v2_gold_hybrid_v3_fixed.parquet"
                    for short, _ in CATS}
    res_prefix = run_config(use_bayes=False, cascade_files=cascade_old)
    print(f"  Headline: {res_prefix['headline']*100:.3f}% on {res_prefix['n']} cells")
    print(f"  LLM calls: {res_prefix['n_abstain']} ({res_prefix['n_abstain']/res_prefix['n']*100:.2f}%)")

    delta_E_plus_arch = (res_no_bayes['headline'] - res_prefix['headline']) * 100
    print(f"\nArchitectural + E contribution (cascade-only):")
    print(f"  Δheadline (no Bayes): {delta_E_plus_arch:+.3f} пп")

    print("\n" + "=" * 80)
    print("СВОДКА — декомпозиция вклада в headline")
    print("=" * 80)
    print(f"  Pre-fix baseline (v3_fixed cascade, no Bayes):     {res_prefix['headline']*100:.2f}%")
    print(f"  +E threshold tuning + arch fixes (no Bayes):       {res_no_bayes['headline']*100:.2f}%  (+{delta_E_plus_arch:.2f} пп)")
    print(f"  +Bayes (current production):                       {res_combined['headline']*100:.2f}%  (+{delta_bayes:.2f} пп от Bayes)")
    print(f"\n  Bayes-vклад изолировано: {delta_bayes:+.3f} пп")


if __name__ == "__main__":
    main()
