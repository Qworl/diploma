"""Записать gold-калиброванные пороги для сценария C: 3 атрибута, q=0.02.

Сценарий C даёт лучшее отношение точности к расходу LLM (0.53 пп / 1% LLM),
по сравнению с прод-настройкой A (silver, contains_nuts only, q=0.05) и наивным
«всегда-вкл». Узкое селективное включение валидатора:

  • chocolate / contains_nuts        — q=0.02 на gold
  • pasta     / is_vegan             — q=0.02 на gold
  • cheeses   / is_pdo               — q=0.02 на gold

На остальных атрибутах валидатор не выдаёт вердикта (запись из JSON удаляется
→ `ValidatorService.validate_value` возвращает None → демо не показывает бейдж).

Структура сети сохраняется как есть (учена на silver, нужна для охвата брендов),
меняем только пороги.

Запуск:
  OMP_NUM_THREADS=1 python -m src.pipeline.bayes.calibrate_validator_gold_c
"""
from __future__ import annotations

import json
import pickle
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination

from src.pipeline.bayes.validate import calibrate_thresholds

PROCESSED = Path("datasets/processed")
MODELS = Path("models")
Q = 0.02

USEFUL_ATTRS = {
    "chocolate_stratified": {"contains_nuts"},
    "pasta_stratified": {"is_vegan"},
    "cheeses_stratified": {"is_pdo"},
}

_ORGANIC_RE = re.compile(r"\b(bio|organic|organique|eco|ecol[oó]gico|ekol|öko)\b", re.I)


def _brand_first(s: str) -> str:
    if not s:
        return "unknown"
    s = str(s).split(",")[0].strip().lower()
    return s or "unknown"


def _brand_organic(s: str) -> str:
    return "True" if _ORGANIC_RE.search(str(s or "")) else "False"


def build_gold_wide(short_cat: str, internal_cat: str) -> pd.DataFrame:
    gold = pd.read_parquet(PROCESSED / "consensus_gold_v2_expanded.parquet")
    gold = gold[(gold["category"] == short_cat) & (~gold["gold_is_null"])].copy()
    gold["code"] = gold["code"].astype(str)
    wide = gold.pivot_table(index="code", columns="attr",
                            values="gold_value", aggfunc="first")

    raw = pd.read_parquet(PROCESSED / f"{internal_cat}_raw.parquet",
                          columns=["code", "brands"])
    raw["code"] = raw["code"].astype(str)
    raw["brands"] = raw["brands"].fillna("").astype(str)
    raw["brand"] = raw["brands"].apply(_brand_first)
    raw["brand_has_organic_marker"] = raw["brands"].apply(_brand_organic)
    raw = raw.set_index("code")

    merged = wide.join(raw[["brand", "brand_has_organic_marker"]], how="left")
    for col in merged.columns:
        if col == "brand":
            merged[col] = merged[col].fillna("unknown").astype(str)
            continue
        merged[col] = merged[col].map(
            lambda v: None if v is None or (isinstance(v, float) and np.isnan(v))
            else str(v)
        )
    return merged.reset_index()


def main() -> int:
    cats = [
        ("pasta", "pasta_stratified"),
        ("chocolate", "chocolate_stratified"),
        ("cheeses", "cheeses_stratified"),
    ]
    for short, internal in cats:
        print(f"\n=== {internal} ===")
        bayes_path = MODELS / f"{internal}_bayesian.pkl"
        thr_path = MODELS / f"{internal}_validation_thresholds.json"

        with open(bayes_path, "rb") as f:
            bayes = pickle.load(f)
        inference = VariableElimination(bayes)
        gold_wide = build_gold_wide(short, internal)
        print(f"  gold rows: {len(gold_wide):,}")

        full_thresholds = calibrate_thresholds(bayes, gold_wide, inference, q=Q)

        useful = USEFUL_ATTRS[internal]
        selective: dict[str, float] = {
            attr: thr for attr, thr in full_thresholds.items() if attr in useful
        }
        print(f"  useful attrs kept: {sorted(selective.keys())}")
        for attr, thr in sorted(selective.items()):
            print(f"    {attr}: thr={thr:.6f}")

        backup = thr_path.with_suffix(".json.silver_q005_backup")
        if thr_path.exists() and not backup.exists():
            shutil.copy(thr_path, backup)
            print(f"  backup of previous thresholds → {backup.name}")

        payload: dict[str, Any] = {
            "category": internal,
            "source": "gold (consensus_gold_v2_expanded.parquet)",
            "q": Q,
            "n_train_rows": int(len(gold_wide)),
            "useful_attrs_only": True,
            "thresholds": selective,
            "all_attrs_thresholds_full_calibration": full_thresholds,
        }
        with open(thr_path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"  wrote {thr_path}")

    print("\nDone. Demo will pick up new thresholds on next start.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
