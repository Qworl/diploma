"""Build per-cell traceback for 5 case-study products.

For each (code, category) pair, walk through layer signals reconstructed from
existing artefacts and dump structured JSON.

Inputs:
  - datasets/processed/cascade_preds_{cat}_gold.parquet
        per (code, attr): cascade_layer + cascade_pred + e2e_pred + router_pred
  - datasets/processed/cascade_raw_with_conf_{cat}.parquet
        per (code, attr): ml_pred, ml_conf, ml_threshold, rule_pred, rule_tier
  - datasets/processed/{cat}_stratified_silver_standard.parquet
        partner-available fields (product_name, brands, ingredients_text,
        quantity) and silver attrs.
  - datasets/processed/manual_gold_consensus.parquet
        consensus gold (agreement_ratio).
  - datasets/processed/cascade_errors_taxonomy_v4.parquet
        error_class annotation for failure cases.

Output: datasets/processed/case_study_traceback.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "datasets" / "processed"
OUT = PROC / "case_study_traceback.json"
CODES_OUT = PROC / "case_study_codes.json"


CASES: list[dict[str, Any]] = [
    {
        "case_id": 1,
        "code": "00154222",
        "category": "pasta",
        "kind": "failure",
        "error_class": "regex-false-positive",
        "title": "Регулярное выражение как ложный сигнал (filled-pasta на сыре)",
        "explanation_ru": (
            "ML слой корректно распознаёт начинку (ml_pred=true, ml_conf=0.97 > порога 0.9). "
            "Однако высокоприоритетное эвристическое правило по продукту \"Three Cheese Ravioli\" "
            "форсирует значение false (предположение, что любое блюдо со словом cheese — это "
            "пицца или сэндвич, а не паста с начинкой). Поскольку правила high-tier перекрывают "
            "решение ML, итоговый ответ системы — false, что не совпадает с эталоном "
            "(agreement_ratio=1.0). Этот пример демонстрирует, что класс regex-false-positive "
            "состоит преимущественно из случаев, когда правило подавляет правильный ML-ответ. "
            "Вывод (зафиксирован в INBOX): правило по cheese на is_filled следует ослабить или "
            "ограничить контекстом (исключить Ravioli, Tortellini, Cappelletti)."
        ),
        "attrs": ["is_filled"],
    },
    {
        "case_id": 2,
        "code": "3184670017166",
        "category": "cheeses",
        "kind": "failure",
        "error_class": "class-confusion",
        "title": "Смешение близких классов (cream vs fresh)",
        "explanation_ru": (
            "Слой ML с высокой уверенностью выбирает класс fresh для мягкого козьего сыра "
            "\"Chevre a tartiner\". Эталон же относит товар к крем-сырам (cream). Граница "
            "между fresh-cheese и cream-cheese размыта на уровне OFF-тегов и проявляется как "
            "смешение классов в матрице ошибок (§3.3.x). Такая ошибка ожидаема и потенциально "
            "снимается экспертной заменой границ в схеме (см. INBOX, ticket по reschema "
            "cheeses.texture)."
        ),
        "attrs": ["texture"],
    },
    {
        "case_id": 3,
        "code": "2000000074755",
        "category": "pasta",
        "kind": "failure",
        "error_class": "silver-noise",
        "title": "Silver-шум: товар не относится к категории",
        "explanation_ru": (
            "Товар \"Pâté de foie de volaille\" попал в выборку pasta из-за тегов OFF, однако "
            "к пасте отношения не имеет. Silver-эталон присвоил is_gluten_free=true, тогда как "
            "каскад абстрагируется (cascade_pred=false по умолчанию из-за низкой уверенности ML). "
            "Эталон в данном случае некорректен (пшеничная мука обычно встречается в подобных "
            "паштетах, поэтому is_gluten_free=true сомнительно). Каскад уверенно выдаёт "
            "is_gluten_free=false (ml_conf=0.98). Низкое значение consensus_agreement=0,67 "
            "подтверждает спорность эталона. Пример иллюстрирует ограничение метода получения "
            "silver-эталона из OFF-тегов и обосновывает необходимость пересчёта метрик на "
            "human-gold консенсусе (§3.3.4)."
        ),
        "attrs": ["is_gluten_free"],
    },
    {
        "case_id": 4,
        "code": "20114992",
        "category": "pasta",
        "kind": "success",
        "title": "Сложный мультиязычный товар, ML слой даёт уверенный ответ по всем атрибутам",
        "explanation_ru": (
            "Товар \"Bio Spaghetti\" бренда Combino — мультиязычный (немецкое Bio + итальянское "
            "Spaghetti). Регулярное выражение распознаёт is_filled=false по правилу высокого "
            "приоритета (нет слов начинки). ML слой уверенно предсказывает все остальные шесть "
            "атрибутов (cuisine_origin=italian, grain_type=wheat, is_organic=true, "
            "is_vegan=true, pasta_shape=spaghetti, is_gluten_free=false) с уверенностью выше "
            "соответствующих порогов. Эскалация в LLM не требуется. Пример демонстрирует "
            "корректность многоязычных эмбеддингов (paraphrase-multilingual-MiniLM-L12-v2) и "
            "адекватность per-attribute порогов."
        ),
        "attrs": [
            "grain_type",
            "is_filled",
            "is_organic",
            "is_gluten_free",
            "pasta_shape",
            "is_vegan",
            "cuisine_origin",
        ],
    },
    {
        "case_id": 5,
        "code": "3250392007737",
        "category": "chocolate",
        "kind": "success",
        "title": "Французская тёмная плитка с начинкой — комбинированная работа правил и ML",
        "explanation_ru": (
            "Шоколад \"Tablette fourrée chocolat au lait caramel à la fleur de sel de Guérande\" "
            "бренда Ivoria — тёмная плитка с начинкой солёная карамель. Правила высокого "
            "приоритета корректно срабатывают на is_filled=true (по слову fourrée) и "
            "contains_nuts=false. Правило низкого приоритета корректно фиксирует "
            "chocolate_extra=with_caramel. Оставшиеся три атрибута (chocolate_type=milk, "
            "is_organic=false, flavor_profile=salty_caramel) корректно предсказывает ML слой. "
            "Эскалация в LLM не требуется. Пример иллюстрирует кооперативную работу слоёв на "
            "товаре со сложным составом."
        ),
        "attrs": [
            "chocolate_type",
            "contains_nuts",
            "is_filled",
            "chocolate_extra",
            "is_organic",
            "flavor_profile",
        ],
    },
]


def safe_str(v: Any) -> str | None:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    s = str(v)
    # Replace ASCII double quotes with French/Russian guillemets for thesis style.
    # Only safe if quotes are paired; for raw fields (ingredients_text) we leave them.
    return s


def abbreviate(text: str | None, max_len: int = 140) -> str | None:
    if text is None:
        return None
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def build_case(case: dict[str, Any]) -> dict[str, Any]:
    code = case["code"]
    cat = case["category"]

    preds = pd.read_parquet(PROC / f"cascade_preds_{cat}_gold.parquet")
    raw = pd.read_parquet(PROC / f"cascade_raw_with_conf_{cat}.parquet")
    silver = pd.read_parquet(PROC / f"{cat}_stratified_silver_standard.parquet")
    consensus = pd.read_parquet(PROC / "manual_gold_consensus.parquet")
    tax = pd.read_parquet(PROC / "cascade_errors_taxonomy_v4.parquet")

    # Partner input (only the four real partner fields)
    silver_row = silver[silver["code"] == code]
    partner: dict[str, str | None] = {
        "product_name": None,
        "brands": None,
        "ingredients_text": None,
        "quantity": None,
    }
    if not silver_row.empty:
        r = silver_row.iloc[0]
        partner["product_name"] = safe_str(r.get("product_name"))
        partner["brands"] = safe_str(r.get("brands"))
        partner["ingredients_text"] = abbreviate(safe_str(r.get("ingredients_text")), 160)
        partner["quantity"] = safe_str(r.get("quantity"))

    cells = []
    for attr in case["attrs"]:
        p = preds[(preds["code"] == code) & (preds["attr"] == attr)]
        r = raw[(raw["code"] == code) & (raw["attr"] == attr)]
        c = consensus[(consensus["code"] == code) & (consensus["attr"] == attr)]

        cell = {
            "attr": attr,
            "ml_pred": None,
            "ml_conf": None,
            "ml_threshold": None,
            "rule_pred": None,
            "rule_tier": None,
            "bayes": "—",  # bayes validator не записан в этих артефактах
            "llm": "—",  # реальный LLM ответ не сохранён в артефактах
            "cascade_layer": None,
            "cascade_pred": None,
            "gold_value": None,
            "consensus_agreement": None,
            "consensus_source": None,
            "correct": None,
        }
        if not r.empty:
            row = r.iloc[0]
            cell["ml_pred"] = safe_str(row.get("ml_pred"))
            mc = row.get("ml_conf")
            cell["ml_conf"] = float(mc) if pd.notna(mc) else None
            mt = row.get("ml_threshold")
            cell["ml_threshold"] = float(mt) if pd.notna(mt) else None
            cell["rule_pred"] = safe_str(row.get("rule_pred"))
            cell["rule_tier"] = safe_str(row.get("rule_tier"))
        if not p.empty:
            row = p.iloc[0]
            cell["cascade_layer"] = safe_str(row.get("cascade_layer"))
            cell["cascade_pred"] = safe_str(row.get("cascade_pred"))
            cell["gold_value"] = safe_str(row.get("gold_value"))
            # if cascade_layer == 'fallback', LLM concept-wise was reached
            if cell["cascade_layer"] == "fallback":
                cell["llm"] = "эскалация (ответ не сохранён в артефакте)"
        if not c.empty:
            row = c.iloc[0]
            if "agreement_ratio" in c.columns:
                ar = row.get("agreement_ratio")
                cell["consensus_agreement"] = float(ar) if pd.notna(ar) else None
            # source col detection
            for src_col in ("source", "gold_source"):
                if src_col in c.columns:
                    cell["consensus_source"] = safe_str(row.get(src_col))
                    break
        if cell["cascade_pred"] is not None and cell["gold_value"] is not None:
            cell["correct"] = cell["cascade_pred"] == cell["gold_value"]

        # attach taxonomy error_class if this cell is in taxonomy
        t = tax[(tax["code"] == code) & (tax["attr"] == attr)]
        if not t.empty:
            cell["error_class"] = safe_str(t.iloc[0].get("error_class"))

        cells.append(cell)

    return {
        "case_id": case["case_id"],
        "code": code,
        "category": cat,
        "kind": case["kind"],
        "title": case["title"],
        "error_class": case.get("error_class"),
        "partner_input": partner,
        "cells": cells,
        "explanation_ru": case["explanation_ru"],
    }


def main() -> None:
    cases_out = [build_case(c) for c in CASES]
    OUT.write_text(
        json.dumps({"cases": cases_out}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {OUT} ({len(cases_out)} cases)")

    codes_summary = [
        {
            "case_id": c["case_id"],
            "code": c["code"],
            "category": c["category"],
            "kind": c["kind"],
            "error_class": c.get("error_class"),
            "title": c["title"],
            "attrs": c["attrs"],
        }
        for c in CASES
    ]
    CODES_OUT.write_text(
        json.dumps({"cases": codes_summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {CODES_OUT}")


if __name__ == "__main__":
    main()
