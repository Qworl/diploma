"""Public entry points for the off_labels layer.

- `apply_off_labels(row, schema)` — выдаёт значения атрибутов, выводимые из
  OFF/OBF тегов (full row including categories_tags/labels_tags).
- `apply_partner_type_f(row, schema)` — только TYPE_F regex по partner-полям
  (без OFF-тегов). Используется как regex-слой для partner-input.
"""

from src.pipeline.off_labels.rules import (
    TYPE_F_RULES,
    _type_a_bool, _type_ac_hybrid, _type_b_multiclass,
    _type_c_numeric, _type_d_direct,
    _type_e_regex, _type_f_regex_multiclass,
    _type_vegan_specific, _type_ultra_processed_specific,
)

PARTNER_FIELDS = {"product_name", "brands", "ingredients_text", "quantity", "code"}


def apply_partner_type_f(row: dict, schema: dict) -> dict:
    """TYPE_F regex применённый только к partner-available полям.

    Используется в regex_only/regex_ml/regex_ml_bayes configs, где OFF-side
    поля (categories_tags, labels_tags) методологически не считаются доступными.
    Те же TYPE_F_RULES, но row фильтруется до PARTNER_FIELDS — паттерны вида
    "en:spray" или "en:body-creams" перестают срабатывать, остаются только
    лексические matches по product_name/ingredients_text.

    Применяется только для атрибутов из TYPE_F_RULES: product_type, form_factor
    (cosmetics) и primary_protein_source (pet food). Остальные TYPE_F правила
    с `requires` на categories_tags корректно вернут None.
    """
    partner_row = {k: v for k, v in row.items() if k in PARTNER_FIELDS}
    result = {}
    for attr in schema:
        if attr not in TYPE_F_RULES:
            continue
        v = _type_f_regex_multiclass(partner_row, attr)
        if v is not None:
            result[attr] = v
    return result


def apply_off_labels(row: dict, schema: dict) -> dict:
    """Заполнить атрибуты которые выводимы из OFF полей до LLM-вызова.

    Returns dict с values только для атрибутов из schema.
    None означает "OFF не определил, нужен LLM" (атрибут не включается в return).
    """
    result = {}
    for attr in schema:
        # Specialized: is_vegan — 3-state детекция через labels + ingredients_analysis
        v = _type_vegan_specific(row, attr)
        if v is not None:
            result[attr] = v
            continue

        # Specialized: is_ultra_processed (cheeses) — nova_group==4
        v = _type_ultra_processed_specific(row, attr)
        if v is not None:
            result[attr] = v
            continue

        # Type AC (hybrid tag-or-numeric, e.g. is_low_sugar, is_high_fibre)
        v = _type_ac_hybrid(row, attr)
        if v is not None:
            result[attr] = v
            continue

        # Try Type A
        v = _type_a_bool(row, attr)
        if v is not None:
            result[attr] = v
            continue

        # Type B
        v = _type_b_multiclass(row, attr)
        if v is not None:
            result[attr] = v
            continue

        # Type C (schema-aware: bucket override may come from schema[attr]["buckets"])
        v = _type_c_numeric(row, attr, schema)
        if v is not None:
            result[attr] = v
            continue

        # Type D
        v = _type_d_direct(row, attr)
        if v is not None:
            result[attr] = v
            continue

        # Type E (regex / traces fallback — contains_nuts)
        v = _type_e_regex(row, attr)
        if v is not None:
            result[attr] = v
            continue

        # Type F (regex multiclass — fragrance_status, form_factor, target_audience,
        # primary_protein_source)
        v = _type_f_regex_multiclass(row, attr)
        if v is not None:
            result[attr] = v
            continue

        # None of the rules applied — leave for LLM
    return result
