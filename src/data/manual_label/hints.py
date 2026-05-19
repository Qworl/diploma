"""
Детерминированные подсказки для ручной разметки.

Цель: снизить когнитивную нагрузку на разметчика, не используя ML/LLM/silver-метку
(иначе будет anchoring bias и hold-out перестанет быть честным якорем).

Источники подсказок:
1. **nutrient** — арифметика над числами `*_100g` по тем же бакетам, что в
   llm_enricher.TYPE_C_RULES. Чисто детерминирована, не зависит от модели.
2. **off_label** — regex по `labels_tags` (en:nutriscore-grade-b, en:organic, …).
   Это тот же source-of-truth, что доступен любому downstream-консьюмеру OFF.
3. **regex** — keyword match в product_name / ingredients_text / categories_tags.

Каждая подсказка возвращается с явным `source`+`detail`, чтобы разметчик видел,
*почему* предложено это значение, и мог отвергнуть.

Никогда НЕ возвращаем подсказку без явного positive-evidence — отсутствие
'organic' label не означает «not organic». В таких случаях hint = None.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

# Numeric bucket rules — копия llm_enricher.TYPE_C_RULES, чтобы избежать кругового импорта
_NUMERIC_RULES = {
    "protein_class": {
        "field": "proteins_100g",
        "buckets": [(1.0, "0"), (5.0, "low"), (15.0, "med"), (float("inf"), "high")],
    },
    "sugar_class": {
        "field": "sugars_100g",
        "buckets": [(0.5, "0"), (5.0, "low"), (10.0, "med"), (float("inf"), "high")],
    },
}


@dataclass
class Hint:
    value: Any
    source: str  # 'nutrient' | 'off_label' | 'regex'
    detail: str  # human-readable evidence


# --- Helpers ---

def _str(row, col) -> str:
    v = row.get(col)
    if pd.isna(v):
        return ""
    return str(v)


def _bucket(value: float, buckets: list[tuple[float, str]]) -> str:
    for threshold, label in buckets:
        if value < threshold:
            return label
    return buckets[-1][1]


def _numeric_hint(row, attr: str) -> Hint | None:
    rule = _NUMERIC_RULES[attr]
    raw = row.get(rule["field"])
    if pd.isna(raw):
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    bucket = _bucket(v, rule["buckets"])
    return Hint(value=bucket, source="nutrient", detail=f"{rule['field']}={v:.1f}")


def _labels_blob(row) -> str:
    return _str(row, "labels_tags").lower()


def _text_blob(row) -> str:
    return " ".join(_str(row, c) for c in ("product_name", "generic_name", "ingredients_text", "categories_tags")).lower()


# --- OFF-label rules ---

_NUTRI_LABEL_RE = re.compile(r"en:nutriscore-grade-([a-e])")
_NOVA_LABEL_RE = re.compile(r"en:nova-group-([1-4])")


def _hint_nutri_score(row) -> Hint | None:
    m = _NUTRI_LABEL_RE.search(_labels_blob(row))
    if m:
        return Hint(value=m.group(1).upper(), source="off_label", detail=f"en:nutriscore-grade-{m.group(1)}")
    return None


def _hint_nova_group(row) -> Hint | None:
    m = _NOVA_LABEL_RE.search(_labels_blob(row))
    if m:
        return Hint(value=int(m.group(1)), source="off_label", detail=f"en:nova-group-{m.group(1)}")
    return None


_ORGANIC_LABEL_RE = re.compile(r"en:(organic|eu-organic|ab-agriculture-biologique|de-eco-bio)\b")
_ORGANIC_TEXT_RE = re.compile(r"\b(bio|biologique|biologico|orgánico|organic|öko|ekologisk)\b", re.IGNORECASE)


def _hint_is_organic(row) -> Hint | None:
    m = _ORGANIC_LABEL_RE.search(_labels_blob(row))
    if m:
        return Hint(value=True, source="off_label", detail=m.group(0))
    if _ORGANIC_TEXT_RE.search(_str(row, "product_name") + " " + _str(row, "brands")):
        return Hint(value=True, source="regex", detail="bio/organic в name/brands")
    return None


_GLUTEN_FREE_LABEL = "en:gluten-free"
_GLUTEN_CONTAINS_LABEL = "en:contains-gluten"
_GLUTEN_FREE_TEXT_RE = re.compile(
    r"\b(sans\s+gluten|gluten[\s-]?free|senza\s+glutine|sin\s+gluten|glutenfrei|без\s+глютена)\b",
    re.IGNORECASE,
)


def _hint_is_gluten_free(row) -> Hint | None:
    lb = _labels_blob(row)
    if _GLUTEN_FREE_LABEL in lb:
        return Hint(value=True, source="off_label", detail=_GLUTEN_FREE_LABEL)
    if _GLUTEN_CONTAINS_LABEL in lb:
        return Hint(value=False, source="off_label", detail=_GLUTEN_CONTAINS_LABEL)
    if _GLUTEN_FREE_TEXT_RE.search(_text_blob(row)):
        return Hint(value=True, source="regex", detail="gluten-free / sans gluten")
    return None


_VEGAN_LABEL = "en:vegan"
_NON_VEGAN_LABEL = "en:non-vegan"
_ANIMAL_INGREDIENTS_RE = re.compile(
    r"\b(milk|lait|latte|leche|milch|молоко|"
    r"egg|œuf|oeuf|uovo|huevo|ei\b|яйц|"
    r"butter|beurre|burro|mantequilla|"
    r"cheese|fromage|formaggio|queso|käse|сыр|"
    r"honey|miel|miele|honig|мёд|мед|"
    r"whey|petit-lait|siero|suero|molke|сыворотк)\w*",
    re.IGNORECASE,
)


def _hint_is_vegan(row) -> Hint | None:
    lb = _labels_blob(row)
    if _VEGAN_LABEL in lb:
        return Hint(value=True, source="off_label", detail=_VEGAN_LABEL)
    if _NON_VEGAN_LABEL in lb:
        return Hint(value=False, source="off_label", detail=_NON_VEGAN_LABEL)
    ingr = _str(row, "ingredients_text")
    if ingr:
        m = _ANIMAL_INGREDIENTS_RE.search(ingr)
        if m:
            return Hint(value=False, source="regex", detail=f"animal ingredient: '{m.group(0)}'")
    return None


_NO_ADDED_SUGAR_LABELS = ("en:no-added-sugar", "en:no-sugar-added", "en:no-sugars-added")
_NO_ADDED_SUGAR_TEXT_RE = re.compile(
    r"\b(sans\s+sucres?\s+ajout|no\s+added\s+sugars?|ohne\s+zuckerzusatz|"
    r"senza\s+zuccheri\s+aggiunti|sin\s+az[uú]cares?\s+a[ñn]adidos?)",
    re.IGNORECASE,
)


def _hint_is_no_added_sugar(row) -> Hint | None:
    lb = _labels_blob(row)
    for tag in _NO_ADDED_SUGAR_LABELS:
        if tag in lb:
            return Hint(value=True, source="off_label", detail=tag)
    if _NO_ADDED_SUGAR_TEXT_RE.search(_text_blob(row)):
        return Hint(value=True, source="regex", detail="'sans sucre ajouté' / similar")
    return None


_PALM_FREE_LABELS = ("en:palm-oil-free", "en:no-palm-oil")
_PALM_CONTAINS_LABEL = "en:contains-palm-oil"
_PALM_TEXT_RE = re.compile(
    r"\b(huile\s+de\s+palme|palm\s+oil|palm[öo]l|olio\s+di\s+palma|aceite\s+de\s+palma)\b",
    re.IGNORECASE,
)


def _hint_palm_oil_status(row) -> Hint | None:
    lb = _labels_blob(row)
    for tag in _PALM_FREE_LABELS:
        if tag in lb:
            return Hint(value="palm-oil-free", source="off_label", detail=tag)
    if _PALM_CONTAINS_LABEL in lb:
        return Hint(value="contains", source="off_label", detail=_PALM_CONTAINS_LABEL)
    ingr = _str(row, "ingredients_text")
    if ingr and _PALM_TEXT_RE.search(ingr):
        return Hint(value="contains", source="regex", detail="palm oil в ingredients")
    return None


_WHOLE_GRAIN_LABEL = "en:whole-grain"
_WHOLE_GRAIN_TEXT_RE = re.compile(
    r"\b(whole[\s-]?grain|wholemeal|whole\s+wheat|"
    r"compl[eèé]te?s?|int[eé]gra(le|l)e?s?|integrale|integrali|integral|integrales|"
    r"vollkorn|wholegrain)\b",
    re.IGNORECASE,
)


def _hint_is_whole_grain(row) -> Hint | None:
    if _WHOLE_GRAIN_LABEL in _labels_blob(row):
        return Hint(value=True, source="off_label", detail=_WHOLE_GRAIN_LABEL)
    blob = _str(row, "product_name") + " " + _str(row, "ingredients_text")
    m = _WHOLE_GRAIN_TEXT_RE.search(blob)
    if m:
        return Hint(value=True, source="regex", detail=f"matched '{m.group(0)}'")
    return None


_NUTS_TEXT_RE = re.compile(
    r"\b(hazelnut|noisette|nocciol|avellan|haseln[üu]ss|"
    r"almond|amande|mandorl|almendra|mandel|"
    r"walnut|noix|noce\b|noci|nuez|nogal|walnuss|"
    r"pecan|pecán|"
    r"pistachio|pistache|pistacchio|pistacho|pistazi|"
    r"cashew|noix\s+de\s+cajou|anacardo|kasch[eu]w|"
    r"peanut|cacahu[èe]te|arachide|cacahuet|erdnuss)\w*",
    re.IGNORECASE,
)


def _hint_contains_nuts(row) -> Hint | None:
    blob = _str(row, "product_name") + " " + _str(row, "ingredients_text")
    m = _NUTS_TEXT_RE.search(blob)
    if m:
        return Hint(value=True, source="regex", detail=f"matched '{m.group(0)}'")
    return None


# --- Wrappers around RegexExtractor (для grain_type/pasta_shape/cocoa/chocolate_type/beverage_type) ---

def _hint_via_regex_extractor(row, attr: str, category: str) -> Hint | None:
    """Используем существующий RegexExtractor — он уже многоязычный."""
    from src.pipeline.regex.extractor import RegexExtractor
    rx = RegexExtractor()
    name = _str(row, "product_name")
    desc = _str(row, "generic_name")
    qty = _str(row, "quantity")
    res = rx.extract_all(name, desc, qty, category=category)
    extracted = res.get(attr)
    if extracted is None or extracted.value is None:
        return None
    return Hint(value=extracted.value, source="regex", detail=f"RegexExtractor.{attr}")


# --- Dispatcher ---

# Map (category, attr) → callable(row) → Hint | None
_HINT_FNS = {
    # universal
    "protein_class": lambda row, cat: _numeric_hint(row, "protein_class"),
    "is_organic":    lambda row, cat: _hint_is_organic(row),
    "is_gluten_free": lambda row, cat: _hint_is_gluten_free(row),
    "is_vegan":      lambda row, cat: _hint_is_vegan(row),
    "nutri_score_grade": lambda row, cat: _hint_nutri_score(row),
    "nova_group":    lambda row, cat: _hint_nova_group(row),
    "sugar_class":   lambda row, cat: _numeric_hint(row, "sugar_class"),
    "is_no_added_sugar": lambda row, cat: _hint_is_no_added_sugar(row),
    "palm_oil_status": lambda row, cat: _hint_palm_oil_status(row),
    "is_whole_grain": lambda row, cat: _hint_is_whole_grain(row),
    "contains_nuts": lambda row, cat: _hint_contains_nuts(row),
    "grain_type":    lambda row, cat: _hint_via_regex_extractor(row, "grain_type", cat),
    "pasta_shape":   lambda row, cat: _hint_via_regex_extractor(row, "pasta_shape", cat),
    "cocoa_percentage": lambda row, cat: _hint_via_regex_extractor(row, "cocoa_percentage", cat),
    "chocolate_type": lambda row, cat: _hint_via_regex_extractor(row, "chocolate_type", cat),
    "beverage_type":  lambda row, cat: _hint_via_regex_extractor(row, "beverage_type", cat),
}


def compute_hints(row, category: str, attrs: list[str]) -> dict[str, Hint]:
    """Возвращает словарь подсказок только для тех атрибутов, где есть positive evidence."""
    out: dict[str, Hint] = {}
    for attr in attrs:
        fn = _HINT_FNS.get(attr)
        if fn is None:
            continue
        try:
            h = fn(row, category)
        except Exception:
            continue
        if h is not None and h.value is not None:
            out[attr] = h
    return out
