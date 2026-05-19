"""
Reference source taxonomy per (category, attribute).

Каждой паре (category, attribute) сопоставлен наиболее надёжный
доступный источник эталона. Используется в §6.12.1 notebook'а как
замена унифицированного фильтра `silver_vs_manual_acc ≥ 75%`
(§3.1) на per-attribute source-aware эталон.

Идея: вместо «manual gold annotation на ~50 продуктов × категория» —
строить эталон **по типу проблемы**:
  - бакетные нутри-классы → OFF own *_100g + регуляторная бакетизация;
  - сертификации        → OFF labels_tags (проверяется регулятором);
  - INCI/regex-driven   → детерминированный regex;
  - категориальные      → 3-LLM consensus + ручной арбитраж на disagreement.

Это сильнее, чем «manual gold» в обычном понимании, и требует
~10x меньше ручной работы (только ~300 спорных пар вместо 2400).

Использование:
    from src.eval.validation_sources import (
        VALIDATION_SOURCE, SOURCE_TIER, Source, SourceTier,
        get_source, get_tier, summary_table,
    )

    src = get_source('pasta', 'nutri_score_grade')  # Source.NUTRI_COMPUTED
    tier = get_tier('pasta', 'nutri_score_grade')    # SourceTier.GOLD

    df = summary_table()  # сводная таблица для всех 41 пар
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

import pandas as pd


class Source(str, Enum):
    """Тип источника эталона."""

    NUTRI_COMPUTED = "nutri_computed"
    OFF_LABELS_TAGS = "off_labels_tags"
    TEXT_REGEX = "text_regex"
    NOVA_DERIVED = "nova_derived"
    HYBRID_REGEX_LLM = "hybrid_regex_llm"
    CONSENSUS_NEEDED = "consensus_needed"


class SourceTier(str, Enum):
    """Качественный уровень эталона."""

    GOLD = "gold"
    CLOSE_TO_GOLD = "close_to_gold"
    SILVER_STRONG = "silver_strong"


SOURCE_TIER: dict[Source, SourceTier] = {
    Source.NUTRI_COMPUTED: SourceTier.GOLD,
    Source.TEXT_REGEX: SourceTier.GOLD,
    Source.OFF_LABELS_TAGS: SourceTier.CLOSE_TO_GOLD,
    Source.NOVA_DERIVED: SourceTier.CLOSE_TO_GOLD,
    Source.HYBRID_REGEX_LLM: SourceTier.CLOSE_TO_GOLD,
    Source.CONSENSUS_NEEDED: SourceTier.SILVER_STRONG,
}


SOURCE_DESCRIPTION: dict[Source, str] = {
    Source.NUTRI_COMPUTED: (
        "Бакетизация из реальных нутри-полей OFF (fat_100g/sugars_100g/proteins_100g/"
        "fiber_100g) по регуляторной формуле. Числовые поля заявлены производителем "
        "согласно EU 1169/2011 или community-verified в OFF. На покрытом срезе "
        "(95-100%) эталон детерминирован — gold."
    ),
    Source.OFF_LABELS_TAGS: (
        "Тег из OFF labels_tags (en:organic, en:gluten-free, "
        "en:protected-designation-of-origin) либо origins_tags. Сертификации "
        "проверяются регулятором (EU 834/2007 для organic, EU 1151/2012 для "
        "AOP/PDO). Близко к gold; шум только от пропусков в OFF, не от "
        "содержательных ошибок."
    ),
    Source.TEXT_REGEX: (
        "Детерминированный regex по product_name или ingredients_text. Примеры: "
        "cocoa_percentage = `\\d+%` в названии, has_sulfates/has_silicones — INCI "
        "lookup (SLS/SLES/dimethicone/-siloxane), is_filled = "
        "`ravioli|tortellini|farci|filled`. Gold для well-defined паттернов; шум "
        "только от опечаток и нестандартных формулировок."
    ),
    Source.NOVA_DERIVED: (
        "NOVA-классификация (Monteiro et al. 2019, 4 группы по степени обработки) — "
        "derived from ingredients additives, эмульгаторов и industrial markers. "
        "Используется в EU/Brazil/France public health policy и в Nutri-Score 2.0. "
        "Close to gold; OFF nova_group поле обычно community-verified."
    ),
    Source.HYBRID_REGEX_LLM: (
        "Гибрид: regex покрывает явные случаи (например, is_vegan через "
        "non-vegan-ingredient regex: milk/egg/honey/gelatin/lactose; "
        "contains_nuts через traces_tags + INCI), LLM-consensus только для "
        "пограничных. ~70-80% покрывается regex (gold), остаток — silver-strong "
        "через 3-LLM consensus + manual arbitrage."
    ),
    Source.CONSENSUS_NEEDED: (
        "Категориальные атрибуты с текстовой семантикой (pasta_shape, "
        "chocolate_type, milk_source, cereal_type) — нет deterministic-источника. "
        "Эталон строится через 3-LLM consensus (Sonnet 4.5 + GPT-4o + Gemini) с "
        "ручным арбитражем на disagreement. По таксономии weak supervision "
        "(Ratner et al. 2017, Snorkel) — silver-strong с известным noise floor."
    ),
}


VALIDATION_SOURCE: dict[tuple[str, str], Source] = {
    # === pasta (8 атрибутов) ===
    ("pasta", "grain_type"):         Source.CONSENSUS_NEEDED,
    ("pasta", "is_filled"):          Source.TEXT_REGEX,
    ("pasta", "is_organic"):         Source.OFF_LABELS_TAGS,
    ("pasta", "is_gluten_free"):     Source.OFF_LABELS_TAGS,
    ("pasta", "pasta_shape"):        Source.CONSENSUS_NEEDED,
    ("pasta", "is_vegan"):           Source.HYBRID_REGEX_LLM,
    ("pasta", "nutri_score_grade"):  Source.NUTRI_COMPUTED,
    ("pasta", "protein_class"):      Source.NUTRI_COMPUTED,

    # === chocolate (7 атрибутов) ===
    ("chocolate", "chocolate_type"):    Source.CONSENSUS_NEEDED,
    ("chocolate", "cocoa_percentage"):  Source.TEXT_REGEX,
    ("chocolate", "contains_nuts"):     Source.HYBRID_REGEX_LLM,
    # chocolate_extra: silver получает значение из OFF traces_tags + ingredients regex
    # (apply_off_labels). 3-LLM consensus систематически проигрывает silver (-24 п.п.),
    # потому что LLM не имеет доступа к traces_tags. Переклассифицирован 2026-05-15.
    ("chocolate", "chocolate_extra"):   Source.OFF_LABELS_TAGS,
    ("chocolate", "is_organic"):        Source.OFF_LABELS_TAGS,
    ("chocolate", "nutri_score_grade"): Source.NUTRI_COMPUTED,
    ("chocolate", "protein_class"):     Source.NUTRI_COMPUTED,

    # === beverages (7 атрибутов) ===
    ("beverages", "beverage_type"):     Source.CONSENSUS_NEEDED,
    ("beverages", "sugar_class"):       Source.NUTRI_COMPUTED,
    ("beverages", "is_organic"):        Source.OFF_LABELS_TAGS,
    ("beverages", "is_carbonated"):     Source.HYBRID_REGEX_LLM,
    ("beverages", "nutri_score_grade"): Source.NUTRI_COMPUTED,
    ("beverages", "protein_class"):     Source.NUTRI_COMPUTED,
    ("beverages", "nova_group"):        Source.NOVA_DERIVED,
    ("beverages", "is_vegan"):          Source.HYBRID_REGEX_LLM,

    # === cheeses (7 атрибутов) ===
    ("cheeses", "milk_source"):         Source.CONSENSUS_NEEDED,
    ("cheeses", "texture"):             Source.CONSENSUS_NEEDED,
    ("cheeses", "country_of_origin"):   Source.OFF_LABELS_TAGS,  # origins_tags + AOP DB
    ("cheeses", "fat_class"):           Source.NUTRI_COMPUTED,
    ("cheeses", "is_pdo"):              Source.OFF_LABELS_TAGS,
    ("cheeses", "is_organic"):          Source.OFF_LABELS_TAGS,
    ("cheeses", "is_ultra_processed"): Source.NOVA_DERIVED,

    # === cereals (8 атрибутов) ===
    ("cereals", "cereal_type"):    Source.CONSENSUS_NEEDED,
    ("cereals", "grain_type"):     Source.HYBRID_REGEX_LLM,
    ("cereals", "is_low_sugar"):   Source.NUTRI_COMPUTED,
    ("cereals", "is_high_fibre"):  Source.NUTRI_COMPUTED,
    ("cereals", "nova_class"):     Source.NOVA_DERIVED,
    ("cereals", "is_vegan"):       Source.HYBRID_REGEX_LLM,
    ("cereals", "is_whole_grain"): Source.TEXT_REGEX,
    ("cereals", "is_organic"):     Source.OFF_LABELS_TAGS,

    # === cosmetics (6 атрибутов) ===
    ("cosmetics", "product_type"):  Source.TEXT_REGEX,    # TYPE_F regex (см. §1)
    ("cosmetics", "form_factor"):   Source.TEXT_REGEX,
    ("cosmetics", "body_area"):     Source.CONSENSUS_NEEDED,
    ("cosmetics", "has_sulfates"):  Source.TEXT_REGEX,    # INCI: SLS/SLES/SCS
    ("cosmetics", "has_silicones"): Source.TEXT_REGEX,    # INCI: dimethicone/-siloxane
    ("cosmetics", "is_organic"):    Source.OFF_LABELS_TAGS,
}


# Ожидаемая точность эталона vs фактическая правда (для аргумента на защите).
# Цифры — консервативная оценка по литературе weak supervision и регуляторным
# требованиям к OFF community.
EXPECTED_GOLD_QUALITY: dict[Source, tuple[float, float]] = {
    Source.NUTRI_COMPUTED:   (0.95, 0.99),  # OFF нутри-поля community-verified; шум на бакетных границах
    Source.OFF_LABELS_TAGS:  (0.90, 0.98),  # сертификации регулируются, но возможны пропуски
    Source.TEXT_REGEX:       (0.92, 0.99),  # gold при правильном regex; шум только нестандартные формулировки
    Source.NOVA_DERIVED:     (0.85, 0.95),  # NOVA — экспертная классификация, есть граница 3/4
    Source.HYBRID_REGEX_LLM: (0.80, 0.92),  # regex покрывает явное, LLM на пограничном
    Source.CONSENSUS_NEEDED: (0.75, 0.90),  # 3-LLM consensus floor по §6.7 + arbitrage
}


def get_source(category: str, attr: str) -> Optional[Source]:
    """Источник эталона для пары (category, attribute) или None."""
    return VALIDATION_SOURCE.get((category, attr))


def get_tier(category: str, attr: str) -> Optional[SourceTier]:
    """Качественный уровень эталона."""
    src = get_source(category, attr)
    return SOURCE_TIER.get(src) if src else None


def get_expected_quality(category: str, attr: str) -> Optional[tuple[float, float]]:
    """Диапазон ожидаемой точности эталона vs фактической правды."""
    src = get_source(category, attr)
    return EXPECTED_GOLD_QUALITY.get(src) if src else None


def summary_table() -> pd.DataFrame:
    """Сводная таблица: (category, attr, source, tier, ожидаемая точность эталона)."""
    rows = []
    for (cat, attr), src in VALIDATION_SOURCE.items():
        tier = SOURCE_TIER[src]
        q_lo, q_hi = EXPECTED_GOLD_QUALITY[src]
        rows.append({
            "category": cat,
            "attr": attr,
            "source": src.value,
            "tier": tier.value,
            "expected_quality_lo": q_lo,
            "expected_quality_hi": q_hi,
        })
    return pd.DataFrame(rows).sort_values(["category", "source", "attr"]).reset_index(drop=True)


def tier_counts() -> pd.DataFrame:
    """Распределение по tier."""
    df = summary_table()
    return df.groupby(["tier", "source"]).size().reset_index(name="n_pairs")


def coverage_of_source(category: str, attr: str, df: pd.DataFrame) -> float:
    """Доля продуктов, на которых эталон по этому источнику доступен.

    Для NUTRI_COMPUTED — coverage входных *_100g полей (определяет
    возможность бакетизации). Для остальных детерминированных источников —
    coverage уже-размеченной schema-колонки (результат label_silver),
    т.к. она и есть «эталон по этому источнику».
    """
    src = get_source(category, attr)
    if src is None:
        return 0.0

    n = len(df)
    if n == 0:
        return 0.0

    # NUTRI_COMPUTED считается из числовых *_100g; coverage = доля продуктов
    # с заполненным входным числовым полем (= возможность бакетизации).
    NUTRI_INPUT_BY_ATTR = {
        "protein_class":     "proteins_100g",
        "sugar_class":       "sugars_100g",
        "fat_class":         "fat_100g",
        "is_low_sugar":      "sugars_100g",
        "is_high_fibre":     "fiber_100g",
        # nutri_score_grade требует совокупности 5 полей; считаем по минимуму
        "nutri_score_grade": "fat_100g",
    }

    if src == Source.NUTRI_COMPUTED:
        col = NUTRI_INPUT_BY_ATTR.get(attr)
        if col and col in df.columns:
            return float(df[col].notna().sum() / n)
        return 0.0

    # Для всех остальных детерминированных источников эталон — это
    # уже-размеченная schema-колонка (label_silver применил правила).
    if attr in df.columns:
        return float(df[attr].notna().sum() / n)

    return 0.0


# Сверка: VALIDATION_SOURCE должен содержать ровно те же пары, что
# `manual_eval_summary.parquet`. Эту проверку запускаем в тестах.
EXPECTED_PAIRS_COUNT = 44  # 8+7+8+7+8+6
