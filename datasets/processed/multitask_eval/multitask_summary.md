# Multitask vs Single XGBoost — итоги

Эксперимент: для каждой пары атрибутов (A, B) обучаем три модели на одном train/test:
- **single** — отдельный XGBoost для каждого атрибута (текущий baseline в проекте)
- **multioutput** — `MultiOutputClassifier(XGBClassifier(...))` поверх (A, B)
- **cartesian** — единый XGBoost с меткой `A × B` (Cartesian product классов)

Признаки: cached SBERT-эмбеддинги (384d, multilingual-MiniLM).
Сплит: brand-disjoint 60/20/20 из `{cat}_gold_split.parquet`.
Метрики на test-сплите интерсекции (оба атрибута размечены).

## Результаты по парам (per-attribute)

| cat | pair | attr | n_test | single acc | multi acc | Δ acc (multi-single) | single F1 | multi F1 | cartesian acc | cartesian F1 |
|-----|------|------|--------|-----------:|----------:|---------------------:|----------:|---------:|--------------:|-------------:|
| pasta | pasta_shape+is_filled | pasta_shape | 141 | 0.816 | 0.801 | -1.4 п.п. | 0.485 | 0.437 | 0.801 | 0.476 |
| pasta | pasta_shape+is_filled | is_filled | 141 | 0.936 | 0.936 | +0.0 п.п. | 0.933 | 0.932 | 0.936 | 0.934 |
| pasta | grain_type+is_gluten_free | grain_type | 136 | 0.941 | 0.934 | -0.7 п.п. | 0.680 | 0.512 | 0.934 | 0.513 |
| pasta | grain_type+is_gluten_free | is_gluten_free | 136 | 0.919 | 0.934 | +1.5 п.п. | 0.758 | 0.786 | 0.926 | 0.730 |
| chocolate | chocolate_extra+contains_nuts | chocolate_extra | 232 | 0.629 | 0.655 | +2.6 п.п. | 0.336 | 0.359 | 0.629 | 0.327 |
| chocolate | chocolate_extra+contains_nuts | contains_nuts | 232 | 0.772 | 0.733 | -3.9 п.п. | 0.710 | 0.659 | 0.828 | 0.800 |
| cheeses | milk_source+texture | milk_source | 62 | 0.806 | 0.823 | +1.6 п.п. | 0.400 | 0.400 | 0.774 | 0.409 |
| cheeses | milk_source+texture | texture | 62 | 0.613 | 0.629 | +1.6 п.п. | 0.384 | 0.394 | 0.694 | 0.610 |
| cheeses | milk_source+country_of_origin | milk_source | 63 | 0.857 | 0.873 | +1.6 п.п. | 0.233 | 0.233 | 0.825 | 0.232 |
| cheeses | milk_source+country_of_origin | country_of_origin | 63 | 0.857 | 0.825 | -3.2 п.п. | 0.700 | 0.672 | 0.873 | 0.848 |
| chocolate | chocolate_type+chocolate_extra | chocolate_type | 194 | 0.851 | 0.861 | +1.0 п.п. | 0.475 | 0.518 | 0.845 | 0.520 |
| chocolate | chocolate_type+chocolate_extra | chocolate_extra | 194 | 0.701 | 0.701 | +0.0 п.п. | 0.482 | 0.429 | 0.639 | 0.468 |
| pasta | is_organic+is_vegan | is_organic | 175 | 0.909 | 0.903 | -0.6 п.п. | 0.863 | 0.856 | 0.920 | 0.880 |
| pasta | is_organic+is_vegan | is_vegan | 175 | 0.897 | 0.909 | +1.1 п.п. | 0.895 | 0.907 | 0.897 | 0.894 |

## Net-effect по парам (сумма по обоим атрибутам)

Положительное значение — выигрыш относительно single. Если ΣAcc одного и того же знака с ΣF1 — сигнал устойчив.

| cat | pair | ΔΣ acc multi-single | ΔΣ acc cart-single | ΔΣ macro_f1 multi-single | ΔΣ macro_f1 cart-single |
|-----|------|--------------------:|-------------------:|-------------------------:|------------------------:|
| pasta | pasta_shape+is_filled | -0.014 | -0.014 | -0.048 | -0.007 |
| pasta | grain_type+is_gluten_free | +0.007 | +0.000 | -0.139 | -0.194 |
| chocolate | chocolate_extra+contains_nuts | -0.013 | +0.056 | -0.028 | +0.082 |
| cheeses | milk_source+texture | +0.032 | +0.048 | +0.010 | +0.235 |
| cheeses | milk_source+country_of_origin | -0.016 | -0.016 | -0.027 | +0.148 |
| chocolate | chocolate_type+chocolate_extra | +0.010 | -0.067 | -0.011 | +0.030 |
| pasta | is_organic+is_vegan | +0.006 | +0.011 | +0.005 | +0.016 |

## Сводка

**MultiOutput vs Single** (per-attribute, по acc, порог ±0.5 п.п.): 7 выигрышей, 5 проигрышей, 2 ничьих.
**Cartesian vs Single** (per-attribute, по acc, порог ±0.5 п.п.): 5 выигрышей, 6 проигрышей.

### Какие пары выиграли в multi-output (по ΣAcc):

- **cheeses/milk_source+texture**: ΔΣAcc multi-single = +0.032, ΔΣF1 = +0.010
- **chocolate/chocolate_type+chocolate_extra**: ΔΣAcc multi-single = +0.010, ΔΣF1 = -0.011

### Какие пары выиграли в cartesian (по ΣAcc):

- **chocolate/chocolate_extra+contains_nuts**: ΔΣAcc cart-single = +0.056, ΔΣF1 = +0.082
- **cheeses/milk_source+texture**: ΔΣAcc cart-single = +0.048, ΔΣF1 = +0.235
- **pasta/is_organic+is_vegan**: ΔΣAcc cart-single = +0.011, ΔΣF1 = +0.016

## Вывод

**Multi-output XGBoost практически не даёт выигрыша над per-attribute baseline.** Из 7 пар: ровно 1 (cheeses/milk_source+texture) даёт устойчивый прирост (ΔΣAcc ≈ +0.03) и в acc, и в macro-F1; остальные — около нуля либо отрицательны. Эффект shared representation в sklearn-обёртке `MultiOutputClassifier` минимален: под капотом обучаются `n_outputs` независимых XGBoost (без общих градиентов между головами), отличие от per-attribute baseline сводится только к разделённой sample-weight стратегии. Поэтому совпадение результатов — ожидаемо.

**Cartesian (объединённый ярлык A×B) показал смешанные результаты:** для пар с малым числом классов и сильной корреляцией (chocolate/chocolate_extra+contains_nuts, cheeses/milk_source+texture, cheeses/milk_source+country_of_origin) даёт заметный прирост macro-F1 (+0.08…+0.24), что вызвано тем, что предсказание совместной метки лучше учитывает корреляции редких комбинаций (например, sheep+hard vs cow+hard). Для пар с большим Cartesian-пространством (pasta_shape×is_filled = 22 класса, chocolate_type×chocolate_extra = 28) — деградация: модель не успевает выучить редкие комбинации, проигрывает per-class accuracy.

**Применимость в production-каскаде:** не рекомендуется заменять per-attribute архитектуру на multi-output для всех пар. Точечно — `cartesian` подход стоит рассмотреть как опциональный лишь для конкретных пар с (а) ≤ ~15 совместных классов, (б) CramerV ≥ 0.5, (в) узким бутылочным горлом по macro-F1. Кандидат — `cheeses/milk_source × texture` (+0.235 ΣF1, +0.048 ΣAcc на n_test=62, т. е. ~3 правильно классифицированных образца — на грани шума при таком n).

## Замечания и оговорки

- Признаки: 384d MiniLM (cached). У production v4 модели — 768d MPNet + TF-IDF SVD-128.
  Цель эксперимента — изолировать архитектурный эффект (single vs multioutput),
  поэтому сравнение fair: оба варианта используют те же признаки.
- Без CalibratedClassifierCV и без early stopping — упрощено, применено единообразно ко всем трём вариантам.
- Test-сплит — пересечение брэнд-дизъюнктного `test` + разметки обоих атрибутов. Числа single-модели здесь НЕ совпадают с production-цифрами (production single-модель учится и оценивается на бо́льшем подмножестве, где разметка может быть только одного из атрибутов).
- На малых пересечениях (cheeses: n_test=62) ширина доверительного интервала Wilson ≈ ±8 п.п. — даже видимые выигрыши/проигрыши в acc лежат внутри шума.
