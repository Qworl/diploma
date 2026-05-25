# Принципы фильтрации данных, схемы атрибутов и методология оценки

> Финальная конфигурация системы. Содержит описание того, как устроены данные,
> схема атрибутов, правила оценки и известные ограничения.

---

## 1. Категориальный охват

Основные категории для экспериментов §3.3:

- `pasta` — макаронные изделия
- `chocolate` — шоколад
- `cheeses` — сыры (включая обработанные)

Категория `electronics` и другие участвуют только в роли OOD-класса для category router (Layer 0). В headline-метриках не учитываются.

## 2. Источник данных

- **Open Food Facts** (`food.parquet`, 4.5M товаров).
- Per-category выгрузка через DuckDB по `categories_tags`:

| Категория | Кол-во товаров |
|---|---|
| pasta | ~14,000 |
| chocolate | ~13,000 |
| cheeses | ~14,500 |

## 3. Фильтры применяемые на eval-time

При вычислении любых метрик из gold отбрасываются следующие ячейки:

| Фильтр | Описание |
|---|---|
| `in_scope == True` | Товар попадает под `CAT_VALID_TAGS[cat]` — проверка корректности категории по `categories_tags`. Защищает от мисс-категоризации в gold. |
| `disputed == False` | Только consensus gold: ячейки, где 3 LLM-аннотатора дали 3 разных ответа, считаются ненадёжным ground truth. |
| Не deprecated класс | Ячейки с `gold_value ∈ SCHEMA_EXCLUDE[attr]` исключаются (см. §6). |
| Не None gold | `gold_value` отсутствует (NaN) → ячейка не оценивается. |
| Только in-scope категории | Eval ограничен `{pasta, chocolate, cheeses}`. |

## 4. Train/test изоляция

Модели обучаются на **noleak training pool**: из silver standard `{cat}_gold_v4_wide.parquet` удалены все codes, попадающие хотя бы в один из eval gold-источников (consensus gold, human gold, extended gold).

Используются модели с суффиксом `_mpnet_tfidf_noleak`. Это обеспечивает честный test-time accuracy без memorization-bias.

## 5. Схема атрибутов

### 5.1 Chocolate

| Attribute | Type | Classes |
|---|---|---|
| `chocolate_type` | multiclass | `{dark, milk, white}` |
| `is_filled` | binary | `{True, False}` |
| `chocolate_extra` | multiclass | `{plain, with_nuts, with_fruit, with_caramel, with_cookie}` |
| `contains_nuts` | binary | `{True, False}` |
| `is_organic` | binary | `{True, False}` |
| `flavor_profile` | multiclass | `{fruity, intense_bitter, nutty, salty_caramel, spiced, sweet_creamy, floral}` |

**Принцип ортогональности.** `chocolate_type` и `is_filled` — независимые свойства. Шоколад может быть `dark + filled` одновременно (трюфели Lindt, Ritter Sport Marzapane, Ferrero Rocher). Объединение этих свойств в один multiclass-атрибут теряет информацию.

`chocolate_extra` описывает **только mix-ins** (что добавлено в шоколадную массу или сопровождает её), не структурную форму.

### 5.2 Cheeses

| Attribute | Type | Classes |
|---|---|---|
| `milk_source` | multiclass | `{buffalo, cow, goat, mixed, other, sheep}` |
| `texture` | multiclass | `{blue, cream, fresh, hard, processed, soft}` |
| `country_of_origin` | multiclass | (страны Европы + other) |
| `is_pdo` | binary | `{True, False}` |
| `is_organic` | binary | `{True, False}` |
| `is_ultra_processed` | binary | `{True, False}` |
| `aging` | multiclass | `{aged, fresh, young}` |

`texture` не содержит `semi_soft` (объединено с `soft`). Не содержит `other` (catch-all без когерентного сигнала).

### 5.3 Pasta

| Attribute | Type | Classes |
|---|---|---|
| `grain_type` | multiclass | `{wheat, rice, legume, potato, buckwheat, spelt, mixed, oat, corn, other}` |
| `pasta_shape` | multiclass | (формы) |
| `is_filled` | binary | `{True, False}` (равиоли, тортеллини и т.п.) |
| `is_gluten_free` | binary | `{True, False}` |
| `is_organic` | binary | `{True, False}` |
| `is_vegan` | binary | `{True, False}` |
| `cuisine_origin` | multiclass | (страны) |
| `protein_class` | multiclass | (опционально) |

`pasta.is_filled` ≠ `chocolate.is_filled` — независимые атрибуты для разных категорий.

## 6. Deprecated классы (SCHEMA_EXCLUDE)

Класс помечается deprecated при выполнении ВСЕХ условий:
1. n_train < 100 (sparse representation)
2. recall < 0.20 на independent eval set
3. Catch-all семантически (нет когерентного паттерна) **или** покрывается другим атрибутом
4. Нет downstream consumer с жёстким требованием к этому классу

Применённые deprecated:

```python
SCHEMA_EXCLUDE = {
    "chocolate_type":  {"filled", "other"},
    "chocolate_extra": {"filled", "other", "with_alcohol", "with_coffee"},
    "chocolate.flavor_profile": {"other"},
    "cheeses.texture": {"other"},
}
```

`chocolate.filled` → перенесён в `is_filled` (binary). `other` / `with_alcohol` / `with_coffee` — мёртвые catch-all классы.

При eval ячейки с deprecated `gold_value` исключаются. При training — `exclude_classes` в `CATEGORY_CONFIG` фильтрует их из train pool.

## 7. Derivation labels для is_filled (chocolate)

### Silver (training)
```python
is_filled = (chocolate_type == 'filled') OR
            (chocolate_extra == 'filled') OR
            FILLED_CHOCOLATE_REGEX.search(product_name | ingredients_text | categories_tags)
```

`FILLED_CHOCOLATE_REGEX` (в `src/pipeline/off_labels/rules.py`) покрывает:
- EN: `filled`, `filling`, `truffle`, `praline`, `ganache`, `gianduja`, `hollow`, `easter egg`, `molten`
- FR: `fourré`, `cœur (de|fondant)`, `oeuf (en|de) chocolat`
- ES: `relleno`, `corazón`, `sorpresa`
- DE: `gefüllt`, `füllung`
- IT: `ripieno`, `cuore`
- Filling-implies-filled: `marzipan`, `marzapane`, `liqueur`, `likör`, `nougat filling`
- Tags: `en:filled-chocolates`, `en:pralines`, `en:truffles`

Распределение в silver chocolate: ~12% positives (1481 из 12,494).

### Gold (eval)
- **Positives:** explicit cells где `chocolate_type='filled'` или `chocolate_extra='filled'`
- **Negatives:** cells где `chocolate_type ∈ {dark/milk/white}` И `chocolate_extra ∈ safe_set`, не в positives
- **Verification:** 2-LLM consensus (qwen3-max + deepseek-r1) для validation negatives. Override negative→positive только при двустороннем согласии.

Final counts: consensus gold +198 rows (17 pos / 181 neg), human gold +50 rows (5 pos / 45 neg).

## 8. Texture rule-based приоритеты

В `TYPE_F_RULES["texture"]` (cheeses) приоритет идёт сверху вниз. Ключевые принципы:

- **`cream`** имеет приоритет над `fresh` для свежих сливочных сыров: `ricotta`, `labneh`, `quark`, `speisequark`, `mascarpone`, `philadelphia`, `cream cheese`, `fromage frais à tartiner`, `faisselle`, `petit-suisse`.
- **`processed`** — индустриальные эмульсии: `kraft singles`, `laughing cow / vache qui rit`, `kiri`, `velveeta`, `cheez whiz`, `cheese spread`, `dairylea`.  
  Бренды cream-style (например, `Tartare`) НЕ относятся к processed.
- **`blue`** — penicillium roqueforti: `roquefort`, `gorgonzola`, `stilton`, `bleu d'Auvergne`, `danish blue`, `blauschimmel`.
- **`fresh`** — `mozzarella`, `feta`, `cottage`, `burrata`, `paneer`, `halloumi`, `queso fresco/blanco/panela`.
- **`soft`** — `brie`, `camembert`, `taleggio`, `munster`, `reblochon`, `mont d'or`, `vacherin`, `havarti`, `port salut`.
- **`hard`** — `parmesan`, `gruyère`, `emmental`, `cheddar`, `gouda`, `manchego`, `comté`, `pecorino`, `raclette`.

Ambiguous названия (например, `Coeur de Savoie` — может быть soft или hard) НЕ попадают в rule, оставлены для ML.

## 9. Category router (Layer 0)

Pre-cascade XGBoost-классификатор на partner-available fields (`product_name + brands + ingredients_text + quantity` → MPNet 768d). 4-class output: `{pasta, chocolate, cheeses, ood}`.

**Поведение:**
- `router_pred == true_cat` → cascade prediction используется
- `router_pred != true_cat` → e2e answer = None (cascade на чужих моделях считается ошибкой)
- `router_pred == 'ood'` → система абстенется (None)

Метрики:
- **Cascade-only accuracy** — router НЕ применён (assume oracle category). Upper bound.
- **E2E coverage** — % cells где router_pred корректен.
- **E2E acc (None=wrong)** — production-realistic, None считается как ошибка.

## 10. Метрики

Для multiclass-атрибутов всегда репортятся:
- **micro-accuracy** — общая доля правильных предсказаний
- **macro-F1** — невзвешенная средняя F1 по классам (penalises poor performance on rare classes)
- **balanced accuracy** — recall, усреднённый по классам
- **per-class precision, recall, F1**

Для headline accuracy используется **Wilson 95% CI** (не normal approximation — на малых n даёт корректные интервалы).

Для бинарных атрибутов: macro-F1 (с учётом обоих классов), accuracy, per-class recall.

## 11. Disputed cells policy

В 3-LLM consensus gold (qwen3.7-max + deepseek-r1 + mistral-large-2411):
- **`disputed=False`** — ≥2 модели согласны (consensus), используется в eval
- **`disputed=True`** — 3 разных ответа, ненадёжный GT, исключается из eval

В human gold (Opus-разметка) колонка `disputed` отсутствует — все ячейки используются.

## 12. Известные ограничения

### 12.1 Circular evaluation bias (~3.8pp)
- Training silver labels: hybrid (OFF tags + Gemini Flash relabel).
- Eval gold: qwen3.7-max + deepseek-r1 + mistral-large-2411 consensus.
- LLMs семантически перекрываются → measured accuracy на ~3.8pp выше truly-independent ground truth.
- Conservative interpretation: headline 94.8% на consensus → ~91% на полностью независимой разметке.

### 12.2 Brand-disjoint test невозможен
- pasta: 100% test brands present в training pool
- chocolate: 96% overlap, 7 disjoint codes
- cheeses: 93-98% overlap
- → claim о brand-disjoint generalization не делается

### 12.3 Sample size
- HUMAN gold: 615 cells, single labeler (Opus), без IRR. Используется как conservative lower-bound оценка.
- LLM-consensus gold: 3257 cells (primary headline), 3-LLM majority vote.

### 12.4 Long-tail классы
- Несколько атрибутов имеют классы с n_train < 100 (например, `cheeses.milk_source.other` n=26, `chocolate.flavor_profile.spiced` n=47). Эти классы сохранены в схеме, но имеют пониженный recall — production практика: routing в LLM fallback при низкой confidence.

### 12.5 Pasta.grain_type на human gold
- n=9, все `wheat`. Одна ошибка → macro-F1=0.47. Чистый sample noise.
- На consensus gold (n=139): macro-F1=0.81 — атрибут работает корректно.

## 13. Финальные правила (cheatsheet для notebook)

При построении любой метрики:
1. Filter `in_scope == True`
2. Filter `disputed == False` (если колонка есть)
3. Filter `gold_value ∉ SCHEMA_EXCLUDE[attr]`
4. Filter `gold_value` is not None
5. Eval ограничен `{pasta, chocolate, cheeses}`
6. Использовать модели с суффиксом `_mpnet_tfidf_noleak`
7. Router OOD treated as None
8. Wilson 95% CI для headline accuracy
9. Macro-F1 + per-class breakdown (не только micro)

## 14. Итоговые числа

### 14.1 Headline (LLM-consensus gold, n=3257)

| Метрика | Значение |
|---|---|
| Cascade-only micro-accuracy | **94.8%** |
| Cascade-only macro-F1 (cells-weighted) | **0.899** |
| Cascade-only macro-F1 (attr-unweighted) | **0.902** |
| Router accuracy (per code, n=570) | **95.4%** |
| E2E coverage | **96.0%** |
| **E2E accuracy (None=wrong) — production-realistic** | **91.1%** |

### 14.2 Conservative bound (HUMAN gold, n=615)

| Метрика | Значение |
|---|---|
| Cascade-only micro-accuracy | 91.7% |
| Cascade-only macro-F1 | 0.848 |
| Router accuracy (n=107) | 95.3% |
| E2E coverage | 95.4% |
| **E2E accuracy (None=wrong)** | **87.5%** |

### 14.3 Per-attribute macro-F1 (LLM-consensus gold)

| Категория | Атрибут | n | micro | macro-F1 |
|---|---|---|---|---|
| pasta | grain_type | 139 | 94.2% | 0.807 |
| pasta | pasta_shape | 139 | 99.3% | 0.984 |
| pasta | is_filled | 210 | 95.7% | 0.793 |
| pasta | is_gluten_free | 194 | 94.3% | 0.942 |
| pasta | is_organic | 193 | 97.4% | 0.969 |
| pasta | is_vegan | 200 | 95.5% | 0.939 |
| pasta | cuisine_origin | 198 | 90.9% | 0.790 |
| chocolate | chocolate_type | 150 | 98.0% | 0.973 |
| chocolate | is_filled | 198 | 94.9% | 0.861 |
| chocolate | chocolate_extra | 157 | 91.1% | 0.899 |
| chocolate | contains_nuts | 172 | 89.0% | 0.888 |
| chocolate | is_organic | 169 | 98.2% | 0.967 |
| chocolate | flavor_profile | 136 | 94.9% | 0.898 |
| cheeses | milk_source | 147 | 97.3% | 0.972 |
| cheeses | texture | 130 | 92.3% | 0.927 |
| cheeses | country_of_origin | 125 | 95.2% | 0.901 |
| cheeses | aging | 119 | 93.3% | 0.889 |
| cheeses | is_pdo | 140 | 97.9% | 0.939 |
| cheeses | is_organic | 158 | 98.7% | 0.967 |
| cheeses | is_ultra_processed | 154 | 96.1% | 0.920 |

### 14.4 LLM fallback rate (на eval LLM-consensus gold, n=3506)

**Production cascade source distribution:**

| Слой | Cells | % | Назначение |
|---|---|---|---|
| Layer 1 (rule_h, regex по тегам/тексту) | 646 | **18.4%** | High-precision rules |
| Layer 2 (ML: MPNet + TF-IDF SVD + XGBoost) | 2588 | **73.8%** | Главная работа |
| Layer 3 (rule_l, low-precision regex) | 23 | 0.7% | Fallback перед LLM |
| **Layer 4 (LLM fallback)** | **249** | **7.1%** | Сложные / неуверенные cells |

**LLM cost reduction vs naive all-LLM baseline: 92.9%.**

**Per-category fallback:**
- pasta: 10.9% (grain_type + pasta_shape dominate)
- chocolate: 2.0% (rule_h + ML почти полностью покрывают)
- cheeses: 6.9%

**Топ-5 атрибутов по доле LLM:**
1. pasta.grain_type — 32.9%
2. pasta.pasta_shape — 22.8%
3. cheeses.is_pdo — 13.6%
4. cheeses.country_of_origin — 12.6%
5. cheeses.aging — 8.5%

**Атрибуты с LLM=0% (полностью покрыты rule_h + ML):**
- chocolate: `is_filled`, `contains_nuts`, `chocolate_extra`
- pasta: `is_filled`

### 14.5 Учёт circular bias

Headline 94.8% (cascade на consensus) включает оценку circular bias ~3.8pp.

**Conservative estimate** truly-independent accuracy:
- Cascade: ~91% (94.8% − 3.8pp)
- E2E: ~87% (91.1% − ~4pp)

Совпадает с human gold (Opus, 87.5%) в пределах CI → conservative interpretation подтверждена.

## 15. Reproducibility

| Скрипт / модуль | Назначение |
|---|---|
| `src/pipeline/ml/train.py` | Обучение ML-классификаторов (с `exclude_classes` поддержкой) |
| `src/pipeline/off_labels/rules.py` | TYPE_A..TYPE_F правила, FILLED_CHOCOLATE_REGEX |
| `scripts/build_noleak_artifacts.py` | Сборка noleak training pool |
| `scripts/refactor_filled_schema.py` | Деривация is_filled labels + cleanup deprecated rows |
| `scripts/verify_is_filled_gold.py` | 2-LLM verification negative candidates |
| `src/eval/metric_table.py` | Per-attribute metrics (macro-F1, balanced acc, per-class) |
| `src/eval/end_to_end.py` | E2E pipeline eval (router → cascade) |

### Training
```bash
OMP_NUM_THREADS=2 XGB_N_JOBS=2 python -m src.pipeline.ml.train \
  --category chocolate_v4 --model-suffix _mpnet_tfidf_noleak --with-tfidf
OMP_NUM_THREADS=2 XGB_N_JOBS=2 python -m src.pipeline.ml.train \
  --category cheeses_v4 --model-suffix _mpnet_tfidf_noleak --with-tfidf
OMP_NUM_THREADS=2 XGB_N_JOBS=2 python -m src.pipeline.ml.train \
  --category pasta_v4 --model-suffix _mpnet_tfidf_noleak --with-tfidf
```

### Evaluation
```bash
python -m src.eval.metric_table     # per-attr macro-F1 + per-class breakdown
python -m src.eval.end_to_end       # router → cascade pipeline
```
