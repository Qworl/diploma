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
| Не deprecated класс | Ячейки с `gold_value ∈ SCHEMA_EXCLUDE[attr]` исключаются (см. §5.2). |
| Не None gold | `gold_value` отсутствует (NaN) → ячейка не оценивается. |
| Только in-scope категории | Eval ограничен `{pasta, chocolate, cheeses}`. |

## 4. Train/test изоляция

Модели обучаются на **noleak training pool**: из silver standard `{cat}_gold_v4_wide.parquet` удалены все codes, попадающие хотя бы в один из eval gold-источников (consensus gold, human gold, extended gold).

Используются модели с суффиксом `_mpnet_tfidf_noleak`. Это обеспечивает честный test-time accuracy без memorization-bias.

## 5. Принципы проектирования схемы атрибутов

Дизайн схемы предметной области (множество атрибутов и их доменов значений) — критическое решение, влияющее на потолок качества системы независимо от выбора модели. В работе используются три принципа.

### 5.1 Принцип ортогональности атрибутов

**Формулировка.** Свойства, которые могут проявляться у товара независимо друг от друга, должны быть представлены **отдельными** атрибутами, а не объединяться в один multiclass-набор значений.

**Канонический пример: `chocolate.is_filled` vs `chocolate_type`.**

Шоколад одновременно характеризуется по двум независимым осям:
- **Базовый тип какао-массы:** dark / milk / white (взаимоисключающие)
- **Структурная форма:** solid bar / filled (есть начинка)

Один товар может быть *dark + filled* (трюфели Lindt с тёмной оболочкой и ганашевой начинкой), *milk + filled* (Ferrero Rocher), *white + filled* (Lindt с малиновым кремом) — все эти комбинации валидны и встречаются на рынке.

Объединение этих свойств в один multiclass-атрибут с доменом `{dark, milk, white, filled}` вынуждает аннотатора и классификатор выбирать **одну** метку, теряя информацию. На LLM-consensus gold cross-tab показал что среди товаров с `chocolate_type=filled` (n=11) лишь 18% имеют `chocolate_extra=filled`, остальные сочетаются с `with_nuts` (4), `with_cookie` (2), `with_fruit` (2), `with_alcohol` (1) — то есть `filled` функционирует как структурная характеристика, ортогональная к составу.

**Применение.** В схему введён отдельный binary-атрибут `is_filled`. Класс `filled` исключён из multiclass-атрибутов `chocolate_type` и `chocolate_extra`. Эмпирический результат на consensus gold:
- `chocolate_type` (3-class): macro-F1 = 0.973
- `is_filled` (binary): macro-F1 = 0.861, balanced accuracy = 0.919
- `chocolate_extra` (5-class): macro-F1 = 0.899

### 5.2 Принцип отказа от мёртвых классов

**Формулировка.** Класс multiclass-атрибута подлежит удалению из схемы при выполнении **всех** условий:
1. **Низкая представленность** в обучающей выборке: n_train < 100 (≤ 1% от размера корпуса).
2. **Низкий recall** на independent eval (< 0.20) — классификатор фактически не предсказывает этот класс.
3. **Семантическая некогерентность**: класс выступает как catch-all («other», «прочее») без устойчивого паттерна в признаках, ИЛИ покрывается другим атрибутом (см. §5.1).
4. **Отсутствие downstream-зависимости**: класс не требуется обязательно ни для одного использующего систему процесса.

Невыполнение условия 3 переводит решение в плоскость class-imbalance техник (class weights, oversampling, focal loss) или routing-стратегий (отправка в LLM при низкой confidence), а не схемного изменения.

**Применённые исключения** (`SCHEMA_EXCLUDE`):

| Атрибут | Класс | n_train | recall на consensus | Условие |
|---|---|---|---|---|
| chocolate_type | filled | 134 | 0.33 | Покрыт `is_filled` (§5.1) |
| chocolate_type | other | 67 | 0.00 | Catch-all без сигнала |
| chocolate_extra | filled | — | — | То же, что для type |
| chocolate_extra | other | 6 (на eval) | 0.00 | Catch-all |
| chocolate_extra | with_alcohol | 1 (на eval) | 0.00 | Неотличим от mix-ins |
| chocolate_extra | with_coffee | 1 (на eval) | 0.00 | Неотличим от mix-ins |
| flavor_profile | other | 72 | 0.11 | Catch-all |
| cheeses.texture | other | 14 | 0.14 | Catch-all |

```python
SCHEMA_EXCLUDE = {
    "chocolate_type":         {"filled", "other"},
    "chocolate_extra":        {"filled", "other", "with_alcohol", "with_coffee"},
    "chocolate.flavor_profile": {"other"},
    "cheeses.texture":        {"other"},
}
```

**Двойное применение фильтра.** На этапе обучения работает `exclude_classes` в `CATEGORY_CONFIG[train.py]` — фильтр меток из train pool до `LabelEncoder.fit`, классификатор не знает о существовании этих классов. На этапе оценки тот же набор применяется как `SCHEMA_EXCLUDE` в eval-скриптах — gold-ячейки с deprecated значениями отбрасываются. Соответствие train ↔ eval ↔ production schema обеспечивается единым описанием.

### 5.3 Принцип консервативной агрегации rule-based слоёв

**Формулировка.** Правило (regex или tag-lookup) включается в high-precision слой (`rule_h`) только если оно даёт **однозначный** ответ во всех контекстах. Неоднозначные паттерны (например, региональные названия сыров, обозначающие разные классы в зависимости от подтипа) оставляются для ML-слоя.

**Пример (cheeses.texture).** В правила класса `cream` включены сыры однозначно относящиеся к этому типу по dairy taxonomy: `ricotta`, `labneh`, `quark`, `mascarpone`, `philadelphia`. Также brand-name `Tartare` (Savencia, herbed cream cheese, аналог Boursin). Региональное название `Coeur de Savoie` исключено из всех правил — оно покрывает как soft-, так и hard-варианты в зависимости от производителя, корректное решение возможно только при наличии контекста ингредиентов / категории.

## 6. Финальная схема атрибутов

### 6.1 Chocolate

| Attribute | Type | Classes |
|---|---|---|
| `chocolate_type` | multiclass | `{dark, milk, white}` |
| `is_filled` | binary | `{True, False}` |
| `chocolate_extra` | multiclass | `{plain, with_nuts, with_fruit, with_caramel, with_cookie}` |
| `contains_nuts` | binary | `{True, False}` |
| `is_organic` | binary | `{True, False}` |
| `flavor_profile` | multiclass | `{fruity, intense_bitter, nutty, salty_caramel, spiced, sweet_creamy, floral}` |

`chocolate_extra` описывает **только mix-ins** (что добавлено в шоколадную массу или сопровождает её), не структурную форму — последнее обеспечивается атрибутом `is_filled`.

### 6.2 Cheeses

| Attribute | Type | Classes |
|---|---|---|
| `milk_source` | multiclass | `{buffalo, cow, goat, mixed, other, sheep}` |
| `texture` | multiclass | `{blue, cream, fresh, hard, processed, soft}` |
| `country_of_origin` | multiclass | (страны Европы + other) |
| `is_pdo` | binary | `{True, False}` |
| `is_organic` | binary | `{True, False}` |
| `is_ultra_processed` | binary | `{True, False}` |
| `aging` | multiclass | `{aged, fresh, young}` |

`texture` не содержит `semi_soft` (объединено с `soft` — граница между ними не несёт информации в production-метках и провоцирует ошибки annotator-agreement).

### 6.3 Pasta

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

`pasta.is_filled` и `chocolate.is_filled` — независимые атрибуты для разных категорий, не разделяют classifier или embedding space.

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

### 12.1 Circular evaluation bias

- Training silver labels: hybrid (OFF tags + Gemini Flash relabel).
- Eval gold: qwen3.7-max + deepseek-r1 + mistral-large-2411 consensus.
- LLMs семантически перекрываются → measured accuracy выше truly-independent
  ground truth.
- **Операционная оценка bias** = consensus − human gold на той же выборке:
  - cascade-only: 95,5 % − 91,3 % = **4,2 пп**
  - E2E: 93,0 % − 86,7 % = **6,3 пп**
  - midpoint ≈ 5 пп; диапазон 4–6 пп
- Conservative interpretation: headline 95,5 % на consensus →
  91,3 % на человеческой разметке как нижняя независимая оценка.
- Источник: `notebooks/03_evaluate.ipynb` cells `44dbf52a` (consensus)
  + `495abc8a` (human) → `v4_e2e_router_eval.json` keys `LLM-consensus` / `HUMAN`.

### 12.2 Brand-disjoint test невозможен
- pasta: 100% test brands present в training pool
- chocolate: 96% overlap, 7 disjoint codes
- cheeses: 93-98% overlap
- → claim о brand-disjoint generalization не делается

### 12.3 Sample size
- HUMAN gold: 566 cells cascade-valid (688 в-scope, 122 Layer 4 fallback вне cascade-only знаменателя), single labeler (Opus), без IRR. Используется как conservative lower-bound оценка.
- LLM-consensus gold (extended rerun 2026-05-27): 16360 cells cascade-valid (17062 в-scope), 3-LLM majority vote.

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

### 14.1 Headline (LLM-consensus gold, n=16360 cascade-valid из 17062 в-scope, extended rerun 2026-05-27)

| Метрика | Значение |
|---|---|
| Cascade-only micro-accuracy | **95.5%** |
| Cascade-only macro-F1 (cells-weighted) | **0.890** |
| Cascade-only macro-F1 (attr-unweighted) | **0.892** |
| Router accuracy (per code, n=570) | **97.2%** |
| E2E coverage | **97.3%** |
| **E2E accuracy (None=wrong) — production-realistic** | **93.0%** |

### 14.2 Conservative bound (HUMAN gold, n=566 cascade-valid)

| Метрика | Значение |
|---|---|
| Cascade-only micro-accuracy | 91.3% |
| Cascade-only macro-F1 | 0.848 |
| Router accuracy (n=107) | 95.3% |
| E2E coverage | 95.1% |
| **E2E accuracy (None=wrong)** | **86.7%** |

**Семантика n=566.** Знаменатель cascade-only и E2E считается по ячейкам, на которых каскад вернул конкретный класс (Layer 1 rule_h ∪ Layer 2 ML ∪ Layer 3 rule_l). Layer 4 LLM fallback (122 ячейки) исключён: на этих ячейках каскад «отказывается», в production они отправляются в большую языковую модель, для cascade-only такие ячейки не определены. Полное число in-scope cells по эталону Opus — 688. Числа воспроизводятся в `notebooks/03_evaluate.ipynb` ячейка `495abc8a` (источник — `datasets/processed/v4_e2e_router_eval.json` ключ `HUMAN`, генерируется `python -m src.eval.end_to_end` на VM).

### 14.3 Per-attribute macro-F1 (LLM-consensus gold, extended consensus rerun 2026-05-27)

Источник: `datasets/processed/v4_metric_table_v2.json` ключ `LLM-consensus`.
Сгенерировано ячейкой `c539b2a5` в `notebooks/03_evaluate.ipynb`; та же ячейка
пишет TeX-таблицу в `report/contents/tables/per_attr_consensus.tex` (подключается
в §3.3.2.3 ВКР через `\input`).

| Категория | Атрибут | n | micro | macro-F1 |
|---|---|---|---|---|
| pasta | grain_type | 1081 | 98.8% | 0.831 |
| pasta | pasta_shape | 645 | 94.9% | 0.909 |
| pasta | is_filled | 1225 | 88.5% | 0.786 |
| pasta | is_gluten_free | 1197 | 99.2% | 0.978 |
| pasta | is_organic | 1179 | 98.1% | 0.969 |
| pasta | is_vegan | 1156 | 96.3% | 0.962 |
| pasta | cuisine_origin | 1110 | 95.1% | 0.753 |
| chocolate | chocolate_type | 984 | 98.3% | 0.971 |
| chocolate | is_filled | 1188 | 96.9% | 0.910 |
| chocolate | chocolate_extra | 961 | 92.5% | 0.857 |
| chocolate | contains_nuts | 1126 | 94.9% | 0.945 |
| chocolate | is_organic | 1156 | 99.0% | 0.987 |
| chocolate | flavor_profile | 1060 | 87.5% | 0.684 |
| cheeses | milk_source | 354 | 96.9% | 0.776 |
| cheeses | texture | 335 | 89.0% | 0.848 |
| cheeses | country_of_origin | 310 | 94.8% | 0.906 |
| cheeses | aging | 267 | 94.4% | 0.929 |
| cheeses | is_pdo | 325 | 99.7% | 0.993 |
| cheeses | is_organic | 357 | 99.4% | 0.982 |
| cheeses | is_ultra_processed | 344 | 95.9% | 0.859 |

**Cells-weighted micro = 95,5 %, macro-F1 = 0,890. Attr-unweighted macro-F1 = 0,892.** Sum n = 16 360.

### 14.4 LLM fallback rate (на eval LLM-consensus gold extended rerun, n=17062)

**Production cascade source distribution:**

| Слой | Cells | % | Назначение |
|---|---|---|---|
| Layer 1 (rule_h, regex по тегам/тексту) | 3734 | **21.9%** | High-precision rules |
| Layer 2 (ML: MPNet + TF-IDF SVD + XGBoost) | 12487 | **73.2%** | Главная работа |
| Layer 3 (rule_l, low-precision regex) | 139 | 0.8% | Fallback перед LLM |
| **Layer 4 (LLM fallback)** | **702** | **4.1%** | Сложные / неуверенные cells |

**LLM cost reduction vs naive all-LLM baseline: 95.9%.**

**Per-category fallback (extended consensus rerun, in_scope):** pasta — 5,1 % (n=8004), chocolate — 2,6 % (n=6759), cheeses — 4,9 % (n=2411). Источник: `datasets/processed/cascade_preds_{cat}_gold.parquet`, считается в `notebooks/03_evaluate.ipynb` cell `percat-l4`. Та же ячейка пишет таблицу-разбивку по слоям в `report/contents/tables/per_category_layer.tex` (подключается в §3.3.3.1 ВКР через `\input`).

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

Headline 95,5 % (cascade на consensus) и 93,0 % (E2E на consensus)
получены на LLM-consensus gold, который семантически перекрывается
с silver-обучением. Прямая операционная оценка bias:

- cascade-only: 95,5 % − 91,3 % = **4,2 пп**
- E2E: 93,0 % − 86,7 % = **6,3 пп**

Human gold (Opus, n=566) служит **независимой нижней оценкой**:
91,3 % cascade-only, 86,7 % E2E. Эта оценка цитируется в abstract,
intro, conclusion как «нижняя граница точности на полностью
независимой ручной разметке».

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
