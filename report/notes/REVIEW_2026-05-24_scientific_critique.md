# Научная рецензия ВКР — «Гибридная каскадная система обогащения товарных атрибутов»

**Дата:** 2026-05-24
**Объект рецензии:** монолитный ноутбук `notebooks/00_thesis_main.ipynb` (deprecated; на 2026-05-25 разбит на `01_dataset.ipynb`, `03_evaluate.ipynb`, `04_demo.ipynb`) + `src/`
**Метод:** аудит кода и ноутбука без оценки оформления; претензии привязаны к файлам/строкам.

**Версия 2 (после мета-верификации `REVIEW_2026-05-24_meta_critique.md`):**
- Опровергнутая претензия (categories_tags в direct_llm) удалена.
- Подтверждённые цифрой претензии (threshold-on-test overlap) дополнены реальными числами.
- Добавлены пропущенные находки (blind-audit не blind на TYPE_A/B, pre-registration post-hoc, DAG instability, demo не запускает LLM).
- Top-3 перестроен с учётом верификации.

Рецензия основана на чтении ноутбука `notebooks/00_thesis_main.ipynb` (исторически — единый монолит; в текущем состоянии разбит на `01_dataset.ipynb`, `03_evaluate.ipynb`, `04_demo.ipynb`), ключевых модулей `src/` (label_silver, off_labels/{apply,rules}, pipeline/ml/train, pipeline/bayes/train, pipeline/llm_fallback/{prompts,enrich}, diagnostics/silver/{audit,compare,self_consistency,leakage_probe}, eval/{run_experiments, layer4_llm, direct_llm, manual_vs_silver, cascade_vs_llm_stats, cost_quality_ci, per_language_eval, blind_silver_audit}, data/split/{brand_disjoint,generate_gold_splits}, manual_label/off_field_filter, electronics/{prepare,cold_start_demo}, llm/client), артефактов `datasets/processed/blind_vs_prefill_*.parquet`, `consensus_gold_v2_expanded.parquet`, `dag_stability_*.parquet`, `demo/ml_service/cascade.py`, а также `docs/thesis/pre_registration_2026-Q2.md`.

---

## 1. Что сделано хорошо (с оговорками)

1. **Blind-audit Opus на 4088 cells — *частично* методологически чистый шумовой пол.** `datasets/processed/blind_vs_prefill_overall.parquet` показывает 95% согласие с silver, κ>0.85 на 17 из 22 пар. Реализация: `src/eval/blind_silver_audit.py`, фильтрация полей: `src/manual_label/off_field_filter.py`.
   **Оговорка (см. §2.10):** blacklist в `off_field_filter.py:17-27` отсекает только `nutriscore_grade`, `nova_group`, `ecoscore_grade`, `ingredients_analysis_tags`. **`labels_tags` и `categories_tags` НЕ blacklist'ed**, поэтому на TYPE_A (`is_organic`, `is_vegan`, `is_gluten_free`, `is_pdo`, `country_of_origin`, …) Opus читает ровно те поля, по которым silver и был выведен → это согласие двух читателей одних данных, не валидация. Похвала применима только к TYPE_C-nutrient и частично к text-derived атрибутам, где у Opus есть дополнительные signals.

2. **Leakage probe (`src/diagnostics/silver/leakage_probe.py`) явно ловит регрессии каскада относительно TF-IDF+LR на partner-полях.** Артефакт `off_leakage_probe.parquet` и scatter в §9 cell 26 честно подсвечивают `chocolate_extra`, где cascade 67.2% против probe 76.6%. Признать собственную регрессию — редкость.

3. **Pre-registration с Bonferroni существует и формально корректен** (`docs/thesis/pre_registration_2026-Q2.md`): H_E1...H_E8, α=0.0125 на 4 гипотезы Phase 1.
   **Оговорка (см. §2.11):** файл коммитнут в первом коммите проекта **2026-05-19** одновременно со всем кодом и parquet-результатами. Внутри стоит подпись «2026-05-15», но это markdown-строка, не cryptographic timestamp. Истории до 2026-05-19 на других ветках/тегах нет. Формально это post-hoc justification; декларация «зафиксировано ДО запуска» технически недоказуема.

4. **Brand-clustered bootstrap CI** в `src/eval/cost_quality_ci.py` ресемплит бренды (не ячейки), 1000 итераций; McNemar χ² с continuity correction + биномиальный exact для малых n в `src/eval/cascade_vs_llm_stats.py`.
   **Оговорка:** ROUTER_ACC при бутстрапе не ресемплируется (см. §3.3); McNemar применён per (cat, attr) без FDR на ~40 одновременных тестов (см. §3.1).

5. **`router_multiplicativity_check.parquet`** эмпирически проверяет `acc_oracle_cat × router_acc` vs реальную e2e — gap −0.14 п.п. Обычно просто умножают, тут проверили.

6. **LLM client детерминирован.** `src/llm/client.py:38` — `"temperature": 0.0`. Single-shot повторяемы; stochastic CI вокруг одного прогона не требуется.

---

## 2. Критические методологические проблемы

### 2.1 [`src/pipeline/ml/train.py:209-223, 380, 469`] — per-attribute thresholds выбираются на test-сплите silver — **подтверждено цифрой**

`find_best_threshold(final_clf, X_test, y_test_enc)` — где `X_test` это 20% random split (`RANDOM_STATE=42`, строка 763). Отдельного val-split для тюнинга порога нет. Порог сохраняется в `{prefix}_thresholds.pkl` и загружается во всех eval-скриптах (`src/eval/run_experiments.py:186-191 load_thresholds`).

**Измеренный overlap `silver_test ∩ brand_disjoint splits`:**

| cat | silver_test (n) | bd_test (n) | overlap test∩test | overlap test∩train |
|---|---|---|---|---|
| pasta | 3 139 | 250 | 42 (**16.8%** bd_test) | 141 (18.8% bd_train) |
| chocolate | 2 694 | 248 | 55 (**22.2%**) | 155 (20.9%) |
| cheeses | 4 242 | 250 | 36 (14.4%) | 134 (17.9%) |
| beverages | 248 | 248 | 49 (**19.8%**) | 157 (21.1%) |
| cereals | 198 | 198 | 37 (18.7%) | 125 (21.1%) |
| cosmetics | 223 | 223 | 52 (**23.3%**) | 128 (19.2%) |

В `bd_test` для всех категорий 14–23% codes уже видны были на этапе тюнинга порога. Headline 91.5% / 93.81% получены с порогами, выбранными на части тех же продуктов. Реальный сдвиг headline вероятно 0.5–2 п.п.

### 2.2 [`src/pipeline/off_labels/rules.py:758-1175` + `src/common.py:20 PARTNER_TEXT_FIELDS`] — структурная утечка на text-derived атрибутах

silver для 19 атрибутов (см. `attribute_signal_taxonomy.parquet`, `signal_type='text_derived'`) выводится регексами по `ingredients_text` + `product_name` (`grain_type`, `pasta_shape`, `is_filled`, `chocolate_type`, `cocoa_percentage`, `contains_nuts`, `chocolate_extra`, `milk_source`, `texture`, `beverage_type`, `is_carbonated`, `cereal_type`, `is_whole_grain`, `product_type`, `form_factor`, `body_area`, `has_sulfates`, `has_silicones`, …). Эти же поля — вход SBERT (`PARTNER_TEXT_FIELDS = [product_name, brands, ingredients_text, quantity]`). ML по факту восстанавливает выход regex из текста, по которому regex применён.

Blind-audit Opus 95% **не отделяет** «silver правильный» от «оба читателя смотрят в один текст» — на этих атрибутах Opus видит `ingredients_text` (он не blacklist'ed). В headline эти атрибуты усреднены с nutrient-derived (TYPE_C) — нужна разбивка с пометкой "regex-derived" и intentionally сниженной weight в narrative.

### 2.3 [`src/eval/cost_quality_ci.py:118-124`] — LLM accuracy на абстейн-ячейках аппроксимирована средней по атрибуту

`proxy = cov*acc_cov + (1-cov)*llm_a`, где `llm_a = llm_acc_on_attr` — точность LLM на полном (cat, attr) тесте из `cascade_plus_llm4_hybrid.parquet`, не на абстейн-ячейках. Абстейн-ячейки систематически «трудные» (поэтому каскад и абстейнится). На них LLM тоже хуже. Headline завышен; реальный замер уже сделан в `src/eval/layer4_llm.py:107-135` и сохранён в `llm_fallback_eval_{cat}.parquet`, но в финальный cost-quality scatter не интегрирован. Эффект — порядка +0.5...+1.5 п.п. систематического оптимизма.

### 2.4 ~~direct LLM baseline получает `categories_tags`~~ — **ОПРОВЕРГНУТО**

Изначально подозревалось, что direct LLM в `src/eval/direct_llm.py` получает на вход весь silver row, включая `categories_tags`/`labels_tags`. Проверка показала обратное:
- `src/eval/direct_llm.py:55` определяет ЛОКАЛЬНЫЙ `INPUT_FIELDS = ["product_name", "brands", "ingredients_text", "quantity"]` — без `categories_tags`.
- Line 104: `product = {f: row.get(f) for f in INPUT_FIELDS if f in row.index}` фильтрует ровно по этим 4 полям.
- `src/pipeline/llm_fallback/prompts.py:_render_product_block` через `if product.get(field)` пропускает отсутствующие поля.

**Вывод:** «+21.2 п.п. cascade vs direct LLM gpt-oss» — НЕ артефакт промпта. Заявление защитимо.

### 2.5 [`src/data/split/generate_gold_splits.py:23-24`] — brand normalization берёт только первый comma-separated brand

`silver["brand_norm"] = silver["brands"].fillna("UNKNOWN").astype(str).str.split(",").str[0].str.strip().str.lower()`. То есть «Carrefour BIO, Carrefour» → `carrefour bio`, а «Carrefour, Carrefour BIO» → `carrefour` — два разных brand-norm для одной сущности. При greedy bin-packing они могут оказаться в разных splits, давая subbrand leakage. Для `is_organic` (где regex ловит «BIO») это прямой leak.

### 2.6 [`src/eval/per_language_eval.py:36-39`] — язык детектится `langdetect` по короткому product_name

`langdetect.detect(product_name)` на «Penne Rigate» → IT/EN/случайно. Утверждение «точность по языкам равная» в §9 нужно подкрепить распределением n_cells и CI, иначе «равенство» — артефакт малого n на DE/IT/ES.

### 2.7 [`src/pipeline/llm_fallback/prompts.py:55-83`] — derivation_block прописывает в промпт точные пороги бакетирования

Для `nutri_score_grade`, `protein_class`, `fat_class`, `cocoa_percentage` LLM получает правила `<5g→low, 5-15g→med, >15g→high` — те же, что TYPE_C_RULES в `rules.py:668-705`. Сравнение «cascade vs LLM» на этих атрибутах вырождается в «apply rule in code vs apply rule in prompt». Их надо отделить в taxonomy.

**Дополнительно:** для `cocoa_percentage` silver-buckets (`70-85`, `50-70`) и LLM-output («70%», «dark chocolate 70») имеют разные labelspace — LLM возвращает число, silver хранит интервал → автоматический mismatch без bucketization downstream.

### 2.8 [`src/pipeline/bayes/train.py:251-283`] — DAG обучается HillClimb+BIC, **структура нестабильна (подтверждено артефактами)**

`learn_and_build` использует `HillClimbSearch(data).estimate(scoring_method="bic-d", max_indegree=3)`. Артефакты `datasets/processed/dag_stability_*.parquet` (генерирует `src/diagnostics/ml/dag_bootstrap.py`):
- **chocolate: 0 STABLE edges из 6 reference.**
- **electronics:** ключевой edge `brand→os`, обосновывающий cold-start демо в §5, имеет bootstrap freq **0.61**; edge `ram_class→brand` (странное направление!) — 0.55.

Структура графа во многом артефакт sample. Это противоречит cold-start нарративу «архитектурная ценность Bayes». В prod-конфигурации (`accuracy_squeeze_holdout.py`) DAG фиксирован, что маскирует instability от читателя ВКР.

### 2.9 [`src/eval/direct_llm.py:84-87`] — random split, не brand-disjoint

`train_test_split(np.arange(len(silver)), test_size=TEST_SIZE, random_state=RANDOM_STATE)`. `cascade_vs_llm_stats.parquet` строит McNemar на этом random split — где cascade имеет brand-overlap преимущество. Все p-values cascade vs direct LLM оптимистично завышены. Brand-disjoint headline 91.5% — на другом сплите. brand_disjoint-режима у direct_llm нет.

### 2.10 [`src/manual_label/off_field_filter.py:17-27`] — blind audit Opus не blind на TYPE_A/tag-derived атрибутах

Blacklist: `nutriscore_grade`, `nova_group`, `ecoscore_grade`, `ingredients_analysis_tags`. **`labels_tags` и `categories_tags` отсутствуют.** silver для tag-derived (13 атрибутов в `attribute_signal_taxonomy.parquet`: `is_organic`, `is_vegan`, `is_gluten_free`, `is_pdo`, `country_of_origin`, `is_ultra_processed`, …) выводится регексом по `labels_tags`/`categories_tags`; blind Opus читает те же колонки. Заявленное «κ=0.975 для chocolate/is_organic» — это согласие двух читателей одних тегов, не валидация silver.

**Эффект:** на 13 из 44 атрибутов blind-audit не валидирует silver. Реальное доказательство качества silver на этих атрибутах требует либо ручной разметки, либо blind audit с расширенным blacklist (с пересчётом LLM-вызовов).

### 2.11 [`docs/thesis/pre_registration_2026-Q2.md` + git history] — pre-registration post-hoc по git-таймстампам

Pre-registration коммитнута в первом коммите проекта **2026-05-19** одновременно со всем кодом и parquet-результатами. Внутри файла указано «2026-05-15», но это markdown-строка, не cryptographic timestamp. Истории до 2026-05-19 на других ветках/тегах нет.

**Эффект:** формально pre-registration **не пре-регистрация** в смысле journal/OSF. Это рискованно на защите: оппонент может потребовать timestamp-доказательство. Стоит либо переименовать в "Phase 2 analysis plan" (честно), либо явно проговорить в ВКР: "оформлена retrospectively на этапе оформления, фиксирует решения, принятые в процессе работы".

### 2.12 [`demo/ml_service/cascade.py:412-429`] — demo не запускает Layer 4 LLM

Для не-покрытых атрибутов demo проставляет `"layer": "llm_fallback"`, `"value": null` — настоящего LLM-вызова в production demo НЕТ. Headline 91.52% — это `cascade+gemini25flash` (ноутбук §3.3.2). Это два разных артефакта; в ВКР §4.1 «как пользоваться» нужно явно проговорить: "демо запускает Layers 1-3, для production-grade покрытия Layer 4 нужно интегрировать API"; иначе «работоспособное демо» путается с «accuracy системы».

---

## 3. Статистические проблемы

- **`cascade_vs_llm_stats.py:97-115`** — нет FDR/Holm-коррекции на ~40 одновременных сравнений (cat, attr). Pre-registration упоминает Bonferroni только для 4 гипотез Phase 1.
- **`find_best_threshold:209-223`** — критерий `f1 * coverage^0.3`. Степень 0.3 выбрана как? Чувствительность headline к этой константе не показана.
- **`cost_quality_ci.py:33 ROUTER_ACC`** — router_acc зафиксирован как константа (`{"pasta": 0.962880, "chocolate": 0.985377, "cheeses": 0.977477}`); при bootstrap'е брендов не ресемплируется → CI занижены, особенно на cheeses (n брендов меньше).
- **`pre_registration_2026-Q2.md`, H_E1** — decision rule по point-estimate (Δ<1.5, Δ≥3.0), а не по CI. На n=1539 разница 1.45 vs 1.55 — внутри шума.
- **`src/diagnostics/silver/self_consistency.py`** — определена только для pasta, n=100, Haiku 4.5. Шумовой пол LLM для остальных категорий неизвестен → «близко к потолку» недоказуемо.
- **Provider-variance не оценена.** `src/llm/client.py` использует OpenRouter; один model_id может попасть на разные провайдеры (vertex/groq/fireworks) с разными квантизациями. Headline 91.5% получен в одном прогоне.

---

## 4. Воспроизводимость и инженерная чистота

- Единый seed=42 — хорошо. `cv_stability_groupkfold.py` существует, но multi-seed std для headline в ноутбуке не найден.
- **`src/experiments/` — 60+ файлов** с `_v2/_v3/_b3/_promptfix/_r2`. Какие использованы для headline, какие — отброшенные ветки? README нет — затруднит защиту.
- **`src/pipeline/ml/train.py:614-622 recompute_calibration_only`** — recomputed ECE пишется и в `ece_raw`, и в `ece_calibrated` (`"ece_raw": ece, "ece_calibrated": ece`). Plot «до vs после» в §6 cell 15 для recomputed строк может вводить в заблуждение.
- **`demo/`** — см. §2.12; заявленная «работоспособность демо» относится к Layers 1-3, не к headline 91.5%.

---

## 5. Новизна и позиционирование

- Каскад regex→ML→LLM — стандарт e-commerce каталогизации (MAVE/TXtract/OpenTag, 2018+). **Реально нового в работе**: static per-attribute policy как замена обучаемому роутеру (negative result H1), pgmpy-Bayes для inter-attribute связей, применение к мультиязычному food-домену.
- В §6.1 (+1.03 п.п. при +5% LLM) сравнение идёт **с собой без validator'а**, не с воспроизведённым TXtract/MAVE. H_E8 в pre-registration о multi-task encoder заявлена, но реализация в `src/experiments/` не найдена.
- **Electronics §5 cold-start** через `P(os|brand=Apple)≈1` для iOS — тривиально (Apple→iOS by definition). Это conditional inference, не transfer learning между категориями. Дополнительно (§2.8): edge `brand→os` бутстрап-стабилен только на 61% — нарратив "Bayes извлекает связь" хрупкий.
- **+1.03 п.п. при n≈4350**: 95% Wilson CI разницы пропорций ≈ ±1.2 п.п. Парный McNemar может быть значим, но `cascade_vs_llm_stats.parquet` показывает cascade vs direct LLM, не «+Bayes vs −Bayes». Нужен explicit парный тест.

---

## 6. Что НЕ проверено

1. `src/eval/router_pre_registered.py`, `router_loco_gold.py`, `router_pareto_gold.py` — критично для негативного результата H1.
2. `src/diagnostics/ml/cv_stability_groupkfold.py` — multi-seed std headline не верифицирован.
3. `src/eval/run_diagnostics.py` — агрегатор.
4. `src/experiments/e1_circularity_analysis.py`, `e2_holdout_split.py`, `e5_cross_domain_summary.py`, `e6_per_pair_bootstrap.py`.
5. `src/pipeline/category_router/train_v3.py` — фичи и OOD-калибровка router v3.
6. `docs/thesis/03_chapter3_implementation.md`, `04_chapter4_results.md` — текст ВКР.
7. `demo/ml_service/validator.py` — если переобучается на test products, утечка в demo.
8. `bootstrap_ci_brand_clustered.py` — детали стратификации не проверены.
9. `src/eval/validator_hypothesis_tests.py` — p-value precision для Bayes-validator.
10. `accuracy_squeeze_holdout.py` — claim, что в prod DAG фиксирован, не верифицирован руками.
11. Реальный sensitivity headline к threshold-on-test — overlap посчитан, но retraining с `threshold=0.5` не запускался.

---

## 7. Top-3 что обязательно фиксить до защиты

### №1 — Structural circularity TYPE_E + не-blind Opus audit

Двойная проблема (§2.2 + §2.10). На 19 text-derived атрибутах silver = regex по `ingredients_text`/`product_name`; ML обучен на эмбеддингах тех же полей; blind Opus читает те же поля. На 13 tag-derived атрибутах blind Opus читает `labels_tags`/`categories_tags`, не отфильтрованные в `off_field_filter.py:17-27` — тот же источник, что для silver. Получается **три ридера одного источника, согласие между ними ≠ валидация**.

**Минимальный фикс без LLM-затрат:** разбить headline по `signal_type` (taxonomy уже в `attribute_signal_taxonomy.parquet`):
- `nutri_derived` (12 attrs): высокий confidence, наименьший circularity risk.
- `tag_derived` (13 attrs): moderate, явно проговорить, что blind-audit на них не валидирует silver.
- `text_derived` (19 attrs): низкий confidence по claim'у «accuracy», явно пометить как regex-rediscovery, не attribute extraction.

Headline по группам читается честнее, чем общая 91.5%.

**Полный фикс (требует пересчёта LLM):** добавить `labels_tags`/`categories_tags` в blacklist `off_field_filter.py`, прогнать blind audit заново на 4088 cells — дорогостояще ($ LLM-calls).

### №2 — Threshold-on-test sensitivity

§2.1 подтверждён цифрой: bd_test содержит 14–23% codes из silver_test, на котором калиброван порог. **Минимальный фикс:** sensitivity analysis — headline с `threshold=0.5` (без тюнинга) vs текущий. Если разница <0.5 п.п. — претензию снять. Если ≥2 п.п. — перекалибровать на отдельном val-сплите. Реализовать как `src/eval/threshold_sensitivity.py` поверх существующих моделей; не требует переобучения.

### №3 — DAG instability в narrative §6 / §5 cold-start

§2.8 подтверждён артефактами (`dag_stability_chocolate.parquet`: 0/6 STABLE; `dag_stability_electronics.parquet`: `brand→os` 61%). **Минимальный фикс — текстовый**: в §6 явно проговорить, что edges нестабильны при bootstrap, и Bayes используется как regularizer/calibrator, а не как «открытие структуры». Cold-start демо в §5 переформулировать с «структурное знание Bayes» на «conditional inference on hand-crafted edges». Артефакты уже в проекте — нужно их вывести в ВКР, а не скрывать.

---

## Итог

Работа методологически выше среднего магистерского уровня: pre-registration с Bonferroni (с оговоркой о post-hoc, §2.11), blind-audit Opus (с оговоркой о не-blind на tag-derived, §2.10), leakage probe, brand-clustered bootstrap, McNemar tests, ECE-калибровка, knn-distance ablation. Большая часть стандартных claim'ов в ноутбуке цифрами подкреплена.

**Главные методологические уязвимости** (после мета-верификации): structural circularity на 19+13 атрибутах ⇒ нужна разбивка по taxonomy; threshold-on-test snooping с подтверждённым overlap 14–23%; DAG instability, противоречащая cold-start нарративу.

**Все три фиксаются без новых LLM-затрат** — таксономическая разбивка и текстовые disclaimers в ВКР + один sensitivity-скрипт. После фикса headline 91.5% разойдётся на nutrient-derived (defensible высокая), tag-derived (с caveat), text-derived (intentionally сниженная weight в narrative). Это сделает работу значительно более защищаемой, чем монолитная цифра 91.5%, которая на защите будет оспорена в первые 10 минут.

**Перед защитой обязательно** нужно либо устранить эти три проблемы, либо иметь готовый ответ на каждую.

---

## Критические файлы для рецензии/доработки

- `src/pipeline/ml/train.py` — `find_best_threshold` на test
- `src/eval/cost_quality_ci.py` — `llm_acc` proxy + `ROUTER_ACC` константа
- `src/pipeline/off_labels/rules.py` — TYPE_E/TYPE_F regex (структурная circularity)
- `src/manual_label/off_field_filter.py` — `labels_tags`/`categories_tags` не blacklist'ed для blind audit
- `src/data/split/generate_gold_splits.py` — brand_norm subbrand leak
- `datasets/processed/dag_stability_chocolate.parquet` — 0/6 STABLE edges
- `datasets/processed/attribute_signal_taxonomy.parquet` — готовая разбивка для §1 фикса
- `docs/thesis/pre_registration_2026-Q2.md` — git timestamp = первый коммит проекта
- `demo/ml_service/cascade.py:412-429` — null вместо реального LLM-вызова
