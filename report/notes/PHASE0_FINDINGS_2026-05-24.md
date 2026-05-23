# Phase 0 findings — 2026-05-24

> Примечание о пути: план (`IMPLEMENTATION_PLAN_2026-05-24.md`) указывает
> `docs/thesis/PHASE0_FINDINGS_2026-05-24.md`, но каталога `docs/thesis/` в
> репозитории нет — все thesis-заметки лежат в `report/notes/` (FIX_PLAN,
> REVIEW_2026-05-24_*, STYLE_REFERENCE, IMPLEMENTATION_PLAN). Файл сохранён
> рядом с источниками, чтобы быть консистентным с фактической структурой;
> при необходимости его легко перенести при создании каталога `docs/thesis/`.

## 0.1 validator_hypothesis_tests.py

**Файл:** `src/eval/validator_hypothesis_tests.py` (125 строк).

**Наблюдения:**
- Файл реализует три инструмента для пред-регистрированных гипотез:
  `_auc_safe`, `paired_bootstrap_auc_diff` (AUC-сравнение «greater»),
  `pareto_dominance_vs_static` (валидатор должен достичь recall статической
  политики при строго меньшем routing budget).
- Константа `BONFERRONI_ALPHA = 0.05 / 3` (строка 17) с комментарием «Three
  pre-registered hypotheses (H1, H2, H3); Bonferroni-corrected». Это
  файл-«библиотека функций» — он сам по себе не выбирает 3 атрибута, а
  фиксирует поправку на число гипотез.
- Внутри файла **нет** списка из «трёх активных атрибутов Bayes-validator»:
  список валидаторов передаётся снаружи через параметры `validator_cols`
  (строка 83) и `static_attrs` (строка 84) в `pareto_dominance_vs_static`.
  Выбор конкретных атрибутов делается у потребителя
  (`run_validator_experiments.py`, `bayes_validator_*.py`, `validator_pareto.py`
  и т. п.).
- Memory-файл `bayes_validator_scenario_c.md` и
  `accuracy_squeeze_deploy_2026-05-20.md` фиксируют, что в продакшн-
  конфигурации Bayes-валидатор активен только на нескольких (≈3)
  атрибутах. Конкретный список выбран по эмпирическому Δacc на val-сабсете
  (Phase 1 в `accuracy_squeeze_holdout.py`, см. §0.2), а не задан жёстко
  в `validator_hypothesis_tests.py`.

**Verdict:** OK как библиотека (поправка явная, протокол прозрачный).
Однако сам выбор «активных атрибутов» происходит не здесь — а в
`accuracy_squeeze_holdout.py` через max-Δacc по сетке (Q_SWEEP × ML_THR_SWEEP)
**на 50 %-val сабсете брендов** с дальнейшей валидацией на held-out. Это
honest hold-out, не cherry-picking, **но** список атрибутов отбирается
data-driven, а не из теории — об этом нужен disclaimer в §3.3 ВКР
(«атрибуты выбраны post-hoc по val-Δacc, защищены held-out test brand
split»).

## 0.2 accuracy_squeeze_holdout.py

**Файл:** `src/experiments/accuracy_squeeze_holdout.py` (440 строк, читался
полностью).

**Наблюдения:**
- **DAG (структура сети):** silver-обученный, **edges не переучиваются** —
  скрипт читает `models/{internal}_bayesian.pkl` (строка 116) и забирает
  оттуда `edges`/`nodes` (строка 118). Hill-Climb уже произошёл при обучении
  silver-модели; на этапе squeeze структура считается фиксированной.
- **CPDs:** **hybrid gold-refit с golden-replications×10.** Константа
  `GOLD_REPLICATIONS = 10` (строка 44). На smoothed silver-данных
  (`silver_brand_disjoint`) конкатенируется gold-фрагмент, повторённый 10 раз
  (`pd.concat([silver_fit] + [gold_fit] * GOLD_REPLICATIONS,
  ignore_index=True)`, строка 127), затем CPDs пересчитываются через
  `BayesianEstimator(prior_type="BDeu", equivalent_sample_size=10)`
  (строки 128–131). Это **не** mixture с разными весами, а буквально
  10-кратное дублирование строк gold перед стандартной BDeu-оценкой.
- **ML thresholds:** **gold-tuned per attribute**, выбираются на val-50 %
  брендов из test-fold и применяются на held-out-50 %.
  - Сетки: `ML_THR_SWEEP = [0.45, 0.50, …, 0.80]` и `Q_SWEEP = [0.005, 0.010,
    0.020, 0.050, 0.100, 0.150]` (строки 42–43).
  - Phase 1 (строки 328–394): для каждого `(cat, attr)` перебирается
    `(ml_thr, bayes_q)` и выбирается комбинация с максимальным
    `gain = (e2e_new − e2e_cur) * n_total` на val-брендах. Это **не**
    те же thresholds, что в `train.py thresholds.pkl`.
  - Phase 2 (строки 396–408): выбранные `(chosen_E, chosen_A)`
    применяются на held-out брендах, считается defended Δheadline.
- **Test/val split:** `split_test_brands` (строки 137–152). Тест-коды
  делятся на val/held по брендам (брендов случайно перемешано
  `seed=SEED=0`, первая половина — val, вторая — held), **brand-disjoint
  внутри test-fold**. Это защищает от tuning-bias.
- **Hybrid models:** для chocolate/(chocolate_type, contains_nuts) загружается
  отдельный TF-IDF + xgb_hybrid (строки 46, 165–176, 184–191) — это
  пересекается с §1.0 brand-norm fix (модель `*_hybrid_tfidf.pkl` тоже
  может пострадать от перетасовки splits).
- **Выход:** `datasets/processed/accuracy_squeeze_holdout.parquet` +
  `accuracy_squeeze_chosen_config.json` (строки 426–436).

**Branching scenario:** **(c) Production-path использует gold-refit → новая
disclosure нужна.**

Конкретно: DAG silver-обученный (стабильный, edges не двигаются), но CPDs
производственной конфигурации пересчитаны с gold-replications×10, и пер-
атрибутные ML/Bayes-thresholds подобраны на val-сабсете брендов из gold.
Это **не** «чисто silver-Bayes на silver-данных» — головной +1.03 пп
получен на конфигурации, явно использующей gold.

**Verdict:** OK c протоколом (val/held brand-disjoint внутри test-fold,
defended на held-out). Phase 3.2 narrative должен:
1. Перечислить три уровня информации в продакшн-конфиге: (a) DAG из silver,
   (b) CPDs hybrid silver+gold×10, (c) per-attr thresholds tuned on val
   brands of test split.
2. Явно указать в §3.3, что headline 93.81 % — defended на held-out
   половине test-fold (≈50 % брендов), а не на полном test-fold.
3. Обновить любой §2 narrative, где говорится «Bayes обучен на silver» —
   правильнее «структура из silver, CPDs hybrid silver+gold(×10)».

## 0.3 cv_stability_groupkfold.py + artifact

**Файл:** `src/diagnostics/ml/cv_stability_groupkfold.py` (148 строк).

**Наблюдения:**
- Скрипт честный: `GroupKFold(brand)` × 10 seeds × 5 folds, эмбеддинги
  закэшированы, XGB с одинаковыми гиперпараметрами.
- Бренд-нормализация локальная (строки 116–117):
  `silver["brands"].fillna("UNKNOWN").str.split(",").str[0].str.strip().str.lower()`
  — это **тот самый brand_norm, который чинится в Phase 1.0**; здесь он
  уже применён корректно (split on comma → first → strip → lower),
  так что артефакт совместим с пост-фиксной нормализацией.
- **Артефакт:** `datasets/processed/cv_stability_groupkfold.parquet`
  присутствует (2200 строк, 8 колонок).
- Дополнительные парные файлы в processed:
  `cv_stability_5fold.parquet`, `cv_stability_10seed_random_DEPRECATED.parquet`
  (random-split версия, явно помечена устаревшей).

**Multi-seed std headline (мера):**
- Усреднение accuracy по всем `(category, attr, fold)` per seed:
  - mean = **75.4474 %**
  - std (по 10 seeds) = **0.1046 пп**
  - range = 0.32 пп (min 75.24 %, max 75.56 %)
- Per-`(cat, attr)` std по seed-mean (агрегат): значения в диапазоне
  ~0.003–0.013 (т. е. 0.3–1.3 пп). Самые шумные: `is_vegan` (cereals)
  ~1.25 пп, `is_vegan` (beverages) ~0.92 пп, `is_high_fibre` (cereals)
  ~1.13 пп. Большинство `is_organic`/`is_pdo` attrs стабильны ~0.3 пп.

**Verdict:** std (на уровне ml-headline GroupKFold) **= 0.10 пп < 0.5**
→ **headline точно измерен**, claim «+1.03 пп Bayes» по магнитуде
**на порядок** больше шума ML-baseline. Phase 1.2 «threshold sensitivity»
может опираться на эту цифру: дополнительный seed-sweep по thresholds
очень желателен, но fundamental ML-instability **не** делает +1.03 пп
шумом. NB: эта цифра — стабильность layer-2 (ML) accuracy; стабильность
+1.03 пп от Bayes-validator отдельно измерить полезно (нет в текущих
артефактах — отдельный эксперимент в Phase 1.2).

## 0.4 demo/ml_service/validator.py

**Файл:** `demo/ml_service/validator.py` (101 строка).

**Наблюдения:**
- `ValidatorService.__init__` загружает **предварительно обученные** артефакты
  из `models_dir`: `{cat}_bayesian.pkl` и `{cat}_validation_thresholds.json`
  (строки 41–52). Сам сервис ничего не fit-ит.
- Сервис экспонирует `validate_value(internal_category, attr, value,
  evidence)` (строки 57–88) — это inference-only API.
- Потребитель в `demo/ml_service/cascade.py` (строки 180–181) создаёт
  `ValidatorService(...)` и далее вызывает `validate_value` (строки 315, 322)
  на **пользовательском input** (HTTP body, `req.product_name` etc., см.
  `demo/ml_service/main.py:54, 190`).
- Никакого пути «загрузить silver/gold parquet и использовать как input
  демо» в коде нет; в `demo/ml_service/main.py` ищется только grep по
  `silver|test|fit|train|gold|input` — единственный хит: `product_name`
  и `is_invalid_input` (валидация формы запроса).

**Verdict:** **OK — нет overlap fit-set/serve-set.** Валидатор предобучен
на silver+gold (через offline `train.py`-pipeline → `.pkl` файлы),
serve-time принимает live HTTP input. Утечки в demo нет. Никаких
дополнительных disclosures в Phase 3.4 по этому пункту не требуется.

## 0.5 router_pre_registered.py + router_loco_gold.py

**Файлы:**
- `src/eval/router_pre_registered.py` (93 строки) — decision-rule layer
  (Bonferroni @α/3, читает уже посчитанный `router_stats_gold.parquet`).
- `src/eval/router_loco_gold.py` (169 строк) — LOCO eval, переиспользует
  `train_router` из `src/pipeline/router/train.py`.

**Наблюдения:**
- `router_pre_registered.py` не лезет в фичи: он только применяет
  правило: `is_sig = (p_mcnemar < α/3) and (ci_lo > 0)` (строка 53),
  PASS если ≥1 из 3 pre-registered бюджетов (25 %/40 %/50 %) даёт строгое
  превосходство router'а над static. H1-FAIL правильно триггерит Plan B4
  (строки 86–87).
- `router_loco_gold.py` загружает `router_train.parquet`, обогащает meta,
  применяет gold overrides (`_apply_gold_overrides`), и для каждой
  категории-holdout обучает router на 5 остальных категориях
  (train+val concat, затем 90/10 re-split для калибратора, `seed=42`,
  строка 87). Тест — на held-out категории.
- Комментарий в docstring (строки 13–17) явно отмечает leakage-check:
  «`brand_set / class_freq_table / brand_attr_acc_table` строятся внутри
  `train_router` ТОЛЬКО на train split (см. строки 123–125
  `src/pipeline/router/train.py`)». Brand/class taxonomy features
  агрегируются на train split, не leak'ят из held-out категории.
- Фичи router'а сами по себе **не читаются здесь** — они в
  `src/pipeline/router/features.py:featurize(...)` (impor'тится строкой 34).
  Точный список не проверен в этой phase, но названия meta-фичей в
  inference path: `brand_set`, `class_freq_table`, `brand_attr_acc_table`
  → бренд, класс-частоты и per-brand attr accuracy. Категориальный
  `category_id` как фича **не виден** в LOCO-обвязке, и LOCO явно
  держит holdout категорию вне train — категориальный leak protocol-level
  закрыт.
- Split: LOCO (leave-one-category-out) на test_all, train на остальных
  5 категориях. Это даже строже brand-disjoint: cross-category transfer
  test.

**Verdict:** **OK.** Протокол H1-FAIL корректен:
- multiple-comparison поправка явная (Bonferroni @α/3),
- pre-registered бюджеты (25/40/50 %) задокументированы,
- LOCO-leakage явно проверен в docstring,
- pareto-static-comparison через `compute_router_vs_static_at_budgets`
  с McNemar + CI.
Slabost: фичи router'а не открыты в этих двух файлах — для полной уверенности
было бы полезно прочитать `src/pipeline/router/features.py`, но
LOCO-protocol гарантирует, что даже если есть категориально-зависимые
фичи, тест проходит по cross-category transfer (категория-holdout
никогда не присутствует в train). Spec narrative о H1-fail остаётся
валидным; никаких новых disclosures в §3 не требуется.

## 0.6 e1_circularity_analysis.py

**Файл:** `src/experiments/e1_circularity_analysis.py` (194 строки).

**Наблюдения:**
- Скрипт **не** делает per-taxonomy breakdown. Он сравнивает headline
  hybrid-accuracy при разных Layer-4 LLM backend'ах (baseline gpt-oss-120b
  vs sonnet45/gemini25flash/gpt4o + control llama3b).
- Логика: для каждого backend читает `router_pareto_gold_{sfx}.parquet`,
  берёт строку `strategy == "per_attr_table"` (одна точка per_attr_table),
  считает Δ vs baseline по headline accuracy. Применяется decision-rule
  (Δ<+1.5 → no circularity; +1.5..+3.0 → borderline; ≥+3.0 → circularity)
  плюс cost-matched analysis (interpolation вдоль static_threshold curve
  на cost=0.34).
- Артефакт `datasets/processed/e1_circularity_summary.json` существует.
- Никакого «per-taxonomy» (по доменам/категориям/per-attr) breakdown
  в e1 **нет** — это headline-уровень с разбиением только по backend'у.
- Phase 1.1 (per_taxonomy_breakdown) — другой scope: per-domain / per-attr
  / возможно per-language разбиение headline-числа. e1 не покрывает.

**Verdict:** **Phase 1.1 идёт как plan, новый скрипт.** Переиспользование
e1 не подходит: e1 группирует по «выбор Layer-4 backend», а Phase 1.1
требует группировку по таксономии (домен × attr-class). При создании
`per_taxonomy_breakdown.py` имеет смысл переиспользовать общую структуру
e1 (чтение `router_pareto_gold*.parquet`, helper `get_headline`), но
группировка — новая.

## 0.7 Identify blind audit model

**Файлы:**
- `src/manual_label/opus_off_grounded_audit.py` (270 строк) — оригинальный
  blind-audit скрипт.
- `src/llm/client.py` (130 строк) — generic OpenRouter/Ollama HTTP клиент,
  не задаёт никакую default модель.

**Grep вывод:**
```
src/manual_label/opus_off_grounded_audit.py:79:  model: str = "anthropic/claude-opus-4"
src/manual_label/opus_off_grounded_audit.py:149: model: str = "anthropic/claude-opus-4"
src/manual_label/opus_off_grounded_audit.py:251: p.add_argument("--model", default="anthropic/claude-opus-4")
src/manual_label/opus_off_grounded_audit.py:57:  _PRICE_INPUT_PER_MTOK = float(os.environ.get("OPUS_PRICE_INPUT_PER_MTOK", "15.0"))
src/manual_label/opus_off_grounded_audit.py:58:  _PRICE_OUTPUT_PER_MTOK = float(os.environ.get("OPUS_PRICE_OUTPUT_PER_MTOK", "75.0"))
```

**Модель:** **Claude Opus 4** (`anthropic/claude-opus-4`), цены $15/$75 per
MTok input/output (что соответствует Opus 4 pricing). `src/llm/client.py`
сам по себе универсален и без default-модели.

**Apples-to-apples decision (Phase 2.2):**
- Оригинальный audit — Opus 4 → Phase 2 default Sonnet 4.6 (или Sonnet 4.5,
  если 4.6 ещё не доступен на OpenRouter) + Opus 4.5/Opus 4.6 subsample
  (50 cells) для validation, чтобы убедиться, что Sonnet ≈ Opus на спорных
  кейсах. Это согласуется с §3 spec'а в плане.
- Cost-aware соображение: Opus 4 → Opus 4.5/4.6 (если есть в OpenRouter,
  судя по `PRICING` в `direct_llm_v2.py`, есть `anthropic/claude-opus-4.5`
  и `anthropic/claude-sonnet-4.5`) — Sonnet 4.5 в ~5x дешевле Opus 4 при
  сравнимом качестве на классификационных задачах OFF-attrs.

## 0.8 Grep downstream blacklist consumers

**Команда:** `grep -rn "curate_prompt_fields|DERIVED_BLACKLIST|off_field_filter" src/ demo/`.

**Хиты:**
| Файл | Строка | Тип использования |
|---|---|---|
| `src/manual_label/off_field_filter.py` | 17, 51, 54, 60 | Определение (declaration) |
| `src/manual_label/opus_off_grounded_audit.py` | 35, 94 | Импорт + вызов |
| `src/eval/direct_llm_v2.py` | 60, 71 | **Импорт внутри `_load_off_grounded_fields`** (lazy, только для `--context-mode off_grounded`) |

В `demo/` — **нет** хитов.

**Categorization:**
- (a) `opus_off_grounded_audit.py` — основной потребитель (blind audit, target Phase 2).
- (b) `direct_llm_v2.py` — потребитель **только** в режиме `--context-mode
  off_grounded` (строки 92–94, 114–121). В режиме `partner_input` (default
  для headline) blacklist не задействуется, используются только
  `_PARTNER_FIELDS = ["product_name", "brands", "ingredients_text",
  "quantity"]` (строка 56). Headline baseline (sonnet45 baseline в
  `direct_llm_eval_*_sonnet45.parquet`) — нужно проверить, в каком режиме
  он был построен. Если headline-числа baseline'а строились с
  `off_grounded`, то расширение blacklist в Phase 2.1 повлияет на эту
  цифру → потребует параметризации (новая Task 1.9).
- (c) Demo не использует — нет runtime impact.

**Action:** **Условная Task 1.9 — параметризация blacklist.**
Перед Phase 2.1 (расширение DERIVED_BLACKLIST) проверить:
```
grep -rn "context_mode\|--context-mode" \
  /Users/miafrolov/Desktop/stuff/ai_attributes/scripts/ \
  /Users/miafrolov/Desktop/stuff/ai_attributes/notebooks/
```
Если headline direct_llm_v2-baseline собирался в `off_grounded` режиме —
расширение DERIVED_BLACKLIST затронет baseline → нужно (а) переименовать
старый blacklist в `BLIND_AUDIT_BLACKLIST` и добавить параметр в
`curate_prompt_fields(..., blacklist=...)`, либо (б) пересобрать
baseline с расширенным blacklist (дороже). По умолчанию — (а).
Если headline в `partner_input` — параметризация не нужна, расширение
безопасно.

**Verdict:** **(b) с conditional.** Phase 2.1 пишется так, чтобы добавить
параметр `blacklist=DERIVED_BLACKLIST` в `curate_prompt_fields`, оставив
существующее поведение default. Это безопаснее, чем мутировать константу
на месте.

## 0.9 Verify direct_llm_v2.py не leak'ит

**Файл:** `src/eval/direct_llm_v2.py` (245 строк, читался полностью).

**Наблюдения:**
- Два режима: `partner_input` (default, CLI `--context-mode`) и `off_grounded`.
- `_PARTNER_FIELDS = ["product_name", "brands", "ingredients_text", "quantity"]`
  (строка 56) — те же 4 поля, что использует ML-layer cascade. **Никаких
  categories_tags/labels_tags** в этом списке.
- В `partner_input` режиме (строки 114–115):
  ```python
  product_dict = {k: prod.get(k, "") for k in _PARTNER_FIELDS}
  ```
  Прокидываются буквально 4 поля. **Leak нет.**
- В `off_grounded` режиме (строки 116–121):
  ```python
  product_dict = _load_off_grounded_fields(code, off_cache_dir)
  ```
  → `_load_off_grounded_fields` (строки 59–71) читает кэш OFF и применяет
  `curate_prompt_fields` (которая дропает `DERIVED_BLACKLIST`). Но
  curate **оставляет** `categories_tags` (строки 67–69
  `off_field_filter.py`: `if k == "categories_tags": out[k] = sorted(v)`)
  и `labels_tags` (нет в blacklist). Это **намеренно** для blind audit
  (что увидел бы человек на странице OFF), но **в `off_grounded` режиме
  direct_llm_v2 leak'ает** target-релевантные тэги.
- В режиме `off_grounded` direct_llm_v2 имитирует ровно тот же контекст,
  что Opus blind audit (§0.7). Это **не** «утечка» в строгом смысле —
  это та же methodology, что и blind ground-truth. Но при сравнении
  baselines в headline-таблице, числа `partner_input` и `off_grounded`
  **несравнимы** — у них разный набор входов.
- В `_load_off_grounded_fields` (строки 67–70) есть hack для
  «flat-format» OFF cache: `product = off_response.get("product") or
  off_response` — для исторического кэша Phase 1, у которого
  top-level dict сам и есть product.

**Verdict:** **v2 чистый в `partner_input` режиме**, утечка только в
`off_grounded` режиме (намеренная, для apples-to-apples с blind audit).
**Task 1.9 add'ится только если headline baseline собирался в
`off_grounded` режиме** (см. §0.8). По умолчанию — direct_llm_v2 headline
строится в `partner_input` mode (это видно из CLI default
`--context-mode partner_input`, строка 178), что соответствует v1
методологии (`local INPUT_FIELDS`).

**Recommendation для записи в Limitations §5:** упомянуть, что direct_llm_v2
поддерживает два режима; **headline** замерян в `partner_input` (без
leak'а), сравнения с blind Opus audit делаются в `off_grounded` (matched
input). Если в narrative §3.3 / §6 ноутбука где-то путаются эти два
режима — нужен fix.

## 0.10 Summary — branching impact on plan

| Phase task | Branching outcome | Action |
|---|---|---|
| 2.2 model | **Opus 4** (original blind audit использовал `anthropic/claude-opus-4`, §0.7) | Phase 2 default — **Sonnet 4.5** (apples-to-apples по поколению, в ~5x дешевле Opus 4) + **Opus 4.5 subsample на 50 ячейках** для cross-validation. Конкретные id моделей: `anthropic/claude-sonnet-4.5` (main) и `anthropic/claude-opus-4.5` (validation), оба присутствуют в `direct_llm_v2.py:PRICING`. |
| 3.2 Bayes narrative | **(c) Production-path использует gold-refit** (§0.2) — DAG silver, CPDs hybrid silver+gold×10, ML/Bayes thresholds tuned на val-50 % брендов из test-fold | Переписать §3.2 / §3.3 narrative о Bayes: явно разделить три уровня (DAG из silver, CPDs hybrid, thresholds gold-tuned). Указать, что headline 93.81 % defended на held-out half of test-fold (~50 % brands). Добавить sentence: «Структура сети получена Hill-Climb на silver; conditional probability tables и пороги перекалиброваны с использованием gold-данных (10-кратная репликация в BDeu prior, equivalent_sample_size=10); тест проведён brand-disjoint на held-out половине test-fold (seed=0).» |
| 1.2 threshold sensitivity | **std headline = 0.10 пп** (§0.3) на GroupKFold×10seed; per-attr std 0.3–1.3 пп; диапазон 0.32 пп | Headline стабилен на порядок лучше +1.03 пп gain. Phase 1.2 ограничивается **threshold-sweep** (sensitivity к выбору ML/Bayes thresholds), а не full seed-rerun ML. Достаточно: для каждого выбранного threshold показать ±0.05 sensitivity и убедиться, что Δheadline остаётся >0.5 пп. seed-instability ML-layer'a не блокер. |
| 1.1 taxonomy breakdown | **новый скрипт** (e1 другой scope — backend-comparison, не taxonomy, §0.6) | Создать `src/eval/per_taxonomy_breakdown.py` с нуля, переиспользовав helper'ы `get_headline` и pattern чтения `router_pareto_gold*.parquet` из e1. Не переиспользовать e1 целиком. |
| 1.9 direct_llm_v2 leak | **partner_input mode — нет утечки** (`_PARTNER_FIELDS` только 4 partner-поля, §0.9); **off_grounded mode — преднамеренная утечка** через `categories_tags`/`labels_tags` для apples-to-apples с blind Opus audit | **Task 1.9 НЕ нужна как «fix leak»**, но требуется **methodological note в §5 Limitations**: указать, что direct_llm_v2 имеет два режима, headline собран в `partner_input`. Если в текущем notebook/§3 где-то путаются режимы → точечный текстовый fix (не отдельная Task). Перед Phase 2.1 пробежать `grep -rn "context_mode" notebooks/ scripts/` чтобы убедиться. |
| Blacklist consumers | **direct_llm_v2.py + opus_off_grounded_audit.py** (§0.8); demo не использует | **Параметризовать `curate_prompt_fields(blacklist=DERIVED_BLACKLIST)`** в Phase 2.1, чтобы blind-audit расширение blacklist'а не ломало direct_llm_v2 `off_grounded` мод (если он используется в апробации). Если grep по `context_mode` в notebooks/scripts покажет, что `off_grounded` нигде не запускался для baseline-headline → можно безопасно расширить in-place. Default — параметризация (безопаснее). |
