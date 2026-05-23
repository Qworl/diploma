# Мета-ревью научной рецензии ВКР

**Дата:** 2026-05-24
**Объект мета-ревью:** `docs/thesis/REVIEW_2026-05-24_scientific_critique.md`
**Метод:** независимая верификация каждой претензии против реального кода и артефактов; поиск пропущенных проблем и ошибочных похвал.

Главный принцип — **не доверять рецензенту**, проверять руками: рецензент мог неправильно прочитать строки, пропустить mitigation в другом файле, раздуть мелкую проблему или проглядеть реальные.

---

## 1. Top-3 — вердикты с цифрами

### Top-3 №1: `find_best_threshold` тюнится на test split — **ПОДТВЕРЖДЕНО ЧАСТИЧНО**

`src/pipeline/ml/train.py:380, 469` действительно вызывают `find_best_threshold(final_clf, X_test, y_test_enc)`. Отдельного val-split для тюнинга порога нет — порог выбирается на том же test, на котором затем считают accuracy. Это classical threshold-on-test snooping.

**Реальный overlap `silver_test ∩ brand_disjoint splits` (посчитан):**

| cat | silver_test (n) | bd_test (n) | overlap test∩test | overlap test∩train |
|---|---|---|---|---|
| pasta | 3 139 | 250 | 42 (**16.8%** bd_test) | 141 (18.8% bd_train) |
| chocolate | 2 694 | 248 | 55 (**22.2%**) | 155 (20.9%) |
| cheeses | 4 242 | 250 | 36 (14.4%) | 134 (17.9%) |
| beverages | 248 | 248 | 49 (**19.8%**) | 157 (21.1%) |
| cereals | 198 | 198 | 37 (18.7%) | 125 (21.1%) |
| cosmetics | 223 | 223 | 52 (**23.3%**) | 128 (19.2%) |

То есть в `bd_test` для всех категорий 14–23% кодов уже видны были на этапе тюнинга порога. Дополнительно 17–21% `bd_train` пересекается с silver_test (двойная утечка: модель обучена на этих кодах и порог калиброван на них же). Тяжесть значимая, но не катастрофическая — реальный сдвиг headline вероятно 0.5–2 п.п.

### Top-3 №2: LLM accuracy на абстейн-выборке аппроксимирована — **ПОДТВЕРЖДЕНО**

`src/eval/cost_quality_ci.py:118-124` делает `proxy = cov*acc_cov + (1-cov)*llm_a`, где `llm_a = llm_acc_on_attr` — точность LLM на полном (cat, attr) тесте из `cascade_plus_llm4_hybrid.parquet`, не на абстейн-ячейках. `src/eval/layer4_llm.py:107-135` уже измеряет LLM на хвосте (`layer='none'`) и сохраняет в `llm_fallback_eval_{cat}.parquet`, но этот артефакт в `cost_quality_ci` НЕ интегрирован. Эффект — ≤1.5 п.п. оптимизма.

### Top-3 №3: direct LLM baseline получает `categories_tags` — **ОПРОВЕРГНУТО**

`src/eval/direct_llm.py:55` определяет ЛОКАЛЬНЫЙ `INPUT_FIELDS = ["product_name", "brands", "ingredients_text", "quantity"]` — БЕЗ `categories_tags`. Line 104: `product = {f: row.get(f) for f in INPUT_FIELDS if f in row.index}` фильтрует ровно по этим 4 полям. Затем `enrich_product → build_prompt → _render_product_block`: line 19 `if product.get(field)` — поле `categories_tags` вернёт `None` и не отрендерится. `_EXTRA_RENDER_FIELDS` (`labels_tags`, `nutriments`...) аналогично: их в product dict нет → не рендерится.

Исходный рецензент явно писал «не дочитан, возможно leak» — после чтения leak нет. **«+21.2 п.п. cascade vs direct LLM» НЕ артефакт промпта**, заявление защитимо.

---

## 2. §2 — что устояло после проверки

| § | Вердикт | Основание |
|---|---|---|
| **2.1** threshold-on-test | ПОДТВЕРЖДЕНО ЧАСТИЧНО | См. Top-3 №1, цифры выше |
| **2.2** TYPE_E/TYPE_F regex circularity | ПОДТВЕРЖДЕНО | `rules.py:758-1175` действительно regex по `ingredients_text`/`product_name`/`labels_tags`/`categories_tags` для всех 11 перечисленных атрибутов; `common.py:20 PARTNER_TEXT_FIELDS = [product_name, brands, ingredients_text, quantity]` — те же поля, что и SBERT input |
| **2.3** LLM acc on abstain | ПОДТВЕРЖДЕНО | см. Top-3 №2 |
| **2.4** direct LLM получает теги | ОПРОВЕРГНУТО | см. Top-3 №3 |
| **2.5** brand-norm split(',')[0] | ПОДТВЕРЖДЕНО | `generate_gold_splits.py:23-24` ровно такая строчка; `brand_disjoint.py` доп. нормализации не делает |
| **2.6** langdetect шум | ПОДТВЕРЖДЕНО | требует подкрепления распределением; не критично |
| **2.7** derivation_block с порогами | ПОДТВЕРЖДЕНО | `prompts.py:66-83` ровно совпадает с TYPE_C_RULES в `rules.py:688-705` («<5g→low, 5-15g→med, >15g→high») |
| **2.8** DAG stability | ЧАСТИЧНО + усугубляется | `dag_bootstrap.py` существует, артефакты `dag_stability_*.parquet` имеются. Прогон даёт: chocolate — **0/6 STABLE** edges; electronics edge `brand→os` (главный для cold-start демо!) — только 61% bootstrap freq, `ram_class→brand` (странное направление) — 55%. Это **усиливает** претензию |
| **2.9** random vs brand-disjoint в direct_llm | ПОДТВЕРЖДЕНО | оба `direct_llm.py:85-87` и `direct_llm_v2.py` — random split; brand_disjoint-режима у direct_llm нет |

---

## 3. Что рецензент пропустил (новые находки)

### 3.1 Blind-audit Opus НЕ blind на TYPE_A/B атрибутах
`src/manual_label/off_field_filter.py:17-27` blacklist'ит только `nutriscore_grade`, `nova_group`, `ecoscore_grade`, `ingredients_analysis_tags`. **`labels_tags` и `categories_tags` НЕ blacklist'ed.** Silver для `is_organic`, `is_vegan` (TYPE_A) выводится регексом по `labels_tags` (`en:organic`, `en:bio`, `en:vegan`); blind Opus видит ту же колонку. «κ=0.975» для `chocolate/is_organic` — это согласие двух читателей одних и тех же тегов, не валидация silver. §1.1 рецензии (где blind-audit назван «методологически чистым шумовым полом») неправ на этих атрибутах.

### 3.2 Pre-registration — post-hoc по git-таймстампам
Pre-registration коммитнута в первом коммите проекта **2026-05-19** (вместе со всем кодом и parquet-результатами). Внутри файла стоит подпись «2026-05-15», но это строка в markdown, не cryptographic timestamp. Истории до 2026-05-19 на других ветках/тегах нет. Декларация «зафиксировано ДО запуска» технически недоказуема. §1.3 рецензии (похвала за pre-reg с Bonferroni) **завышена**: формально это post-hoc justification.

### 3.3 Demo не запускает Layer 4 LLM — расхождение system-under-test
`demo/ml_service/cascade.py:412-429` для не-покрытых атрибутов просто проставляет `"layer": "llm_fallback"`, `"value": null` — настоящего LLM-вызова в production demo НЕТ. Headline 91.52% — это `cascade+gemini25flash` (ноутбук §3.3.2). Это два разных артефакта; в ВКР это должно быть явно проговорено, чтобы «работоспособность демо» не путалась с «accuracy системы».

### 3.4 DAG нестабильны — критично для нарратива «архитектурная ценность Bayes»
Артефакты `dag_stability_chocolate.parquet`: 0 STABLE edges из 6 reference. Для electronics ключевой edge `brand→os`, обосновывающий cold-start демо в §5, имеет bootstrap freq 0.61; `ram_class→brand` — 0.55 (странное направление). Структура графа во многом артефакт sample. Рецензент назвал это claim'ом («может быть cherry-picked»); цифры подтверждают.

### 3.5 LLM client temperature=0.0, single-shot
`src/llm/client.py:38` — `"temperature": 0.0`. Это значит: (a) повторные запуски детерминированы, поэтому stochastic CI вокруг single-shot не нужен (это **смягчает** обычную претензию); (b) НО provider-routing через OpenRouter недетерминирован — один и тот же ID модели может попасть на разные провайдеры (vertex/groq/fireworks) с разными квантизациями. Headline 91.5% получен в одном прогоне — provider-variance не оценена.

### 3.6 `cost_quality_ci.py ROUTER_ACC` — захардкоженные константы
Line 33: `ROUTER_ACC = {"pasta": 0.962880, "chocolate": 0.985377, "cheeses": 0.977477}`. При bootstrap'е брендов (142-150) `ROUTER_ACC` не ресемплируется. CI занижены, особенно на cheeses (n брендов меньше). Рецензент упомянул в §3, но не подсветил в Top-3.

### 3.7 `cocoa_percentage` — разные таксономии в silver и LLM-промпте
Silver buckets: «70-85», «50-70». В prompt LLM получает: «parse from product_name/ingredients_text, common patterns: "70%", "dark chocolate 70"». LLM не знает буккет-границы → возвращает «70», сравнивается с silver «70-85» → автоматический mismatch. Если accuracy `cocoa_percentage` в LLM eval низкая — артефакт labelspace, не качества LLM.

---

## 4. Что рецензент ошибочно похвалил

1. **§1.1** Blind-audit Opus как «методологически чистый шумовой пол» — см. 3.1: на TYPE_A/B Opus видит те же `labels_tags`/`categories_tags`, что silver-extractor; на TYPE_E (regex по `ingredients_text`) — оба читают один текст. Не настоящий blind-эксперимент.
2. **§1.3** Pre-registration с Bonferroni — см. 3.2: первый коммит = single-shot import всего проекта, timestamp недоказуем.
3. **§1.4** «McNemar χ² + биномиальный exact для малых n» — корректно реализовано (`cascade_vs_llm_stats.py:97-115`), но применено per-attr × per-cat **без FDR-коррекции** (~40 одновременных тестов), что эту корректность нивелирует.

---

## 5. Что осталось непроверенным даже после мета-ревью

1. **Реальный sensitivity headline к threshold-on-test** — не запускал retraining с `threshold=0.5`, только посчитал overlap.
2. **Bayes validator p-value precision** — `src/eval/validator_hypothesis_tests.py` не прочитан.
3. **per_language_eval с реальным распределением n** — не проверил per-lang CI.
4. **`accuracy_squeeze_holdout.py`** — claim, что в prod DAG фиксирован, не верифицирован.
5. **`recompute_calibration_only` bug** — code lines 615-621 пишут `"ece_raw": ece, "ece_calibrated": ece` (оба ECE равны). Plot before/after может вводить в заблуждение. Эффект на читателя не оценён.
6. **`demo/ml_service/validator.py`** не прочитан. Если validator переобучается на test products — утечка в demo.
7. **`bootstrap_ci_brand_clustered.py`** не прочитан; детали стратификации не проверены.

---

## 6. Финальный ранг Top-3 ПОСЛЕ верификации

### №1 (новый) — Structural circularity TYPE_E/TYPE_F + не-blind Opus аудит

Силовая комбинация §2.2 рецензента + 3.1 мета-ревью. На 11 атрибутах из ~22 (`contains_nuts`, `has_sulfates`, `has_silicones`, `product_type`, `form_factor`, `fragrance_status`, `milk_type`, `feeding_purpose`, `primary_protein_source`, `chocolate_extra`, `is_whole_grain`) silver = regex по `ingredients_text`/`product_name`; ML обучен на эмбеддингах ровно тех же полей; blind Opus читает те же поля + `labels_tags`/`categories_tags`. Это **три ридера одного источника, согласие между ними ≠ валидация**. Headline-разделение «TYPE_A/B vs TYPE_C/D/E/F» в ВКР отсутствует — это методологически критично для защиты. Фикс — разбивка headline и явная пометка regex-derived attrs, а не code changes.

### №2 — Threshold-on-test + brand-disjoint overlap (исходный Top-3 №1)

Подтверждено цифрой: bd_test содержит 14–23% codes из silver_test, на котором калиброван порог. Защита потребует sensitivity analysis (headline с `threshold=0.5` vs текущий). Если разница <0.5 п.п. — претензия снимается; если ≥2 п.п. — переобучение.

### №3 — DAG instability разрушает нарратив «Bayes как архитектурная ценность»

Из 3.4: для chocolate 0/6 ref edges STABLE; для electronics ключевой `brand→os` всего 61%. Это прямо противоречит cold-start демо §5. Артефакты уже посчитаны (`dag_stability_*.parquet`) — рецензент об этом не знал, но цифры уже **внутри проекта**.

### Что ушло из Top-3

- Top-3 №3 рецензента (categories_tags в direct LLM) — **ОПРОВЕРГНУТО**.
- Top-3 №2 рецензента (LLM acc on abstain) — **ПОДТВЕРЖДЕНО**, но эффект мал (≤1.5 п.п.) и легко фиксится; переезжает в "secondary".

### Honorable mentions (вне top-3, требуют внимания)

- Pre-registration post-hoc (3.2) — argumentative liability на защите.
- Demo ≠ system-under-test (3.3) — лёгкий фикс через явное проговаривание в ВКР.
- `recompute_calibration_only` bug (`ece_raw == ece_calibrated`, train.py:618-619) — plot «before/after» может быть искажающим.
- Per-attr McNemar без FDR (~40 тестов) — pre-reg покрывает только 4 phase-1 гипотезы.

---

## Файлы, на которые стоит смотреть в первую очередь при доработке

- `src/pipeline/ml/train.py` — `find_best_threshold` на test
- `src/eval/cost_quality_ci.py` — `llm_acc` proxy + `ROUTER_ACC` константа
- `src/pipeline/off_labels/rules.py` — TYPE_E/TYPE_F regex (структурная circularity)
- `src/manual_label/off_field_filter.py` — `labels_tags`/`categories_tags` не blacklist'ed для blind audit
- `src/data/split/generate_gold_splits.py` — brand_norm subbrand leak
- `datasets/processed/dag_stability_chocolate.parquet` — 0/6 STABLE edges (критично для нарратива)
- `docs/thesis/pre_registration_2026-Q2.md` — git timestamp = первый коммит проекта
