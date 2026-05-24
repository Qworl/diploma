---
name: Научная рецензия 2026-05-24
description: Критика ВКР по коду+ноутбуку, Top-3 блокера до защиты (threshold-on-test, LLM-accuracy proxy, direct_llm categories leak).
type: project
originSessionId: 6e2d20b5-1ea6-4789-8430-806901e58512
---
Исходная рецензия: `docs/thesis/REVIEW_2026-05-24_scientific_critique.md`.
Мета-ревью (верификация): `docs/thesis/REVIEW_2026-05-24_meta_critique.md`.

**Top-3 ПОСЛЕ верификации (исходный Top-3 №3 опровергнут):**

1. **Structural circularity TYPE_E/TYPE_F + не-blind Opus аудит** — silver для 11 атрибутов (contains_nuts, has_sulfates, has_silicones, product_type, form_factor, fragrance_status, milk_type, feeding_purpose, primary_protein_source, chocolate_extra, is_whole_grain) выводится регексом по `ingredients_text`/`product_name` (см. `src/pipeline/off_labels/rules.py:758-1175`); ML обучен на тех же полях; `src/manual_label/off_field_filter.py:17-27` НЕ blacklist'ит `labels_tags`/`categories_tags`, поэтому blind Opus тоже не blind. Headline-разбивка regex-derived vs nutrient-derived в ВКР отсутствует.

2. **Threshold-on-test + brand-disjoint overlap** — `src/pipeline/ml/train.py:380, 469` тюнит порог на silver test split. Реальный overlap silver_test ∩ bd_test: 14–23% codes (посчитано: pasta 16.8%, chocolate 22.2%, beverages 19.8%, cosmetics 23.3%, cheeses 14.4%, cereals 18.7%). Требуется sensitivity analysis с `threshold=0.5`.

3. **DAG instability разрушает narrative «Bayes — архитектурная ценность»** — `datasets/processed/dag_stability_chocolate.parquet` показывает 0/6 STABLE edges; для electronics ключевой `brand→os` (основа cold-start демо §5) — bootstrap freq всего 61%. Артефакты уже внутри проекта.

**Опровергнутое (из исходного Top-3):** `src/eval/direct_llm.py:55` имеет ЛОКАЛЬНЫЙ `INPUT_FIELDS` без `categories_tags`, фильтрация на line 104 корректна. «+21.2 п.п. cascade vs direct LLM» НЕ артефакт промпта.

**Honorable mentions (вне top-3, требуют ответа на защите):**
- Pre-registration post-hoc: первый git-коммит проекта = single-shot import всего кода + parquet'ов + pre-reg одновременно (2026-05-19). Timestamp «2026-05-15» внутри markdown недоказуем.
- `cost_quality_ci.py:118-124` LLM acc на абстейн-выборке = средняя по атрибуту (`layer4_llm.py` уже измеряет правильно, но не интегрирован) — эффект ≤1.5 п.п.
- Demo (`demo/ml_service/cascade.py:412-429`) не запускает LLM — для непокрытых атрибутов ставит `value: null`. «Работоспособное демо» ≠ «headline 91.5%».
- `train.py:614-622` `recompute_calibration_only` пишет одинаковое значение в `ece_raw` и `ece_calibrated` — plot «до/после» может быть искажающим.
- McNemar per (cat, attr) без FDR на ~40 тестах.

**How to apply:** Если пользователь идёт чинить — приоритет ровно в этом порядке (TYPE_E разбивка → threshold sensitivity → DAG разговор в ВКР). Не повторять опровергнутую претензию про direct_llm leak.
