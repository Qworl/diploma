# GOLDEN_STRINGS — идентификаторы, которые НЕ должны меняться

Наполнено в этапе -1. Регистрозависимое сравнение, нормализация пробелов через `re.sub(r'\s+', ' ', ...)`. Якорь засчитан, если найден хотя бы в одном из источников (DOCX или notebook).

## 1. Имена моделей и библиотек

- [x] `paraphrase-multilingual-MiniLM-L12-v2`
- [x] `gpt-oss-120b`
- [x] `Claude Sonnet 4.5`
- [x] `Gemini 2.5 Flash`
- [x] `XGBoost`
- [x] `pgmpy`
- [x] `SBERT`
- [x] `gpt-4o`

## 2. Parquet-файлы и артефакты экспериментов

- [x] `headline_v3e_after_fix.parquet`
- [x] `consensus_gold_v2_expanded.parquet`
- [x] `grand_acc_summary_after_fix.parquet`
- [x] `cascade_layer0_eval.parquet`
- [x] `router_pareto_gold.parquet`
- [x] `tier_breakdown.parquet`
- [x] `knn_distance_ablation.parquet`
- [x] `bayes_validator_demote_metric.parquet`
- [x] `ece_calibration_table.parquet`
- [x] `per_language_eval.parquet`
- [x] `off_leakage_probe.parquet`

## 3. Технические стандарты

- [x] `ГОСТ 7.32`
- [x] `ГОСТ 7.1`

## 4. Ключевые URL из списка литературы

- [x] `https://world.openfoodfacts.org/data`
- [x] `https://akeneo.com`
- [x] `https://pimcore.com`
- [x] `https://www.salsify.com`
- [x] `https://arxiv.org/abs/1908.10084`
- [x] `https://arxiv.org/abs/2305.05176`
- [x] `https://cyberleninka.ru/article/n/aktivnoe-obuchenie-i-kraudsorsing-obzor-metodov-optimizatsii-razmetki-dannyh`

## Чек-листы

- [x] этап -1: 4 категории наполнены из `_baseline.docx`
- [x] `check_regression.py` зелёный на полном списке (true positive)
- [ ] (после этапа 25) добавить `ОД-093-СМК-ПОЛ-001-Ф` когда форма МАИ будет вписана в титульный
