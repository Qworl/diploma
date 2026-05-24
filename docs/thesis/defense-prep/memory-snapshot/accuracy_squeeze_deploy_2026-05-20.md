---
name: Accuracy-squeeze deploy (hybrid Bayes + per-attr ML thr)
description: 2026-05-20 — финальная prod-конфигурация Bayes/ML-thresholds после A+E squeeze. Загружать когда трогаем models/{cat}_bayesian.pkl, models/{cat}_thresholds.pkl, models/{cat}_validation_thresholds.json, или нужны актуальные headline-числа.
type: project
originSessionId: e4355493-787d-41ac-a3a3-de828031510b
---
2026-05-20 после accuracy-squeeze A+E задеплоена новая prod-конфигурация валидатора и ML-порогов.

**Финальная конфигурация (10 useful Bayes-пар + 13 ML-thresholds, holdout-validated):**

| механика | (категория, атрибут) | значение |
|---|---|---|
| Bayes-validator A | pasta/is_organic | q=0.10 |
| | pasta/is_gluten_free | q=0.01 |
| | chocolate/chocolate_type | q=0.05 |
| | chocolate/cocoa_percentage | q=0.10 |
| | chocolate/contains_nuts | q=0.10 |
| | chocolate/is_organic | q=0.15 |
| | chocolate/nutri_score_grade | q=0.005 |
| | chocolate/protein_class | q=0.10 |
| | cheeses/texture | q=0.05 |
| | cheeses/is_ultra_processed | q=0.10 |
| ML threshold E | pasta/pasta_shape | 0.70 → 0.75 |
| | pasta/is_filled | 0.85 → 0.45 |
| | pasta/is_organic | 0.70 → 0.45 |
| | pasta/is_gluten_free | 0.55 → 0.45 |
| | pasta/nutri_score_grade | 0.70 → 0.45 |
| | chocolate/chocolate_type | 0.65 → 0.60 |
| | chocolate/cocoa_percentage | 0.60 → 0.45 |
| | chocolate/chocolate_extra | 0.65 → 0.45 (через THRESHOLD_OVERRIDES в regen — фактически 0.65) |
| | chocolate/is_organic | 0.50 → 0.45 |
| | chocolate/nutri_score_grade | 0.70 → 0.45 |
| | cheeses/country_of_origin | 0.55 → 0.75 |
| | cheeses/fat_class | 0.50 → 0.75 |
| | cheeses/is_pdo | 0.65 → 0.80 |

**Headline-результаты (после deploy):**

| метрика | значение |
|---|---|
| Headline (cascade + gemini-flash) | 92.78 % → **93.81 % (+1.03 пп holdout-defended; +1.12 пп in-sample)** |
| Bayes flagging | 213 cells flagged, TP=72 / FP=141 (precision 33.8 % vs ~10 % baseline) |
| LLM call rate | 3.3 % → 8.2 % (+4.9 пп) |
| Cost/quality | 0.23 пп / 1 % дополнительных LLM-вызовов |

**Топ-вклады per-attr (holdout):**
- cheeses/is_ultra_processed: +9.8 пп локально (Bayes q=0.10 ловит 14 confident-wrong)
- chocolate/contains_nuts: +3.5 пп (Bayes q=0.10)
- chocolate/is_organic: +3.7 пп (комбо E thr 0.45 + Bayes q=0.15)
- chocolate/nutri_score_grade: +2.5 пп (E thr 0.45, без Bayes)
- pasta/is_organic: +2.0 пп (комбо E thr 0.45 + Bayes q=0.10)
- chocolate/chocolate_extra: +1.4 пп (E thr 0.45, без Bayes)
- pasta/is_gluten_free: +0.6 пп (комбо)

**Bayes сеть (production):**
- DAG (структура) — от silver (Hill Climb + BIC на 15K silver brand-disjoint vs test)
- CPD-таблицы — fit на CONCAT(silver brand-disjoint, gold-train brand-disjoint × 10), BDeu prior
- Файлы: `models/{cat}_bayesian.pkl` (replaced); silver-версия как `.silver_backup.pkl`
- Пороги `_validation_thresholds.json` — selective per-attr q (см. таблицу выше)

**Why hybrid CPD-fit (silver + gold × 10):**
- Чистый gold (~650 train rows) → CPD-таблицы разрежены, brand-coverage узкая
- Чистый silver → выучивает шумовые корреляции, флажит ML-правильные ячейки
- Replication factor saturates at N≥5 — gold-CPD доминирует, silver обеспечивает brand-padding на редких case'ах
- Sweep по N ∈ {1, 5, 10, 20, 40} проведён, чувствительности нет

**How to apply:**
- Скрипты обучения и проверки:
  - `src/pipeline/bayes/refit_hybrid_holdout.py` — sweep по N (для выбора оптимума)
  - `src/experiments/accuracy_squeeze_ae.py` — per-attr q + ML-threshold sweep
  - `src/experiments/accuracy_squeeze_holdout.py` — val/held-out split + selection
  - `src/experiments/deploy_squeeze_config.py` — финальный deploy
- Артефакт chosen-config: `datasets/processed/accuracy_squeeze_chosen_config.json`
- Артефакт holdout-result: `datasets/processed/accuracy_squeeze_holdout.parquet`
- Бэкапы:
  - `_bayesian.silver_backup.pkl` — оригинальный silver-trained Bayes
  - `_validation_thresholds.scenario_c_backup.json` — предыдущая scenario-C config (gold-cal q=0.02, 3 useful attrs)
  - `_thresholds.scenario_c_backup.pkl` — предыдущие ML thresholds

**Слайды презентации обновлены:** 10 (доля LLM 3,3 → 8,2 %), 13 (Gemini 92,8 → 93,8 %, ≈720× → ≈290×, +9,0 → +10,0 пп vs direct Sonnet), 14 (Bayes block +0,3 → +1,0 пп, обновлена подпись), 15 (gpt-oss 92,3 → 93,3 %, +23,0 → +24,0 пп), 18 (итоги).

**Что НЕ обновлено и может всплыть на защите:**
- Thesis text (главы 3.3.2, 3.3.3, 3.3.4) — всё ещё содержит цифры 92,8 % / 90,5 % / 8,6 % / 3,3 %. Если защита требует консистентности — нужна отдельная правка по 4 источникам в `docs/thesis/*.md`.
- `cascade_preds_*_after_fix.parquet` обновлены regen'ом, но `tier_breakdown.parquet` остаётся на старом v3_fixed (намеренно — для §3.3.1.1 leakage-check).
- Headline_v3e_after_fix.parquet регенерирован, но GRAND ACC SUMMARY рапортит cascade-only=90.5% (не включает Bayes-demote, который добавляет +1 пп при runtime).
