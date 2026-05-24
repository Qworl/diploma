# AI Attributes Project Memory

## Index
- [Состояние ВКР и реквизиты](thesis_state.md) — актуальный статус дипломной работы, текущий план глав, ключевые числа.
- [Router H1 verdict 2026-05-13](router_h1_decision.md) — FINAL H1 FAIL на strong-seed brand-disjoint pre-registered eval; Plan B4 active.
- [Ревью thesis 2026-05-15](review_findings_2026-05-15.md) — актуальная критика после устранения блокеров: router target circularity, per-attr-table comparison, scaling-law on n=4.
- [Научная рецензия 2026-05-24](review_findings_2026-05-24.md) — Top-3 блокера: threshold-on-test в train.py:380/469, LLM-accuracy proxy в cost_quality_ci.py:118, потенциальный leak categories_tags в direct_llm.py.
- [Senior-ML ревью 2026-05-14](../../docs/thesis/REVIEW_2026-05-14_senior_ml.md) — `docs/thesis/REVIEW_2026-05-14_senior_ml.md`: согласование нарратива с темой «каскадная система», 10 правок, оценка Layer 1 (regex), SOTA-позиционирование, ответы на Q&A.
- [SOTA позиционирование](sota_positioning.md) — TXtract/MAVE/OpenTag/FrugalGPT/AutoMix/Hybrid LLM/RouterBench: у кого есть router, формулировка новизны.
- [Критические находки ревью 2026-05-12](review_findings_2026-05-12.md) — старая ревизия (блокеры закрыты, оставлено как историческая запись).
- [Category Router v2 (Layer 0)](category_router_v2.md) — pre-cascade XGBoost-классификатор, метрики после балансировки + LOCO; §6.15 ноутбука.
- [Long scripts must show progress](feedback_long_scripts_progress.md) — никаких `| grep | tail` для лонг-ранов, всегда incremental log в файл.
- [Bayes-validator на gold-порогах (сценарий C)](bayes_validator_scenario_c.md) — 2026-05-20 (УСТАРЕЛО, см. accuracy_squeeze_deploy) пороги перекалиброваны на gold q=0.02, валидатор активен только на 3 атрибутах.
- [Layer 1 расширен на сыры](cheeses_regex_2026-05-20.md) — 2026-05-20 в regex добавлены milk_source/is_pdo/is_ultra_processed на сырах, 100 % precision (47/47), cascade-only +3.38 п.п.
- [Accuracy-squeeze deploy 2026-05-20](accuracy_squeeze_deploy_2026-05-20.md) — финальная prod-конфигурация: hybrid Bayes (silver-DAG + gold-CPD×10) на 10 атрибутах + per-attr ML thresholds. Headline 92.78 → 93.81 % (+1.03 пп holdout-defended) при +5 % LLM cost.
- [Ноутбук 00_thesis_main намеренно монолитен](notebook_monolith_intentional.md) — это §3.3 диссертации в исполняемом виде; дробить нельзя, кейс EngineerXL/master-diploma не применим.

## Project
- Дипломная работа: гибридная система обогащения товарных данных (Regex → ML → Bayes → LLM)
- Рабочая папка: /Users/miafrolov/Desktop/stuff/ai_attributes
- Python 3.14, venv в .venv/, XGBoost нужен `brew install libomp`

## Key Decisions
- Работаем с DataFrame напрямую, без InternalProduct маппера
- OFF CSV основной источник (4.5M products), GroceryDB и Zerotox скачаны как дополнительные
- КБЖУ корреляции — тривиальные, не использовать. Осмысленные: P(fat|brand), P(organic|brand), P(sugar|brand)
- pgmpy 1.1.2: BayesianNetwork deprecated → DiscreteBayesianNetwork, fit через estimate_cpd()
- code колонка в OFF — кастить в str (overflow), completeness — в numeric

## Current State
- Состав: 6 food (pasta/chocolate/beverages/cheeses/cereals/cosmetics) + electronics (§5).
- Layer 0 (category router) добавлен 2026-05-14: XGBoost на 7 known + threshold-OOD. v2 balanced: acc=0.867, F1_macro=0.875, OOD AUROC=0.800.
- Notebook `notebooks/00_thesis_main.ipynb` — 59 cells (на 2026-05-24, monolith by design = §3.3 диссертации в исполняемом виде).
- **Главный вклад** (после Plan B4, §6.14.7): гибридный каскад regex → ML+XGBoost → per-attr static policy → gpt-oss-120b. 82.1% @ 34% LLM-cost на brand-disjoint test (n=1539). H1 для обучаемого XGBoost-роутера ОТКЛОНЕНА.
- Bayes исключён из финальной конфигурации (6% решений, 53% acc).
- XGBoost: regularized (subsample=0.8, colsample=0.8, early stopping, gamma=0.1).
- Per-attribute confidence thresholds → {category}_thresholds.pkl.
- Все артефакты на месте: calibration JSONs, router_pareto/stats/loco (`_gold`, `_gpt4omini`, `_gptoss`, `_llama3b`), cv_stability_10seed, manual_eval_summary, consensus_gold.parquet, ml_retrain_gold_eval.parquet.

## User Preferences
- Общение на русском
- Предпочитает venv, а не --user pip install
- Хочет видеть прогресс (tqdm)
- Предпочитает скачивать всё, фильтровать потом
