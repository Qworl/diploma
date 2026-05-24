---
name: Category Router (Layer 0) — v2 → v3 production
description: Layer 0 — pre-cascade XGBoost classifier; production = Router v3, включён в headline через router_v3_correct_frac на per-category mult.
type: project
originSessionId: 4c818f12-459d-499a-8deb-9def7fd5c5da
---
Layer 0 (category router) — отдельный pre-cascade классификатор. В production используется **Router v3** (артефакты `models/category_router_v3_lgbm.pkl`, `category_router_v3_vec.pkl`, `category_router_v3_threshold.json`). Per-category accuracy измерена в `datasets/processed/cascade_layer0_eval.parquet`:

- pasta: 96.29 %
- chocolate: 98.54 %
- cheeses: 97.75 %

Layer 0 включён в headline через мультипликатор: `acc_with_router = acc_oracle_cat × router_acc_v3` (колонка `acc_hybrid_with_router` в `cascade_plus_llm4_summary.parquet`). На итоговый scatter §3.3.2 каждая точка `cascade+X` уже учитывает соответствующий router-штраф 1.4–3.7 п.п.

**Историческая справка (v1/v2 — учебные итерации, не в production):**
- v1 (imbalanced, n_per_class=5000): test_acc=0.932, F1_macro=0.892, OOD_AUROC=0.887. Завышено за счёт перекоса в сторону electronics.
- v2 (balanced, все классы по 988, commit `a421d9f`): test_acc=0.867, F1_macro=0.875, OOD_AUROC=0.800. Это была методологически чистая цифра на 6 категориях; **не актуальна для текущей 3-категорийной production-конфигурации**.

How to apply:
- При обсуждении точности Layer 0 в защите ссылаться на Router v3 (96–99 % per cat).
- При изменении состава известных категорий — перетренировать `python -m src.pipeline.category_router.train`.
- Подсекция ноутбука «3b. Маршрутизатор категорий Layer 0» (после §3a tier-breakdown) — единственное place где Layer 0 явно отображён в ноутбуке.

Why это в memory: рецензент в обзоре 2026-05-18 спросил «memory mentions §6.15 о Layer 0, но в ноутбуке его нет». Уточнение: §6.15 был в pre-refactor 175-cell ноутбуке; в ВКР-grade 41-cell ноутбуке Layer 0 представлен подсекцией §3b (брэнчево-disjoint метрики per-cat). Memory обновлена 2026-05-18.
