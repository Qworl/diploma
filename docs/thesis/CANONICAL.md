# CANONICAL — Single Source of Truth (VKR Defense Polish)

> **Авторитетный источник** для всех числовых заявлений, схем атрибутов, формулировок стоимости и терминологии в thesis (`report/**/*.tex`), слайдах (`slides/main.tex`) и сопутствующих документах. Создан 2026-05-26 для месячной полировки ВКР, обновлён 2026-05-27 после extended 3-LLM consensus rerun.
>
> Любая правка thesis/slides обязана сверяться против этого файла. Несоответствие — ошибка.
>
> Backing source: `datasets/processed/v4_e2e_router_eval.json` + `v4_metric_table_v2.json` (pulled 2026-05-27 11:03 UTC), `docs/thesis/data_methodology.md` §14.

## 1. Headline accuracy

Базовый эталон — **LLM-consensus gold (extended rerun 2026-05-27)**, n=16360 cascade-valid (из 17062 в-scope), 20 атрибутов, code-disjoint train/test (не brand-disjoint, см. §6). Консервативный эталон — **human gold** (Opus blind), n=566 cascade-valid (из 688 в-scope) — без изменений.

| Метрика | LLM-consensus (n=16360) | Human gold (n=566) | Артефакт |
|---|---|---|---|
| Cascade-only micro-accuracy | **95,5%** | 91,3% | `v4_e2e_router_eval.json` |
| Cascade-only macro-F1 (cells-weighted) | **0,890** | 0,848 | `v4_metric_table_v2.json` (VM) |
| Cascade-only macro-F1 (attr-unweighted) | 0,892 | — | (тот же) |
| Router accuracy (per code) | 97,2% (n=570) | 95,3% (n=107) | `v4_e2e_router_eval.json` |
| E2E coverage | 97,3% | 95,1% | `v4_e2e_router_eval.json` |
| **E2E accuracy (None=wrong)** | **93,0%** | **86,7%** | `v4_e2e_router_eval.json` |

**Семантика n cascade-valid.** Знаменатель — ячейки, на которых каскад вернул конкретный класс (Layer 1 rule_h ∪ Layer 2 ML ∪ Layer 3 rule_l). Layer 4 LLM fallback (702 cells LLM-consensus / 122 cells HUMAN) исключён из cascade-only знаменателя по определению — на этих ячейках каскад «отказывается» и в production отправляет в LLM. Поэтому n=16360/566 (cascade-valid), не n=17062/688 (всего в-scope). Воспроизводится в `notebooks/03_evaluate.ipynb` ячейка `44dbf52a`, источник — `datasets/processed/v4_e2e_router_eval.json` ключи `LLM-consensus`/`HUMAN`.

## 2. LLM fallback distribution

| Слой | Cells (n=17062) | Доля | Назначение |
|---|---|---|---|
| Layer 1 (rule_h, regex по тегам/тексту) | 3734 | **21,9%** | High-precision rules |
| Layer 2 (ML: MPNet + TF-IDF SVD + XGBoost) | 12487 | **73,2%** | Главная работа |
| Layer 3 (rule_l, low-precision regex) | 139 | 0,8% | Fallback перед LLM |
| **Layer 4 (LLM fallback)** | **702** | **4,1%** | Сложные / неуверенные cells |

**LLM cost reduction vs naive all-LLM baseline: 95,9%.**

**Per-category fallback rate (extended consensus rerun, in_scope, source `cascade_preds_{cat}_gold.parquet`):** pasta — 5,1 % (n=8004), chocolate — 2,6 % (n=6759), cheeses — 4,9 % (n=2411). Воспроизводится в `notebooks/03_evaluate.ipynb` cell `percat-l4`. Сумма по категориям (17174) на 112 ячеек больше n=17062 из таблицы §2 — это разница между фильтром «cascade_preds in_scope» и «manual_gold_consensus passing consensus»; на per-category проценты не влияет.

## 3. Cost framing — три явных термина

В thesis (особенно §2.1.4, §3.3.2, §4.3.2) **всегда** называть тип снижения стоимости одним из трёх:

1. **«Архитектурное сокращение»** = 24× (≈95,9%) — каскад против «one LLM на всё» при ОДНОЙ И ТОЙ ЖЕ модели Layer 4. Чистый эффект каскадной композиции (1 / 0,0411 ≈ 24,33×, округление 24×).
2. **«Комбинированное сокращение»** = 580× (Gemini Flash), 24× (Sonnet/каскад), 33× (GPT-4o/каскад), 810× (gpt-oss-120b), 2700× (llama-3b) — каскад против all-Sonnet baseline. Включает удешевление модели + архитектурный вклад. Формула: `combined = 1 / (fallback_rate × COST_REL[model])`, где `fallback_rate = 0,0411`, `COST_REL` — тариф модели относительно Sonnet 4.5 (1,000 sonnet / 0,727 gpt-4o / 0,042 gemini-flash / 0,030 gpt-oss-120b / 0,009 llama-3b). Воспроизводится в `notebooks/03_evaluate.ipynb` cell `cost-recompute` (см. §10). Предыдущие значения 333× / 471× / 1571× были выведены при старом `fallback_rate = 0,071` до extended consensus rerun 2026-05-27.
3. **«Эффективное сокращение по сравнению с прямой моделью»** = архитектурный вклад × выбранная модель Layer 4. Используется в матрице «точность-стоимость» (рис. 3.3).

**Запрещено:** говорить «720×» без contextual qualifier — это устаревшее число.

## 4. Схема атрибутов

**Total — 21 атрибут в production схеме.** В headline-таблице — 20 (без `pasta.protein_class`, помечен как опциональный).

| Категория | Атрибутов | Полный перечень |
|---|---|---|
| pasta | **8** | `grain_type`, `pasta_shape`, `is_filled`, `is_gluten_free`, `is_organic`, `is_vegan`, `cuisine_origin`, `protein_class`* |
| chocolate | **6** | `chocolate_type`, `is_filled`, `chocolate_extra`, `contains_nuts`, `is_organic`, `flavor_profile` |
| cheeses | **7** | `milk_source`, `texture`, `country_of_origin`, `aging`, `is_pdo`, `is_organic`, `is_ultra_processed` |

\* `pasta.protein_class` — опциональный, в headline-таблице 20 атрибутов не включён.

## 5. Тестовые эталоны

| Эталон | n cells | Источник | Назначение |
|---|---|---|---|
| LLM-consensus gold (extended) | 16360 cascade-valid (17062 в-scope) | 3 LLM (qwen3.7-max + deepseek-r1 + mistral-large-2411), правило ≥2/3 | Primary headline |
| Human gold | 566 cascade-valid (688 в-scope) | Opus 4 blind на спорных | Conservative lower bound |
| Router gold | 570 / 107 | per-product, для оценки Layer 0 | Router accuracy |
| H1 router test | 1539 (~284 codes × 6 категорий) | preregistered, `router_train.parquet` | Только для H1 теста (см. §7) |

## 6. Train/test split — code-disjoint, не brand-disjoint

**Реальный split** (`scripts/build_noleak_artifacts.py`):
- Удаляются товарные коды (codes), попавшие в любой из eval gold (consensus / human / extended).
- Бренды остаются: brand overlap train ↔ eval составляет **82,5% pasta / 84,9% chocolate / 82,7% cheeses** (по brand tokens; ниже, чем 100%/96%/93% в `data_methodology.md` §12.2 из-за разной токенизации, но направление то же).

**В thesis формулировки:**
- ✅ «без пересечения товарных кодов (новые SKU)»
- ✅ «code-disjoint train/test»
- ✅ «новые SKU из существующих и новых брендов»
- ❌ «без пересечения брендов» / «brand-disjoint» — снято
- ❌ «выборка без пересечения брендов» — снято

**В §3.2.4 — добавить явный абзац:** «Разбиение выполнено по идентификатору товара (code-disjoint), а не по бренду. Brand overlap между обучающим пулом и тестовой выборкой составляет 82–85% по категориям и обусловлен концентрацией брендов в открытом каталоге OFF. Brand-disjoint режим невозможен без переразметки эталона; в §3.3.7 приводится sensitivity-check на brand-disjoint подмножестве (17/186 pasta, 19/301 chocolate, 28/297 cheeses) как доп. проверка устойчивости.»

## 7. H1 router preregistration

H1 (отрицательный результат: обучаемый XGBoost-маршрутизатор не превосходит статическое правило при равных бюджетах LLM):
- Preregistered commit: `cd9ac7a` (2026-05-13 02:11), за 1ч 11мин до `router_pareto_gold.parquet` (03:22).
- n = 1539 ≈ 284 кода × 6 категорий, через `by_product_split(seed=42)` (`src/pipeline/router/data.py:89`).
- Split: **code-disjoint, не brand-disjoint** (brand overlap 27–70%).
- MDE ≈ 4,4 пп (парный McNemar при n=1539, α/3 ≈ 0,0167).

**n = 1539 — это frozen preregistered snapshot, материализованный в кэше `router_train.parquet` на момент коммита `cd9ac7a`.** Текущий live recompute (`_apply_gold_overrides` поверх обновлённых `{cat}_gold_split.parquet`) даёт `n_test = 1574` (+35 строк из переразметки эталона). Принцип предварительной регистрации требует фиксации именно зарегистрированного значения, поэтому в TeX оставлено 1539; live 1574 — для внутренних диагностик и Q&A. MDE 4,4 пп устойчив на n ∈ [1539, 1635]; качественный вывод H1 не меняется. См. §3.3.5 footnote и `docs/thesis/defense-prep/qa_prep.md`.

В thesis (§3.3.5) формулировка должна явно говорить «code-disjoint», не «brand-disjoint».

## 8. ТЗ counts

**Канонически: 6 ФР / 5 НФР / 3 А.** Источник — `report/contents/2-chapter1-analysis.tex` §1.3.

| Группа | Количество | Items |
|---|---|---|
| Функциональные (ФР) | **6** | Ф-1..Ф-6 |
| Нефункциональные (НФР) | **5** | НФ-1..НФ-5 |
| Архитектурные (А) | **3** | А-1..А-3 |

Слайды сейчас показывают N1–N4 + A1–A3. Conclusion говорит «4 архитектурных» — обе версии устарели.

## 9. Models / pipeline

- Слой 0 категоризатор: LightGBM поверх TF-IDF (n-граммы 1–2) + OOD-порог
- Слой 1: regex (`src/pipeline/regex/extractor.py`)
- Слой 2: MPNet `paraphrase-multilingual-mpnet-base-v2` (768d) ⊕ TF-IDF SVD-128 ⇒ XGBoost (`_mpnet_tfidf_noleak`)
- Слой 3: байесовская сеть (Hill Climb + BIC, pgmpy 1.1.2) — **в production включена селективно только на chocolate/contains_nuts**
- Слой 4: OpenRouter — Sonnet 4.5 / GPT-4o / Gemini 2.5 Flash / gpt-oss-120b / llama-3.2-3b

## 10. Терминология

| Термин | Что означает |
|---|---|
| Cascade-only accuracy | Точность системы на покрытых ячейках (95,5% / 91,3%); знаменатель = cascade-valid (n=16360 / 566); upper bound при идеальной маршрутизации категории |
| E2E accuracy | Точность с учётом ошибок маршрутизации, None=wrong (93,0% / 86,7%); тот же знаменатель что у cascade-only; production-realistic |
| Производственная цепочка | = E2E (синоним в Chapter 4) |
| Архитектурное снижение стоимости | Эффект каскадной композиции при той же модели Layer 4 (24×) |
| Комбинированное снижение | Эффект каскад × выбранная модель Layer 4 (24× для каскад+Sonnet, 580× для каскад+Gemini, 810× для каскад+gpt-oss-120b, 2700× для каскад+llama-3b — fallback 4,11%) |
| Циркулярное смещение | Operational bias 4,2 пп (cascade) / 6,3 пп (E2E) — gap consensus vs human gold; midpoint ≈5 пп |

## 11. Стоп-лист — устаревшие числа и формулировки

При любом редактировании `.tex` / `.md` искать и заменять/удалять следующие:

| Устаревшее | Заменить на | Источник устарелости |
|---|---|---|
| `94,8 \%` (cascade headline) | `95,5 \%` | 2026-05-27 extended consensus rerun |
| `91,1 \%` (E2E headline) | `93,0 \%` | 2026-05-27 extended consensus rerun |
| `95,4 \%` (router accuracy) | `97,2 \%` | 2026-05-27 extended consensus rerun |
| `96,0 \%` (E2E coverage) | `97,3 \%` | 2026-05-27 extended consensus rerun |
| `18,4 \%` (Layer 1 share) | `21,9 \%` | 2026-05-27 extended consensus rerun |
| `73,8 \%` (Layer 2 share) | `73,2 \%` | 2026-05-27 extended consensus rerun |
| `0,7 \%` (Layer 3 share) | `0,8 \%` | 2026-05-27 extended consensus rerun |
| `7,1 \%` (Layer 4 share / LLM fallback) | `4,1 \%` | 2026-05-27 extended consensus rerun |
| `92,9 \%` (LLM cost reduction) | `95,9 \%` | 2026-05-27 extended consensus rerun |
| `14×` (architectural cost) | `24×` | 2026-05-27 (1/0,0411 ≈ 24,33×) |
| `0,899` (macro-F1 cells-weighted) | `0,890` | 2026-05-27 extended consensus rerun |
| `0,902` (macro-F1 attr-unweighted) | `0,892` | 2026-05-27 extended consensus rerun |
| `n=3257` (cascade-valid LLM-consensus) | `n=16360` | 2026-05-27 extended consensus rerun |
| `n=3506` (in-scope LLM-consensus) | `n=17062` | 2026-05-27 extended consensus rerun |
| `92,8 \%` (как sole headline) | `93,0 \%` E2E / `95,5 \%` cascade-only | post-v4 refactor |
| `720` (как cost factor) | `580` (для каскад+Gemini) / `24` (архитектурное) | 2026-05-27 extended consensus rerun |
| `333` (для каскад+Gemini, fallback 7,1%) | `580` (fallback 4,11%) | 2026-05-27 |
| `471` (для каскад+gpt-oss, fallback 7,1%) | `810` (fallback 4,11%) | 2026-05-27 |
| `1571` (для каскад+llama-3b, fallback 7,1%) | `2700` (fallback 4,11%) | 2026-05-27 |
| `4350` cells | `16360` (LLM-consensus) / `566` (human gold cascade-valid) | post-v4 refactor |
| `n=615` (human gold) | `n=566` cascade-valid (live из `v4_e2e_router_eval.json`) | 2026-05-26 drift fix |
| `91,7%` cascade-only human | `91,3%` (live) | 2026-05-26 drift fix |
| `87,5%` E2E human | `86,7%` (live) | 2026-05-26 drift fix |
| `3,3 \%` LLM | `4,1 \%` LLM | 2026-05-27 extended consensus rerun |
| `9,0 п. п.` (vs Sonnet) | `7,3..9,2 п. п.` (E2E, в зависимости от base) | post-v4 refactor |
| `23,0 п. п.` (gpt-oss) | `21,3 п. п.` | post-v4 refactor |
| `2,3 п. п.` LLM contribution | `+1,4..+1,7 п. п.` (per-category) | post-v4 refactor |
| `24 поатрибутных XGBoost` | `21 XGBoost (8/6/7)` | post-v4 refactor + schema reform |
| `22 пары` (слайды) | `20 в headline-таблице / 21 в схеме` | uncertainty |
| `выборка без пересечения брендов` | `выборка без пересечения товарных кодов (code-disjoint)` | brand overlap 82–85% |
| `brand-disjoint test` | `code-disjoint test (new SKUs)` | то же |
| `4 НФР` | `5 НФР` | спец §1.3 |
| `4 архитектурных` | `3 архитектурных` | спец §1.3 |
| `96,7 \%` (старое каскад coverage) | `97,3 \%` E2E coverage | 2026-05-27 extended consensus rerun |
| `90,5 \%` (старая средняя точность) | (зависит от контекста — обычно `95,5 \%` cascade-only) | 2026-05-27 |

## 12. Известные ограничения (для §«Ограничения» conclusion)

1. **Brand overlap 82–85%.** Эталон — code-disjoint, не brand-disjoint. Sensitivity на brand-disjoint subset вынесена в §3.3.7 (C-4).
2. **Circular bias 4,2 пп (cascade) / 6,3 пп (E2E).** Обучающий silver частично пересекается семантически с consensus gold (оба — LLM-derived). Operational gap = consensus − human на той же выборке. Human gold (91,3% cascade / 86,7% E2E) даёт independent lower bound.
3. **Sample size человеческого эталона.** n=566 cascade-valid (из 688 в-scope, остальные 122 — Layer 4 LLM fallback), single labeler (Opus 4), без IRR.
4. **Calibration не в production.** ECE 0,070 → 0,043 при isotonic CV на gold проверена (§3.3.3.3); не включена в production.
5. **Bayes в production — на 1 атрибуте.** Из 20 пар селективный сигнал работает только на chocolate/contains_nuts. Остальные — flat либо отрицательный прирост.
6. **Языковые слабые места.** cheeses/texture/es (−47 пп от лучшего), pasta_shape/de и др. — четыре конкретных hot-spot'а в §3.3.7.1.
7. **Cross-domain validity** ограничена food. Electronics добавлена как cross-domain control (§3.4 после E-3), но не как полноценная replication.
8. **Lic.** Open Food Facts ODbL — требует attribution share-alike в производных.

## 13. Артефакты, на которые опираются числа

### 13.A. Число → notebook cell → backing artifact

| Число | Notebook · cell | Артефакт |
|---|---|---|
| 95,5 % / 93,0 % / 0,890 macro-F1 (consensus extended) | `03_evaluate.ipynb` cell `44dbf52a` | `v4_e2e_router_eval.json` (ключ `LLM-consensus`, refresh 2026-05-27) + `v4_metric_table_v2.json` |
| 86,7 % / 91,3 % / 0,843 F1 (HUMAN gold) | `03_evaluate.ipynb` cell `495abc8a` | `v4_e2e_router_eval.json` ключ `HUMAN` (live из `python -m src.eval.end_to_end`) |
| 97,2 % router accuracy (n=570) | `03_evaluate.ipynb` cell `router-cell` | `v4_e2e_router_eval.json` ключ `LLM-consensus.router_acc` |
| 4,1 % LLM share (per-layer) | `03_evaluate.ipynb` cell `be873f72` | `v4_e2e_router_eval.json` ключ `LLM-consensus.per_layer_pct.fallback` |
| Per-category fallback (pasta 5,1 / choc 2,6 / cheese 4,9 %) | `03_evaluate.ipynb` cell `percat-l4` | `cascade_preds_{cat}_gold.parquet` |
| Per-attribute micro/macro-F1 (20 строк) | `03_evaluate.ipynb` cell `c539b2a5` | `v4_metric_table_v2.json` ключ `LLM-consensus` |
| Per-category cascade-only (95,8 / 94,9 / 95,8 %) | `03_evaluate.ipynb` cell `t42-percat` | `v4_metric_table_v2.json` агрегирован по cat |
| Macro-precision 0,918 / macro-recall 0,881 | `03_evaluate.ipynb` (рендер inline) | `v4_metric_table_v2.json` per_class блоки |
| Cost-quality scatter (1× / 24× / 580× / 810×) | `03_evaluate.ipynb` cell `cost-quality-cell` | hardcode из `data_methodology.md` §3 + `v4_e2e_router_eval.json` |
| McNemar p=0,74 / 0,49 / 1,00 (router H1) | `03_evaluate.ipynb` cell `c7e260ae` (cell 29) | `router_pareto_gold.parquet` |
| XGBoost / MLP / RF / LogReg сравнение (92,02 / 92,67 / 89,94 / 91,85 %) | `05_method_comparison.ipynb` cell 18 (по индексу — нет ID) | `method_comparison_results.parquet` |
| Brand overlap 82–85 % | вне notebook'ов — verification agent | `defense-prep/2026-05-26-brand-overlap-verification.md` |
| H1 предрегистрация | git commit `cd9ac7a` (2026-05-13) | — |
| Прямые LLM 83,8 % / 69,3 % (Sonnet / gpt-oss) | hardcode в `4-chapter3-implementation.tex:189-192` | `cascade_plus_llm4_v4.parquet` (per-cell raw, агрегаты в TeX) |

### 13.B. Notebook cell → TeX partial (\input)

| Cell | Output partial | Подключается в TeX |
|---|---|---|
| `03_evaluate.ipynb` cell `c539b2a5` | `report/contents/tables/per_attr_consensus.tex` | `4-chapter3-implementation.tex:248` (§3.3.2.3) |
| `03_evaluate.ipynb` cell `percat-l4` | `report/contents/tables/per_category_layer.tex` | `4-chapter3-implementation.tex:268` (§3.3.3.1) |
| `03_evaluate.ipynb` cell `t42-percat` | `report/contents/tables/cascade_per_category.tex` | `5-chapter4-results.tex:94` (§4.3.1 таблица 4.2) |
| `05_method_comparison.ipynb` cell 18 | `report/contents/tables/method_comparison.tex` | `4-chapter3-implementation.tex` §3.3.7.5 |

### 13.C. Картинка → producer

| Картинка | Producer |
|---|---|
| `images/tier_breakdown.png` | `03_evaluate.ipynb` cell `c539b2a5` |
| `images/layer_per_attribute.png` | `src/figures/render_layer_per_attribute.py` (standalone, читает `cascade_preds_*_gold.parquet`) |
| `images/layer_contribution.png` | `03_evaluate.ipynb` cell `be873f72` |
| `images/cost_quality_scatter.png` | `03_evaluate.ipynb` cell `cost-quality-cell` |
| `images/method_comparison_bar.png` | `05_method_comparison.ipynb` cell 11 |
| `images/method_comparison_boxplot.png` | `05_method_comparison.ipynb` cell 13 |
| `images/method_comparison_tradeoff.png` | `05_method_comparison.ipynb` cell 15 |
| `images/fig_2_1_functional_model.png` | вне notebook'ов — drawio/manual asset; cropped 2026-05-27 |
| `images/fig_3_5_bayes_dag.png` | `src/figures/render_bayes_dag.py` (если есть) или manual |
| `images/fig_4_1_demo_ui.png` | screenshot демо-комплекса |

### 13.D. Принцип live regen

Перезапуск `03_evaluate.ipynb` (cells 0..15) или `05_method_comparison.ipynb` (cells 0..18) **полностью** регенерирует все TeX-partials и PNG-картинки из таблиц 13.B/13.C, читая исходные артефакты `v4_e2e_router_eval.json`, `v4_metric_table_v2.json`, `cascade_preds_*_gold.parquet`, `method_comparison_results.parquet`.

**Если v4-артефакты отсутствуют локально** (см. cleanup commit `07fcd04` от 2026-05-25) — нужно либо: (а) подтянуть с VM (158.160.88.176), либо (б) пересобрать через `reproduce.sh` (C-1). Поэтому C-0 / C-1 — критические для верификации.

### 13.E. Cell ID notation

- `03_evaluate.ipynb` — все cells имеют stable string IDs (`c539b2a5`, `percat-l4`, `t42-percat` и т. п.), Jupyter ≥6 формат.
- `05_method_comparison.ipynb` — IDs отсутствуют (старый формат); ссылаемся по cell index. При следующей правке нотбука желательно мигрировать на string IDs (Jupyter автоматически проставит при save в современном окружении).

---

**Дата создания:** 2026-05-26
**Обновлено:** 2026-05-27 (extended 3-LLM consensus rerun, 17062 cells)
**v6 alignment:** числа отражают v6 schema (chocolate.is_filled orthogonal binary; cheeses.texture без `other`; 21 prod / 20 headline). Подтверждено пользователем 2026-05-26: «v6 и canonical — это одно и то же».
**Последняя верификация:** 2026-05-27 (extended consensus rerun + headline updates)
**Defense window:** 2 месяца, ≈2026-07-26
**Hard print deadline:** ≈2026-07-12 (уточнить в деканате)
**Следующая верификация:** после первого critic-agent deep pass (~Day 25)
