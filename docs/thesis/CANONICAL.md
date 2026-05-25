# CANONICAL — Single Source of Truth (VKR Defense Polish)

> **Авторитетный источник** для всех числовых заявлений, схем атрибутов, формулировок стоимости и терминологии в thesis (`report/**/*.tex`), слайдах (`slides/main.tex`) и сопутствующих документах. Создан 2026-05-26 для месячной полировки ВКР.
>
> Любая правка thesis/slides обязана сверяться против этого файла. Несоответствие — ошибка.
>
> Backing source: `docs/thesis/data_methodology.md` §14 + верификация brand-overlap агентом 2026-05-26.

## 1. Headline accuracy

Базовый эталон — **LLM-consensus gold**, n=3257, 20 атрибутов, code-disjoint train/test (не brand-disjoint, см. §6). Консервативный эталон — **human gold** (Opus blind), n=615.

| Метрика | LLM-consensus (n=3257) | Human gold (n=615) | Артефакт |
|---|---|---|---|
| Cascade-only micro-accuracy | **94,8%** | 91,7% | `consensus_gold_v4.parquet` / `v4_eval_human_gold.parquet` |
| Cascade-only macro-F1 (cells-weighted) | **0,899** | 0,848 | (тот же) |
| Cascade-only macro-F1 (attr-unweighted) | 0,902 | — | (тот же) |
| Router accuracy (per code) | 95,4% (n=570) | 95,3% (n=107) | `router_eval_v4.parquet` |
| E2E coverage | 96,0% | 95,4% | (consensus / human gold) |
| **E2E accuracy (None=wrong)** | **91,1%** | **87,5%** | production-realistic |

## 2. LLM fallback distribution

| Слой | Cells (n=3506) | Доля | Назначение |
|---|---|---|---|
| Layer 1 (rule_h, regex по тегам/тексту) | 646 | **18,4%** | High-precision rules |
| Layer 2 (ML: MPNet + TF-IDF SVD + XGBoost) | 2588 | **73,8%** | Главная работа |
| Layer 3 (rule_l, low-precision regex) | 23 | 0,7% | Fallback перед LLM |
| **Layer 4 (LLM fallback)** | **249** | **7,1%** | Сложные / неуверенные cells |

**LLM cost reduction vs naive all-LLM baseline: 92,9%.**

**Per-category fallback rate:**
- pasta: 10,9% (grain_type + pasta_shape dominate)
- chocolate: 2,0%
- cheeses: 6,9%

## 3. Cost framing — три явных термина

В thesis (особенно §2.1.4, §3.3.2, §4.3.2) **всегда** называть тип снижения стоимости одним из трёх:

1. **«Архитектурное сокращение»** = 14× (≈92,9%) — каскад против «one LLM на всё» при ОДНОЙ И ТОЙ ЖЕ модели Layer 4. Чистый эффект каскадной композиции.
2. **«Комбинированное сокращение»** = 333× (Gemini Flash), 14× (Sonnet/каскад), 471× (gpt-oss-120b), 1571× (llama-3b) — каскад против all-Sonnet baseline. Включает удешевление модели + архитектурный вклад.
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
| LLM-consensus gold | 3257 | 3 LLM (qwen3.7-max + deepseek-r1 + mistral-large-2411), правило ≥2/3 | Primary headline |
| Human gold | 615 | Opus 4 blind на спорных | Consensive lower bound |
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
| Cascade-only accuracy | Точность системы на покрытых ячейках (94,8% / 91,7%); upper bound при идеальной маршрутизации категории |
| E2E accuracy | Sigма со всеми ошибками маршрутизации и None=wrong (91,1% / 87,5%); production-realistic |
| Производственная цепочка | = E2E (синоним в Chapter 4) |
| Архитектурное снижение стоимости | Эффект каскадной композиции при той же модели Layer 4 (14×) |
| Комбинированное снижение | Эффект каскад × выбранная модель Layer 4 (14× для каскад+Sonnet, 333× для каскад+Gemini, ...) |
| Циркулярное смещение | Bias ~3,8 пп от семантического перекрытия LLM-консенсуса и обучающих сигналов |

## 11. Стоп-лист — устаревшие числа и формулировки

При любом редактировании `.tex` / `.md` искать и заменять/удалять следующие:

| Устаревшее | Заменить на | Источник устарелости |
|---|---|---|
| `92,8 \%` (как sole headline) | `91,1 \%` E2E / `94,8 \%` cascade-only | post-v4 refactor |
| `720` (как cost factor) | `333` (для каскад+Gemini) / `14` (архитектурное) | post-v4 refactor |
| `4350` cells | `3257` (LLM-consensus) / `615` (human gold) | post-v4 refactor |
| `3,3 \%` LLM | `7,1 \%` LLM | post-v4 refactor |
| `9,0 п. п.` (vs Sonnet) | `7,3 п. п.` (E2E) | post-v4 refactor |
| `23,0 п. п.` (gpt-oss) | `21,3 п. п.` | post-v4 refactor |
| `2,3 п. п.` LLM contribution | `+1,4..+1,7 п. п.` (per-category) | post-v4 refactor |
| `24 поатрибутных XGBoost` | `21 XGBoost (8/6/7)` | post-v4 refactor + schema reform |
| `22 пары` (слайды) | `20 в headline-таблице / 21 в схеме` | uncertainty |
| `выборка без пересечения брендов` | `выборка без пересечения товарных кодов (code-disjoint)` | brand overlap 82–85% |
| `brand-disjoint test` | `code-disjoint test (new SKUs)` | то же |
| `4 НФР` | `5 НФР` | спец §1.3 |
| `4 архитектурных` | `3 архитектурных` | спец §1.3 |
| `96,7 \%` (старое каскад coverage) | `96,0 \%` E2E coverage | post-v4 |
| `90,5 \%` (старая средняя точность) | (зависит от контекста — обычно `94,8 \%` cascade-only) | post-v4 |

## 12. Известные ограничения (для §«Ограничения» conclusion)

1. **Brand overlap 82–85%.** Эталон — code-disjoint, не brand-disjoint. Sensitivity на brand-disjoint subset вынесена в §3.3.7 (C-4).
2. **Circular bias ~3,8 пп.** Обучающий silver частично пересекается семантически с consensus gold (оба — LLM-derived). Human gold (87,5% E2E) даёт independent lower bound.
3. **Sample size человеческого эталона.** n=615, single labeler (Opus 4), без IRR.
4. **Calibration не в production.** ECE 0,070 → 0,043 при isotonic CV на gold проверена (§3.3.3.3); не включена в production.
5. **Bayes в production — на 1 атрибуте.** Из 20 пар селективный сигнал работает только на chocolate/contains_nuts. Остальные — flat либо отрицательный прирост.
6. **Языковые слабые места.** cheeses/texture/es (−47 пп от лучшего), pasta_shape/de и др. — четыре конкретных hot-spot'а в §3.3.7.1.
7. **Cross-domain validity** ограничена food. Electronics добавлена как cross-domain control (§3.4 после E-3), но не как полноценная replication.
8. **Lic.** Open Food Facts ODbL — требует attribution share-alike в производных.

## 13. Артефакты, на которые опираются числа

| Число | Артефакт |
|---|---|
| 94,8 / 91,1 / 0,899 (consensus) | `datasets/processed/consensus_gold_v4.parquet` + `headline_v4_*` (если v4 нет — fallback `headline_v3e_final.parquet` + manual recompute) |
| 87,5 / 91,7 (human gold) | `datasets/processed/v4_eval_human_gold.parquet` |
| 95,4 router | `datasets/processed/router_eval_v4.parquet` |
| Brand overlap 82–85% | computed 2026-05-26 верификационным агентом (отчёт в `defense-prep/2026-05-26-brand-overlap-verification.md` — TODO Phase 1) |
| H1 cd9ac7a | git commit cd9ac7a |
| 7,1% LLM | `headline_v4_*` или recomputed |

**Если v4-артефакты отсутствуют локально** (см. cleanup commit `07fcd04` от 2026-05-25) — нужно либо: (а) подтянуть с VM (158.160.88.176), либо (б) пересобрать через `reproduce.sh` (C-1). Поэтому C-0 / C-1 — критические для верификации.

---

**Дата создания:** 2026-05-26
**Последняя верификация:** 2026-05-26 (brand-overlap агент)
**Следующая верификация:** после закрытия Phase 1 (A-1, A-3)
