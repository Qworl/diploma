# Phase 0 — Infrastructure Implementation Plan

> **ARCHIVE / SUPERSEDED:** Этот документ — снимок плана инфраструктуры на момент создания CANONICAL.md. **Цифры внутри (91,7 % / 87,5 % / n=615) устарели после drift-fix 2026-05-26** (см. `docs/po/tickets/2026-05-26-human-gold-drift.md`). Актуальные headline-числа — в `docs/thesis/CANONICAL.md` (91,3 % cascade-only / 86,7 % E2E / n=566 cascade-valid human gold).
>
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Развернуть persistent infrastructure (source-of-truth, backlog, decisions, slash command, memory, CLAUDE.md обновление) для месячного PO-цикла полировки ВКР.

**Architecture:** Шесть документов в `docs/thesis/` и `docs/thesis/defense-prep/` + одна slash-команда + одна memory-запись + обновление `CLAUDE.md`. Никакого кода в проектном `src/`. Цель — чтобы любая будущая сессия (включая после context compaction) могла подхватить работу без потери контекста.

**Tech Stack:** Markdown, simple file I/O, git.

**Spec:** [`2026-05-26-final-polish-design.md`](../2026-05-26-final-polish-design.md)

**Покрываемые items спека:** F-0, F-1, F-2, F-3, F-4 (P0). F-5 (critic cron, P1) — отложен на Phase 1 Day 3.

---

## File Structure

| Файл | Создаётся / правится | Роль |
|---|---|---|
| `docs/thesis/CANONICAL.md` | create | Single source of truth для чисел / терминов / стоп-листа |
| `docs/thesis/defense-prep/STATE.md` | create | Live operational state: что закрыто, что в работе, что блокировано |
| `docs/thesis/defense-prep/BACKLOG.md` | create | Приоритизированный backlog всех item'ов спека |
| `docs/thesis/defense-prep/RETROSPECTIVES.md` | create | Phase-level ретроспективы для обучения по проекту |
| `docs/thesis/defense-prep/DECISIONS.md` | create | ADR-style реестр решений (стартовая запись — 3 принципа схемы) |
| `.claude/commands/next.md` | create | Slash-команда `/next` для PO-loop'а |
| `CLAUDE.md` | modify (append) | Добавить раздел «VKR Defense Polish» + «PO Mode» |
| `/Users/miafrolov/.claude/projects/-Users-miafrolov-Desktop-stuff-ai-attributes/memory/vkr_canonical_2026.md` | create | Memory entry с canonical-цифрами для подгрузки в будущих сессиях |
| `/Users/miafrolov/.claude/projects/-Users-miafrolov-Desktop-stuff-ai-attributes/memory/MEMORY.md` | modify (append) | Один index-line к новому memory-файлу |

---

## Task 1: Create `docs/thesis/CANONICAL.md` (F-0)

**Files:**
- Create: `docs/thesis/CANONICAL.md`

- [ ] **Step 1: Write CANONICAL.md**

Write the following exact content to `docs/thesis/CANONICAL.md`:

```markdown
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
| Brand overlap 82–85% | computed 2026-05-26 verification агентом, протокол в `defense-prep/2026-05-26-brand-overlap-verification.md` (создать в Phase 1) |
| H1 cd9ac7a | git commit cd9ac7a |
| 7,1% LLM | `headline_v4_*` или recomputed |

**Если v4-артефакты отсутствуют локально** (см. cleanup commit `07fcd04` от 2026-05-25) — нужно либо: (а) подтянуть с VM (158.160.88.176), либо (б) пересобрать через `reproduce.sh` (C-1). Поэтому C-0 / C-1 — критические для верификации.

---

**Дата создания:** 2026-05-26
**Последняя верификация:** 2026-05-26 (brand-overlap агент)
**Следующая верификация:** после закрытия Phase 1 (A-1, A-3)
```

- [ ] **Step 2: Verify file exists**

Run: `ls -la docs/thesis/CANONICAL.md`
Expected: file exists, size >5KB.

- [ ] **Step 3: Commit**

```bash
git add docs/thesis/CANONICAL.md
git commit -m "$(cat <<'EOF'
docs(canonical): single source of truth for VKR polish

All headline numbers, schemas, cost framings, stop-list of deprecated
numbers, terminology, and known limitations consolidated from
data_methodology.md §14 + brand-overlap verification (2026-05-26).

Used as canonical input for every subsequent edit to report/**/*.tex
and slides/main.tex.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Create `docs/thesis/defense-prep/STATE.md` (F-1.1)

**Files:**
- Create: `docs/thesis/defense-prep/STATE.md`

- [ ] **Step 1: Write STATE.md**

Write the following exact content to `docs/thesis/defense-prep/STATE.md`:

```markdown
# STATE — VKR Defense Polish

> Live operational state. Обновляется после каждого закрытого item.
>
> Spec: [2026-05-26-final-polish-design.md](2026-05-26-final-polish-design.md)
> Canonical: [../CANONICAL.md](../CANONICAL.md)
> Backlog: [BACKLOG.md](BACKLOG.md)

## Текущая фаза

**Phase 0** — инфраструктура (F-0..F-4). Стартовано 2026-05-26.

## In progress

— (заполняется по мере работы)

## Только что закрыто (последние 5)

— (заполняется)

## Заблокировано

— (none)

## Ключевые анкеры

- **Day 1:** 2026-05-27
- **Hard print deadline:** ≈Day 17 (2026-06-12) — уточнить в деканате
- **Advisor pre-review gate (D-4):** Day 15 (~2026-06-10)
- **Defense:** ≈Day 31 (2026-06-26)

## Текущие headline-числа

(Mirror из CANONICAL.md §1 для быстрого reference)

| Метрика | LLM-consensus (n=3257) | Human gold (n=615) |
|---|---|---|
| Cascade-only micro | 94,8% | 91,7% |
| Cascade macro-F1 | 0,899 | 0,848 |
| E2E (None=wrong) | **91,1%** | **87,5%** |
| LLM fallback | 7,1% | — |

## Phase progress

- [ ] **Phase 0** (Days 1–2): F-0..F-4 — инфраструктура
- [ ] **Phase 1** (Days 3–6): A-0..A-5, A-19, B-0..B-8, D-0 — canonical alignment + advisor comments
- [ ] **Phase 2** (Days 7–11): A-6..A-8, B-9..B-14, C-0..C-2, D-1..D-2, E-0..E-1
- [ ] **Phase 3** (Days 12–14): A-16..A-18, C-3..C-4, E-2..E-3, D-3
- [ ] **D-4 ADVISOR GATE** (Day 15)
- [ ] **Phase 4** (Days 16–17): apply advisor edits, A-9..A-15, E-4
- [ ] **HARD DEADLINE — Day 17**
- [ ] **Phase 5** (Days 18–28): slides finalization, mock defense v2, repro verify
- [ ] **Phase 6** (Days 29–31): final polish, buffer
- [ ] **DEFENSE — Day 31**

## Open blockers (на пользователя)

| # | Блокер | Статус |
|---|---|---|
| 1 | Hard print deadline | Уточнить в деканате Day 1 |
| 2 | pptx vs latex для замечания научрука #7 | По умолчанию latex; при уточнении возможен пересмотр |
```

- [ ] **Step 2: Commit**

```bash
git add docs/thesis/defense-prep/STATE.md
git commit -m "docs(state): live operational state skeleton for Phase 0"
```

---

## Task 3: Create `docs/thesis/defense-prep/BACKLOG.md` (F-1.2)

**Files:**
- Create: `docs/thesis/defense-prep/BACKLOG.md`

- [ ] **Step 1: Write BACKLOG.md**

Write the following exact content:

```markdown
# BACKLOG — VKR Defense Polish

> Приоритизированный backlog. Источник item'ов — [spec §4](2026-05-26-final-polish-design.md).
>
> Приоритеты: **P0** блокер, **P1** сильно усиливает, **P2** nice-to-have, **P3** пропускаемо.

## P0 (блокирует защиту)

### Workstream F — Infrastructure
- [ ] F-0 CANONICAL.md (Phase 0)
- [ ] F-1 STATE / BACKLOG / RETROSPECTIVES / DECISIONS (Phase 0)
- [ ] F-2 /next slash command (Phase 0)
- [ ] F-3 CLAUDE.md PO mode block (Phase 0)
- [ ] F-4 Memory entry vkr_canonical_2026.md (Phase 0)

### Workstream A — Text (canonical alignment)
- [ ] A-0 Brand-disjoint → code-disjoint в `.tex` (Phase 1)
- [ ] A-1 Sync headline-чисел в abstract / intro / conclusion (Phase 1)
- [ ] A-2 ТЗ count alignment 6/5/3 (Phase 1)
- [ ] A-3 Число атрибутов: 20 в headline / 21 в схеме (Phase 1)
- [ ] A-4 Cost framing — 3 явных термина (Phase 1)
- [ ] A-5 §«Ограничения» расширить (Phase 1)
- [ ] A-16 Full thesis re-read (Phase 3)
- [ ] A-19 Bayes honest framing (1 attribute) (Phase 1)

### Workstream B — Slides P0
- [ ] B-0 Sync чисел из CANONICAL (Phase 1)
- [ ] B-1 Real demo screenshots (Phase 1, depends C-2)
- [ ] B-2 Bayes DAG in main deck (Phase 1)
- [ ] B-3 ТЗ count F1–F6 / N1–N5 / A1–A3 (Phase 1)
- [ ] B-4 «Цель и задачи» — 5 задач (Phase 1)
- [ ] B-5 «Заключение» таблица — 5 строк (Phase 1)
- [ ] B-6 Формальная постановка — в буллет (Phase 1)
- [ ] B-7 ГОСТ-контекст со слайда 14 (Phase 1)
- [ ] B-8 Аналоги — убрать спорные ✓ (Phase 1)

### Workstream C — Code / Repro / Demo P0
- [ ] C-0 verify_numbers.py (Phase 2)
- [ ] C-1 reproduce.sh audit + rebuild (Phase 2)
- [ ] C-2 Real demo screenshots + recording (Phase 2)

### Workstream D — Q&A P0
- [ ] D-0 Initial 8–10 черновых вопросов (Phase 1)
- [ ] D-4 Advisor pre-review gate (Day 15)

## P1 (сильно усиливает)

- [ ] A-6 Random Forest abzac (Phase 2)
- [ ] A-7 Bayes DAG в текст §2.2.5 + §3.3.4 (Phase 2)
- [ ] A-8 «Апробация и внедрение» через integration-readiness (Phase 2)
- [ ] A-17 Print-ready proofread (Phase 3)
- [ ] A-18 Bibliography audit (Phase 3)
- [ ] B-9 «1 классификатор на атрибут (21)» в слайд алгоритмов (Phase 2)
- [ ] B-10 Backup B8: multitask эксперимент (Phase 2)
- [ ] B-11 B7 Electronics — real numbers (Phase 2/3, depends E-1)
- [ ] B-12 B5 Многоязычность — мини-таблица (Phase 2)
- [ ] B-13 Шрифты ≥10pt (Phase 5)
- [ ] B-14 Speaker notes — обновить (Phase 5)
- [ ] C-3 RF baseline empirical (Phase 3)
- [ ] C-4 Brand-disjoint subset sensitivity (Phase 3)
- [ ] C-5 Electronics PhoneDB pipeline (Phase 2/3)
- [ ] D-1 15 вопросов с pointer'ами (Phase 2)
- [ ] D-2 Verified answers (Phase 2)
- [ ] D-3 Mock defense (Phase 3 + Phase 5)
- [ ] E-0 PhoneDB gold construction (Phase 2)
- [ ] E-1 Electronics cascade eval (Phase 2)
- [ ] E-2 Bayes-on-electronics replication (Phase 3)
- [ ] E-3 §3.4 текст в thesis (Phase 3)
- [ ] E-4 Update abstract / intro / conclusion (Phase 4)
- [ ] F-5 Critic cron daily + event-triggered (Day 3+)

## P2 (nice-to-have)

- [ ] A-9 Обновить таблицу аналогов §1.2.3 (Phase 4)
- [ ] A-10 Confusion matrices трудных атрибутов (Phase 4)
- [ ] A-11 Language heatmap §3.3.7.1 (Phase 4)
- [ ] A-12 TCO в $ §4.3.2 (Phase 4)
- [ ] A-13 Глоссарий (Phase 4)
- [ ] A-14 ODbL атрибуция OFF (Phase 4)
- [ ] D-5 Speaker notes на презентации (Phase 5)

## P3 (пропускаемо)

- [ ] A-15 Поднять число рисунков до ≥12 (Phase 4 buffer)
- [ ] C-6 Pre-commit hook на verify_numbers.py (отложен)

## Закрыто

— (заполняется по мере работы; формат: `- [x] X-N описание (YYYY-MM-DD, commit <hash>)`)
```

- [ ] **Step 2: Commit**

```bash
git add docs/thesis/defense-prep/BACKLOG.md
git commit -m "docs(backlog): initialize prioritized backlog from spec"
```

---

## Task 4: Create `docs/thesis/defense-prep/RETROSPECTIVES.md` (F-1.3)

**Files:**
- Create: `docs/thesis/defense-prep/RETROSPECTIVES.md`

- [ ] **Step 1: Write RETROSPECTIVES.md**

```markdown
# RETROSPECTIVES — VKR Defense Polish

> Phase-level retros: что сработало, что упустил, что делать иначе.
>
> Формат: каждая запись 5–10 строк, в конце phase'ы / при крупных открытиях.

## 2026-05-26 — Pre-Phase 0 critique

Что упало в первое review этой работы:
- Все headline-числа рассогласованы (abstract 92,8% vs Chapter 3 91,1%).
- «Brand-disjoint» в thesis vs реальный code-disjoint (brand overlap 82–85%, проверено агентом).
- ТЗ описывается разными способами (6/5/3 vs 6/4/3).
- Число атрибутов плавает (20 / 21 / 22 / 24).
- 7 пунктов замечаний научрука не отработаны.
- Demo-скриншоты в слайдах помечены TODO.
- `reproduce.sh` существует, но содержимое не верифицировано.

**Lesson:** между artifactами (parquet'ы, methodology doc) и thesis накопился drift. Нужен single source of truth + verify-script + daily critic. PO-loop инфраструктура — необходимое условие.

## Phase 0 — closure entry будет добавлен в Task 10 Step 3
```

- [ ] **Step 2: Commit**

```bash
git add docs/thesis/defense-prep/RETROSPECTIVES.md
git commit -m "docs(retro): seed with pre-Phase 0 critique findings"
```

---

## Task 5: Create `docs/thesis/defense-prep/DECISIONS.md` (F-1.4)

**Files:**
- Create: `docs/thesis/defense-prep/DECISIONS.md`

- [ ] **Step 1: Write DECISIONS.md**

```markdown
# DECISIONS — VKR Defense Polish

> ADR-style реестр архитектурных решений с обоснованием.
>
> Формат каждой записи: ID / дата / решение / альтернативы / обоснование / последствия.

## D-001: Per-product code-disjoint split (не brand-disjoint)

**Дата:** 2026-05-26 (зафиксировано постфактум).
**Решение:** train/test split выполняется по идентификатору товара (code), не по бренду. Brand overlap 82–85%.
**Альтернативы:** (а) полноценный brand-disjoint через `src/data/split/brand_disjoint.py` — требует переразметки gold (~2 недели работы); (б) hybrid с brand-stratification.
**Обоснование:** gold собран дорого через LLM-consensus на v4 (~$40 на 60k через VM). Переразметка нереальна за месяц. Code-disjoint даёт честную проверку обобщения на новые SKU; brand-disjoint sensitivity-check на 17/19/28 кодов остаётся как доп. проверка (C-4).
**Последствия:** все формулировки «brand-disjoint» в thesis заменяются на «code-disjoint» (A-0). Это слабее заявление, но честное.

## D-002: Bayes как селективный валидатор, не классификатор

**Дата:** ~2026-05-20.
**Решение:** байесовская сеть в production включена только на chocolate/contains_nuts. На остальных 19 атрибутах либо отключена, либо порог такой, что не отсекает.
**Альтернативы:** (а) убрать Bayes-слой полностью; (б) сохранить классификаторную роль.
**Обоснование:** классификаторная точность Bayes на трудном остатке после Layer 2 — ~52% (ниже majority baseline). На остальных парах атрибутов селективность отбраковки не превосходит базовый уровень.
**Последствия:** тема ВКР заявляет «отбраковку фактических ошибок» — Bayes формально это делает на одной паре. Honest framing в §3.3.4 + §«Ограничения» (A-19) обязательно.

## D-003: Принцип ортогональности атрибутов

**Дата:** ~2026-05-25 (v6 schema refactor).
**Решение:** свойства, проявляющиеся у товара независимо, выделены в отдельные атрибуты, а не объединены в multiclass-домен.
**Канонический пример:** `chocolate.is_filled` выделен как binary, отделён от `chocolate_type` (3-class {dark, milk, white}) и `chocolate_extra` (5-class mix-ins).
**Обоснование:** объединение даёт macro-F1 0,50 за счёт catch-all класса `filled` с recall 0,33. Раздельные атрибуты: type 0,973, is_filled 0,861, extra 0,899.
**Источник:** `docs/thesis/data_methodology.md` §5.1.

## D-004: Принцип отказа от мёртвых классов

**Дата:** ~2026-05-25.
**Решение:** класс multiclass-атрибута удаляется при выполнении ВСЕХ условий: n_train < 100, recall < 0,20 на independent eval, semantic catch-all без устойчивого паттерна, нет downstream-зависимости.
**Применённые удаления:** `chocolate_type.other`, `chocolate_extra.{other,with_alcohol,with_coffee}`, `chocolate.flavor_profile.other`, `cheeses.texture.other`. `chocolate_type.filled` удалён (покрыт `is_filled`).
**Обоснование:** см. data_methodology.md §5.2.
**Двойной фильтр:** `exclude_classes` в train + `SCHEMA_EXCLUDE` в eval (одно описание).

## D-005: Принцип консервативной агрегации rule-based слоёв

**Дата:** ~2026-05-25.
**Решение:** правило в `rule_h` допускается только при однозначности ответа во всех контекстах. Ambiguous паттерны (e.g., `Coeur de Savoie` — soft|hard в зависимости от производителя) переданы на ML-слой.
**Обоснование:** см. data_methodology.md §5.3.
**Последствия:** на cheeses Layer 1 покрытие низкое (~3%), но точность 100% (47/47).

## D-006: PO-цикл с инфраструктурой single source of truth

**Дата:** 2026-05-26.
**Решение:** для месячной полировки разворачивается persistent infrastructure (CANONICAL.md, STATE, BACKLOG, RETROSPECTIVES, DECISIONS, /next, CLAUDE.md PO block, memory entry, critic cron daily).
**Альтернативы:** ad-hoc правки по чек-листу.
**Обоснование:** drift между artifactами и thesis уже накопился значительно. Без SoT каждая правка рискует ввести новое рассогласование. PO-loop позволяет менять приоритеты по итогам каждого закрытого item'а.
**Последствия:** Phase 0 уходит на инфраструктуру (1–2 дня) перед началом контентной правки.
```

- [ ] **Step 2: Commit**

```bash
git add docs/thesis/defense-prep/DECISIONS.md
git commit -m "docs(decisions): seed ADR registry with 6 foundational decisions"
```

---

## Task 6: Create `.claude/commands/next.md` (F-2)

**Files:**
- Create: `.claude/commands/next.md`

- [ ] **Step 1: Verify `.claude/commands/` directory exists or create it**

Run: `mkdir -p .claude/commands`

- [ ] **Step 2: Write slash command file**

Write the following to `.claude/commands/next.md`:

```markdown
---
description: PO review — find next most-valuable thing to do for VKR polish
---

Запустить PO-loop цикл по проекту полировки ВКР. Алгоритм:

## 1. Загрузить контекст

Прочитай в указанном порядке:

1. `docs/thesis/CANONICAL.md` — single source of truth (числа, термины, стоп-лист)
2. `docs/thesis/defense-prep/STATE.md` — текущее состояние, фаза, что в работе
3. `docs/thesis/defense-prep/BACKLOG.md` — приоритизированный список item'ов
4. `docs/thesis/defense-prep/2026-05-26-final-polish-design.md` — спек (только §5 sequencing + §12 fallback)
5. `docs/thesis/defense-prep/2026-05-25-advisor-comments.md` — комментарии руководителя
6. Запусти `git log --since="3 days ago" --oneline` — что произошло

## 2. Оценить ситуацию

- Если в STATE.md есть `in_progress` item — отчитайся о прогрессе, предложи следующий конкретный шаг внутри него.
- Если последний коммит закрыл какой-то item — пометь его как закрытый в BACKLOG.md (`- [x] X-N ... (YYYY-MM-DD, commit <hash>)`), добавь 2–3-строчный entry в RETROSPECTIVES.md, обнови STATE.md.
- Если все P0 в текущей фазе закрыты — переходи к следующей фазе по sequencing §5 спека.
- Если backlog пуст в текущем P-уровне → подняться на уровень ниже (P0 → P1 → P2) либо запустить **deep critique pass**:
  - Прочитать целевой раздел thesis (по контексту последних правок) или весь thesis при пустом backlog
  - Найти ≥3 новые проблемы (рассогласования, устаревшие числа, слабые формулировки, упущенные ограничения)
  - Записать в BACKLOG.md с приоритетом
  - Записать в RETROSPECTIVES.md что нашли и почему упустили раньше

## 3. Предложить следующий шаг

В конце одно сообщение пользователю:

1. **Текущее состояние:** какая фаза, какой item открыт, сколько % фазы закрыто.
2. **Что предлагаю сделать дальше:** конкретный item с ID, его описание, оценка времени.
3. **Почему именно это:** по приоритету / по последовательности / по риску.
4. **Что нужно от пользователя** (если что-то блокирует): явное указание блокера или «ничего, начинаю».

## 4. Если auto mode активен

Не жди подтверждения — начинай выполнение item'а, если он не требует человеческого решения (например, не A-8 без реквизитов работодателя).

## 5. После каждого item

- Обнови STATE.md
- Поставь галочку в BACKLOG.md
- Запусти `python scripts/verify_numbers.py` (когда он будет создан в C-0)
- Если изменения P0 — commit и report
```

- [ ] **Step 3: Commit**

```bash
git add .claude/commands/next.md
git commit -m "feat(commands): /next slash command for PO review loop"
```

---

## Task 7: Update `CLAUDE.md` with PO Mode section (F-3)

**Files:**
- Modify: `CLAUDE.md` (append at end)

- [ ] **Step 1: Append PO Mode section to CLAUDE.md**

Read existing `CLAUDE.md` first (already loaded in context), then append the following block at the very end of the file:

```markdown

## VKR Defense Polish (active until ~2026-06-26)

**Spec:** `docs/thesis/defense-prep/2026-05-26-final-polish-design.md`
**Canonical (source of truth):** `docs/thesis/CANONICAL.md`
**Live state:** `docs/thesis/defense-prep/STATE.md`
**Backlog:** `docs/thesis/defense-prep/BACKLOG.md`
**Decisions:** `docs/thesis/defense-prep/DECISIONS.md`
**Retrospectives:** `docs/thesis/defense-prep/RETROSPECTIVES.md`

### Whenever editing thesis or slides
1. Cross-check all numbers against `CANONICAL.md` before writing. Treat `CANONICAL.md` as the only authoritative source for headline numbers, attribute counts, cost framings, and split type (code-disjoint, not brand-disjoint).
2. Forbidden substrings (stop-list — see CANONICAL §11): `92,8 \%` as headline, `720`-cost factor, `4350` cells, `3,3 \%` LLM, `24 поатрибутных XGBoost`, `4 НФР`, `brand-disjoint test`, `выборка без пересечения брендов`.
3. After edit, run `python scripts/verify_numbers.py` (when it exists — C-0). Exit code 0 required before commit.
4. If you discover a stale number, ALSO log it in `RETROSPECTIVES.md` under «Phase X / late catches».

### PO Mode (operating principle)

You operate as product owner, not just executor. After completing any meaningful unit of work:
1. Update `STATE.md` with what's closed.
2. Add a 5–10 line retrospective to `RETROSPECTIVES.md` if Phase boundary or non-trivial discovery.
3. Re-scan `BACKLOG.md` and propose next-most-valuable item.
4. If nothing in backlog scores P0/P1, run a deep critique pass before saying done.

Never declare "all done" without a critique pass — there's always one more thing worth checking.

### Standing inputs before proposing next step

- `docs/thesis/defense-prep/STATE.md`
- `docs/thesis/defense-prep/BACKLOG.md`
- `docs/thesis/defense-prep/2026-05-25-advisor-comments.md`
- `git log --since="3 days ago"` (recent changes)
- `report/notes/` newest 3 files (any incoming critique)

### Auto behaviors

- Critic cron (when set up in F-5): runs once daily 20:00 gated by `git log --since="24h"`, plus event-triggered after closing any P0 item. Writes to `defense-prep/critic-YYYY-MM-DD.md`.
- Slash command `/next` triggers PO-loop manually any time.
```

- [ ] **Step 2: Verify append succeeded**

Run: `tail -50 CLAUDE.md`
Expected: last lines show the new VKR Defense Polish section.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude): add VKR Defense Polish + PO Mode block

Points every future session to canonical numbers, backlog, decisions.
Defines stop-list for forbidden stale strings + PO-loop operating principle.
"
```

---

## Task 8: Create memory entry + update MEMORY.md index (F-4)

**Files:**
- Create: `/Users/miafrolov/.claude/projects/-Users-miafrolov-Desktop-stuff-ai-attributes/memory/vkr_canonical_2026.md`
- Modify: `/Users/miafrolov/.claude/projects/-Users-miafrolov-Desktop-stuff-ai-attributes/memory/MEMORY.md`

- [ ] **Step 1: Write memory file**

Write the following to `/Users/miafrolov/.claude/projects/-Users-miafrolov-Desktop-stuff-ai-attributes/memory/vkr_canonical_2026.md`:

```markdown
---
name: VKR Canonical Numbers 2026 (active until defense ~2026-06-26)
description: Single source of truth for all headline numbers, schemas, cost framings, and stop-list of deprecated numbers in the VKR. Mirror of docs/thesis/CANONICAL.md. Use when editing any thesis text or slides.
type: project
---

# VKR Canonical 2026

**Spec:** `docs/thesis/defense-prep/2026-05-26-final-polish-design.md`
**Full canonical:** `docs/thesis/CANONICAL.md`

## Headline (LLM-consensus gold, n=3257, code-disjoint)

- Cascade-only micro: **94,8%** (macro-F1 0,899)
- Router accuracy: 95,4% (n=570)
- E2E coverage: 96,0%
- **E2E (None=wrong): 91,1%** ← production-realistic

## Conservative bound (human gold, n=615, Opus blind)

- Cascade-only: 91,7%
- **E2E: 87,5%**

## LLM fallback

- Layer 1 regex: 18,4% (646 cells)
- Layer 2 ML (MPNet+TF-IDF+XGB): **73,8%** (2588)
- Layer 3 rule_l: 0,7% (23)
- Layer 4 LLM: **7,1%** (249)
- Cost reduction vs all-LLM: **92,9% (≈14× архитектурное)**

## Cost framings (always specify which one!)

- **Архитектурное снижение** = 14× — каскад vs all-Sonnet при той же модели Layer 4
- **Комбинированное** = 333× (каскад+Gemini), 14× (каскад+Sonnet), 471× (каскад+gpt-oss), 1571× (каскад+llama-3b)
- **Эффективное (vs прямая модель)** — архитектурное × выбранная модель Layer 4

## Schema (21 attrs, headline table 20)

- pasta: 8 — grain_type, pasta_shape, is_filled, is_gluten_free, is_organic, is_vegan, cuisine_origin, protein_class*
- chocolate: 6 — chocolate_type, is_filled, chocolate_extra, contains_nuts, is_organic, flavor_profile
- cheeses: 7 — milk_source, texture, country_of_origin, aging, is_pdo, is_organic, is_ultra_processed
- (* protein_class опциональный, в headline-таблице 20 не включён)

## ТЗ counts

- 6 ФР / 5 НФР / 3 А (Ф-1..Ф-6, НФ-1..НФ-5, А-1..А-3)

## Split — code-disjoint, не brand-disjoint!

Brand overlap train↔eval: pasta 82,5% / chocolate 84,9% / cheeses 82,7%. Никогда не писать «brand-disjoint» или «без пересечения брендов» — это снято решением D-001.

## Стоп-лист (forbidden stale strings)

- 92,8% as headline → 91,1% / 94,8%
- 720× cost → 14× архитектурное / 333× комбинированное
- 4350 cells → 3257 / 615
- 3,3% LLM → 7,1%
- 24 поатрибутных XGBoost → 21
- 4 НФР → 5 НФР
- brand-disjoint → code-disjoint

## Известные ограничения

1. Brand overlap 82–85% (code-disjoint, не brand-disjoint)
2. Circular bias ~3,8 пп (LLM-consensus частично сем-перекрывается с silver)
3. Sample size human gold n=615, single labeler Opus 4
4. Calibration не в production (проверена, не включена)
5. Bayes в production только на chocolate/contains_nuts (1 из 20)
6. Language hot-spots: cheeses/texture/es (−47 пп) и 3 других
7. Cross-domain (food only); electronics — control, не replication

**Backing source:** `docs/thesis/data_methodology.md` §14 + brand-overlap верификация агентом 2026-05-26.
```

- [ ] **Step 2: Read existing MEMORY.md index**

Run: `head -10 /Users/miafrolov/.claude/projects/-Users-miafrolov-Desktop-stuff-ai-attributes/memory/MEMORY.md`

- [ ] **Step 3: Append index line to MEMORY.md**

Use Edit tool to add the following line at the TOP of the Index section in MEMORY.md (after `## Index`):

```markdown
- [VKR Canonical Numbers 2026 — ACTIVE](vkr_canonical_2026.md) — single source of truth for headline numbers, schemas, stop-list; load whenever editing thesis/slides.
```

The line must appear FIRST under `## Index` (highest priority — это ACTIVE working SoT).

- [ ] **Step 4: Verify**

Run: `head -5 /Users/miafrolov/.claude/projects/-Users-miafrolov-Desktop-stuff-ai-attributes/memory/MEMORY.md && echo "---" && ls /Users/miafrolov/.claude/projects/-Users-miafrolov-Desktop-stuff-ai-attributes/memory/vkr_canonical_2026.md`
Expected: index line visible, file exists.

- [ ] **Step 5: No commit needed**

Memory directory is outside repo. No git action.

---

## Task 9: Manual verification of /next slash command

**Files:** none (verification only)

- [ ] **Step 1: Notify user that /next is now available**

Print message: «Slash command `/next` создана. Триггер вручную в любой сессии. В этой сессии можно протестировать, набрав `/next` в следующем сообщении (опционально).»

- [ ] **Step 2: No file change, no commit.**

This task is a checkpoint, not work.

---

## Task 10: Close Phase 0

**Files:**
- Modify: `docs/thesis/defense-prep/STATE.md`
- Modify: `docs/thesis/defense-prep/BACKLOG.md`
- Modify: `docs/thesis/defense-prep/RETROSPECTIVES.md`

- [ ] **Step 1: Mark Phase 0 as closed in STATE.md**

Edit `docs/thesis/defense-prep/STATE.md`:

Change line:
```
- [ ] **Phase 0** (Days 1–2): F-0..F-4 — инфраструктура
```
to:
```
- [x] **Phase 0** (Days 1–2): F-0..F-4 — инфраструктура ✅ closed 2026-05-26
```

Change «Текущая фаза» from:
```
**Phase 0** — инфраструктура (F-0..F-4). Стартовано 2026-05-26.
```
to:
```
**Phase 1** — canonical alignment + advisor comments + start Q&A. Стартует 2026-05-27.
```

- [ ] **Step 2: Mark F-0..F-4 closed in BACKLOG.md**

Edit `docs/thesis/defense-prep/BACKLOG.md`:

For each of F-0, F-1, F-2, F-3, F-4 — change `- [ ]` to `- [x]` and append ` (2026-05-26, Phase 0 closure)`.

Add to «Закрыто» section:
```
- [x] F-0 CANONICAL.md (2026-05-26, commit <hash>)
- [x] F-1 STATE / BACKLOG / RETROSPECTIVES / DECISIONS (2026-05-26)
- [x] F-2 /next slash command (2026-05-26)
- [x] F-3 CLAUDE.md PO mode block (2026-05-26)
- [x] F-4 Memory entry vkr_canonical_2026.md (2026-05-26)
```

- [ ] **Step 3: Write Phase 0 retrospective**

Append to `docs/thesis/defense-prep/RETROSPECTIVES.md` after «## Phase 0 — (закрыть при завершении)»:

```markdown
## Phase 0 closure — 2026-05-26

**Что сделано:**
- F-0 CANONICAL.md создан, в нём 13 секций (headline / fallback / cost framings / schema / эталоны / split / H1 / ТЗ / models / терминология / стоп-лист / ограничения / артефакты)
- F-1 четыре документа (STATE / BACKLOG / RETROS / DECISIONS) seeded из спека и data_methodology.md
- F-2 slash command `/next` запущена
- F-3 CLAUDE.md дополнен PO mode + VKR Polish блоком
- F-4 memory entry vkr_canonical_2026.md + индекс-строка в MEMORY.md

**Что сработало:**
- Spec → plan → atomic tasks дал чёткий ход без блоков.
- Brand-overlap верификация агентом подтвердила направление §12.2 doc'а и зафиксировала точные цифры до старта правок.

**Что упустил / делать иначе:**
- Не дополнил CANONICAL.md ссылкой на actual brand-overlap verification report — нужно создать `2026-05-26-brand-overlap-verification.md` в Phase 1 как backing artifact.
- F-5 (critic cron) отложен на Day 3 — стоит проверить, что инфраструктура `CronCreate` действительно доступна.

**Lessons for Phase 1:**
- Перед началом content-правок (A-0..A-5) — один сквозной diff sweep: «какие .tex / .md содержат строки из стоп-листа?», чтобы не пропускать места.
- Speaker notes для слайдов будут отставать — лучше начать тонкий feeds туда параллельно.
```

- [ ] **Step 4: Commit Phase 0 closure**

```bash
git add docs/thesis/defense-prep/STATE.md docs/thesis/defense-prep/BACKLOG.md docs/thesis/defense-prep/RETROSPECTIVES.md
git commit -m "docs(state): close Phase 0 — infrastructure complete

F-0..F-4 done. STATE / BACKLOG / RETROSPECTIVES updated.
Next: Phase 1 (Days 3-6) — A-0 brand-disjoint fix first.
"
```

---

## Phase 0 Acceptance Criteria

- [ ] `docs/thesis/CANONICAL.md` exists with all 13 sections
- [ ] `docs/thesis/defense-prep/STATE.md` exists, Phase 0 marked closed
- [ ] `docs/thesis/defense-prep/BACKLOG.md` exists with full prioritized backlog
- [ ] `docs/thesis/defense-prep/RETROSPECTIVES.md` has pre-Phase 0 + Phase 0 closure entries
- [ ] `docs/thesis/defense-prep/DECISIONS.md` has 6 ADR entries
- [ ] `.claude/commands/next.md` exists
- [ ] `CLAUDE.md` has new «VKR Defense Polish» section at end
- [ ] `memory/vkr_canonical_2026.md` exists, indexed in `MEMORY.md`
- [ ] 7 commits made (one per major task + Phase closure)
- [ ] `/next` invocation in a fresh session loads CANONICAL + STATE + BACKLOG (manual check optional)

## Out of scope for Phase 0 (deferred to Phase 1+)

- F-5 critic cron (Day 3 trigger; depends on stable cron infrastructure)
- Verification script (C-0, Phase 2)
- `reproduce.sh` audit (C-1, Phase 2)
- Brand-overlap backing artifact (`2026-05-26-brand-overlap-verification.md`) — record agent output in Phase 1
