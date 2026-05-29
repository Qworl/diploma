# VKR Final Polish — Design Spec

**Дата:** 2026-05-26 (обновлено 2026-05-26: timeline расширен до 2 месяцев)
**Окно работы:** до защиты (~2026-07-26, ≈8 недель)
**Hard deadline печати:** ≈2026-07-12 (типография МАИ, ~14 дней до защиты — уточнить в деканате)
**Effective working window:** ≈35 рабочих дней до hard deadline + 10–14 дней до защиты (slides + Q&A + mock rounds)
**Scope:** A (полировка трёх категорий) + C (electronics как cross-domain replication, §3.4) — теперь твёрдо в scope, не P1-риск
**Out of scope:** расширение на beverages/cereals/cosmetics, production deployment, real user testing
**v6 = CANONICAL alignment:** числа в `docs/thesis/CANONICAL.md` отражают v6 schema (chocolate `is_filled` orthogonal binary, cheeses.texture без `other`, 21 atts в production / 20 в headline). Подтверждено пользователем 2026-05-26.

---

## 1. Контекст

На 2026-05-26 thesis собирается (`report/main.pdf`, 91 стр.), слайды собираются (`slides/main.pdf`), демо-комплекс существует но без живых скриншотов в слайдах. Архитектура и эмпирика стабильны (cascade 94,8% / E2E 91,1% / human gold E2E 86,7%); рассогласования между artefacts, thesis-текстом и слайдами накапливались итеративно и не были выровнены.

Авторитетные источники:
- `docs/thesis/data_methodology.md` §14 — текущие headline-числа, схемы атрибутов, ограничения
- `report/contents/4-chapter3-implementation.tex` — детали implementation, ближе всего к артефактам
- MEMORY.md V6 entry — состояние после v4 schema refactor

Расхождения, выявленные первичной критикой 2026-05-26:
- abstract / introduction / conclusion цитируют устаревшие числа (92,8% / 720× / 4350 / 3,3% / 24 XGBoost)
- thesis заявляет «brand-disjoint test»; реальные данные — code-disjoint (brand overlap 82–85%); подтверждено верификационным агентом 2026-05-26
- ТЗ описывается по-разному (6/5/3 в conclusion vs 6/4/3 в слайдах vs 6/5/3 в §1.3)
- число атрибутов плавает (20 / 21 / 22 / 24)
- 7 пунктов замечаний научного руководителя 2026-05-25 не отработаны
- demo-скриншоты в слайдах помечены TODO, не заменены реальными
- `reproduce.sh` существует, но содержимое не верифицировано

## 2. Цель

К дате защиты:
1. Все цифры в thesis / slides / артефактах согласованы с одним источником правды (CANONICAL.md).
2. Все заявления, которые не подтверждаются данными (в первую очередь «brand-disjoint test»), либо приведены в соответствие с данными, либо переформулированы честно.
3. Все 7 пунктов замечаний научного руководителя отработаны.
4. Демонстрационный комплекс показуется живыми скриншотами и записанной демо-сессией; `reproduce.sh` действительно регенерирует headline-числа.
5. Подготовлен и репетирован Q&A (10–15 верифицированных ответов).
6. Electronics добавлена как §5 (cross-domain репликация), с собственным gold-эталоном и оценкой Bayes-валидатора как control.

## 3. Source of Truth

`docs/thesis/CANONICAL.md` (создаётся в F-0) — единственный источник правды для:
- headline-чисел (cascade 94,8% / E2E 91,1% / human gold E2E 86,7% / LLM fallback 7,1% / router 95,4% / Layer 1: 18,4% / Layer 2: 73,8% / Layer 3: 0,7%)
- размеров эталонов (consensus gold n=3257 cascade-valid, human gold n=566 cascade-valid)
- числа атрибутов (схемы 21 = pasta 8 + chocolate 6 + cheeses 7; headline-таблица 20 — `pasta.protein_class` опциональный)
- формулировок стоимости (14× архитектурное / 333× с Gemini / каскад+gpt-oss 471× / каскад+llama-3b 1571×)
- терминологии (cascade-only / E2E / production-realistic — что есть что)
- split-режима (code-disjoint, не brand-disjoint; brand overlap 82–85%)
- стоп-листа устаревших чисел (92,8% / 720× / 4350 / 3,3% / 24 поатрибутных XGBoost / 9,0 п.п.)

Данные за каждым числом — pinned к конкретному parquet/JSON-артефакту.

CANONICAL.md загружается в каждую subagent-задачу как обязательный входной контекст.

## 4. Рабочие потоки

Каждый item: ID (`X-N`), приоритет (P0 блокер / P1 сильно усиливает / P2 nice-to-have / P3 пропускаемо), краткое описание.

### Workstream A: Текст ВКР

| ID | Приоритет | Описание |
|---|---|---|
| **A-0** | **P0** | Brand-disjoint → code-disjoint везде. Прицельная правка формулировок в `report/contents/*.tex` и `docs/thesis/`. Добавить явный абзац в §3.2.4 о реальном split (brand overlap 82–85%). Контекст-зависимая редактура, не sed. |
| **A-1** | **P0** | Sync headline-чисел: abstract / introduction / conclusion → CANONICAL.md. Удалить все «92,8 %» / «720×» / «4350» / «3,3 %» / «9,0 п. п.» / «24 поатрибутных XGBoost» / «23,0 п. п.». Каждое употребление переписать с учётом контекста, не slep-replace. |
| **A-2** | **P0** | ТЗ count alignment. Зафиксировать «6 ФР / 5 НФР / 3 А» (по §1.3) и протянуть в conclusion и слайды. |
| **A-3** | **P0** | Число атрибутов: 21 в схеме, 20 в headline-таблице. Снять «22» / «24» из conclusion и сводных формулировок. |
| **A-4** | **P0** | Cost framing. Ввести 3 явных термина в §2.1.4 (архитектурное снижение / комбинированное / эффективное) и протянуть. |
| **A-5** | **P0** | §«Ограничения» в conclusion расширить: brand overlap, circular bias ~3,8 пп, sample size, language hot-spots. Должно отражать §12 data_methodology.md. |
| **A-6** | **P1** | Добавить абзац «Почему XGBoost, а не Random Forest» в §2.2.4 (замечание научрука 1–2). Эмпирический RF baseline на pasta как иллюстрация (см. C-3). |
| **A-7** | **P1** | Bayes DAG в текст §2.2.5 + §3.3.4 (замечание научрука 3). Использовать существующий `images/fig_3_5_bayes_dag.png`. Добавить пояснение, что Bayes — валидатор, селективный, работает на 3 парах из 20. |
| **A-8** | **P1** | Переформулировать «Апробация и внедрение» (замечание научрука 4). Конкретный работодатель не упоминается (по решению пользователя 2026-05-26). Вместо этого — акцент на integration-readiness: «спроектирована как интегрируемый компонент в системы PIM российских интернет-магазинов; готовый к развёртыванию демо-комплекс реализует типичный partner→catalog flow». Снимает «не внедрено» как формулировку. |
| **A-9** | **P2** | Обновить таблицу аналогов §1.2.3: Akeneo AI (2023+), TXtract «локальное развёртывание» — снять или оговорить. |
| **A-10** | **P2** | Confusion matrices для 3 трудных атрибутов (chocolate/contains_nuts, cheeses/aging, pasta/grain_type) — рисунок в §3.3.2 или §3.3.3. |
| **A-11** | **P2** | Heatmap «категория × атрибут × язык» с language hot-spot'ами в §3.3.7.1 (cheeses/texture/es −47 пп и др.). |
| **A-12** | **P2** | TCO в $: расчёт на 100K и 1M SKU/месяц в §4.3.2. Заменить «порядки величины» на конкретные цифры. |
| **A-13** | **P2** | Глоссарий в `report/contents/glossary.tex`: cascade-only / E2E / production-realistic / архитектурное снижение etc. |
| **A-14** | **P2** | Цитата ODbL атрибуция Open Food Facts в §3.2.1 (правовая чистота). |
| **A-15** | **P3** | Поднять число рисунков с 9 до ≥12 (по кафедральной норме «много рисунков в гл 2/3»). |
| **A-16** | **P0** | **Full thesis re-read** — после всех правок один сквозной проход от title до conclusion. Цель: transition issues, потерянные параграфы, broken cross-references, последовательность мыслей. Делается в Phase 4. |
| **A-17** | **P1** | Print-ready proofread: ё/е, длинные тире vs дефисы, неразрывные пробелы (числа + единицы), формат подписей таблиц/рисунков по ГОСТ 7.32-2018, выравнивание, hanging punctuation. |
| **A-18** | **P1** | Bibliography audit: исправить mismatch «46 vs 52 источника» (introduction vs abstract); проверить ГОСТ 7.1-2003 оформление каждой записи; убедиться, что все ref_X в тексте есть в main.bib. |
| **A-19** | **P0** | Bayes honest framing — отдельный абзац в §3.3.4 и в §«Ограничения» conclusion: «байесовский валидатор в production включён точечно на одной паре (chocolate/contains_nuts); на остальных селективность сигнала недостаточна для архитектурного включения». Сейчас в conclusion это упомянуто как принцип «изменения роли», но не как ограничение. |

### Workstream B: Слайды

| ID | Приоритет | Описание |
|---|---|---|
| **B-0** | **P0** | Sync чисел из CANONICAL.md. После завершения A-1 — пробежаться по `slides/main.tex` и привести к единому источнику. |
| **B-1** | **P0** | Заменить TODO-картинку `fig_4_1_demo_ui.png` на реальные скриншоты живого UI: pasta / chocolate / cheeses (см. C-2). |
| **B-2** | **P0** | Bayes DAG в основную деку (после «Функциональной модели» либо рядом с «Ключевыми алгоритмами»). Замечание научрука 3. |
| **B-3** | **P0** | ТЗ count alignment: F1–F6 / N1–N5 / A1–A3 (сейчас N1–N4 и A1–A3). |
| **B-4** | **P0** | «Цель и задачи» — 5 задач (сейчас 6, лишний пункт «Рекомендации»). |
| **B-5** | **P0** | «Заключение: задачи и полученные результаты» — 5 строк (сейчас 6). |
| **B-6** | **P0** | Сместить формальную постановку в один буллет / убрать (замечание научрука 5). |
| **B-7** | **P0** | ГОСТ-контекст со слайда 14 (замечание научрука 7). Уточнить, какой файл — после установления убрать. |
| **B-8** | **P0** | Аналоги: убрать спорные ✓ (TXtract «локально», MAVE «локально»), заменить на «частично / нет». |
| **B-9** | **P1** | Добавить пометку «1 классификатор на атрибут (21 модель)» в слайд алгоритмов (замечание научрука 6). |
| **B-10** | **P1** | Backup-слайд B8: multitask эксперимент (уже сделан, артефакт `datasets/processed/multitask_eval/`). |
| **B-11** | **P1** | B7 Electronics: положить реальные числа (после E-1). |
| **B-12** | **P1** | B5 Многоязычность: мини-таблица из §3.3.7.1 в backup. |
| **B-13** | **P1** | Поднять шрифты ≤9pt до ≥10pt на слайдах «Состав требуемых функций», «Аналоги», «Главный результат», «Заключение». |
| **B-14** | **P1** | Speaker notes (`SPEAKER_NOTES.md`) обновить под все слайды основной деки + 8 backup'ов. |
| **B-15** | — | ~~Приватность github~~ — снято решением пользователя 2026-05-26 (репо остаётся публичным). |

### Workstream C: Код / Repro / Demo

| ID | Приоритет | Описание |
|---|---|---|
| **C-0** | **P0** | `src/eval/verify_numbers.py`: grep по report/**/*.tex и slides/main.tex на стоп-листе; assertion-check присутствия canonical-чисел; exit 0/1. Запускается вручную, не как hook. |
| **C-1** | **P0** | Аудит `reproduce.sh`: открыть, проверить, что команды реально регенерируют headline. Дописать недостающее. Прогон на чистом venv. |
| **C-2** | **P0** | Реальные скриншоты живого демо: запустить ml_service + Go gateway + frontend, сделать ≥3 скриншота (pasta / chocolate / cheeses) + короткое видео demo.mp4 для backup. |
| **C-3** | **P1** | Random Forest baseline эмпирический: обучить RF на pasta (тот же noleak training pool, MPNet+TF-IDF), сравнить с XGBoost по micro/macro-F1/ECE. Результат в A-6. |
| **C-4** | **P1** | Brand-disjoint subset sensitivity check: вычислить cascade accuracy на brand-disjoint подмножестве (17 pasta / 19 chocolate / 28 cheeses кодов). Результат в A-5 / §3.3.7. |
| **C-5** | **P2** | Electronics PhoneDB pipeline: gold construction + cascade eval (для E-1..E-2). |
| **C-6** | **P3** | Pre-commit hook на verify_numbers.py — отложен, чтобы не блокировать. |

### Workstream D: Q&A Подготовка

**Sequencing note.** Подготовка Q&A сама находит дыры в тексте (нет ответа на вопрос → дыра в §X). Поэтому D-1 / D-2 стартуют в Phase 1 параллельно с canonical alignment, а не на Phase 3.

| ID | Приоритет | Описание |
|---|---|---|
| **D-0** | **P0** | Initial list — пройтись по «острым местам» (brand overlap, Bayes на 1 атрибуте, calibration не в проде, OpenRouter vs локально, circular bias, fake 4350/720× в abstract на ранних версиях) и сформулировать 8–10 черновых вопросов в Phase 1. |
| **D-1** | **P1** | Полная подборка 15 вопросов. На каждый — pointer на артефакт / раздел thesis. Документ `defense-prep/qa-2026-06.md`. Расширение D-0. |
| **D-2** | **P1** | Верифицированный ответ для каждого с проверкой против CANONICAL.md и parquet'ов. |
| **D-3** | **P1** | Mock defense run (sub-agent в роли скептичного рецензента) — найти gap'ы в ответах. Phase 3 + повтор в Phase 4. |
| **D-4** | **P0** | **Advisor pre-review gate**: финальную версию thesis отправить научруку за ≥7 дней до сдачи в типографию. Зарезервировать буфер на правки. |
| **D-5** | **P2** | Speaker notes на самой презентации: что говорить кроме того, что на слайде. |

### Workstream E: Electronics (cross-domain replication)

**Структурное примечание.** Кафедральная структура — 4 главы (см. CLAUDE.md). Добавление полноценной «Главы 5» нарушает шаблон. Electronics включается одним из трёх способов: (а) как §3.4 «Кросс-доменная репликация» внутри гл.3; (б) как §4.4 в гл.4 (применение / переносимость); (в) как Приложение Б. Окончательное место — по согласованию с научруком в Phase 2. По умолчанию — вариант (а).

| ID | Приоритет | Описание |
|---|---|---|
| **E-0** | **P1** | PhoneDB gold construction: схема атрибутов уже в `src/pipeline/schemas/electronics.py`. Собрать gold n≥100 через LLM-consensus (qwen3+ds-r1+mistral) или Opus blind. |
| **E-1** | **P1** | Cascade eval на electronics: cascade + LLM, без Layer 1 regex (нет правил). Headline accuracy, macro-F1, LLM-доля. |
| **E-2** | **P1** | Bayes-replication: обучить Bayes на electronics (если есть достаточно данных), проверить, селективно ли работает. Усиливает аргумент «Bayes — общее ограничение архитектуры, не quirk food-домена». |
| **E-3** | **P1** | Текст в thesis ~3–5 страниц как §3.4 (default) или альтернатива. Подключение в conclusion как «Третий пункт научной новизны — cross-domain replication подтверждена». |
| **E-4** | **P1** | Update abstract / introduction / conclusion: добавить electronics как cross-domain control. |

### Workstream F: Инфраструктура (cross-cutting)

| ID | Приоритет | Описание |
|---|---|---|
| **F-0** | **P0** | `docs/thesis/CANONICAL.md` — single source of truth. Создать, заполнить из data_methodology.md §14 + результата верификации brand-disjoint. |
| **F-1** | **P0** | `docs/thesis/defense-prep/STATE.md`, `BACKLOG.md`, `RETROSPECTIVES.md`, `DECISIONS.md` — создать скелеты. BACKLOG проинициализировать всеми items из этого спека. DECISIONS заполнить тремя принципами проектирования схемы (§5.1–5.3 data_methodology.md). |
| **F-2** | **P0** | `.claude/commands/next.md` — слэш-команда `/next` для PO-loop'a. |
| **F-3** | **P0** | Дополнение `CLAUDE.md` — блок «VKR Defense Polish» + «PO Mode» с pointer'ами на CANONICAL / STATE / BACKLOG и инструкцией всегда верифицировать числа против CANONICAL. |
| **F-4** | **P0** | Memory entry `vkr_canonical_2026.md` — short copy of CANONICAL для подгрузки в будущих сессиях. |
| **F-5** | **P1** | Critic cron: **раз в сутки** (20:00) gated по `git log --since="24h"` + event-triggered (после закрытия любого P0-item). Запись в `defense-prep/critic-YYYY-MM-DD.md`. Через `CronCreate`. Изменено с 2h на 24h по итогам self-critique (200 пасов/мес → ~30 пасов/мес). |

## 5. Sequencing (обновлено под 2 месяца)

Анкеры: **Day 1 = 2026-05-27**, **Hard print deadline ≈ Day 47 (2026-07-12)**, **Defense ≈ Day 60 (2026-07-26)**.

Phase 0 уже частично закрыт в первой сессии 2026-05-26: CANONICAL.md, PO mode, 6 тикетов advisor, A-0/A-1/A-2/A-3/A-19 защитные правки текста. См. retro в `docs/po/critique/` или RETROSPECTIVES (если был).

```
[ЗАКРЫТО Day 0]  Pre-Phase 0: spec + plan + critic              ✅
                          ▼
[ЗАКРЫТО Day 0]  Phase 0: F-0..F-4 инфраструктура                ✅
                          ▼
[ЗАКРЫТО Day 0]  Phase 1 P0 text alignment: A-0/A-1/A-2/A-3/A-19  ✅
                          ▼
Phase 2 (Days 1–10):      B-1..B-8 slides P0 (numbers / demo screenshots / Bayes DAG в основную деку /
                                                ТЗ counts / advisor comments #5–7),
                                              + advisor-comment-bayes-viz ticket (P0)
                                              + advisor-comment-rf ticket: text + RF empirical baseline (C-3)
                                              + C-0 verify_numbers.py
                                              + D-0 initial Q&A list
                          ▼
Phase 3 (Days 11–20):     C-1 reproduce.sh audit/rebuild + C-2 real demo screenshots + recording
                                              + A-6..A-8 (RF text final, Bayes DAG в thesis,
                                                          integration-readiness refrasing — A-8 готов к рефреймингу
                                                          без реквизитов работодателя)
                                              + D-1..D-2 polished Q&A 15 questions
                                              + C-4 brand-disjoint subset sensitivity
                          ▼
Phase 4 (Days 21–30):     **Electronics E-0..E-3 полноценно** (PhoneDB gold construction n≥100,
                                              cascade eval, Bayes replication, §3.4 текст 5–7 страниц)
                                              + первый critic-agent deep pass
                          ▼
Phase 5 (Days 31–37):     A-9..A-15 P2 polish (analogs table обновить, confusion matrices,
                                                language heatmap, TCO в $, glossary, bib audit)
                                              + B-13..B-14 slides typography + speaker notes
                                              + D-3 mock defense round 1 (subagent в роли критика)
                                              + E-4 abstract/intro/conclusion обновлены под electronics
                          ▼
Phase 6 (Days 38–42):     A-16 full thesis re-read + A-17 print-ready proofread + A-18 bib audit
                                              + Apply mock defense findings
                                              + D-3 mock defense round 2
                          ▼
**D-4 ADVISOR PRE-REVIEW GATE** — Day 43. Финальная версия → научруку, минимум 4 дня буфер на правки.
                          ▼
Phase 7 (Days 44–46):     Apply advisor edits + final compile
                          ▼
**HARD DEADLINE — Day 47 (~2026-07-12)**: thesis → типография.
                          ▼
Phase 8 (Days 48–55):     Slides finalization (B-15 — но github уже public, snm OK)
                                              + D-3 mock defense round 3 (post-print)
                                              + speaker notes finalize
                                              + repro verification на чистом venv ещё раз
                          ▼
Phase 9 (Days 56–59):     Defense day prep: dry run полным форматом, тайминги, Q&A drill, edge cases
                          ▼
**DEFENSE — Day 60 (~2026-07-26)**
```

`F-5` (critic cron) включается на Day 5 — после стабилизации Phase 2 первых правок слайдов.

**Что меняется vs 1-месячный план:**
- Phase 4 (electronics) теперь полноценно отдельная фаза, не зажата в Phase 3
- Три раунда mock defense (Days 32, 41, 51) вместо одного
- A-6 RF делается **эмпирическим baseline'ом** (C-3), не только текстом
- A-12 TCO в $ и A-10 confusion matrices теперь точно в scope
- Phase 9 (4 дня финальной отладки перед защитой) — раньше не было

**Buffer policy.** Если какой-либо item задерживается на > 50% оценки — флагуется через /next и приоритизируется в следующей фазе либо выкидывается по fallback ranking §12.

## 6. Verification gates

| Когда | Что проверяется |
|---|---|
| После каждого P0-item | `python src/eval/verify_numbers.py` |
| После Phase 1 | Полный grep по стоп-листу; ручная сверка abstract / intro / conclusion |
| После Phase 2 | Прогон `reproduce.sh` на чистом venv |
| После Phase 3 | Critic agent: «найди ≥3 новых проблемы»; mock defense Q&A |
| После Phase 4 | Полный финальный pass critic'а + final compile thesis.pdf + slides.pdf |

## 7. Acceptance criteria

Защита считается «готовой к запуску», когда:
- [ ] `verify_numbers.py` exit 0 на всех .tex
- [ ] Все P0 items закрыты в BACKLOG (checkbox done)
- [ ] Все 7 пунктов замечаний научрука закрыты (см. `2026-05-25-advisor-comments.md`)
- [ ] thesis.pdf компилируется без warning'ов
- [ ] slides.pdf компилируется, шрифты ≥10pt
- [ ] reproduce.sh успешно регенерирует headline числа
- [ ] Скриншоты живого demo вставлены в слайды
- [ ] Bayes DAG в thesis и slides
- [ ] §5 electronics добавлена (если Phase 3 завершён в срок)
- [ ] D-3 mock defense проведён, gap'ы закрыты
- [ ] CANONICAL.md соответствует финальной thesis 1:1

## 8. Open blockers

| Блокер | Что нужно от пользователя | Влияет на |
|---|---|---|
| ~~Реквизиты работодателя~~ | Снято 2026-05-26 — рефрейм через A-8 (integration-readiness) | — |
| pptx или latex слайды для замечания #7 | По умолчанию принято: LaTeX (`slides/main.tex` — активный). При уточнении возможен пересмотр. | B-7 |
| ~~GitHub приватность~~ | Снято 2026-05-26 — репо остаётся публичным | — |
| Hard print deadline | Уточнить в деканате на Day 1 — точный date sdacha v типографию | sequencing §5 |

## 9. Out of scope

- Расширение на beverages / cereals / cosmetics (вариант B из брейншторма)
- Production deployment в PIM-системе
- Реальное user testing с контент-менеджерами
- Перестроение eval gold под brand-disjoint (требует переразметки)
- Calibration в production (вынесено в перспективы)

## 10. Risks

| Риск | Митигация |
|---|---|
| Electronics не успевает в срок | Phase 3 buffer; в крайнем случае оставить в «перспективах», как сейчас |
| `reproduce.sh` глубоко сломан | C-1 разбит на audit (0,5 дня) + rebuild по необходимости (2–5 дней); запланировано на Phase 2 |
| `brand-disjoint subset` (C-4) покажет резкое падение | Это всё равно полезная информация для §3.3.7; не блокирует A-0 (переименование) |
| Critic agent найдёт новый P0 ближе к print deadline | Phase 3 буфер + D-4 advisor gate — оба дают возможность вставить экстренные правки |
| Demo recording (C-2) сломается на демо-стенде | Fallback — screenshot-фоллбэк + statically rendered video; в крайнем случае показ только из слайдов без live |
| E-0 PhoneDB gold < n=100 | Снизить до n=50 + явно пометить как «pilot replication, не full eval» |
| Hard print deadline сдвинется | Уточнить в Day 1; sequencing §5 имеет вариант с +7 днями |
| Advisor pre-review (D-4) даст крупные правки | Phase 4 буфер 2 дня; в крайнем случае — Phase 5 переходит на доработку текста |

## 11. Метрика прогресса

Update `STATE.md` после каждого закрытого item. Update `RETROSPECTIVES.md` после закрытия каждой Phase. Update `BACKLOG.md` непрерывно (приоритеты могут меняться).

## 12. Fallback ranking (что выбрасываем первым)

Если в Phase 2/3 станет ясно, что не успеваем всё — отказываемся в строгом порядке снизу вверх:

**Сохраняем всегда (защита не пройдёт без этого):**
- F-0..F-4 (инфраструктура)
- A-0..A-5, A-16, A-19 (canonical alignment + brand-disjoint fix + Bayes honest framing + full re-read)
- B-0..B-8 (slides P0 — числа, скриншоты, замечания научрука)
- C-0, C-1 (verify_numbers + reproduce.sh)
- D-0, D-2, D-4 (Q&A core + advisor gate)

**Отказываемся первым (если упрёмся в время — порядок выбрасывания):**
1. A-15 (поднять число рисунков до 12) — кафедральная norma, но не критично
2. A-13 (глоссарий) — nice-to-have
3. A-11 (language heatmap) — усиливает, но не блокирующий
4. A-10 (confusion matrices) — усиливает, не блокирующий
5. A-12 (TCO в $) — усиливает практическую часть
6. C-3 (RF baseline эмпирический) — замечание научрука 1-2 можно закрыть только текстом A-6
7. A-9 (обновление таблицы аналогов) — устаревшее знание о Akeneo AI; рецензент может не заметить
8. E-2, E-3, E-4 (electronics Bayes-repl, §3.4 текст, abstract update) — пакет electronics; если падает — остаётся «в перспективах»
9. D-5 (speaker notes) — можно справиться без них
10. A-17 (print-ready proofread) — типография обычно толерантна к мелочам
11. A-18 (bib audit) — рецензент скорее всего не считает источники

Условие: если выбросили item ≥ 5 в этом списке → critic agent + повторный self-review в конце Phase 3.
