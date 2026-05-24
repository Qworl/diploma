# Session Resume — Defense Prep Brainstorm (2026-05-24)

> Документ для возобновления работы на новом хосте. Содержит **всё**, что нужно знать сессии, продолжающей подготовку к защите.

## Контекст

Магистерская ВКР МАИ (кафедра 806, научный руководитель Судаков В.А.). Тема: гибридная каскадная система обогащения товарных данных (Regex → ML → Bayes → LLM).

Дата защиты: ≥ 1 месяц от 2026-05-24.

Работа находится в активной переработке — финальные числа (headline accuracy, LLM call rate, cost ratio) могут меняться. На момент 2026-05-24 в тексте ВКР: **92,8 % @ 720× cheaper, 3,3 % LLM, n=4350**. В memory зафиксирован post-deploy: **93,81 % @ 290×, 8,2 % LLM** (см. `memory-snapshot/accuracy_squeeze_deploy_2026-05-20.md`).

Решение по headline отложено user'ом до стабилизации переработки.

## Что сделано в текущей сессии

1. **Брейнсторминг defense-prep** — 4 секции (slides / notebook / report / defense materials).
2. **Дизайн-спецификация v3** — `2026-05-24-design.md` в этом же каталоге.
3. **Два прохода критика** — оба зафиксированы в `CRITIC_REVIEWS.md`.
4. **Memory snapshot** — критические memory файлы скопированы в `memory-snapshot/` (memory лежит локально в `~/.claude/projects/...`, при переезде не уедет автоматически).
5. **Обновление user memory**: создан `notebook_monolith_intentional.md` (зафиксировал, что `00_thesis_main.ipynb` — это §3.3 в исполняемом виде, дробить нельзя).

## Что НЕ сделано (pending)

1. **Передача в writing-plans skill** — спека дизайна готова, но детальный implementation plan не построен. Делать на новом хосте.
2. **Number sync** — отложено пока работа в переработке (см. §3 спеки).
3. **Бюрократический tracker** (§13 спеки) — все статусы `❓`, нужно начать заполнять.
4. **6 открытых вопросов руководителю** (§14 спеки) — нужно задать Судакову В.А.

## Ключевые решения, принятые в брейнсторме

| Решение | Обоснование |
|---|---|
| Структуру глав МАИ не трогаем | Жёсткий регламент кафедры 806; реф (EngineerXL/master-diploma) с иной структурой не применим |
| Notebook `00_thesis_main.ipynb` — монолит | Это §3.3 ВКР в исполняемом виде, дробить = сломать ссылки отчёт↔ноутбук |
| Slides 5 → 14 main + 7 backup | Уровень детализации референса, под МАИ магистра |
| 1 QR appendix (как у реф), не 3 | GitHub-репо содержит всё; индирекция упрощает |
| Speaker notes = plain MD, не beamer `\note{}` | МАИ-комнаты обычно без presenter view |
| Live demo на сцене — НЕТ | Слайд 11 с pre-baked скриншотами |
| 3 main домена = pasta + chocolate + cheeses | Зафиксировано в §3.1 текста ВКР |
| Числа в спеке = placeholders | User объявил работу в флюксе |
| TikZ-диаграмма — только D-cascade (1, не 4) | Effort:value при 1-мес бюджете; PNG переиспользуем |
| H1 router fail — softer фрейм «не обнаружено превосходства» | Affirming-the-consequent на MDE ≈ 4,4 пп не работает |
| Headline statistical defense — paired McNemar, не Wilson CI overlap | Overlap test не доказывает значимость |

## Ключевые находки критика (см. CRITIC_REVIEWS.md)

### Pass 1 (на v1)
- 5 задач циркулярны → переформулировка с измеримыми артефактами
- H1 router fail = footgun для МАИ → позитивный фрейм
- Headline без CI → добавить Wilson CI + McNemar
- Бюрократия отсутствует → §13 tracker
- 3 main vs 7 — нужно preempt question
- TZ задание = front-matter, не appendix

### Pass 2 (на v2)
- Task 3 и Task 5 ЕЩЁ циркулярны → доработка (v3 fixed)
- H1 reframe «доказано static достаточна» = affirming the consequent → softer (v3 fixed)
- Wilson CI overlap НЕ доказывает значимость → paired McNemar vs pre-deploy (v3 fixed)
- Frankenstein-числа (mix post/pre deploy) → placeholders в v3
- Recommendation: rollback на 92,8 % @ 720× — user отложил решение

## Файлы в этом каталоге

- `2026-05-24-design.md` — основная спецификация v3 (актуальная)
- `CRITIC_REVIEWS.md` — full content обоих проходов критика
- `SESSION_RESUME.md` — этот файл
- `memory-snapshot/` — копия project-relevant memory с момента 2026-05-24

## Что делать на новом хосте

### Сразу при подключении

1. **Восстановить локальные файлы**, которые НЕ в git:
   - `~/.claude/projects/-Users-miafrolov-Desktop-stuff-ai-attributes/memory/` — скопировать с старого хоста, **либо** использовать `memory-snapshot/` из этого репо как стартовый набор
   - `CLAUDE.md` в корне репо — gitignored, нужно скопировать с старого хоста (содержит структурные требования МАИ и project conventions)
   - `.claude/` в корне репо — gitignored, локальная конфигурация Claude Code
   - `.env` в корне репо — gitignored, секреты API
2. **Установить venv**:
   ```bash
   python3.14 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   brew install libomp  # macOS
   ```

### Возобновление работы по defense-prep

1. Прочитать **`2026-05-24-design.md`** (спека дизайна v3) полностью.
2. Прочитать **`SESSION_RESUME.md`** (этот файл) и **`CRITIC_REVIEWS.md`** для контекста решений.
3. Определиться с **headline числами** (см. §3 спеки и §11 «Риски»):
   - Вариант A: откат на 92,8 % @ 720× (рекомендация критика)
   - Вариант B: forward на 93,81 % @ 290× (требует sync в 15+ местах + McNemar)
   - Вариант C: пере-эксперимент с новой конфигурацией
4. **Передать спеку в writing-plans skill** для построения детального implementation-плана.
5. **Запустить Phase 0 бюрократия** (§10 спеки) асинхронно:
   - Запросить PDF задания у Крылова С.С.
   - Подтвердить тайминг защиты с Судаковым В.А.
   - Узнать рецензента и регистрацию ГИА
   - Аудит `reproduce.sh`
6. **Phase 1-3** (отчёт + слайды без чисел + доп. артефакты) — можно делать параллельно с переработкой работы.
7. **Phase 4 number sync** — ПОСЛЕ стабилизации.
8. **Phase 5 pre-defense** — за 2-3 недели.

## Контактные точки

- Научный руководитель: **Судаков Владимир Анатольевич**
- Зав. кафедрой 806: **Крылов Сергей Сергеевич**
- Кафедра: 806 «Вычислительная математика и программирование», Институт № 8 МАИ

## Версионирование

- v1 спеки — устранены circular tasks, footgun H1, missing CI, missing bureaucracy
- v2 спеки — Frankenstein numbers fixed, n=1539≠4350, expanded tracker
- v3 спеки — placeholders for numbers, Tasks 3/5 properly fixed, softer H1, paired McNemar, normokontrol added (**текущая**)
