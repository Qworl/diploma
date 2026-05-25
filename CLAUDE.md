# AI Attributes — Hybrid Enrichment System

Дипломный проект: гибридная система автоматизированного обогащения товарных данных для интернет-магазина.
Архитектура: 4-слойный pipeline (Regex → ML → Bayes → LLM fallback) на открытых данных.

## Environment

### Local (macOS)
- Python 3.14, venv: `.venv/` в корне проекта
- Активация: `source .venv/bin/activate`
- Зависимости: `pip install -r requirements.txt`
- XGBoost требует `brew install libomp` (macOS)
- sentence-transformers (~2GB с torch)
- pgmpy 1.1.2: `DiscreteBayesianNetwork`, estimator через `BayesianEstimator(model, data).estimate_cpd()`
- **OMP_NUM_THREADS=1** при запуске скриптов (segfault из-за libomp + torch на macOS)

### Remote VM (для heavy compute и LLM batch labelling)

- **Host:** `158.160.88.176` (Yandex Cloud, Ubuntu 24)
- **User:** `miafrolov`
- **Project path:** `~/Desktop/diploma/` (rsync с local repo)
- **VM specs:** 8 CPU, 23 GB RAM, 152 GB free disk, Python 3.12
- **Network:** 92-140 MB/s download (vs local ~80 KB/s) — критично для OFF dump downloads
- **VM venv:** `~/Desktop/diploma/.venv/` (Python 3.12); deps: `pandas`, `pyarrow`, `duckdb`, `huggingface_hub`, `hf_transfer`, `sklearn`, `requests`, `tqdm`, `statsmodels`, `matplotlib`
- **VM .env:** `~/Desktop/diploma/.env` содержит `HF_TOKEN` и `OPENROUTER_API_KEY` (mode 600)
- **Дополнительная папка:** `~/off_work/` — старый workspace (food.parquet 7GB OFF dump, intermediate {cat}_off_full.parquet, {cat}_relabel_*.parquet)

#### Persistent SSH connection (ускоряет работу 5-10×)
```bash
ssh -M -S ~/.ssh/control/yandex-vm -fNT -o ControlPersist=30m miafrolov@158.160.88.176
# Все последующие ssh/scp используют тот же tunnel:
ssh -S ~/.ssh/control/yandex-vm miafrolov@158.160.88.176 'команда'
scp -o ControlPath=~/.ssh/control/yandex-vm файл miafrolov@158.160.88.176:путь
```
Latency: ~74ms через мультиплекс (vs ~500ms per fresh ssh).

#### Sync local ↔ VM
```bash
# Local → VM (push свежий код/данные)
rsync -az --delete --exclude='.venv/' --exclude='__pycache__/' --exclude='models/' \
  -e "ssh -S ~/.ssh/control/yandex-vm" \
  ./ miafrolov@158.160.88.176:~/Desktop/diploma/

# VM → Local (pull результаты LLM-relabel etc)
rsync -az -e "ssh -S ~/.ssh/control/yandex-vm" \
  miafrolov@158.160.88.176:~/Desktop/diploma/datasets/processed/vm_relabel/ \
  ./datasets/processed/vm_relabel/
```

#### Когда использовать VM
- **Always:** OFF parquet download (7 GB), full LLM relabel (60 parallel workers), ML retrain
- **Параллелизм:** ThreadPoolExecutor(20+ workers per cat) для LLM-вызовов через OpenRouter
- **Background работа:** запускать через `nohup ... > log 2>&1 &` — продолжит при разрыве SSH
- **Local:** только мелкие правки + просмотр результатов

## Project Structure

```
ai_attributes/
├── .venv/
├── datasets/
│   ├── raw/
│   │   ├── en.openfoodfacts.org.products.parquet  — dump OFF (526MB, читается за секунды)
│   │   ├── GroceryDB/
│   │   └── zerotox-datasets/
│   └── processed/
│       ├── {cat}_stratified_silver_standard.parquet  — серебро из тегов OFF (pasta/choc/bev)
│       ├── {cat}_stratified_raw.parquet              — исходные данные
│       ├── {cat}_stratified_embeddings.npy           — кэш multilingual embeddings
│       └── ...                                       — parquet результатов экспериментов
├── models/
│   ├── {cat}_stratified_*_xgb.pkl / *_le.pkl        — ML classifiers
│   ├── {cat}_stratified_bayesian.pkl                 — Bayesian network
│   └── {cat}_stratified_*_calibration.json          — калибровка (ECE per attribute)
├── notebooks/
│   └── 00_thesis_main.ipynb                          — главный артефакт
├── src/
│   ├── common.py                  — shared utilities (logging, embeddings, constants)
│   ├── data/
│   │   ├── download.py            — скачивание OFF
│   │   ├── filter.py              — фильтрация CSV по категориям → parquet
│   │   ├── label_silver.py        — эталонная разметка из тегов OFF/OBF
│   │   ├── sample.py              — стратифицированная выборка
│   │   ├── convert.py             — конвертации форматов
│   │   └── manual_label/          — CLI для ручной разметки
│   ├── pipeline/
│   │   ├── regex/extractor.py     — Layer 1: regex (fat, age, measure, shape, grain...)
│   │   ├── ml/{train,infer}.py    — Layer 2: embeddings + XGBoost
│   │   ├── bayes/{train,infer}.py — Layer 3: Bayesian network (pgmpy)
│   │   ├── llm_fallback/{prompts,enrich}.py  — Layer 4: LLM enrichment
│   │   ├── off_labels/{apply,rules,tags}.py  — правила из тегов OFF
│   │   └── schemas/               — SCHEMA + EXAMPLES для 7 доменов
│   ├── llm/
│   │   ├── client.py              — OpenRouter + Ollama API
│   │   └── parsing.py             — парсинг ответов LLM
│   ├── eval/
│   │   ├── run_experiments.py     — сравнительные эксперименты
│   │   ├── run_diagnostics.py     — диагностики (CV, ablation, fairness)
│   │   ├── run_transfer.py        — transfer learning
│   │   ├── layer4_llm.py          — оценка Layer 4 LLM
│   │   ├── direct_llm.py          — direct LLM baseline
│   │   ├── manual_vs_silver.py    — ручное vs серебро
│   │   └── cascade_vs_llm_stats.py
│   ├── diagnostics/
│   │   ├── ml/                    — dag_bootstrap, cv_stability, feature_ablation...
│   │   ├── silver/                — audit, compare, self_consistency...
│   │   └── language/              — per_language_analysis
│   └── electronics/
│       ├── prepare.py             — подготовка датасета смартфонов
│       └── cold_start_demo.py     — cold-start демо
├── tests/
│   ├── test_regex.py
│   ├── test_llm_enricher.py
│   └── test_off_labels.py
└── requirements.txt
```

## Entry Points (python -m src.<module>)

```bash
# Data
python -m src.data.label_silver --category pasta_stratified
python -m src.data.filter --category pasta

# Pipeline training
python -m src.pipeline.ml.train --category pasta_stratified
python -m src.pipeline.bayes.train --category pasta_stratified

# Evaluation
python -m src.eval.run_experiments --category pasta_stratified
python -m src.eval.run_diagnostics
python -m src.eval.layer4_llm --all --max-per-cat 100

# Diagnostics
python -m src.diagnostics.ml.dag_bootstrap --category pasta --n-bootstrap 200
python -m src.diagnostics.ml.feature_ablation --category pasta_stratified
python -m src.diagnostics.silver.audit --category pasta_stratified

# Electronics
python -m src.electronics.prepare
python -m src.electronics.cold_start_demo
```

## Разделение полей: Вход vs Выход

### Что приходит от партнёра (ВХОД для системы)
- `product_name` — всегда
- `brands` — обычно
- `quantity` — иногда
- `ingredients_text` — иногда
- `code` (баркод) — всегда

### Что система должна заполнить (ВЫХОД)
- Все целевые атрибуты (milk_type, grain_type, is_organic, ...)
- `categories_tags` — НЕ входное поле, это результат обогащения
- `labels_tags` — аналогично
- `fat_100g`, `sugars_100g` и другие нутриенты

**ВАЖНО**: ML модель использует ТОЛЬКО partner-available поля для эмбеддингов:
`product_name + brands + ingredients_text + quantity`. НЕ использует categories_tags, labels_tags.

## Архитектура Pipeline

### Layer 1: Regex (`src/pipeline/regex/extractor.py`)
Парсит product_name и quantity. Извлекает:
- fat_content, minimal_age, measure, cooking_time, grain_type, pasta_shape

### Layer 2: ML (`src/pipeline/ml/`)
- Вход: `product_name + brands + ingredients_text + quantity` → SentenceTransformer → 384-dim vector
- Модель: `paraphrase-multilingual-MiniLM-L12-v2` (50+ языков)
- Классификатор: XGBoost (regularized: subsample=0.8, colsample=0.8, early stopping)
- Per-attribute confidence threshold (0.5–0.75), saved in `{category}_thresholds.pkl`

### Layer 3: Bayesian Network (`src/pipeline/bayes/`)
- Evidence: `brand` + ML predictions (inter-attribute)
- Структура: Hill Climb + BIC (pgmpy)

### Layer 4: LLM Fallback (`src/pipeline/llm_fallback/`)
- Бэкенды: OpenRouter API (`src/llm/client.py`), Ollama
- Используется для ~4-6% атрибутов где Layers 1-3 не уверены

## Schemas (`src/pipeline/schemas/`)

7 доменов: `pasta`, `chocolate`, `beverages`, `cheeses`, `cereals`, `cosmetics`, `electronics`.
Каждый модуль экспортирует `{DOMAIN}_SCHEMA` и `{DOMAIN}_EXAMPLES`.

## Conventions

- Silver standard данные и embeddings коммитятся (`datasets/processed/*silver_standard*`, `*embeddings*`)
- Raw данные и модели не коммитятся (.gitignore)
- Язык кода: Python, комментарии допустимы на русском
- Entry points: `python -m src.<module>.<script>`, а не `python src/<path>.py`
- OMP_NUM_THREADS=1 на macOS из-за конфликта libomp (torch + xgboost)

## Требования кафедры к ВКР (КРИТИЧНО — соблюдать структуру)

### Официальные документы и регламенты МАИ (источники форм и требований)
- **Формы документов ГИА (бакалавриат/специалитет/магистратура)** — ОД-093-СМК-ПОЛ-001-Ф, утв. 28.06.2021: https://mai.ru/unit/ouk/docs/ОД-093-СМК-ПОЛ-001-Ф_Формы_1.0.pdf
- **Положение о порядке проведения ГИА в МАИ** — ОД-093-СМК-ПОЛ-001, ред. 2.0: https://mai.ru/unit/ouk/docs/ОД-093-СМК-ПОЛ-001_Положение%20о%20порядке%20проведения%20ГИА_2.0.pdf
- **Регламент проверки на заимствования и размещения ВКР в НТБ МАИ** — приказ № 151 от 06.04.2021: https://mai.ru/upload/iblock/d96/151-ot-06.04.2021_O-vvedenii-v-deystvie-Reglamenta-proverki-i-razmeshcheniya-VKR_NTB-MAI_kontrol-I.S.Medovaya.pdf
- **ГОСТ 7.32-2018 «Отчёт о НИР»** (структура и оформление): https://files.stroyinf.ru/Data2/1/4293742/4293742537.pdf
- **ГОСТ 7.1-2003 «Библиографическая запись. Библиографическое описание»** (оформление ссылок и литературы): https://docs.cntd.ru/document/1200063713?marker=7D20K3

Бакалаврская ВКР (прошлая работа, как образец оформления): `~/Downloads/ВКР_Фролов.docx`.

### Технические требования
- **Минимальный объём основной части: 60 страниц**
- **Оригинальность текста: не менее 80%**, заимствования — не более 15%
- Ссылки на литературу — по тексту, оформлены по ГОСТу
- **Работающий код** (демо-система обязательна)
- **Соответствие правилам академического русского языка** — безличные конструкции, без первого лица
- **Замена англицизмов** терминами на русском там, где возможно (cascade → каскад, pipeline → конвейер, embedding → векторное представление, threshold → порог, fallback → запасной слой, abstain → отказ от ответа, и т.д.)
- **Ссылки на ресурсы, размещённые на территории России** (РИНЦ, КиберЛенинка, eLibrary, ГОСТ — НЕ Google Scholar / arxiv напрямую как primary ссылка; но цитировать зарубежные работы через перевод/локальный mirror допустимо)

### Структура ВКР (формальное содержание)

**Введение (5-7 стр.):**
- Актуальность (научная + практическая, предпосылки темы)
- Цель ВКР (каков должен быть результат — не повторяет тему)
- Задачи (4-6 шагов, исследование не является задачей — это инструмент)
- Объект (что исследуется) и предмет (с какой точки зрения)
- Теоретические и методические основы
- Новизна и основные результаты (каждой задаче — хотя бы один результат)
- Апробация и внедрение
- Библиографическое описание

**Глава 1 — Развёрнутая актуальность темы:**
1. Описание «боли» — проблема, не решённая на практике
2. Исследование существующих подходов (теория + практика)
3. Подробное обоснование целей и задач
- §1.1 — UX/UI и CJM
- §1.2 — классический литературный обзор с цитатами по ГОСТу
- §1.3 — техническое задание на ВКР
- Анализ аналогов в таблице (Критерий × Аналог 1/2/3)
- 2-3 вывода в конце главы

**Глава 2 — Теоретическое/методическое обоснование решения:**
1. Принципы, математическая/логическая модель
2. Обоснование и описание архитектуры и стека технологий
3. Ключевые алгоритмы и формулы
- §2.1 — логика работы разрабатываемого ПО
- §2.2 — архитектура и стек + обоснование выбора
- §2.3 — как должно работать ПО
- Много рисунков и таблиц
- 2-3 вывода в конце главы

**Глава 3 — Реализация решения:**
1. Программная реализация (ключевые фрагменты кода как **рисунки** с комментариями)
2. Данные — источник, структура, предобработка
3. **Тестирование во всех смыслах** (работоспособность + юзабилити) ← эмпирические эксперименты и результаты идут сюда
- §3.1 — описание ПО
- §3.2 — описание данных с фрагментами/примерами
- §3.3 — **доказательство работоспособности и применимости** (метрики, ablation, brand-disjoint test, cost-quality matrix — всё сюда)
- 2-3 вывода в конце главы

**Глава 4 — Описание результатов и применения разработки:**
1. Техническая документация (руководства пользователей с ролями)
2. Порядок внедрения (в какие бизнес-процессы, кем, как)
3. **Характеристика достигнутых результатов** (повышено / ускорено / сокращено) с фактами
- §4.1 — как пользоваться разработкой
- §4.2 — кто и где может пользоваться (типовые роли)
- §4.3 — что позволяет использование разработки (бизнес-эффект, итоговые цифры headline)
- Желательно: справка о внедрении/использовании

**Заключение (~5 стр.):**
- Собрать выводы каждой главы
- Связный текст
- Перспективы развития темы

### Важно для нашей работы
- **Эмпирика (cascade vs LLM, ablation, cost matrix, headline 91.5%) идёт в §3.3** «доказательство работоспособности» — это **не отдельная research-chapter**, это часть тестирования.
- **Глава 4 — про пользователя**, не про эксперименты. §4.3 — итоговый бизнес-эффект «достигнуто X, сокращено Y».
- **Демо описывается дважды**: в §3.1 «описание ПО» (что есть архитектурно) и в §4.1 «как пользоваться» (UI, пресеты, скриншоты).
- Кафедральная структура **не предполагает отдельной research-driven Chapter 4** в стиле arxiv-paper. Это инженерная ВКР с фокусом на «есть рабочее решение → вот как оно делается → вот какой результат».

## Known Issues

### Методологические
- **Silver standard**: ground truth из тегов OFF/OBF, не ручная разметка
- **Bayesian сеть**: добавляет coverage (+6-8%), но может снижать accuracy на 1-3%
- **Мультиязычность**: датасет ~40% FR, ~10% EN, ~7% DE — работает через sentence-transformers

### Технические
- OFF CSV: tab-separated, `on_bad_lines='skip'`
- `code` колонка: кастить в str (overflow int64)
- `completeness`: приходит как string → `pd.to_numeric(errors='coerce')`
- OMP_NUM_THREADS=1 на macOS из-за конфликта libomp (torch + xgboost)
- Lazy import sentence-transformers в `src/pipeline/ml/train.py`

## PO Mode (active until 2026-06-26)

You operate as product owner of the ai_attributes thesis project, not just
executor. After completing any meaningful unit of work:

1. Update the relevant ticket in `docs/po/tickets/`:
   - mark plan checkboxes,
   - append new sub-steps if the plan needs extension,
   - if closing — fill the "Результат" section (≥2 lines: what was done,
     what was learned, follow-ups added to INBOX).
2. Re-scan via `/next` logic, propose next-most-valuable item with reason.
3. New findings during execution → add as bullet to `docs/po/INBOX.md`
   (one line, with `[track]` prefix if known). Don't break the current
   ticket's flow.

**«Done» bears two meanings — distinguish them:**
- «Ticket done» — текущий тикет закрыт, Результат заполнен. Свободно
  отвечай «тикет закрыт» и предлагай следующее через `/next`.
- «Project done» — НИКОГДА не объявляй до защиты ВКР. Если пользователь
  спрашивает «всё ли готово?», отвечай состоянием: «активных тикетов N,
  blocked M, в INBOX K строк, последний /critic был X дней назад».

Standing inputs to check before proposing next step:
- `docs/po/ACTIVE.md`
- `docs/po/tickets/` (status: in_progress | blocked | backlog)
- `docs/po/INBOX.md`
- `git log --since="3 days ago"`
- `docs/thesis/defense-prep/2026-05-25-advisor-comments.md`
  (read-only источник входящих от руководителя; новые pending → INBOX,
  существующий файл не модифицировать)

`/critic` запускается ТОЛЬКО по явному вызову пользователя или через
`/next`, если все остальные источники работы пусты. Никаких
auto-on-session-start триггеров.
