# Gold validation: ручная разметка

## Workflow (post-2026-05-15)

Each `(row, attribute)` cell has two fields:

- **`status`**: one of
  - `empty` — not reviewed
  - `auto` — silver pre-filled, awaiting human review
  - `confirmed` — human reviewed; value agrees with silver
  - `override` — human reviewed; value differs from a non-empty silver
  - `manual_only` — silver was empty; human filled the value
  - `unsure` — human reviewed but expresses doubt (sticky)
- **`mode`**: one of `blind` (typed without pre-fill visible) or
  `prefill` (silver pre-filled in input). Persists once set; never
  cleared.

The `mode` field is **the anchoring-bias control**: 34 pre-pivot
labels are `blind` (typed from scratch); fresh labels are `prefill`
(silver shown). Comparing `override_rate(blind)` vs
`override_rate(prefill)` directly measures whether pre-fill biases
the annotator toward confirming.

### Per-cell controls

- `c` key (while focused on input) — confirm the cell (auto → confirmed)
- Small `✓` button next to chips (visible only when status=auto) —
  same as `c`
- `s` key — copy silver value into input
- `u` key (or `?` button) — mark cell as unsure
- 1-9 keys — pick from chip list

**There is no batch confirmation** — this is intentional. One-click
confirm-all would be the worst form of anchoring bias.

### Server-side derivation

`src/manual_label/status.py:derive_status(silver, manual, prev)` is
a pure function used by the server and the migration script. The
client mirrors it in JS (see `deriveStatus` in `app.py`).

### Migration to new schema

```bash
python -m src.manual_label.migrate_to_prefill \
    --csv datasets/manual_label/<file>.csv
```

Idempotent. Backup written to `<file>.csv.bak`. Atomic via
`os.replace` (no truncation on crash).

---

## Что в этой папке

| Файл | Назначение |
|---|---|
| `<cat>_to_label.csv` | Sample 30/150 продуктов, отобранных из test split. Колонки `silver_*` заполнены автоматически из OFF/OBF/OPFF тегов. Колонки `manual_*` пустые — их вы заполняете. |
| `<cat>_labeled.csv` | После `llm_assisted_label.py` (Sonnet 4.5): добавлен первый проход. Колонки `manual_*` уже предзаполнены Sonnet-разметкой. |
| `<cat>_disagreement.csv` | Только пары (silver ≠ Sonnet). На них фокусируется внимание. |

## Workflow

### 1. Создание sample (автоматически, я делаю)

```bash
python scripts/sample_for_manual_label.py --category baby_stratified --n 150
python scripts/sample_for_manual_label.py --category cosmetics_stratified --n 150
python scripts/sample_for_manual_label.py --category pet_food_stratified --n 150
```

### 2. Первый проход Sonnet 4.5 (автоматически, я делаю; ~$2 на 600 продуктов)

```bash
python scripts/llm_assisted_label.py --category baby --model anthropic/claude-sonnet-4.5
python scripts/llm_assisted_label.py --category cosmetics --model anthropic/claude-sonnet-4.5
python scripts/llm_assisted_label.py --category pet_food --model anthropic/claude-sonnet-4.5
```

### 3. Ручная верификация — ВАШ ШАГ

Откройте `<cat>_labeled.csv` в Excel/Numbers/Google Sheets (или VS Code).

**Структура колонок** (на примере baby):
```
code | product_name | brands | ingredients | quantity |
silver_milk_type | manual_milk_type | silver_minimal_age | manual_minimal_age | ...
```

- `silver_*` — что система автоматически вытащила из тегов OFF (immutable, для справки).
- `manual_*` — это нужно проверить и проставить правильное значение. Sonnet уже залил предсказания, ваша задача — верифицировать.

**Темп:** ~30 секунд на продукт. На 150 продуктов = ~75 минут на категорию.

### Тактика верификации

**Шаг А (быстрый, 2/3 продуктов).** Открыть строку в CSV, посмотреть:
- product_name (например, "Hipp Bio Lait 1er age 0-6 mois")
- silver_milk_type = `cow`, manual_milk_type = `cow` → **оставляете как есть** ✅
- silver_minimal_age = NaN, manual_minimal_age = `0-3m` → Sonnet прав (1er age = 0-3m), **оставляете** ✅
- silver_is_organic = `True`, manual_is_organic = `True` → ✅

Если все 6 атрибутов согласуются между silver и Sonnet — **30 секунд работы, идём дальше**.

**Шаг Б (медленный, 1/3 продуктов).** Когда silver и Sonnet расходятся (см. `<cat>_disagreement.csv`):
- Открыть продукт в OFF: `https://world.openfoodfacts.org/product/<barcode>` (для cosmetics: openbeautyfacts.org, для pet food: openpetfoodfacts.org)
- Посмотреть фото упаковки
- Прочитать состав
- Проставить **правильное** значение в `manual_*`

Пример из baby pilot:
```
Banana apple & carrot strained baby food
silver_is_lactose_free = False
manual_is_lactose_free = True (Sonnet: банан+яблоко+морковь, лактозы нет)
```
Открываете OFF, смотрите состав — действительно нет молочного. **Меняете manual_is_lactose_free на True**, либо оставляете Sonnet'овское True если согласны.

**Спорные случаи** — оставить пустым или поставить `?`. Я при eval автоматически пропущу.

### 4. Финальная оценка (автоматически, я делаю; ~1 минута)

```bash
python scripts/eval_manual_vs_silver.py --all
```

Создаёт `datasets/processed/manual_eval_summary.parquet` с тремя метриками:
- **silver_vs_manual_acc_on_covered** — насколько эталон совпал с правдой
- **cascade_vs_manual_acc_on_covered** — фактическая точность regex_ml_bayes
- **llm_vs_manual_acc_on_covered** — фактическая точность direct LLM

После этого ячейки §6.12 notebook'а наполняются данными.

---

## Pilot результат для baby (30 продуктов)

✅ Запущено `sample_for_manual_label.py` + `llm_assisted_label.py` для baby.

**Файлы готовы:**
- `baby_to_label.csv` — 30 продуктов, manual_* пустые
- `baby_labeled.csv` — 30 продуктов, manual_* предзаполнены Sonnet 4.5
- `baby_disagreement.csv` — 85 расхождений на 30 продуктов (≈3 расхождения на продукт)

**Распределение расхождений по атрибутам:**
| Атрибут | Расхождений | Тип |
|---|---|---|
| minimal_age | 26 | silver обычно NaN, Sonnet находит в product_name |
| is_lactose_free | 18 | silver=False (по умолчанию), Sonnet=True для растительного |
| flavour | 17 | silver обычно NaN, Sonnet видит fruit/vegetable |
| milk_type | 13 | в твёрдом детском питании Sonnet корректно говорит NaN |
| is_gluten_free | 10 | silver=False, Sonnet=True для rice-based |
| is_organic | **1** ✅ | Сильное согласие (organic tag в OFF надёжный) |

**Главный вывод pilot'а:** силвер (OFF) часто имеет False (по умолчанию) или NaN на нутри-метках, тогда как Sonnet даёт более точное прочтение из product_name + ingredients_text. Это **ровно та проблема**, которую gold validation должна закрыть.

---

## Бюджет

| Шаг | Время автоматического | Время ваше | Стоимость API |
|---|---|---|---|
| Sample 150×4 cats | 5 мин | 0 | 0 |
| Sonnet first-pass | ~30 мин (API) | 0 | ~$2–5 |
| **Manual verification** | 0 | **~5 часов на 600 продуктов** | 0 |
| eval_manual_vs_silver | 1 мин | 0 | 0 |
| **Итого** | ~40 мин | 5 часов | ~$5 |

Можно делать по 50 продуктов в день — закроется за ~2 недели в спокойном темпе.

---

## Когда не делать

- Если на защите ВКР согласятся на n=50 (текущий объём) — gold validation не критична. Можно оставить как есть и явно сказать «Wilson-CI ±13 п.п.» в §3.1.
- Если защита через 1–2 недели — оставить как есть, указать в §7.4 как направление дальнейшей работы.

Если защита через 3+ недели — gold validation на n=150 заметно усиливает работу.

  .venv/bin/python webapp/app.py --port 8000

---

## Pasta gold annotation (Trek D)

Specialised flow for the 250-product gold annotation (spec
`docs/superpowers/specs/2026-05-15-contribution-plan-design.md` §3).

### Build the sample
```bash
.venv/bin/python -m src.manual_label.sample_pasta_gold \
    --out datasets/manual_label/pasta_gold_250.csv
```
Stratification: 150 brand-disjoint test + ≤50 cascade/LLM disagreement + ≤50 gold-tier control (real run produced 150/39/50 = 239 due to disagreement pool exhaustion — see commit `28e7b1e`).

### Label
```bash
OMP_NUM_THREADS=1 .venv/bin/python datasets/manual_label/app.py --port 8000
```

Open <http://127.0.0.1:8000/pasta>. Keyboard shortcuts (when focus is on a manual-value input):
- `Tab` — next attribute input
- `1..9` — pick Nth chip in the focused attribute
- `s` — copy silver value into focused input
- `u` — toggle the focused attribute as `unsure`
- `Enter` — AJAX save + jump to next product

Filters via URL: `?only=empty`, `?only=disagreed`, `?only=<attr>_empty`.

Status per attribute: `empty | confident | unsure | conflict`. Auto-flips on input; `?` button toggles unsure manually.

Progress banner shows `done / total`, median typing pace, and ETA in hours (after ≥4 timestamps recorded).

### Cross-check via proxy-LLM (after labelling all 250)
```bash
.venv/bin/python -m src.eval.manual_label_proxy \
    --in datasets/manual_label/pasta_gold_250.csv \
    --out datasets/manual_label/pasta_gold_250_proxy.csv \
    --model qwen/qwen-2.5-72b-instruct
.venv/bin/python -m src.eval.manual_label_iaa \
    --gold datasets/manual_label/pasta_gold_250.csv \
    --proxy datasets/manual_label/pasta_gold_250_proxy.csv \
    --out datasets/processed/pasta_gold_iaa.parquet
```

Qwen 2.5 72B is chosen as proxy because it is outside the consensus_gold model family (Sonnet 4.5, GPT-4o, Gemini 2.5 Flash); human-vs-proxy Cohen's κ provides an independent cross-check.

Defensible κ planks: ≥ 0.70 for all attrs (≥ 0.60 acceptable for `pasta_shape` due to 12 classes with frequent boundary cases).
