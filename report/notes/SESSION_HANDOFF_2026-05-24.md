# Session Handoff — 2026-05-24

> Сохранение состояния для продолжения на более мощной машине.

## Главный pivot, который решён

**Заменить silver полностью на LLM-derived gold + conflict resolution с OFF tags.**

Старый silver формировался из OFF tags + regex, имеет ~20-30% noise на noisy attrs (pasta_shape 71% vs gold, chocolate_extra 65%, nutri_score_grade 71%). Новая стратегия:

1. Скачать OFF parquet (7.5GB) через DuckDB/HF
2. Filter по 3 main cats (pasta / chocolate / cheeses) + required fields
3. LLM-разметить весь clean subset через слабую модель (Gemini 2.5 Flash, ~$25-50)
4. Conflict resolution per attr type:
   - LLM vs OFF tag mismatch → log + apply policy (TYPE_A trust tag; TYPE_E trust LLM; TYPE_C compute from raw nutriments если есть)
5. Save as new gold-equivalent, заменить silver везде

## Что сделано в этой сессии

### Reviews / planning
- `report/notes/REVIEW_2026-05-24_scientific_critique.md` — научная рецензия v2 (с учётом мета-верификации)
- `report/notes/REVIEW_2026-05-24_meta_critique.md` — мета-ревью рецензии (опровергло Top-3 №3 про direct_llm leak)
- `report/notes/FIX_PLAN_2026-05-24.md` — план Approach C (3 итерации правок, 8/10 confidence)
- `report/notes/IMPLEMENTATION_PLAN_2026-05-24.md` — детальный plan для writing-plans skill
- `report/notes/PHASE0_FINDINGS_2026-05-24.md` — verification findings (commit `9053da4`)

### Code changes committed
| commit | что |
|---|---|
| `cb6e6a2` | `src/common.py`: добавлены `MAIN_CATEGORIES = ["pasta","chocolate","cheeses"]` и `ALL_CATEGORIES` |
| `26d47a0` | `src/data/split/generate_gold_splits.py`: brand_norm fix (canonical multi-brand sorted-join). Backups `*_pre_brand_norm_fix.parquet` сохранены |
| `46c7dbe` | docs(plan): correct paths after report/ reorg |
| `9053da4` | docs(thesis): Phase 0 verification findings |
| `fb10d5a` | `src/eval/per_taxonomy_breakdown.py` + `datasets/processed/headline_by_taxonomy.parquet` |

### Untracked (нужно решить — оставить или удалить)
- `datasets/processed/{pasta,chocolate,cheeses}_clean_full.parquet` — promezhutochnye filter results, можно удалить после нового подхода

## Critical findings (НЕ повторять старый подход)

### 1. Brand-norm fix имеет HUGE downstream impact
**Jaccard pre/post = 0.08-0.16** (не 0.85-0.95 как ожидалось). Brand_disjoint_split greedy bin-packer cascades. ML модели и cascade_preds_* сейчас на старых splits — формально stale. Headline на silver не пересчитан.

Но! cascade_preds_* на gold (250 manual annotated codes), а не на silver test. Поэтому headline на gold не drift'нул сильно.

### 2. Silver покрывает только ~8% данных
Из 15-21k продуктов в `*_stratified_silver_standard.parquet`, реально с silver labels — только **1250 per cat** (8%). Остальные **92% — unlabelled** в silver. `b3_full_gemini` уже labels ~25% (Gemini 2.5 Flash, sunk cost $5).

### 3. Silver vs manual gold disagreement
| attr | silver==gold agreement |
|---|---|
| pasta is_vegan | 100% |
| pasta is_organic, is_filled, is_gluten_free | 97-98% |
| pasta nutri_score_grade | **71%** |
| pasta pasta_shape | **71%** |
| chocolate chocolate_extra | **65%** |
| cheeses fat_class | **73%** |

Silver bug examples:
- `pasta_shape`: defaults to "noodles" если specific shape tag отсутствует (Tagliatelles aux oeufs → silver "noodles" вместо "tagliatelle")
- `nutri_score_grade`: silver = OFF `nutriscore_grade` field, иногда stale vs current OFF algorithm
- `chocolate_extra`: 35% disagree — structural bug в TYPE_F regex

### 4. Gemini Flash quality validated
Gemini-2.5-flash на 717 manual-gold-overlapping codes:
- pasta overall: **97.0%** match с gold
- chocolate: **90.8%**
- cheeses: **93.2%**

⚠ Но текущие b3 prompt имеет nulls на complex attrs:
- nutri_score_grade pasta: **97% null** (data limitation — proteins/fat не в большинстве silver-derived products, **salt_100g и fiber_100g вообще не в silver_standard**)
- protein_class chocolate: 53% null
- pasta_shape: 19.5% null

### 5. Raw OFF dumps имеют ВСЕ нутриенты
`{cat}_raw.parquet` (pasta_raw 38k, cheeses_raw 81k) содержат `salt_100g`, `fiber_100g`, `proteins_100g`, и т.д. Silver_standard эти поля обрезал. Используя raw → можно labelать с full nutriments → Gemini сможет реально compute Nutri-Score.

⚠ `chocolate_raw` MISSING (deleted при reorg `9341036`). Нужно либо restore из git history, либо regen через OFF download.

### 6. Phase 0 findings (relevant для prod conf)
- prod-Bayes использует **silver-DAG + hybrid gold-CPDs×10 + thresholds gold-tuned на val 50% brand-disjoint** (scenario c, defended на held 50%)
- ML headline std на 10 seeds = **0.10 п.п.** → ML очень устойчива, brand-norm fix вряд ли drift'нет headline более чем на 0.5 п.п.
- Original blind audit использовал **claude-opus-4** (default в `src/manual_label/opus_off_grounded_audit.py`)
- `direct_llm_v2.py` тоже импортирует `curate_prompt_fields` — patch off_field_filter может изменить direct_llm baseline. **Нужна параметризация blacklist**.

## Что delать в новой сессии

### Priority 1: Download OFF parquet

DuckDB hf:// pattern не справился с XET redirect (`TProtocolException: Invalid data`). HuggingFace datasets streaming работает (с warning HF_TOKEN), но медленно на моём интернете.

**На быстрой машине:**
```bash
# Вариант A — DuckDB query напрямую (если поддерживается):
.venv/bin/python -c "
import duckdb
con = duckdb.connect()
con.execute('INSTALL httpfs; LOAD httpfs;')
res = con.execute('''
    SELECT * FROM \"hf://datasets/openfoodfacts/product-database/food.parquet\"
    WHERE list_contains(categories_tags, 'en:chocolates')
       OR list_contains(categories_tags, 'en:pastas')
       OR list_contains(categories_tags, 'en:cheeses')
    LIMIT 10
''').fetchdf()
print(res)
"

# Вариант B — Download full file once (7.5GB):
huggingface-cli download openfoodfacts/product-database food.parquet --repo-type=dataset
# Затем DuckDB локально:
.venv/bin/python -c "
import duckdb
con = duckdb.connect()
# Filter to 3 cats with required fields
con.execute('''
    COPY (
      SELECT code, product_name, brands, ingredients_text, quantity, categories_tags, labels_tags,
             countries_tags, allergens_tags,
             nutriments['fat_100g'] AS fat_100g,
             nutriments['proteins_100g'] AS proteins_100g,
             nutriments['sugars_100g'] AS sugars_100g,
             nutriments['carbohydrates_100g'] AS carbohydrates_100g,
             nutriments['salt_100g'] AS salt_100g,
             nutriments['fiber_100g'] AS fiber_100g,
             nutriscore_grade, nova_group
      FROM food.parquet
      WHERE list_contains(categories_tags, 'en:chocolates')
         OR list_contains(categories_tags, 'en:pastas')
         OR list_contains(categories_tags, 'en:cheeses')
    ) TO 'datasets/processed/off_3cats_full.parquet' (FORMAT PARQUET);
''')
"
```

### Priority 2: Filter + LLM relabel

После получения off_3cats_full.parquet:

1. Filter on required fields: product_name, brands, ingredients_text, categories_tags, labels_tags, all 6 nutriments
2. Split per cat: `pasta_full.parquet`, `chocolate_full.parquet`, `cheeses_full.parquet`
3. LLM-разметить через `src/eval/direct_llm_v2.py` (или modified version):
   - Model: `google/gemini-2.5-flash` (proven 93-97% agreement с manual gold)
   - Mode: `partner_input` (без tags в prompt — чистая baseline)
   - Schema: `PASTA_SCHEMA / CHOCOLATE_SCHEMA / CHEESES_SCHEMA` из `src/pipeline/schemas`
4. **Conflict resolution per attr**:
   - **TYPE_A (is_organic, is_vegan, is_pdo, country_of_origin, milk_source)**: tag wins, LLM как fallback when tag absent
   - **TYPE_E (pasta_shape, chocolate_type, chocolate_extra, texture, contains_nuts)**: LLM wins (silver bug в tags)
   - **TYPE_C (nutri_score_grade, protein_class, fat_class, cocoa_percentage)**: compute from raw nutriments if present, else LLM
5. Save consolidated as `datasets/processed/{cat}_llm_gold_v1.parquet`
6. Log conflicts per attr → `conflicts_{cat}_{attr}.parquet` для последующего анализа

### Priority 3: Retrain ML on new gold

После llm_gold_v1:
- Retrain XGBoost (`src/pipeline/ml/train.py`) на новой ground truth
- Re-run cascade
- Headline на gold subset (250 codes) должен достичь ≥95% (sanity check)
- Если хуже — investigate

### Priority 4: Что остаётся из старого IMPLEMENTATION_PLAN

После замены silver:
- Phase 1.2 (threshold sensitivity) — становится тривиальной, можно retain
- Phase 1.3 (cost_quality_ci LLM honest) — relevant
- Phase 1.4 (FDR coreкция) — relevant
- Phase 1.5 (recompute_calibration bug) — relevant
- Phase 1.6 (class balance audit) — relevant
- Phase 1.8 (cocoa bucketize) — **уже не нужен**: новый gold-флоу унифицирует labelspace
- **Phase 2 (re-blind audit) — НЕ НУЖЕН**: новый LLM-gold replaces blind audit как validation
- Phase 3 (text disclaimers) — нужно переписать **полностью**: narrative «silver noise problem» вместо «circularity»
- Phase 4 (notebook updates) — нужно переписать
- Phase 5 (verification) — relevant

## Stack ссылок

- HF dataset: `openfoodfacts/product-database`, file `food.parquet` (7.5GB)
- URL: `hf://datasets/openfoodfacts/product-database@main/food.parquet`
- Альтернатива: `https://huggingface.co/datasets/openfoodfacts/product-database/resolve/main/food.parquet` (302 → XET CDN)
- DuckDB version: 1.5.3 installed в `.venv/bin/`
- Существующий LLM client: `src/llm/client.py` (OpenRouter + Ollama)
- Existing labelling code: `src/eval/direct_llm_v2.py` (supports `partner_input` и `off_grounded` modes)
- Pricing for Gemini 2.5 Flash: $0.30 / $2.50 per 1M tok input/output

## Git status

```
On branch main
Ahead of origin/main by 7 commits.
Untracked: datasets/processed/{pasta,chocolate,cheeses}_clean_full.parquet
```

7 commits до push'а — но push требует git push. Содержат code fixes + planning docs.

## SSH ключ

Стандартное место на macOS: `~/.ssh/` (содержит `id_ed25519` или `id_rsa`). Используется через `ssh-add` для аутентификации.
