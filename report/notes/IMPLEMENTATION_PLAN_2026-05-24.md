# Thesis Methodology Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Устранить методологические проблемы ВКР по Approach C (cheap code fixes + один LLM re-blind audit + текстовые disclaimers) на 3 main categories (pasta/chocolate/cheeses); подготовить headline-числа к защищаемому состоянию.

**Architecture:** Pipeline из 6 phases: (0) verification reads → (1) code fixes parallel → (2) blind audit re-run background → (3) text disclaimers → (4) notebook updates → (5) verification + commit. Phase 1.0 (brand-norm fix) идёт первым в Phase 1, так как меняет brand_disjoint splits и invalidates downstream. Скрипты параметризованы константой MAIN_CATEGORIES — следующая итерация expand'ит scope сменой одной константы.

**Tech Stack:** Python 3.14, pandas, sklearn, statsmodels (FDR correction), pgmpy (Bayes), Anthropic Sonnet 4.6 (blind audit re-run), matplotlib (notebook plots), git.

**Spec:** `/Users/miafrolov/Desktop/stuff/ai_attributes/docs/thesis/FIX_PLAN_2026-05-24.md`
**Source critique:** `REVIEW_2026-05-24_scientific_critique.md` + `REVIEW_2026-05-24_meta_critique.md`

**MAIN_CATEGORIES:** `["pasta", "chocolate", "cheeses"]` (3 main); остальные 4 (beverages/cereals/cosmetics/electronics) — следующая итерация.

**Что НЕ покрывает план:** sync `.md` глав в `VKR_Frolov_2026.docx` (отложено до LaTeX-конверсии); cross-category transfer ML; русский OOD; TXtract/MAVE/OpenTag baselines.

---

## File Structure

**Files to create:**
- `src/eval/per_taxonomy_breakdown.py` — Phase 1.1
- `src/eval/threshold_sensitivity.py` — Phase 1.2
- `src/eval/class_balance_audit.py` — Phase 1.6
- `src/eval/cocoa_percentage_labelspace_fix.py` — Phase 1.8
- `docs/thesis/PHASE0_FINDINGS_2026-05-24.md` — Phase 0 output report

**Files to modify:**
- `src/common.py` — добавить `MAIN_CATEGORIES` (shared constant)
- `src/data/split/generate_gold_splits.py:23-24` — brand_norm fix (Phase 1.0)
- `src/eval/cost_quality_ci.py:118-124` — LLM proxy honest measurement (Phase 1.3)
- `src/eval/cascade_vs_llm_stats.py` — добавить BH FDR (Phase 1.4)
- `src/pipeline/ml/train.py:614-622` — recompute_calibration ECE bug fix (Phase 1.5)
- `src/manual_label/off_field_filter.py:17-27` — расширить blacklist (Phase 2.1)
- `docs/thesis/03_chapter3_implementation.md` — disclaimers + sections (Phase 3)
- `docs/thesis/04_chapter4_results.md` — demo disclaimer (Phase 3.4)
- `docs/thesis/05_conclusion.md` — Limitations section (Phase 3.6)
- `docs/thesis/pre_registration_2026-Q2.md` → `phase2_analysis_plan_2026-Q2.md` (Phase 3.5)
- `notebooks/00_thesis_main.ipynb` — новые ячейки (Phase 4)
- `demo/ml_service/cascade.py` — restrict до 3 main cats (Phase 3.4)

---

## Phase 0 — Verification reads (1.5 дня)

**Цель:** Прочитать 10 неподтверждённых файлов и собрать findings в `PHASE0_FINDINGS_2026-05-24.md`. Findings определяют branching в Phase 1.2, 2.2, 3.2.

### Task 0.1: Read validator_hypothesis_tests.py

**Files:**
- Read: `src/eval/validator_hypothesis_tests.py`
- Output: append section в `docs/thesis/PHASE0_FINDINGS_2026-05-24.md`

- [ ] **Step 1: Read full file**

Read tool на `/Users/miafrolov/Desktop/stuff/ai_attributes/src/eval/validator_hypothesis_tests.py`.

- [ ] **Step 2: Extract answers to specific questions**

Записать в findings:
- Какие 3 атрибута активны для Bayes-validator? (constant или from-config?)
- Как выбраны? (комментарий в коде или git blame на decision commit)
- Verdict: `OK` (если criterion явный) / `cherry-picking concern` (если post-hoc selection)

- [ ] **Step 3: Append to findings file**

Создать или append:
```markdown
# Phase 0 findings — 2026-05-24

## 0.1 validator_hypothesis_tests.py
- Активны атрибуты: [список]
- Criterion: [описание]
- Verdict: [OK / cherry-picking concern / blocker]
```

### Task 0.2: Read accuracy_squeeze_holdout.py

**Files:**
- Read: `src/experiments/accuracy_squeeze_holdout.py` (NOT `src/eval/` как в spec; реально лежит в experiments)
- Output: append `PHASE0_FINDINGS_2026-05-24.md`

- [ ] **Step 1: Read full file**

- [ ] **Step 2: Identify prod-config code path**
- DAG: silver-обученный (HillClimb), gold-refit (refit на 200-250 cells), или hand-crafted?
- CPDs: gold-CPD×10 (10× upweight), silver-CPD, или mixture?
- Thresholds: те же из train.py thresholds.pkl, или новые gold-tuned?

- [ ] **Step 3: Branching determination**

Записать verdict с **outcome label** (a/b/c) из spec §0.2:
- (a) DAG silver-обученный, фиксирован, edges стабильны → Phase 3.2 идёт как plan.
- (b) DAG silver-обученный, нестабильный → escalate, Phase 3.2 переписывается.
- (c) Production-path использует gold-refit → новая disclosure нужна.

### Task 0.3: Read cv_stability_groupkfold.py + artifact

**Files:**
- Read: `src/diagnostics/ml/cv_stability_groupkfold.py`
- Read artifact: `datasets/processed/cv_stability_*` (если есть)

- [ ] **Step 1: Read script**

- [ ] **Step 2: Check artifacts**
```bash
ls /Users/miafrolov/Desktop/stuff/ai_attributes/datasets/processed/ | grep -i "cv_stab\|multi_seed\|10seed"
```

- [ ] **Step 3: Extract multi-seed std**

Загрузить parquet (если есть) и посчитать std headline по seed'ам. Записать число в findings.

- [ ] **Step 4: Verdict**

- std < 0.5 п.п. → headline точно измерен, claim «+1.03 пп Bayes» статистически устойчив.
- std 0.5–1.5 → in range, нужен явный disclosure в §3.
- std ≥ 1.5 → claim «+1.03 пп» **не значим**, narrative переписывается.

### Task 0.4: Read demo/ml_service/validator.py

**Files:**
- Read: `demo/ml_service/validator.py`

- [ ] **Step 1: Read file**

- [ ] **Step 2: Determine fit/serve overlap**

Вопросы:
- На каком dataset обучается validator? (silver / gold / nothing — pre-trained .pkl?)
- Какие products потом показываются в demo? (LIVE input или из silver test split?)
- Есть ли overlap?

- [ ] **Step 3: Verdict**

- OK (no overlap, validator pre-trained) → idti dalshe.
- Leak в demo → add disclosure в Phase 3.4.

### Task 0.5: Read router_pre_registered.py + router_loco_gold.py

**Files:**
- Read: `src/eval/router_pre_registered.py`
- Read: `src/eval/router_loco_gold.py`

- [ ] **Step 1: Read оба файла**

- [ ] **Step 2: Verify протокол H1 негативного результата**
- Какие фичи использует router? (`grep -n "feature_cols\|FEATURES" router_*.py`)
- Если категориальные leaks (например, `category_id` как feature) — это слабость.
- Какой split: brand-disjoint или random?

- [ ] **Step 3: Verdict**

- OK → spec narrative о H1 fail остаётся.
- Issue (categorical leak или random split) → add в Limitations.

### Task 0.6: Read e1_circularity_analysis.py

**Files:**
- Read: `src/experiments/e1_circularity_analysis.py`

- [ ] **Step 1: Read file**

- [ ] **Step 2: Determine overlap с Phase 1.1**
- Делает ли e1 per-taxonomy breakdown?
- Если **уже делает** — Phase 1.1 (`per_taxonomy_breakdown.py`) переиспользует e1 вместо нового скрипта. Update Task 1.1 accordingly.

- [ ] **Step 3: Verdict**

- e1 дублирует → reuse: Phase 1.1 становится `python -m src.experiments.e1_circularity_analysis` + sanity check.
- e1 другой scope → Phase 1.1 идёт как plan.

### Task 0.7: Identify blind audit model

**Files:**
- Read: `src/manual_label/opus_off_grounded_audit.py`
- Grep: `grep -rn "claude\|opus\|sonnet\|haiku\|gpt" src/manual_label/ src/llm/client.py | head -30`

- [ ] **Step 1: Read opus_off_grounded_audit.py**

- [ ] **Step 2: Find model_id used in original audit**

Команда:
```bash
grep -rn "model.*=\|MODEL\s*=" /Users/miafrolov/Desktop/stuff/ai_attributes/src/manual_label/opus_off_grounded_audit.py /Users/miafrolov/Desktop/stuff/ai_attributes/src/llm/client.py
```

- [ ] **Step 3: Record findings + model decision**

Record: какая модель в оригинале. Apples-to-apples decision:
- Если Opus → Phase 2 default Sonnet 4.6 + Opus subsample (50 cells) для validation.
- Если Sonnet → Phase 2 default Sonnet 4.6 без validation.
- Если Haiku → Phase 2 default Haiku 4.5 (apples-to-apples).

### Task 0.8: Grep downstream blacklist consumers

- [ ] **Step 1: Run grep**

```bash
grep -rn "curate_prompt_fields\|DERIVED_BLACKLIST\|off_field_filter" \
  /Users/miafrolov/Desktop/stuff/ai_attributes/src/ \
  /Users/miafrolov/Desktop/stuff/ai_attributes/demo/
```

- [ ] **Step 2: Categorize consumers**

Для каждого хита определить:
- (a) Только `opus_off_grounded_audit.py` → patch безопасен.
- (b) `direct_llm_v2.py` consumer и использован в headline → нужна параметризация blacklist (Task 1.9 add'ится).
- (c) demo consumer → проверить runtime impact.

### Task 0.9: Verify direct_llm_v2.py не leak'ит

**Files:**
- Read: `src/eval/direct_llm_v2.py`

- [ ] **Step 1: Read full file**

- [ ] **Step 2: Check INPUT_FIELDS construction**

Найти строки, где product dict передаётся в `enrich_product` или `build_prompt`. Проверить, фильтруются ли поля до partner-only (как в v1, line 55).

- [ ] **Step 3: Verdict**

- v2 чистый → OK.
- v2 leak'ит → add Task 1.9 (rebuild headline без leak'нутого prompt).

### Task 0.10: Pre-flight summary commit

- [ ] **Step 1: Write summary in PHASE0_FINDINGS_2026-05-24.md**

В конце добавить:
```markdown
## Summary — branching impact on plan

| Phase task | Branching outcome | Action |
|---|---|---|
| 2.2 model | [Opus/Sonnet/Haiku from 0.7] | [decision] |
| 3.2 Bayes narrative | [a/b/c from 0.2] | [decision] |
| 1.2 threshold sensitivity | [std value from 0.3] | [decision] |
| 1.1 taxonomy breakdown | [reuse e1 or new from 0.6] | [decision] |
| 1.9 direct_llm_v2 leak | [yes/no from 0.9] | [add task if yes] |
| Blacklist consumers | [from 0.8] | [parametrize or not] |
```

- [ ] **Step 2: Commit findings file**

```bash
git add docs/thesis/PHASE0_FINDINGS_2026-05-24.md
git commit -m "docs(thesis): Phase 0 verification findings"
```

---

## Phase 1.0 — Brand-norm fix (FIRST, 3 hours)

**Critical:** идёт первым, так как меняет brand_disjoint splits → invalidates все downstream brand-disjoint метрики.

### Task 1.0.1: Add MAIN_CATEGORIES to common.py

**Files:**
- Modify: `src/common.py` — append after line 20

- [ ] **Step 1: Edit src/common.py**

После `PARTNER_TEXT_FIELDS = [...]` (line 20) добавить:

```python
# 3 main categories для текущей итерации fix-cycle.
# Остальные 4 (beverages, cereals, cosmetics, electronics) — следующая итерация.
MAIN_CATEGORIES = ["pasta", "chocolate", "cheeses"]
ALL_CATEGORIES = ["pasta", "chocolate", "beverages", "cheeses", "cereals", "cosmetics"]
```

- [ ] **Step 2: Commit**

```bash
git add src/common.py
git commit -m "feat(common): add MAIN_CATEGORIES constant for scoped fixes"
```

### Task 1.0.2: Backup current brand_disjoint splits

**Files:**
- Backup: `datasets/processed/*_gold_split.parquet` → `*_gold_split_pre_brand_norm_fix.parquet`

- [ ] **Step 1: Backup script**

```bash
cd /Users/miafrolov/Desktop/stuff/ai_attributes
for cat in pasta chocolate cheeses; do
  cp "datasets/processed/${cat}_gold_split.parquet" \
     "datasets/processed/${cat}_gold_split_pre_brand_norm_fix.parquet"
done
ls -la datasets/processed/*_gold_split*.parquet
```

Expected: 6 files (3 original + 3 backups).

- [ ] **Step 2: Verify backups loadable**

```bash
.venv/bin/python -c "
import pandas as pd
for cat in ['pasta', 'chocolate', 'cheeses']:
    df = pd.read_parquet(f'datasets/processed/{cat}_gold_split_pre_brand_norm_fix.parquet')
    print(f'{cat}: {len(df)} rows, splits: {df.split.value_counts().to_dict()}')
"
```

### Task 1.0.3: Patch generate_gold_splits.py

**Files:**
- Modify: `src/data/split/generate_gold_splits.py:23-24`

- [ ] **Step 1: Edit brand_norm logic**

Заменить строки 23-24:
```python
        silver["brand_norm"] = silver["brands"].fillna("UNKNOWN").astype(str) \
                                .str.split(",").str[0].str.strip().str.lower()
```

на:
```python
        # Canonical multi-brand norm: «Carrefour, Carrefour BIO» и «Carrefour BIO, Carrefour»
        # дают одинаковый brand_norm «carrefour|carrefour bio», исключая subbrand leak.
        silver["brand_norm"] = (
            silver["brands"].fillna("UNKNOWN").astype(str).str.lower()
            .str.split(",")
            .apply(lambda parts: "|".join(sorted(p.strip() for p in parts if p.strip())))
        )
```

- [ ] **Step 2: Restrict FOOD_CATS к MAIN_CATEGORIES для этого прогона**

Заменить line 14:
```python
FOOD_CATS = ["pasta", "chocolate", "beverages", "cheeses", "cereals", "cosmetics"]
```

на:
```python
from src.common import MAIN_CATEGORIES
FOOD_CATS = MAIN_CATEGORIES  # 3 main cats для текущей итерации; для следующей — ALL_CATEGORIES
```

### Task 1.0.4: Regenerate brand_disjoint splits for 3 cats

- [ ] **Step 1: Run patched script**

```bash
cd /Users/miafrolov/Desktop/stuff/ai_attributes
OMP_NUM_THREADS=1 .venv/bin/python -m src.data.split.generate_gold_splits
```

Expected log lines (примерно):
```
[pasta] train=... val=... test=... → datasets/processed/pasta_gold_split.parquet
[chocolate] train=... val=... test=... → datasets/processed/chocolate_gold_split.parquet
[cheeses] train=... val=... test=... → datasets/processed/cheeses_gold_split.parquet
```

- [ ] **Step 2: Compare pre/post split sizes**

```bash
.venv/bin/python -c "
import pandas as pd
for cat in ['pasta', 'chocolate', 'cheeses']:
    old = pd.read_parquet(f'datasets/processed/{cat}_gold_split_pre_brand_norm_fix.parquet')
    new = pd.read_parquet(f'datasets/processed/{cat}_gold_split.parquet')
    old_codes = set(old[old.split=='test'].code.astype(str))
    new_codes = set(new[new.split=='test'].code.astype(str))
    print(f'{cat}: pre_test={len(old_codes)} post_test={len(new_codes)} jaccard={len(old_codes & new_codes)/len(old_codes | new_codes):.3f}')
"
```

Expected: jaccard 0.85–0.95 (мало кодов перешло между splits — большинство brand'ов канонизируются одинаково).

### Task 1.0.5: Measure affected products

- [ ] **Step 1: Count subbrand-affected products**

```bash
.venv/bin/python -c "
import pandas as pd
for cat in ['pasta', 'chocolate', 'cheeses']:
    silver = pd.read_parquet(f'datasets/processed/{cat}_stratified_silver_standard.parquet')
    silver['brand_old'] = silver['brands'].fillna('UNKNOWN').astype(str).str.split(',').str[0].str.strip().str.lower()
    silver['brand_new'] = (silver['brands'].fillna('UNKNOWN').astype(str).str.lower()
                           .str.split(',')
                           .apply(lambda parts: '|'.join(sorted(p.strip() for p in parts if p.strip()))))
    diff = (silver['brand_old'] != silver['brand_new']).sum()
    multi_brand = silver['brands'].astype(str).str.contains(',').sum()
    print(f'{cat}: total={len(silver)} multi_brand_products={multi_brand} norm_diff={diff} ({100*diff/len(silver):.1f}%)')
"
```

Записать числа — пригодятся для disclosure в Phase 3.

- [ ] **Step 2: Commit brand-norm fix**

```bash
git add src/data/split/generate_gold_splits.py datasets/processed/{pasta,chocolate,cheeses}_gold_split.parquet datasets/processed/{pasta,chocolate,cheeses}_gold_split_pre_brand_norm_fix.parquet
git commit -m "fix(splits): canonical multi-brand norm closes subbrand leak (3 main cats)"
```

### Task 1.0.6: Regenerate downstream brand-disjoint artefacts

Brand_disjoint splits изменились — downstream артефакты, использующие splits, нужно пересчитать. Найдём их и пересчитаем.

- [ ] **Step 1: Find consumers of *_gold_split.parquet**

```bash
grep -rn "_gold_split\.parquet\|gold_split_" \
  /Users/miafrolov/Desktop/stuff/ai_attributes/src/ \
  /Users/miafrolov/Desktop/stuff/ai_attributes/notebooks/ | head -30
```

Записать список consumer-скриптов.

- [ ] **Step 2: For each consumer, decide action**

Для каждого consumer:
- Если скрипт читает split + строит метрики → re-run.
- Если скрипт читает split только для filtering display → skip (он автоматически подхватит новый split при следующем notebook re-run).

Минимальный обязательный re-run: `cost_quality_ci.py` (см. Phase 1.3), `cascade_vs_llm_stats.py` (см. Phase 1.4), `bootstrap_ci_brand_clustered.py` (если есть).

- [ ] **Step 3: Run minimal essential regen**

```bash
# Re-run cost_quality_ci с новыми splits (содержит свой bootstrap)
OMP_NUM_THREADS=1 .venv/bin/python -m src.eval.cost_quality_ci
# Output: datasets/processed/cost_quality_ci.parquet (overwritten)
```

- [ ] **Step 4: Commit regenerated artefacts**

```bash
git add datasets/processed/cost_quality_ci.parquet
git commit -m "fix(eval): regenerate cost_quality_ci with corrected brand splits"
```

---

## Phase 1.1 — Per-taxonomy headline breakdown

**Depends on:** Task 0.6 outcome — если e1_circularity_analysis уже делает breakdown, заменить новый скрипт reuse'ом.

### Task 1.1.1: Read existing taxonomy artifact

- [ ] **Step 1: Verify artifact structure**

```bash
.venv/bin/python -c "
import pandas as pd
tax = pd.read_parquet('datasets/processed/attribute_signal_taxonomy.parquet')
print('Columns:', list(tax.columns))
print('signal_types:', tax.signal_type.value_counts().to_dict())
print('Main 3 cats:')
print(tax[tax.category.isin(['pasta','chocolate','cheeses'])][['category','attr','signal_type']].to_string())
"
```

### Task 1.1.2: Write per_taxonomy_breakdown.py

**Files:**
- Create: `src/eval/per_taxonomy_breakdown.py`

- [ ] **Step 1: Write script**

```python
"""Per-taxonomy headline breakdown for MAIN_CATEGORIES.

Группирует accuracy и coverage по signal_type (nutri_derived / tag_derived / text_derived).
Это разбивает монолитный headline 91.5% на три уровня confidence:
- nutri_derived: наименьший circularity risk (silver из raw нутриентов).
- tag_derived: moderate (silver из labels_tags/categories_tags; blind audit на этих не валидирует — см. §2.10).
- text_derived: высокий circularity risk (silver = regex по ingredients_text/product_name; ML — embeddings тех же полей).

Output: datasets/processed/headline_by_taxonomy.parquet
Columns: category, signal_type, n_cells, accuracy, ci_lo, ci_hi, n_brands
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import MAIN_CATEGORIES, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

PROCESSED = Path(PROCESSED_DIR)
N_BOOTSTRAP = 1000
RNG_SEED = 42


def load_per_product_with_taxonomy(cat: str) -> pd.DataFrame:
    """Per-product cascade results with signal_type joined."""
    exp = pd.read_parquet(PROCESSED / f"experiment_per_product_{cat}_stratified.parquet")
    tax = pd.read_parquet(PROCESSED / "attribute_signal_taxonomy.parquet")
    tax_cat = tax[tax.category == cat][["attr", "signal_type"]]
    brands = pd.read_parquet(
        PROCESSED / f"{cat}_stratified_silver_standard.parquet",
        columns=["code", "brands"],
    )
    brands["code"] = brands["code"].astype(str)
    exp["code"] = exp["code"].astype(str)
    merged = exp.merge(tax_cat, on="attr", how="inner").merge(brands, on="code", how="left")
    merged["correct"] = (merged["pred"] == merged["gt"]).astype(int)
    merged["non_null_gt"] = merged["gt"].notna().astype(int)
    return merged


def bootstrap_ci(df: pd.DataFrame, n_iter: int = N_BOOTSTRAP, seed: int = RNG_SEED) -> tuple[float, float, float]:
    """Brand-clustered bootstrap: resampling брендов, не cells."""
    df = df[df.non_null_gt == 1].copy()
    if len(df) == 0:
        return float("nan"), float("nan"), float("nan")
    brands = df.brands.unique()
    by_brand = {b: df[df.brands == b] for b in brands}
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(n_iter):
        sampled = rng.choice(brands, size=len(brands), replace=True)
        parts = [by_brand[b] for b in sampled]
        boot = pd.concat(parts, ignore_index=True)
        if len(boot) > 0:
            accs.append(boot.correct.mean())
    central = df.correct.mean()
    return float(central), float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def main():
    rows = []
    for cat in MAIN_CATEGORIES:
        logger.info("Processing %s", cat)
        merged = load_per_product_with_taxonomy(cat)
        for st in ["nutri_derived", "tag_derived", "text_derived"]:
            sub = merged[merged.signal_type == st]
            if len(sub) == 0:
                continue
            n_brands = sub.brands.nunique()
            acc, lo, hi = bootstrap_ci(sub)
            n_cells = (sub.non_null_gt == 1).sum()
            rows.append({
                "category": cat,
                "signal_type": st,
                "n_cells": int(n_cells),
                "n_brands": int(n_brands),
                "accuracy": acc,
                "ci_lo": lo,
                "ci_hi": hi,
            })
            logger.info("  %s/%s: n=%d acc=%.4f CI=[%.4f, %.4f]",
                        cat, st, n_cells, acc, lo, hi)
    out = pd.DataFrame(rows)
    out_path = PROCESSED / "headline_by_taxonomy.parquet"
    out.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(out))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run script**

```bash
cd /Users/miafrolov/Desktop/stuff/ai_attributes
OMP_NUM_THREADS=1 .venv/bin/python -m src.eval.per_taxonomy_breakdown
```

Expected: 3 cats × до 3 signal_types = до 9 rows.

- [ ] **Step 3: Print breakdown table**

```bash
.venv/bin/python -c "
import pandas as pd
df = pd.read_parquet('datasets/processed/headline_by_taxonomy.parquet')
print(df.to_string(index=False))
"
```

- [ ] **Step 4: Commit**

```bash
git add src/eval/per_taxonomy_breakdown.py datasets/processed/headline_by_taxonomy.parquet
git commit -m "feat(eval): per-taxonomy headline breakdown for 3 main cats"
```

---

## Phase 1.2 — Threshold sensitivity analysis

**Approach:** использовать `conf` колонку из `experiment_per_product_*.parquet` (уже есть), пере-фильтровать по разным thresholds без re-prediction — намного проще чем загружать модели.

### Task 1.2.1: Verify conf column exists

- [ ] **Step 1: Check column**

```bash
.venv/bin/python -c "
import pandas as pd
for cat in ['pasta', 'chocolate', 'cheeses']:
    df = pd.read_parquet(f'datasets/processed/experiment_per_product_{cat}_stratified.parquet')
    print(f'{cat}: cols={list(df.columns)} layers={df.layer.unique().tolist()}')
"
```

Expected: includes `conf` column.

### Task 1.2.2: Write threshold_sensitivity.py

**Files:**
- Create: `src/eval/threshold_sensitivity.py`

- [ ] **Step 1: Write script**

```python
"""Sensitivity analysis: headline as function of confidence threshold.

Phase 1.2 fix for §2.1 (threshold-on-test snooping). Использует уже сохранённые
per-product conf'ы из experiment_per_product_*.parquet — не требует re-prediction.

Output: datasets/processed/threshold_sensitivity.parquet
Columns: category, attr, threshold, n_covered, accuracy_on_covered, accuracy_overall

Critical check (см. spec): сравнивать delta(current vs uniform 0.5) с half-width
brand-clustered 95% CI (из bootstrap_ci_brand_clustered.parquet). Если delta < CI/2 —
concern в пределах шума.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import MAIN_CATEGORIES, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

PROCESSED = Path(PROCESSED_DIR)
THRESHOLDS = [0.0, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8]


def threshold_metrics(df: pd.DataFrame, threshold: float) -> dict:
    """Apply threshold to ML layer predictions only; regex/bayes layers pass through."""
    eligible = df[df.layer.isin(["ml"])].copy()
    other = df[~df.layer.isin(["ml"])].copy()
    # ML predictions below threshold abstain
    eligible_covered = eligible[eligible.conf >= threshold]
    # Coverage = ml_covered + regex + bayes (those don't have threshold)
    covered = pd.concat([eligible_covered, other[other.layer != "none"]], ignore_index=True)
    covered = covered.dropna(subset=["gt"])
    n_covered = len(covered)
    if n_covered == 0:
        return {"n_covered": 0, "accuracy_on_covered": float("nan"), "accuracy_overall": float("nan")}
    acc_cov = (covered.pred == covered.gt).mean()
    # Overall: assume LLM 0% on abstained (lower bound) → conservative
    n_total = df.dropna(subset=["gt"]).shape[0]
    acc_overall = (covered.pred == covered.gt).sum() / n_total if n_total else 0
    return {
        "n_covered": int(n_covered),
        "accuracy_on_covered": float(acc_cov),
        "accuracy_overall": float(acc_overall),
    }


def main():
    rows = []
    for cat in MAIN_CATEGORIES:
        df = pd.read_parquet(PROCESSED / f"experiment_per_product_{cat}_stratified.parquet")
        for attr in sorted(df.attr.unique()):
            sub = df[df.attr == attr]
            for t in THRESHOLDS:
                m = threshold_metrics(sub, t)
                rows.append({
                    "category": cat,
                    "attr": attr,
                    "threshold": t,
                    **m,
                })
    out = pd.DataFrame(rows)
    out_path = PROCESSED / "threshold_sensitivity.parquet"
    out.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(out))

    # Summary: delta(tuned current vs 0.5) per (cat, attr)
    pivot = out.pivot_table(index=["category", "attr"], columns="threshold",
                             values="accuracy_overall")
    if 0.5 in pivot.columns:
        for col in pivot.columns:
            if col != 0.5:
                pivot[f"delta_{col}_vs_0.5"] = pivot[col] - pivot[0.5]
        logger.info("Pivot summary:\n%s", pivot.to_string())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run script**

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m src.eval.threshold_sensitivity
```

- [ ] **Step 3: Compute delta vs CI**

```bash
.venv/bin/python -c "
import pandas as pd
ts = pd.read_parquet('datasets/processed/threshold_sensitivity.parquet')
# Per category aggregated headline
agg = ts.groupby(['category', 'threshold']).agg(
    total_covered=('n_covered', 'sum'),
    total_correct=('accuracy_overall', lambda x: (x * ts.loc[x.index, 'n_covered']).sum()),
).reset_index()
print('Per-category headline by threshold:')
pivot = ts.groupby(['category', 'threshold'])['accuracy_overall'].mean().unstack('threshold')
print(pivot.round(4))
print()
print('Delta 0.5 vs current (assuming current ~0.7):')
if 0.5 in pivot.columns and 0.7 in pivot.columns:
    print((pivot[0.7] - pivot[0.5]).round(4))
"
```

- [ ] **Step 4: Commit**

```bash
git add src/eval/threshold_sensitivity.py datasets/processed/threshold_sensitivity.parquet
git commit -m "feat(eval): threshold sensitivity analysis using saved conf cols"
```

---

## Phase 1.3 — cost_quality_ci honest LLM measurement

### Task 1.3.1: Verify llm_fallback_eval artifact

- [ ] **Step 1: Check structure**

```bash
.venv/bin/python -c "
import pandas as pd
for cat in ['pasta', 'chocolate', 'cheeses']:
    df = pd.read_parquet(f'datasets/processed/llm_fallback_eval_{cat}_stratified.parquet')
    print(f'{cat}: cols={list(df.columns)} n={len(df)}')
    print(df.head(3).to_string(index=False))
    print()
"
```

Expected: columns include `correct`, `attr`, possibly `model`.

### Task 1.3.2: Patch cost_quality_ci.py

**Files:**
- Modify: `src/eval/cost_quality_ci.py:118-124`

- [ ] **Step 1: Add helper to load abstain LLM accuracy**

В начало `cost_quality_ci.py` (после `load_llm_per_attr` definition или рядом) добавить функцию:

```python
def load_llm_acc_on_abstain() -> dict[str, dict[tuple[str, str], float]]:
    """Honest LLM accuracy измеренная на cells где cascade abstained.
    
    Артефакт llm_fallback_eval_{cat}_stratified.parquet содержит per-cell LLM
    результаты на абстейн-выборке (layer='none' filter уже применён в
    layer4_llm.py:86). Возвращаем dict[model][cat, attr] = mean(correct).
    """
    result: dict[str, dict[tuple[str, str], float]] = {}
    for cat in CATEGORIES:
        path = PROCESSED / f"llm_fallback_eval_{cat}_stratified.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        # Assume single model per file; if multi-model, group by 'model' column
        if "model" in df.columns:
            for model, mdf in df.groupby("model"):
                result.setdefault(str(model), {})
                grp = mdf.groupby("attr")["correct"].mean()
                for attr, acc in grp.items():
                    result[str(model)][(cat, str(attr))] = float(acc)
        else:
            # No model column — assume llm_fallback model default
            result.setdefault("llm_fallback", {})
            grp = df.groupby("attr")["correct"].mean()
            for attr, acc in grp.items():
                result["llm_fallback"][(cat, str(attr))] = float(acc)
    return result
```

- [ ] **Step 2: Wire into compute_grand**

Найти `compute_grand` function. Перед циклом `for model in LLM_MODELS:` добавить:

```python
    llm_abstain_acc = load_llm_acc_on_abstain()
```

В loop'е (line ~118-124) заменить:

```python
        # cascade_plus_<m>: proxy * router_acc, weighted by n_test
        num = denom = 0.0
        for cat, attr, n, cov, acc_cov in casc_stats:
            llm_a = per_attr.get((cat, attr), 0.0)
            proxy = cov * acc_cov + (1 - cov) * llm_a
            proxy *= ROUTER_ACC.get(cat, 1.0)
            num += proxy * n
            denom += n
        out[f"cascade_plus_{model}"] = float(num / denom) if denom else 0.0
```

на:

```python
        # cascade_plus_<m>: proxy * router_acc, weighted by n_test
        # HONEST: на абстейн-ячейках используем реально измеренную LLM accuracy
        # (а не среднюю по всему attr), потому что абстейн-ячейки систематически трудные.
        abstain_per = llm_abstain_acc.get(model) or llm_abstain_acc.get("llm_fallback") or {}
        num = denom = 0.0
        for cat, attr, n, cov, acc_cov in casc_stats:
            llm_a_abstain = abstain_per.get((cat, attr))
            if llm_a_abstain is None:
                # Fallback на старое поведение если honest measurement отсутствует
                llm_a_abstain = per_attr.get((cat, attr), 0.0)
            proxy = cov * acc_cov + (1 - cov) * llm_a_abstain
            proxy *= ROUTER_ACC.get(cat, 1.0)
            num += proxy * n
            denom += n
        out[f"cascade_plus_{model}"] = float(num / denom) if denom else 0.0
```

- [ ] **Step 3: Backup old output, regenerate**

```bash
cp datasets/processed/cost_quality_ci.parquet datasets/processed/cost_quality_ci_pre_honest_llm.parquet
OMP_NUM_THREADS=1 .venv/bin/python -m src.eval.cost_quality_ci
```

- [ ] **Step 4: Compare pre/post**

```bash
.venv/bin/python -c "
import pandas as pd
old = pd.read_parquet('datasets/processed/cost_quality_ci_pre_honest_llm.parquet')
new = pd.read_parquet('datasets/processed/cost_quality_ci.parquet')
print('OLD:')
print(old[['config', 'acc']].to_string(index=False))
print()
print('NEW:')
print(new[['config', 'acc']].to_string(index=False))
print()
merged = old.merge(new, on='config', suffixes=('_old', '_new'))
merged['delta'] = merged.acc_new - merged.acc_old
print('Deltas:')
print(merged[['config', 'acc_old', 'acc_new', 'delta']].to_string(index=False))
"
```

Expected: cascade_plus_* values shift by -0.5 to -1.5 п.п. (downward, more honest).

- [ ] **Step 5: Commit**

```bash
git add src/eval/cost_quality_ci.py datasets/processed/cost_quality_ci.parquet datasets/processed/cost_quality_ci_pre_honest_llm.parquet
git commit -m "fix(eval): honest LLM accuracy on abstain cells in cost-quality CI"
```

---

## Phase 1.4 — FDR correction in cascade_vs_llm_stats

### Task 1.4.1: Check statsmodels availability

- [ ] **Step 1: Verify install**

```bash
.venv/bin/python -c "import statsmodels.stats.multitest as mt; print(mt.multipletests.__doc__[:200])"
```

Если ImportError:
```bash
.venv/bin/pip install statsmodels
```

### Task 1.4.2: Patch cascade_vs_llm_stats.py

**Files:**
- Modify: `src/eval/cascade_vs_llm_stats.py`

- [ ] **Step 1: Find where p-values aggregated**

```bash
grep -n "p_value\|pvalue" /Users/miafrolov/Desktop/stuff/ai_attributes/src/eval/cascade_vs_llm_stats.py
```

- [ ] **Step 2: After p-values aggregated into rows, add FDR before save**

Найти `def main()` или место, где `rows` собираются в DataFrame с p-values. Перед `df.to_parquet(...)` добавить:

```python
    # BH FDR коррекция на ~40 per-(cat, attr) сравнений.
    # Заявленные «significant» без коррекции дают ~2 false positives на α=0.05.
    from statsmodels.stats.multitest import multipletests
    df_out = pd.DataFrame(rows)
    if "p_value" in df_out.columns and len(df_out) > 0:
        valid = df_out["p_value"].notna() & (df_out["p_value"] <= 1.0)
        if valid.any():
            _, p_adj, _, _ = multipletests(
                df_out.loc[valid, "p_value"].values,
                alpha=0.05,
                method="fdr_bh",
            )
            df_out["p_value_fdr_bh"] = float("nan")
            df_out.loc[valid, "p_value_fdr_bh"] = p_adj
            df_out["significant_at_0.05_fdr"] = (df_out["p_value_fdr_bh"] < 0.05).astype("Int64")
        else:
            df_out["p_value_fdr_bh"] = float("nan")
            df_out["significant_at_0.05_fdr"] = pd.NA
```

(Adapt to actual code structure — `rows` may already be DataFrame; adjust accordingly.)

- [ ] **Step 3: Re-run script**

```bash
cp datasets/processed/cascade_vs_llm_stats.parquet datasets/processed/cascade_vs_llm_stats_pre_fdr.parquet
OMP_NUM_THREADS=1 .venv/bin/python -m src.eval.cascade_vs_llm_stats
```

- [ ] **Step 4: Compare significance count pre/post FDR**

```bash
.venv/bin/python -c "
import pandas as pd
df = pd.read_parquet('datasets/processed/cascade_vs_llm_stats.parquet')
if 'significant_at_0.05' in df.columns:
    print('Without FDR significant:', (df['significant_at_0.05'] == 1).sum())
if 'significant_at_0.05_fdr' in df.columns:
    print('With BH-FDR significant:', (df['significant_at_0.05_fdr'] == 1).sum())
print('Total tests:', len(df))
"
```

- [ ] **Step 5: Commit**

```bash
git add src/eval/cascade_vs_llm_stats.py datasets/processed/cascade_vs_llm_stats.parquet datasets/processed/cascade_vs_llm_stats_pre_fdr.parquet
git commit -m "fix(eval): add BH FDR correction for per-(cat,attr) McNemar tests"
```

---

## Phase 1.5 — recompute_calibration ECE bug fix

**Honest framing:** raw clf predictions без isotonic не сохранены. Единственный возможный fix — записать `ece_raw: None` и в plot скрыть «before» точки.

### Task 1.5.1: Patch train.py:614-622

**Files:**
- Modify: `src/pipeline/ml/train.py:614-622`

- [ ] **Step 1: Edit lines 614-624**

Заменить:
```python
        prev_path = os.path.join(MODELS_DIR, f"{prefix}_{attr_name}_calibration.json")
        out = {
            "attr": attr_name,
            "ece_raw": ece,
            "ece_calibrated": ece,
            "bins_raw": bins,
            "bins_calibrated": bins,
            "regenerated_calibration_only": True,
        }
```

на:
```python
        prev_path = os.path.join(MODELS_DIR, f"{prefix}_{attr_name}_calibration.json")
        out = {
            "attr": attr_name,
            # HONEST: raw clf predictions (без isotonic) не сохранены — recomputed ECE
            # отражает только calibrated state. Plot «до vs после» для recomputed строк
            # показывает только calibrated; before-точки исключаются на чтении.
            "ece_raw": None,
            "ece_calibrated": ece,
            "bins_raw": None,
            "bins_calibrated": bins,
            "regenerated_calibration_only": True,
        }
```

- [ ] **Step 2: Commit**

```bash
git add src/pipeline/ml/train.py
git commit -m "fix(train): record ece_raw=None for recomputed cal (raw clf not avail)"
```

**Note:** notebook plot update — в Phase 4.1.

---

## Phase 1.6 — Class balance + NaN audit

### Task 1.6.1: Write class_balance_audit.py

**Files:**
- Create: `src/eval/class_balance_audit.py`

- [ ] **Step 1: Write script**

```python
"""Class balance + NaN audit для honest reporting.

Показывает, что accuracy 93% на is_organic (majority=False ~95%) — это «лучше
тривиального baseline на 2 п.п.», а не «модель работает».

Output: datasets/processed/class_balance.parquet
Columns: category, attr, n_total, n_null_gt, majority_class, majority_baseline_acc,
         cascade_accuracy, cascade_f1_macro, cascade_balanced_acc, f1_macro_reliable
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.common import MAIN_CATEGORIES, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

PROCESSED = Path(PROCESSED_DIR)


def per_attr_stats(df: pd.DataFrame) -> dict:
    valid = df.dropna(subset=["gt"]).copy()
    n_total = len(df)
    n_null = n_total - len(valid)
    if len(valid) == 0:
        return {"n_total": n_total, "n_null_gt": n_null}
    gt = valid.gt.astype(str)
    pred = valid.pred.fillna("__abstain__").astype(str)
    class_counts = gt.value_counts()
    majority_class = class_counts.idxmax()
    majority_n = int(class_counts.max())
    majority_baseline_acc = majority_n / len(valid)
    minority_n = int(class_counts.drop(majority_class).sum())
    # F1_macro надёжен только если минор класс имеет n>=10
    f1_reliable = minority_n >= 10
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        f1_macro = float(f1_score(gt, pred, average="macro", zero_division=0))
        bal_acc = float(balanced_accuracy_score(gt, pred))
    accuracy = float((pred == gt).mean())
    return {
        "n_total": n_total,
        "n_null_gt": n_null,
        "n_valid": len(valid),
        "majority_class": str(majority_class),
        "majority_n": majority_n,
        "minority_n": minority_n,
        "majority_baseline_acc": majority_baseline_acc,
        "cascade_accuracy": accuracy,
        "cascade_f1_macro": f1_macro,
        "cascade_balanced_acc": bal_acc,
        "f1_macro_reliable": f1_reliable,
    }


def main():
    rows = []
    for cat in MAIN_CATEGORIES:
        df = pd.read_parquet(PROCESSED / f"experiment_per_product_{cat}_stratified.parquet")
        for attr in sorted(df.attr.unique()):
            sub = df[df.attr == attr]
            stats = per_attr_stats(sub)
            rows.append({"category": cat, "attr": attr, **stats})
            logger.info("  %s/%s: acc=%.3f baseline=%.3f f1_macro=%.3f%s",
                        cat, attr,
                        stats.get("cascade_accuracy", 0),
                        stats.get("majority_baseline_acc", 0),
                        stats.get("cascade_f1_macro", 0),
                        "" if stats.get("f1_macro_reliable", True) else " (UNRELIABLE single-class)")
    out = pd.DataFrame(rows)
    out_path = PROCESSED / "class_balance.parquet"
    out.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(out))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run script**

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m src.eval.class_balance_audit
```

- [ ] **Step 3: Examine output**

```bash
.venv/bin/python -c "
import pandas as pd
df = pd.read_parquet('datasets/processed/class_balance.parquet')
df['lift_over_baseline'] = df.cascade_accuracy - df.majority_baseline_acc
print(df[['category','attr','cascade_accuracy','majority_baseline_acc','lift_over_baseline','f1_macro_reliable']].round(3).to_string(index=False))
"
```

- [ ] **Step 4: Commit**

```bash
git add src/eval/class_balance_audit.py datasets/processed/class_balance.parquet
git commit -m "feat(eval): class balance audit shows lift over majority baseline"
```

---

## Phase 1.8 — cocoa_percentage labelspace fix

### Task 1.8.1: Inspect silver cocoa labelspace

- [ ] **Step 1: Get silver buckets**

```bash
.venv/bin/python -c "
import pandas as pd
silver = pd.read_parquet('datasets/processed/chocolate_stratified_silver_standard.parquet')
if 'cocoa_percentage' in silver.columns:
    print('silver cocoa_percentage values:')
    print(silver.cocoa_percentage.value_counts(dropna=False).head(20))
"
```

Записать список unique buckets (например: `lt_50`, `50_70`, `70_85`, `85_plus`).

### Task 1.8.2: Write bucketize_cocoa fix

**Files:**
- Create: `src/eval/cocoa_percentage_labelspace_fix.py`

- [ ] **Step 1: Write script**

```python
"""Bucketize LLM raw cocoa percentage to silver labelspace.

Silver: «50-70», «70-85» (или подобные buckets) — derived from rules.py:677-687.
LLM (direct/fallback): возвращает raw число «70», «70%», «dark chocolate 70%».
Без bucketize evaluation тривиально mismatch'ит.

Output: regenerated direct_llm_eval_chocolate.parquet и cascade_plus_llm4_hybrid.parquet
с corrected `pred` для cocoa_percentage.

Auto-derive buckets from silver schema (no hardcoded thresholds).
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import pandas as pd

from src.common import PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

PROCESSED = Path(PROCESSED_DIR)
ATTR = "cocoa_percentage"
CAT = "chocolate"


def parse_percent(raw) -> float | None:
    """Extract numeric percent from LLM output: '70%', '70', 'dark chocolate 70%'."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip()
    m = re.search(r"(\d{1,3})(?:\s*%|\s|$)", s)
    if not m:
        return None
    n = float(m.group(1))
    if 0 <= n <= 100:
        return n
    return None


def derive_buckets_from_silver() -> list[tuple[str, float, float]]:
    """Auto-derive bucket edges from silver labelspace.
    
    Returns list of (label, lo, hi) where intervals are [lo, hi).
    """
    silver = pd.read_parquet(PROCESSED / f"{CAT}_stratified_silver_standard.parquet")
    if ATTR not in silver.columns:
        raise RuntimeError(f"silver lacks {ATTR}")
    labels = sorted(silver[ATTR].dropna().unique().tolist())
    buckets = []
    for label in labels:
        # Match patterns like "50-70", "70_85", "lt_50", "85_plus"
        m = re.match(r"(\d+)[-_](\d+)$", str(label))
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            buckets.append((str(label), lo, hi))
            continue
        m_lt = re.match(r"lt_?(\d+)$", str(label), re.IGNORECASE)
        if m_lt:
            buckets.append((str(label), 0.0, float(m_lt.group(1))))
            continue
        m_plus = re.match(r"(\d+)_?(?:plus|\+)$", str(label), re.IGNORECASE)
        if m_plus:
            buckets.append((str(label), float(m_plus.group(1)), 101.0))
            continue
        logger.warning("Unrecognized cocoa bucket label: %r — skipped", label)
    buckets.sort(key=lambda x: x[1])
    logger.info("Derived buckets: %s", buckets)
    return buckets


def bucketize(value, buckets: list[tuple[str, float, float]]) -> str | None:
    n = parse_percent(value)
    if n is None:
        return None
    for label, lo, hi in buckets:
        if lo <= n < hi:
            return label
    return None


def patch_parquet(path: Path, buckets: list[tuple[str, float, float]],
                  pred_col: str = "pred") -> tuple[int, int]:
    """In-place bucketize for cocoa_percentage rows; returns (changed, total)."""
    df = pd.read_parquet(path)
    if "attr" not in df.columns or pred_col not in df.columns:
        logger.warning("%s: missing attr/%s columns — skipped", path, pred_col)
        return 0, 0
    mask = df.attr == ATTR
    total = int(mask.sum())
    if total == 0:
        return 0, 0
    original = df.loc[mask, pred_col].copy()
    df.loc[mask, pred_col] = df.loc[mask, pred_col].apply(lambda v: bucketize(v, buckets))
    changed = int((original.astype(str) != df.loc[mask, pred_col].astype(str)).sum())
    # Backup
    backup = path.with_suffix(".pre_cocoa_fix.parquet")
    if not backup.exists():
        pd.read_parquet(path).to_parquet(backup, index=False)
    df.to_parquet(path, index=False)
    return changed, total


def main():
    buckets = derive_buckets_from_silver()
    targets = [
        PROCESSED / "direct_llm_eval_chocolate_stratified.parquet",
        PROCESSED / "cascade_plus_llm4_hybrid.parquet",
        PROCESSED / "llm_fallback_eval_chocolate_stratified.parquet",
    ]
    for path in targets:
        if not path.exists():
            logger.warning("Missing: %s — skipped", path)
            continue
        changed, total = patch_parquet(path, buckets)
        logger.info("%s: %d/%d cocoa rows bucketized", path.name, changed, total)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run script**

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m src.eval.cocoa_percentage_labelspace_fix
```

- [ ] **Step 3: Verify accuracy change**

```bash
.venv/bin/python -c "
import pandas as pd
for fname in ['direct_llm_eval_chocolate_stratified', 'cascade_plus_llm4_hybrid']:
    new = pd.read_parquet(f'datasets/processed/{fname}.parquet')
    old = pd.read_parquet(f'datasets/processed/{fname}.pre_cocoa_fix.parquet')
    if 'attr' in new.columns:
        cocoa_new = new[new.attr == 'cocoa_percentage']
        cocoa_old = old[old.attr == 'cocoa_percentage']
        if len(cocoa_new) > 0 and 'gt' in cocoa_new.columns and 'pred' in cocoa_new.columns:
            acc_new = (cocoa_new.pred.astype(str) == cocoa_new.gt.astype(str)).mean()
            acc_old = (cocoa_old.pred.astype(str) == cocoa_old.gt.astype(str)).mean()
            print(f'{fname}: cocoa acc {acc_old:.3f} -> {acc_new:.3f}')
"
```

- [ ] **Step 4: Commit**

```bash
git add src/eval/cocoa_percentage_labelspace_fix.py datasets/processed/*cocoa_fix* datasets/processed/direct_llm_eval_chocolate_stratified.parquet datasets/processed/cascade_plus_llm4_hybrid.parquet datasets/processed/llm_fallback_eval_chocolate_stratified.parquet
git commit -m "fix(eval): bucketize LLM cocoa_percentage to silver labelspace"
```

---

## Phase 2 — Re-blind audit on TYPE_A only

### Task 2.1: Patch off_field_filter.py blacklist

**Files:**
- Modify: `src/manual_label/off_field_filter.py:17-27`

- [ ] **Step 1: Edit DERIVED_BLACKLIST**

Заменить (lines 17-27):
```python
DERIVED_BLACKLIST: frozenset[str] = frozenset({
    "nutriscore_grade",
    "nutriscore_score",
    "nutriscore_data",
    "nova_group",
    "nova_groups_tags",
    "ecoscore_grade",
    "ecoscore_score",
    "ecoscore_data",
    "ingredients_analysis_tags",
})
```

на:
```python
# Top-level OFF fields that ARE target attributes — must NOT enter the prompt.
# Extended 2026-05-24: labels_tags, categories_tags, manufacturing_places,
# countries_tags, allergens_tags также blacklist'нуты, поскольку silver для
# TYPE_A tag-derived атрибутов (is_organic, is_vegan, is_pdo,
# country_of_origin, ...) выводится регексом по этим полям; их видение Opus
# в blind prompt = leak источника silver.
DERIVED_BLACKLIST: frozenset[str] = frozenset({
    "nutriscore_grade",
    "nutriscore_score",
    "nutriscore_data",
    "nova_group",
    "nova_groups_tags",
    "ecoscore_grade",
    "ecoscore_score",
    "ecoscore_data",
    "ingredients_analysis_tags",
    # Tag-derived silver sources (added 2026-05-24):
    "labels_tags",
    "categories_tags",
    "manufacturing_places",
    "countries_tags",
    "allergens_tags",
})
```

- [ ] **Step 2: Remove special case for categories_tags in curate_prompt_fields**

Lines 67-69 в текущем файле имеют:
```python
        if k == "categories_tags" and isinstance(v, list):
            out[k] = sorted(v)
            continue
```

Этот блок НЕ нужен теперь когда categories_tags в blacklist — он будет отфильтрован в DERIVED_BLACKLIST check на line 60. Удалить lines 67-69 (если они существуют) или убедиться, что blacklist check срабатывает раньше.

- [ ] **Step 3: Run dependent test (если есть)**

```bash
grep -rn "test.*off_field_filter\|off_field_filter.*test" /Users/miafrolov/Desktop/stuff/ai_attributes/tests/ 2>&1 | head
```

Если тест есть — запустить. Если нет — sanity manual:

```bash
.venv/bin/python -c "
from src.manual_label.off_field_filter import curate_prompt_fields
example = {
    'product_name': 'Bio organic pasta',
    'brands': 'Carrefour BIO',
    'ingredients_text': 'durum wheat',
    'labels_tags': ['en:organic', 'en:bio'],
    'categories_tags': ['en:pasta'],
    'nutriments': {'fat_100g': 1.2, 'nutriscore_grade': 'a'},
}
result = curate_prompt_fields(example)
assert 'labels_tags' not in result, f'labels_tags leaked: {result.get(\"labels_tags\")}'
assert 'categories_tags' not in result, f'categories_tags leaked: {result.get(\"categories_tags\")}'
assert 'nutriscore_grade' not in result.get('nutriments', {}), 'nutriscore_grade leaked in nutriments'
assert result['product_name'] == 'Bio organic pasta'
assert 'fat_100g' in result['nutriments']
print('blacklist working correctly:', result)
"
```

Expected: assertion passes.

- [ ] **Step 4: Apply Phase 0.8 outcome — parametrize if needed**

Если Phase 0.8 показал, что `direct_llm_v2.py` или другой code использует `curate_prompt_fields` и нужен старый (narrower) blacklist для direct LLM baseline — параметризовать:

```python
def curate_prompt_fields(
    off_product: dict[str, Any],
    blacklist: frozenset[str] = DERIVED_BLACKLIST,
) -> dict[str, Any]:
    ...
    for k, v in off_product.items():
        if k in blacklist:
            continue
        ...
```

И добавить `LEGACY_BLACKLIST` (старая 9-keys frozenset) для caller'ов, которым нужен старый blacklist.

- [ ] **Step 5: Commit**

```bash
git add src/manual_label/off_field_filter.py
git commit -m "fix(blind-audit): blacklist labels_tags/categories_tags (silver sources)"
```

### Task 2.2: Identify and prep TYPE_A attrs subset

- [ ] **Step 1: List TYPE_A attrs from taxonomy**

```bash
.venv/bin/python -c "
import pandas as pd
tax = pd.read_parquet('datasets/processed/attribute_signal_taxonomy.parquet')
tag_attrs = tax[(tax.signal_type == 'tag_derived') & (tax.category.isin(['pasta','chocolate','cheeses']))]
print(tag_attrs[['category','attr']].to_string(index=False))
print(f'Total: {len(tag_attrs)} (cat, attr) pairs')
"
```

Expected: ~9 pairs (is_organic × 3, is_vegan × 1–2, is_gluten_free × 1, is_pdo × 1, country_of_origin × 1, is_ultra_processed × 1).

### Task 2.3: Re-run opus_off_grounded_audit with Sonnet 4.6

**Note:** Phase 0.7 определил модель. По умолчанию Sonnet 4.6. Если оригинал был Opus и нужна apples-to-apples — отдельная Opus subsample (50 cells).

- [ ] **Step 1: Backup existing blind audit artifacts**

```bash
cp datasets/processed/blind_vs_prefill_overall.parquet datasets/processed/blind_vs_prefill_overall_pre_blacklist_fix.parquet
cp datasets/processed/blind_vs_prefill_per_attr.parquet datasets/processed/blind_vs_prefill_per_attr_pre_blacklist_fix.parquet
cp datasets/processed/blind_vs_prefill_flip.parquet datasets/processed/blind_vs_prefill_flip_pre_blacklist_fix.parquet 2>/dev/null || echo "flip not present, skip"
```

- [ ] **Step 2: Run audit on TYPE_A subset**

Reference Phase 0.7 finding для exact entry-point invocation. Generic shape:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m src.manual_label.opus_off_grounded_audit \
    --categories pasta,chocolate,cheeses \
    --attrs-filter tag_derived \
    --model claude-sonnet-4-6 \
    --output datasets/manual_label/opus_audit_2026-05-24_blacklist_fix.csv \
    --log /Users/miafrolov/Desktop/stuff/ai_attributes/logs/blind_audit_2026-05-24.log
```

(Adapt CLI args to actual script interface — Phase 0.7 read даст signature.)

Long-running command (часы wall time). Run in background; monitor through incremental log file (per memory feedback).

- [ ] **Step 3: After completion, regenerate aggregate parquets**

Найти скрипт, генерирующий `blind_vs_prefill_*.parquet`:
```bash
grep -rn "blind_vs_prefill" /Users/miafrolov/Desktop/stuff/ai_attributes/src/ | head
```

Запустить его на новых данных:
```bash
OMP_NUM_THREADS=1 .venv/bin/python -m src.eval.<aggregator_script>
```

- [ ] **Step 4: Compare pre/post κ**

```bash
.venv/bin/python -c "
import pandas as pd
old = pd.read_parquet('datasets/processed/blind_vs_prefill_per_attr_pre_blacklist_fix.parquet')
new = pd.read_parquet('datasets/processed/blind_vs_prefill_per_attr.parquet')
m = old.merge(new, on=['attr','category'], suffixes=('_old','_new'))
m['delta_kappa'] = m['cohen_kappa_new'] - m['cohen_kappa_old']
m['delta_agreement'] = m['agreement_new'] - m['agreement_old']
tag_attrs = ['is_organic','is_vegan','is_gluten_free','is_pdo','country_of_origin','is_ultra_processed']
print('TYPE_A delta:')
print(m[m.attr.isin(tag_attrs)][['category','attr','cohen_kappa_old','cohen_kappa_new','delta_kappa','agreement_old','agreement_new','delta_agreement']].round(3).to_string(index=False))
"
```

- [ ] **Step 5: Determine outcome scenario (a/b/c per spec §2.4)**

- (a) κ просел до 0.4–0.6 → narrative shift: "silver слабо обоснован".
- (b) κ просел до 0.7–0.85 → "substantial agreement даже без leak'нутых tags".
- (c) κ остался ≥0.9 → strongest case: "Opus извлекает signal из text без tags".

Записать в `PHASE0_FINDINGS_2026-05-24.md` секцию `## Phase 2 outcome`.

- [ ] **Step 6: Commit**

```bash
git add datasets/processed/blind_vs_prefill_*.parquet datasets/manual_label/opus_audit_2026-05-24_blacklist_fix.csv
git commit -m "fix(blind-audit): re-run with extended blacklist on TYPE_A (3 main cats)"
```

---

## Phase 3 — Text disclaimers in ВКР

### Task 3.1: §3.3 — taxonomy breakdown, threshold sensitivity, blind audit, LLM honest, class balance

**Files:**
- Modify: `docs/thesis/03_chapter3_implementation.md`

- [ ] **Step 1: Find §3.3 location**

```bash
grep -n "^##\|^###" /Users/miafrolov/Desktop/stuff/ai_attributes/docs/thesis/03_chapter3_implementation.md | head -30
```

Identify section "Доказательство работоспособности" или подобная (§3.3).

- [ ] **Step 2: Add taxonomy breakdown paragraph**

В соответствующую секцию (после current headline statement) добавить:

```markdown
### Декомпозиция метрики по таксономии источников разметки

Headline-метрика, представленная выше, агрегирует атрибуты с различной природой
эталонной разметки. Атрибуты делятся на три типа (taxonomy в
`datasets/processed/attribute_signal_taxonomy.parquet`):
- **nutri_derived** ({N1} атрибутов на 3 основных категориях): эталон выводится
  бакетизацией числовых нутриентов; circularity risk минимальный.
- **tag_derived** ({N2} атрибутов): эталон выводится регексом по `labels_tags` /
  `categories_tags`; circularity risk умеренный, см. §3.3 о повторной слепой
  валидации с расширенным blacklist'ом.
- **text_derived** ({N3} атрибутов): эталон выводится регексом по `ingredients_text` /
  `product_name`; circularity risk высокий, метрика частично отражает воспроизведение
  правил регекса векторной моделью.

Декомпозиция показана в таблице {table-headline-by-taxonomy} (значения из
`headline_by_taxonomy.parquet`, доверительные интервалы через brand-clustered
bootstrap, 1000 итераций):
- nutri_derived: {acc1:.3f} (CI: {ci1lo:.3f}..{ci1hi:.3f})
- tag_derived: {acc2:.3f} (CI: ...)
- text_derived: {acc3:.3f} (CI: ...)
```

**Note:** Конкретные числа `{acc1}`, `{N1}` подставить руками из вывода Task 1.1.2 Step 3. Не оставлять placeholders — заполнить актуальными цифрами при write.

- [ ] **Step 3: Add threshold sensitivity paragraph**

```markdown
### Чувствительность к выбору порога уверенности

Пороги достоверности для ML-слоя были откалиброваны на test-разбиении silver-эталона
(`src/pipeline/ml/train.py:380`). Тот же test пересекается с brand-disjoint test
на 14–23% кодов (см. таблицу в `PHASE0_FINDINGS_*.md`), что создаёт потенциал
для оптимистической оценки.

Анализ чувствительности (`src/eval/threshold_sensitivity.py`, артефакт
`threshold_sensitivity.parquet`) показывает: при переходе на uniform threshold 0.5
без калибровки headline сдвигается на {delta:.2f} п.п. Это {within/exceeds} половину
ширины brand-clustered 95% CI ({ci_halfwidth:.2f} п.п.), что характеризует эффект
как {в пределах статистического шума / реальное смещение, требующее коррекции}.
```

**Note:** Подставить delta и решение «within/exceeds» из Task 1.2.2 Step 3.

- [ ] **Step 4: Add LLM honest measurement paragraph**

```markdown
### Точность LLM на абстейн-выборке

В предыдущих версиях анализа точность LLM на ячейках, где каскад abstain'ится,
аппроксимировалась средней по всему атрибуту. Это приводит к систематическому
завышению финальной комбинированной метрики, поскольку абстейн-ячейки
систематически труднее (именно поэтому каскад на них и отказывается отвечать).

После замены аппроксимации на реально измеренную LLM-точность на абстейн-выборке
(`llm_fallback_eval_{cat}_stratified.parquet`, артефакт обновлён в
`cost_quality_ci.py`), headline для cascade+LLM сдвигается с {old} % на {new} %
(дельта -{delta} п.п.). Это honest нижняя оценка.
```

- [ ] **Step 5: Add brand-norm fix paragraph**

```markdown
### Корректировка нормализации брендов

Нормализация бренда в `src/data/split/generate_gold_splits.py` ранее использовала
первый comma-separated элемент колонки `brands`, что приводило к subbrand leak:
«Carrefour BIO, Carrefour» и «Carrefour, Carrefour BIO» получали разные канонические
имена и могли попадать в разные splits. Для атрибутов вроде `is_organic` (regex
ловит «BIO» в названии бренда) это прямая утечка между train и test.

Канонизация заменена на сортировку и join всех comma-separated элементов
(см. commit `fix(splits): canonical multi-brand norm`). Affected продуктов: {N}
({pct} % от общего числа). После пересчёта brand_disjoint splits headline
сдвигается на {delta} п.п.
```

- [ ] **Step 6: Add class balance paragraph**

```markdown
### Baseline majority-класса

Для контекста: тривиальный baseline «всегда предсказывать majority-класс» даёт
точность от {min_baseline} до {max_baseline} в зависимости от атрибута (атрибуты
`is_organic`, `is_pdo` имеют majority до 95-99%). Lift каскада над этим baseline
показан в таблице {table-class-balance} (артефакт `class_balance.parquet`,
колонка `lift_over_baseline`). F1_macro и balanced_accuracy там же — для атрибутов
с n_minority < 10 F1_macro помечен как `unreliable`.
```

- [ ] **Step 7: Commit**

```bash
git add docs/thesis/03_chapter3_implementation.md
git commit -m "docs(thesis): §3.3 disclaimers — taxonomy, threshold, LLM honest, brand-norm, baseline"
```

### Task 3.2: §6 Bayes narrative branching

**Files:**
- Modify: `docs/thesis/03_chapter3_implementation.md` (если §6 там) или другой
- Depends on: Phase 0.2 outcome (a/b/c)

- [ ] **Step 1: Find §6 location**

```bash
grep -rn "Bayes\|байес\|байесовск" /Users/miafrolov/Desktop/stuff/ai_attributes/docs/thesis/*.md | head -10
```

- [ ] **Step 2: Apply outcome-specific text**

**(a)** DAG silver-обученный, stable: keep current narrative + добавить:
```markdown
Bootstrap-стабильность edges в HillClimb+BIC проверена в
`datasets/processed/dag_stability_*.parquet`. Для main категорий стабильность ≥ 0.7 на
ключевых edges, что характеризует структуру как воспроизводимую при пересиде.
```

**(b)** Silver-DAG нестабилен, prod-конфиг defensible:
```markdown
Bootstrap-стабильность edges в HillClimb+BIC проверена в
`datasets/processed/dag_stability_*.parquet`. Часть edges имеет стабильность ниже 0.7
(для chocolate — 0 из 6 reference edges с порогом STABLE), что отражает (а) силу
sample noise при n≈200-250, и (б) ограничения BIC-критерия на категориальных данных
с десятком переменных. Bayes-сеть в финальной конфигурации используется не как
«открытие структуры», а как regularizer над hand-crafted edges, отражающими
известные domain-зависимости между атрибутами.
```

**(c)** Production gold-refit:
```markdown
Финальная prod-конфигурация (`src/experiments/accuracy_squeeze_holdout.py`) использует
gold-refit DAG: структура переобучается на gold-данных (n≈200-250 cells per category),
что даёт более устойчивую структуру при меньшей выборке, но создаёт overlap risk
между gold-fit и gold-test (см. Limitations).
```

- [ ] **Step 3: Commit**

```bash
git add docs/thesis/03_chapter3_implementation.md
git commit -m "docs(thesis): §6 Bayes narrative reflects DAG stability findings"
```

### Task 3.3: §5 electronics cold-start reformulation

**Files:**
- Modify: главу с описанием electronics cold-start

- [ ] **Step 1: Find §5 location**

```bash
grep -rn "cold.start\|electronics" /Users/miafrolov/Desktop/stuff/ai_attributes/docs/thesis/*.md | head -10
```

- [ ] **Step 2: Replace narrative**

Заменить упоминание «Bayes обнаруживает связь brand→os» на:

```markdown
В демонстрационном эксперименте на категории «электроника» используется
hand-crafted edge `brand → os` (бренд определяет операционную систему: Apple → iOS,
Samsung → Android, и т.д.). Bayes-сеть демонстрирует conditional inference на этом
edge — recovery P(os | brand) ≈ 1 для известных брендов. Это иллюстрация
архитектурной возможности cold-start через P(target | input)
для атрибутов с сильной conditional structure, не результат discovery of structure
(bootstrap-стабильность edge `brand → os` составила {0.61}, что отражает
ограничения автоматического обучения структуры на малых выборках).
```

- [ ] **Step 3: Commit**

```bash
git add docs/thesis/*.md
git commit -m "docs(thesis): §5 electronics cold-start as illustration not discovery"
```

### Task 3.4: §4.1 demo disclaimer + restrict demo to 3 main cats

**Files:**
- Modify: `docs/thesis/04_chapter4_results.md`
- Modify: `demo/ml_service/cascade.py` (или router init)

- [ ] **Step 1: Find demo entry point**

```bash
grep -n "categories\|supported\|allowed_cat" /Users/miafrolov/Desktop/stuff/ai_attributes/demo/ml_service/cascade.py | head -10
```

- [ ] **Step 2: Add whitelist in demo**

В `demo/ml_service/cascade.py` ближе к началу класса/функции, обрабатывающей запрос:

```python
from src.common import MAIN_CATEGORIES

SUPPORTED_CATEGORIES = set(MAIN_CATEGORIES)


def _check_category_supported(category: str) -> dict | None:
    """Return error dict if category unsupported; None otherwise."""
    if category not in SUPPORTED_CATEGORIES:
        return {
            "error": "unsupported_category",
            "category": category,
            "supported": sorted(SUPPORTED_CATEGORIES),
            "note": "Категория не поддерживается в текущей итерации демо; "
                    "следующая итерация добавит beverages, cereals, cosmetics, electronics.",
        }
    return None
```

И в основной handler перед запуском cascade:
```python
    err = _check_category_supported(category)
    if err is not None:
        return err
```

(Exact patch — adapt to actual cascade.py structure из Step 1.)

- [ ] **Step 3: Add disclaimer in ВКР §4.1**

В `04_chapter4_results.md` секцию «Как пользоваться» добавить:

```markdown
### Описание production-демо

Production demo (`demo/ml_service/cascade.py`) предоставляет REST API,
включающий слои 0-3: категориальный роутер (Layer 0), regex (Layer 1),
ML-классификаторы (Layer 2), Bayes-сеть (Layer 3). Demo поддерживает 3 main
category: pasta, chocolate, cheeses; остальные категории (beverages, cereals,
cosmetics, electronics) — research scope текущей итерации, поддержка в demo —
следующая итерация.

Слой 4 (LLM fallback) для непокрытых атрибутов в локальном demo возвращает
`value: null` — реальный LLM-вызов требует OpenRouter API key и не включён
в локальное развёртывание.

При защите основным defendable artifact'ом выступает headline-конфигурация
`cascade + gemini25flash` (см. §3.3, ноутбук §3.3.2); demo — minimal-deployable
proof of architecture, где Layer 4 отключён по cost reasons. Эти два artefact'а
сознательно разделены: headline доказывает методологию, demo — инженерную
реализуемость.
```

- [ ] **Step 4: Test demo restriction works**

```bash
# Smoke test: запустить demo вручную, send тест-request с unsupported cat
# (если demo требует runtime — пропустить или сделать в memory test)
.venv/bin/python -c "
from demo.ml_service.cascade import _check_category_supported
assert _check_category_supported('beverages') is not None
assert _check_category_supported('pasta') is None
print('Demo cat whitelist works')
"
```

- [ ] **Step 5: Commit**

```bash
git add docs/thesis/04_chapter4_results.md demo/ml_service/cascade.py
git commit -m "feat(demo): restrict to 3 main cats + disclaimer in §4.1"
```

### Task 3.5: Rename pre_registration → phase2_analysis_plan

**Files:**
- Rename: `docs/thesis/pre_registration_2026-Q2.md` → `phase2_analysis_plan_2026-Q2.md`
- Update all references

- [ ] **Step 1: git mv**

```bash
cd /Users/miafrolov/Desktop/stuff/ai_attributes
git mv docs/thesis/pre_registration_2026-Q2.md docs/thesis/phase2_analysis_plan_2026-Q2.md
```

- [ ] **Step 2: Update references**

Найти все упоминания:
```bash
grep -rln "pre_registration_2026" docs/ notebooks/ src/ .claude/projects/ 2>/dev/null
```

Для каждого — заменить `pre_registration_2026-Q2.md` на `phase2_analysis_plan_2026-Q2.md`:

```bash
grep -rln "pre_registration_2026" docs/ notebooks/ src/ 2>/dev/null | while read f; do
    sed -i.bak "s|pre_registration_2026-Q2\.md|phase2_analysis_plan_2026-Q2.md|g" "$f"
    rm "$f.bak"
done
```

Также — найти и заменить упоминания «pre-registration» в тексте (там, где это уже не верное название):
```bash
grep -rln "pre-registration\|пре-?регистрация" docs/thesis/ | head -10
```

Manual review каждого хита; заменить на «зафиксированный план анализа Phase 2» где уместно.

- [ ] **Step 3: Add honest timing header в новый файл**

В начало `docs/thesis/phase2_analysis_plan_2026-Q2.md` добавить (после title):

```markdown
> **Phase 2 analysis plan.** Документ оформлен ретроспективно на этапе сводки
> результатов; фиксирует hypotheses, decision rules и correction methods, которые
> мы готовы защищать. Не претендует на статус formal pre-registration (OSF/journal).
> Зафиксированные decision rules применены ко всем измерениям единообразно.
```

- [ ] **Step 4: Commit**

```bash
git add docs/thesis/phase2_analysis_plan_2026-Q2.md docs/thesis/*.md notebooks/ src/
git commit -m "refactor(docs): rename pre-registration → phase2 analysis plan (honest timing)"
```

### Task 3.6: Limitations section in Заключение

**Files:**
- Modify: `docs/thesis/05_conclusion.md`

- [ ] **Step 1: Add Limitations section**

В конце `05_conclusion.md` добавить:

```markdown
## Ограничения работы

В данном разделе сводно перечислены известные ограничения настоящего исследования
и их количественная оценка, где она была получена.

**Scope текущей итерации.** Все методологические правки и метрики в §3.3 относятся
к трём основным категориям: pasta, chocolate, cheeses. Категории beverages,
cereals, cosmetics, electronics — secondary scope; результаты на них приведены
как preliminary до завершения следующей итерации. Демо также ограничено тремя
основными категориями.

**Snooping порогов уверенности.** Пороги откалиброваны на test-разбиении silver,
которое пересекается с brand-disjoint test на 14–23% кодов. Чувствительность
к этому пересечению измерена в `threshold_sensitivity.parquet`: переход на
uniform 0.5 без калибровки сдвигает headline на ≤ {X} п.п., что {в пределах /
больше} половины ширины brand-clustered 95% CI.

**Structural circularity для text-derived атрибутов.** Для 19 из 44 атрибутов
silver-эталон выводится регексом по тем же текстовым полям (`ingredients_text`,
`product_name`), которые используются как вход ML-слою. Метрика на этих атрибутах
частично отражает воспроизведение правил регекса векторной моделью.

**Slepая валидация silver для tag-derived атрибутов.** Расширение blacklist'а
blind-аудита (`src/manual_label/off_field_filter.py`: добавлены `labels_tags`,
`categories_tags`, `manufacturing_places`, `countries_tags`, `allergens_tags`)
повторный прогон даёт κ {Y} для is_organic / is_vegan / is_pdo (см. §3.3,
сценарий {a/b/c}). Это {интерпретация} согласия silver с независимой оценкой.

**TYPE_C silver не валиден через blind-аудит.** Для 12 атрибутов nutri-derived
(nutri_score_grade, protein_class, fat_class, sugar_class, ...) refusal_rate
LLM в blind-режиме составляет 0.82–0.93: модель не способна бакетизировать
сырые нутриенты без явного derivation_block в промпте (который сам по себе
был бы leak). Silver на этих атрибутах валидируется через rule-consistency
с raw nutriments (`src/diagnostics/silver/audit.py`), не через blind LLM.

**Нестабильность структуры Bayes-сети.** HillClimb+BIC даёт bootstrap-стабильность
edges <0.7 на 3+ из 6 reference edges для chocolate; для электроники ключевой
edge `brand → os` имеет стабильность 0.61 (`dag_stability_*.parquet`). Bayes
используется в финальной конфигурации как regularizer над hand-crafted edges,
не как механизм открытия структуры.

**Direct LLM baseline.** Сравнение «cascade vs direct LLM» проведено только
для модели gpt-oss-120b; более сильные модели (Opus, GPT-4o, Gemini 2.5 Pro)
не тестировались как direct baseline. Также direct LLM eval использует
random split, не brand-disjoint, что может завышать lift каскада из-за
brand-overlap преимущества cascade.

**Brand-norm после fix.** После корректировки `generate_gold_splits.py`
(canonical multi-brand norm) brand_disjoint splits пересчитаны; affected
продуктов: {N} ({pct}%); headline сдвигается на {delta} п.п.

**Cocoa percentage labelspace.** Mismatch между silver-buckets и raw LLM-output
устранён в `src/eval/cocoa_percentage_labelspace_fix.py`. Accuracy на этом
атрибуте до фикса частично отражала labelspace mismatch, не качество модели.

**ROUTER_ACC в cost-quality CI.** Router accuracy зафиксирован как константа
({"pasta": 0.962880, "chocolate": 0.985377, "cheeses": 0.977477}) и не
ресемплируется в brand-clustered bootstrap. Доверительные интервалы для
cost-quality scatter могут быть занижены, особенно на cheeses
(меньшее число брендов).

**Self-consistency baseline.** Шумовой пол LLM измерен только на pasta n=100
(Haiku 4.5); для chocolate/cheeses self-consistency baseline отсутствует.

**Provider-variance.** Headline получен в одном прогоне; OpenRouter маршрутизирует
один model_id на разные провайдеры (vertex/groq/fireworks) с разными
квантизациями, provider-variance не оценена.

**Кросс-категорийный transfer ML.** LOCO-ablation проведена только для роутера,
не для attribute classifiers. Cross-category transfer ML не измерен.

**Multilingual fairness.** Per-language accuracy измерена с помощью `langdetect`
на коротких product_name — детекция шумна, особенно для близких языков.
Утверждение о равной точности по языкам требует подкрепления per-language
n_cells и CI.

**Pre-registration timing.** Файл `phase2_analysis_plan_2026-Q2.md` (ранее
`pre_registration_2026-Q2.md`) оформлен ретроспективно на этапе сводки.
Decision rules применены единообразно, но формально документ не является
pre-registration в смысле OSF/journal.

**DOCX-синхронизация.** Правки в данной итерации внесены в `.md`-главы.
Ручной перенос в `VKR_Frolov_2026.docx` — отдельная задача, выполняемая
при финальной LaTeX-конверсии.

**Демо vs headline.** Демо (`demo/ml_service/cascade.py`) запускает Layers 0-3;
Layer 4 возвращает `null`. Headline 91.5% (или новое post-fix значение)
соответствует конфигурации `cascade + gemini25flash`, не локальному demo.
```

**Note:** Подставить актуальные `{X}`, `{Y}`, `{N}`, `{a/b/c}` из выходов Phases 0–2 перед commit.

- [ ] **Step 2: Commit**

```bash
git add docs/thesis/05_conclusion.md
git commit -m "docs(thesis): Limitations section in conclusion"
```

---

## Phase 4 — Notebook updates

### Task 4.1: Add new cells to 00_thesis_main.ipynb

**Files:**
- Modify: `notebooks/00_thesis_main.ipynb`

- [ ] **Step 1: Find anchor cells**

```bash
.venv/bin/python -c "
import json
nb = json.load(open('notebooks/00_thesis_main.ipynb'))
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        src = ''.join(cell['source'])[:80]
        print(f'{i:3d}: MD {src!r}')
" | grep -iE "3\.3|cost.quality|bayes|cold.start|limitations"
```

Identify cell positions для anchors: §3.3.X, §4.X cost-quality, §6.X Bayes/DAG, §10 Limitations.

- [ ] **Step 2: Insert taxonomy breakdown cell**

После §3.3 anchor добавить markdown + code cell:

```markdown
### §3.3.X Декомпозиция headline по taxonomy
```

```python
import pandas as pd
import matplotlib.pyplot as plt
tax_df = pd.read_parquet('datasets/processed/headline_by_taxonomy.parquet')
print(tax_df.to_string(index=False))
fig, ax = plt.subplots(figsize=(8, 4))
pivot = tax_df.pivot_table(index='signal_type', columns='category', values='accuracy')
pivot.plot.bar(ax=ax, ylim=(0.5, 1.0))
ax.set_ylabel('Accuracy on covered cells')
ax.set_title('Headline by signal_type (3 main cats)')
ax.legend(loc='lower right')
plt.tight_layout()
plt.show()
```

Use Read tool + Edit tool на notebook JSON, либо используйте `nbformat`:

```bash
.venv/bin/python <<'EOF'
import nbformat
nb_path = 'notebooks/00_thesis_main.ipynb'
nb = nbformat.read(nb_path, as_version=4)

new_md = nbformat.v4.new_markdown_cell(source="### §3.3.X Декомпозиция headline по taxonomy\n\nИспользует `datasets/processed/headline_by_taxonomy.parquet`.")
new_code = nbformat.v4.new_code_cell(source="""import pandas as pd
import matplotlib.pyplot as plt
tax_df = pd.read_parquet('datasets/processed/headline_by_taxonomy.parquet')
print(tax_df.to_string(index=False))
fig, ax = plt.subplots(figsize=(8, 4))
pivot = tax_df.pivot_table(index='signal_type', columns='category', values='accuracy')
pivot.plot.bar(ax=ax, ylim=(0.5, 1.0))
ax.set_ylabel('Accuracy on covered cells')
ax.set_title('Headline by signal_type (3 main cats)')
ax.legend(loc='lower right')
plt.tight_layout()
plt.show()
""")
# Find anchor cell with §3.3 in markdown
anchor_idx = None
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'markdown' and '3.3' in ''.join(cell.source):
        anchor_idx = i
        break
if anchor_idx is None:
    anchor_idx = len(nb.cells) - 1
nb.cells.insert(anchor_idx + 1, new_md)
nb.cells.insert(anchor_idx + 2, new_code)
nbformat.write(nb, nb_path)
print(f'Inserted at position {anchor_idx + 1}')
EOF
```

- [ ] **Step 3: Insert threshold sensitivity cell**

Аналогично, anchor — после taxonomy breakdown:

```python
ts = pd.read_parquet('datasets/processed/threshold_sensitivity.parquet')
pivot = ts.groupby(['category', 'threshold'])['accuracy_overall'].mean().unstack('threshold')
fig, ax = plt.subplots(figsize=(8, 4))
pivot.T.plot(ax=ax, marker='o')
ax.set_xlabel('Confidence threshold')
ax.set_ylabel('Headline accuracy')
ax.set_title('Threshold sensitivity (3 main cats)')
plt.tight_layout()
plt.show()
```

- [ ] **Step 4: Insert blind audit pre/post κ comparison cell**

```python
old = pd.read_parquet('datasets/processed/blind_vs_prefill_per_attr_pre_blacklist_fix.parquet')
new = pd.read_parquet('datasets/processed/blind_vs_prefill_per_attr.parquet')
m = old.merge(new, on=['attr', 'category'], suffixes=('_old', '_new'))
print('Pre/post blacklist fix (TYPE_A only relevant):')
print(m[['category','attr','cohen_kappa_old','cohen_kappa_new','agreement_old','agreement_new']].round(3).to_string(index=False))
```

- [ ] **Step 5: Insert cost-quality updated scatter cell**

После existing cost-quality cell или его обновить:

```python
cq_new = pd.read_parquet('datasets/processed/cost_quality_ci.parquet')
cq_old = pd.read_parquet('datasets/processed/cost_quality_ci_pre_honest_llm.parquet')
print('Pre/post honest LLM measurement:')
m = cq_old.merge(cq_new, on='config', suffixes=('_old', '_new'))
m['delta'] = m.acc_new - m.acc_old
print(m[['config','acc_old','acc_new','delta']].round(4).to_string(index=False))
```

- [ ] **Step 6: Insert DAG stability cell (§6)**

```python
import glob
stab_files = sorted(glob.glob('datasets/processed/dag_stability_*.parquet'))
if stab_files:
    for f in stab_files:
        cat = f.split('dag_stability_')[1].replace('.parquet', '')
        stab = pd.read_parquet(f)
        print(f'\n{cat}:')
        print(stab.head(10).to_string(index=False))
```

- [ ] **Step 7: Insert class balance cell**

```python
cb = pd.read_parquet('datasets/processed/class_balance.parquet')
cb['lift_over_baseline'] = cb.cascade_accuracy - cb.majority_baseline_acc
print(cb[['category','attr','cascade_accuracy','majority_baseline_acc','lift_over_baseline','f1_macro_reliable']].round(3).to_string(index=False))
```

- [ ] **Step 8: Insert brand-norm fix comparison cell**

```python
for cat in ['pasta', 'chocolate', 'cheeses']:
    old = pd.read_parquet(f'datasets/processed/{cat}_gold_split_pre_brand_norm_fix.parquet')
    new = pd.read_parquet(f'datasets/processed/{cat}_gold_split.parquet')
    old_test = set(old[old.split=='test'].code.astype(str))
    new_test = set(new[new.split=='test'].code.astype(str))
    print(f'{cat}: pre_test={len(old_test)} post_test={len(new_test)} '
          f'jaccard={len(old_test & new_test)/len(old_test | new_test):.3f}')
```

- [ ] **Step 9: Insert top-of-notebook limitations disclaimer**

В самое начало (cell 0 или 1):

```markdown
## ⚠️ Известные методологические ограничения

В данном ноутбуке применены fixes из `docs/thesis/FIX_PLAN_2026-05-24.md`.
Все метрики ниже относятся к **3 main категориям** (pasta, chocolate, cheeses).
Полный список ограничений — в `docs/thesis/05_conclusion.md` (Limitations) и
`docs/thesis/REVIEW_2026-05-24_meta_critique.md`.
```

- [ ] **Step 10: Update cost-quality narrative throughout**

```bash
grep -n "91\.5\|34%\|cost.quality" notebooks/00_thesis_main.ipynb | head
```

Manual review каждого хита; обновить, если число изменилось (после Phase 1.3 cost-quality акценты сдвинулись).

- [ ] **Step 11: Commit notebook**

```bash
git add notebooks/00_thesis_main.ipynb
git commit -m "feat(notebook): new cells for taxonomy/threshold/blind-audit/DAG/balance"
```

### Task 4.2: Selective re-execute of new cells

- [ ] **Step 1: Execute only new cells (without retraining)**

Use VS Code or jupyter UI to manually run the new cells. Alternative — копия notebook + nbconvert:

```bash
cp notebooks/00_thesis_main.ipynb /tmp/test_run.ipynb
jupyter nbconvert --to notebook --execute /tmp/test_run.ipynb --output /tmp/test_run_executed.ipynb --ExecutePreprocessor.timeout=300 --ExecutePreprocessor.allow_errors=True
# Inspect /tmp/test_run_executed.ipynb для errors
```

- [ ] **Step 2: If new cells fail, fix paths/imports**

Common issues: missing parquet (Phase 1.X не дал output), stale references. Fix and re-execute.

---

## Phase 5 — Verification + commit cycle

### Task 5.1: Cross-category mass balance assertion

- [ ] **Step 1: Write assertion script**

```bash
.venv/bin/python <<'EOF'
import pandas as pd

MAIN_CATEGORIES = ["pasta", "chocolate", "cheeses"]
tax = pd.read_parquet('datasets/processed/headline_by_taxonomy.parquet')

for cat in MAIN_CATEGORIES:
    sub = tax[tax.category == cat]
    if sub.empty:
        print(f'{cat}: NO DATA — skip')
        continue
    weighted = (sub.accuracy * sub.n_cells).sum() / sub.n_cells.sum()
    print(f'{cat}: weighted_acc_by_taxonomy = {weighted:.4f} (n_cells_total={sub.n_cells.sum()})')

# Compare with overall headline from experiment_per_product
print()
print('--- Overall accuracy from experiments (sanity check) ---')
for cat in MAIN_CATEGORIES:
    exp = pd.read_parquet(f'datasets/processed/experiment_per_product_{cat}_stratified.parquet')
    exp_valid = exp.dropna(subset=['gt'])
    exp_covered = exp_valid[exp_valid.layer != 'none']
    if len(exp_covered) > 0:
        acc = (exp_covered.pred == exp_covered.gt).mean()
        print(f'{cat}: overall_acc = {acc:.4f} (n_covered={len(exp_covered)})')
EOF
```

Sanity: weighted_acc_by_taxonomy и overall_acc должны быть близки (±0.5% — small diff из-за разной weighting).

### Task 5.2: Cross-document consistency grep

- [ ] **Step 1: Find numerical references**

```bash
echo "--- headline numbers in docs ---"
grep -rnE "9[0-3]\.[0-9]+\s*%|\+?[0-9]+\.[0-9]+\s*п\.п\.|34\s*%\s*LLM" \
  /Users/miafrolov/Desktop/stuff/ai_attributes/docs/thesis/*.md 2>&1 | head -40

echo "--- in notebook ---"
grep -nE "91\.5|92\.78|93\.81|\+1\.03|34%" \
  /Users/miafrolov/Desktop/stuff/ai_attributes/notebooks/00_thesis_main.ipynb | head -20
```

- [ ] **Step 2: Manual review**

Для каждого хита: соответствует ли число актуальной post-fix реальности? Если нет — fix in place.

### Task 5.3: Update memory

- [ ] **Step 1: Update review findings**

```bash
cat > /tmp/memory_update.md <<'EOF'
---
name: Fix-cycle 2026-05-24 results
description: Все Top-3 + honorable mentions из REVIEW v2 + meta_critique устранены или disclosed. Headline post-fix: <NEW_NUMBER>.
type: project
---

После выполнения `docs/thesis/IMPLEMENTATION_PLAN_2026-05-24.md`:

**Закрыто фиксами:**
- Top-3 №1 (TYPE_E circularity): taxonomy breakdown в headline_by_taxonomy.parquet; tag_derived disclaimed после re-blind (см. ниже).
- Top-3 №2 (threshold-on-test): sensitivity analysis в threshold_sensitivity.parquet; effect <X п.п.
- Brand-norm subbrand leak (§2.5): fixed in generate_gold_splits.py; affected products N (X%); headline delta Y п.п.
- LLM proxy в cost_quality_ci: заменён на honest measurement из llm_fallback_eval_*.parquet; headline -0.X п.п.
- FDR в cascade_vs_llm_stats: BH-FDR added; significant tests count: before Z → after W.
- recompute_calibration bug: ece_raw set to None для recomputed rows.
- cocoa_percentage labelspace mismatch: bucketized to silver labelspace.
- Blind audit blacklist fix: re-run с расширенным blacklist на TYPE_A; outcome scenario (a/b/c).
- Pre-registration honesty: renamed to phase2_analysis_plan_2026-Q2.md + honest header.
- Demo restricted to 3 main cats with disclaimer in §4.1.
- Class balance audit: per-attr lift over majority baseline in class_balance.parquet.
- Limitations section в 05_conclusion.md.

**Scope:** 3 main categories (pasta, chocolate, cheeses). Остальные 4 — следующая итерация (поменять MAIN_CATEGORIES в src/common.py и re-run Phase 1.0–1.6).

**Deferred:** DOCX sync (до LaTeX), TXtract/MAVE/OpenTag baselines, Russian OOD, cross-cat ML transfer.
EOF
cp /tmp/memory_update.md /Users/miafrolov/.claude/projects/-Users-miafrolov-Desktop-stuff-ai-attributes/memory/fix_cycle_results_2026-05-24.md
```

- [ ] **Step 2: Add to MEMORY.md index**

Append line:
```markdown
- [Fix-cycle 2026-05-24 results](fix_cycle_results_2026-05-24.md) — все Top-3 + honorable mentions устранены или disclosed.
```

### Task 5.4: Final tag and tree cleanup

- [ ] **Step 1: Check git status**

```bash
cd /Users/miafrolov/Desktop/stuff/ai_attributes
git status
```

Должно быть либо clean, либо явно untracked файлы, не относящиеся к fix-cycle.

- [ ] **Step 2: Tag**

```bash
# Verify tag не существует
git tag | grep "v2026-05-fix-cycle-3cats" || echo "tag clear"
git tag -a v2026-05-fix-cycle-3cats -m "Fix-cycle 2026-05-24: thesis methodology cleanup on 3 main cats"
```

- [ ] **Step 3: Final report**

```bash
echo "=== Fix-cycle 2026-05-24 summary ==="
git log --oneline v2026-05-fix-cycle-3cats~20..v2026-05-fix-cycle-3cats 2>/dev/null | head -25
echo ""
echo "=== Generated/updated artifacts ==="
ls -la datasets/processed/ | grep -E "headline_by_taxonomy|threshold_sensitivity|class_balance|cost_quality_ci|blind_vs_prefill_|_gold_split_pre" | head -15
```

---

## Self-Review Notes

**Spec coverage:**
- ✅ Phase 0 verification — Tasks 0.1–0.10
- ✅ Phase 1.0 brand-norm fix — Tasks 1.0.1–1.0.6
- ✅ Phase 1.1 taxonomy breakdown — Tasks 1.1.1–1.1.2
- ✅ Phase 1.2 threshold sensitivity — Tasks 1.2.1–1.2.2
- ✅ Phase 1.3 cost_quality LLM proxy — Tasks 1.3.1–1.3.2
- ✅ Phase 1.4 FDR — Tasks 1.4.1–1.4.2
- ✅ Phase 1.5 recompute_calibration — Task 1.5.1
- ✅ Phase 1.6 class balance — Task 1.6.1
- ✅ Phase 1.8 cocoa fix — Tasks 1.8.1–1.8.2
- ✅ Phase 2.1 blacklist patch — Task 2.1
- ✅ Phase 2.2 re-blind audit — Tasks 2.2–2.3
- ✅ Phase 3.1 §3.3 disclaimers — Task 3.1
- ✅ Phase 3.2 §6 Bayes — Task 3.2
- ✅ Phase 3.3 §5 cold-start — Task 3.3
- ✅ Phase 3.4 §4.1 demo + restrict — Task 3.4
- ✅ Phase 3.5 pre-reg rename — Task 3.5
- ✅ Phase 3.6 Limitations — Task 3.6
- ✅ Phase 4 notebook — Tasks 4.1–4.2
- ✅ Phase 5 verification — Tasks 5.1–5.4

**Defense Q&A coverage:**
- ✅ Brand-norm subbrand leak: Phase 1.0
- ✅ Threshold sensitivity vs CI width: Phase 1.2 (note in Step 3 of task 3.1)
- ✅ Demo vs headline: Phase 3.4 disclaimer
- ✅ Blind audit not blind on TYPE_A: Phase 2
- ✅ Pre-reg post-hoc: Phase 3.5 rename
- ✅ DAG instability narrative: Phase 3.2
- ✅ Class imbalance: Phase 1.6 + Limitations 3.6
- ✅ gpt-oss vs stronger models: Limitations 3.6 (acknowledged)
- ✅ Russian OOD: Limitations 3.6 (acknowledged)

**Not covered (explicit scope cut):**
- DOCX sync (до LaTeX)
- ML retrain с честным val split (sensitivity analysis в 1.2 даёт upper bound)
- TXtract/MAVE/OpenTag baselines
- Cross-category transfer для ML
- Russian OOD test
- pytest CI

**Risks acknowledged in plan:**
- Phase 0 finding может изменить scope (branching написан явно)
- Phase 1.0 brand-norm fix invalidates downstream → handled by Tasks 1.0.5–1.0.6
- Phase 2 LLM cost: $3–10 Sonnet, $50 if Opus apples-to-apples
- Notebook execution: selective only (cheap cells), no full retrain
