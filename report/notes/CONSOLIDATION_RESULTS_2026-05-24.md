# Consolidated LLM-Gold v1 — Results

## Концепция

Замена silver на **multi-source consolidated gold**: объединить все existing labelled sources (silver + 5 Gemini batches) и применить majority vote / priority resolution per (code, attr).

## Sources merged (free, sunk cost ~$5)

Per category:
- silver_standard (OFF tags → columns, 1.2k labelled per cat)
- b3_full_gemini (Gemini 2.5 Flash, off_grounded mode, 2-5k records)
- b3_promptfix (3 parts, ~600 each — improved prompt iteration)
- b3_promptfix_retry (4 batches × 3-6k, further improved)
- b3_r2 (4 parts × 2k — refined v2)
- gemini_validation (239 records на gold-overlapping subset)

Priority order: silver > gemini_validation > b3_r2 > b3_promptfix_retry > b3_promptfix > b3_full.

## Resolution policy

Per (code, attr) group:
- 1 source → 'single'
- ≥2/3 majority → 'consensus' or 'majority' (если только 1 disagreement)
- иначе → 'priority' (highest-priority source wins, conflict flagged)

## Coverage

| cat | unique (code, attr) | consensus | single | priority (conflict) |
|---|---|---|---|---|
| pasta | 98353 | 76752 (78%) | 18330 (19%) | 3271 (3.3%) |
| chocolate | 72577 | 50459 (70%) | 17947 (25%) | 4171 (5.7%) |
| cheeses | 102385 | 80616 (79%) | 15970 (16%) | 5799 (5.7%) |

Coverage product-level — 70-95% от 13-21k silver:

| cat | silver_total | combined labelled | coverage |
|---|---|---|---|
| pasta | 15691 | 14861 | **94.7%** |
| chocolate | 13469 | 12855 | **95.4%** |
| cheeses | 21208 | 14967 | **70.6%** |

## Validation vs manual gold (250 codes per cat)

### Pasta — overall **+1.4%**

| attr | silver | consolidated | diff |
|---|---|---|---|
| pasta_shape | 70.5% | **87.6%** | **+17.1%** |
| nutri_score_grade | 71.2% | **78.1%** | +6.9% |
| grain_type | 89.4% | 92.3% | +2.9% |
| protein_class | 80% | 82.2% | +2.2% |
| is_filled, is_gluten_free, is_organic | 97-98% | 96-97% | ≤-1% |
| is_vegan | 100% | 97.6% | -2.4% |
| **OVERALL** | **91.8%** | **93.2%** | **+1.4%** |

### Chocolate — overall **+1.4%**

| attr | silver | consolidated | diff |
|---|---|---|---|
| chocolate_extra | 64.7% | **68.0%** | +3.3% |
| cocoa_percentage | 81.4% | **87.8%** | +6.5% |
| chocolate_type | 87.8% | 89.0% | +1.2% |
| is_organic | 91.1% | 93.2% | +2.1% |
| nutri_score_grade | 85.1% | 86.9% | +1.8% |
| contains_nuts | 84.5% | 82.9% | -1.6% |
| protein_class | 86.7% | 85.3% | -1.4% |
| **OVERALL** | **82.6%** | **84.0%** | **+1.4%** |

### Cheeses — overall **-4.5%** ⚠

| attr | silver | consolidated | diff |
|---|---|---|---|
| country_of_origin | 100% | 98.8% | -1.2% |
| is_organic | 99.4% | 99.3% | -0.1% |
| is_pdo | 96.6% | 96.4% | -0.2% |
| is_ultra_processed | 85.7% | 84.0% | -1.7% |
| milk_source | 97.4% | 94.4% | -3.0% |
| texture | 91.1% | 86.3% | -4.8% |
| **fat_class** | **72.5%** | **52.3%** | **-20.2%** |
| **OVERALL** | **90.9%** | **86.4%** | **-4.5%** |

## Confidence stratification работает чисто

На pasta для всех attrs:
- `consensus` (multi-source agree): **99-100% accuracy** vs gold
- `single` (only 1 source): 78-98%
- `priority` (conflict): 0-50% — conflicts are genuinely hard cases

То есть **consensus-only subset = very high quality**. Conflicts надо изучать отдельно.

## Critical bug — cheeses fat_class regression

**Root cause:** `silver_standard.parquet` имеет **stale** TYPE_C labels:
- silver_standard.fat_class column: 52% accuracy vs gold (computed in old run)
- experiment_per_product.gt (recomputed inline with current `rules.py`): 72% accuracy

Consolidated подцепил stale silver column. Gemini majority (21k votes) overrules silver (1.2k votes). Result ≈ Gemini accuracy ≈ 52%.

**Fix:** для TYPE_C attrs (nutri_score_grade, protein_class, fat_class, cocoa_percentage) recompute из raw nutriments через `src/pipeline/off_labels/rules.py:_type_c_buckets`, use as priority=1000 source.

## Что делать дальше

### Quick wins
1. **Fix cheeses fat_class** — special-case TYPE_C: recompute через rules.py, не использовать stale silver column.
2. После fix expected: cheeses overall +2-3% vs silver (recovering 20pp на fat_class).

### Use this как новый ground truth
1. Replace silver gt в `experiment_per_product_*.parquet` с consolidated gold.
2. Retrain ML на new gt → ожидается accuracy boost ~2-5% per cat on critical attrs (pasta_shape, chocolate_extra, cocoa_percentage, nutri_score_grade).

### Что НЕ нужно делать
- ❌ Full OFF download (7.5GB) — coverage уже 70-95%, marginal benefit низкий
- ❌ Phase 2 re-blind audit — consolidated gold заменяет blind audit как ground truth
- ❌ Phase 1.8 cocoa labelspace fix — silver и Gemini оба already в same labelspace, mismatch не наблюдается

### Что остаётся актуальным из IMPLEMENTATION_PLAN
- Phase 1.2 threshold sensitivity — still relevant on new gold
- Phase 1.3 cost_quality LLM honest — still relevant
- Phase 1.4 FDR — still relevant
- Phase 1.5 recompute_calibration bug — still relevant
- Phase 1.6 class balance audit — should re-run на new gold
- Phase 3 disclaimers — narrative нужно переписать: "consolidated multi-source gold с conflict resolution" вместо "circularity concerns"
- Phase 4 notebook — update tables с new numbers
- Phase 5 verification — relevant

## Артефакты

Создано:
- `src/eval/consolidate_llm_gold.py` — скрипт consolidation
- `datasets/processed/{cat}_labels_all_sources_long.parquet` — long-format всех меток (178k-189k rows per cat)
- `datasets/processed/{cat}_consolidated_gold_v1.parquet` — resolved gold

Не сделано (next iteration):
- TYPE_C fix через rules.py recompute
- Replacement of silver gt in experiment_per_product
- ML retrain
