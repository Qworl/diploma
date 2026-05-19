# models/

## Структура

```
models/
├── *_xgb_hybrid.pkl + *_le_hybrid.pkl  ← production: копия v3e_off_tier0 (48 файлов) ⭐
├── v2_clean/                            ← Tier1+2 only, no Tier3 (44 файла, 32MB)
├── v3_clean/                            ← + Tier3 gemini OLD prompt, uniform weights (44 файла, 42MB)
├── v3b_weighted/                        ← v3 + tier weights s=1,t1=6,t2=4,t3=2 (44 файла, 42MB)
├── v3c_no_silver/                       ← v3b с silver=0 (44 файла, 41MB)
├── v3c_silver0/                         ← v3c вариант (early version, 44 файла, 41MB)
├── v3d_promptfix/                       ← Opus 4.5 + gemini promptfix (44 файла, 42MB)
├── v3e_off_tier0/                       ← v3d + Tier0 deterministic OFF (48 файлов) ⭐ production
└── full_snapshot_2026-05-17/            ← полный исторический снимок 523 файлов (229MB)
```

## Какой когда использовать

| Use case | Variant |
|---|---|
| Production cascade (current default) | root `models/*.pkl` (= v3e_off_tier0) |
| Best on OFF-OOD deterministic (89.1% acc, +7pp над v3b) | `v3e_off_tier0/` ⭐ |
| Best Opus-labeled benchmark (96.0% with leak caveat) | `v3d_promptfix/` |
| Best on Tier1+2 small-clean | `v3b_weighted/` |
| Baseline (no Tier3 expansion) | `v2_clean/` |
| Ablation: silver=0 | `v3c_no_silver/` |
| Full historical reproducibility | `full_snapshot_2026-05-17/` |

## Восстановление варианта в production

```bash
# Переключить production на конкретный вариант
rm models/*_xgb_hybrid.pkl models/*_le_hybrid.pkl
cp models/v3b_weighted/*.pkl models/    # или другой вариант
```

## Источник лейблов

- **v2/v3**: Phase 1 Opus (OLD prompt без нутриентов) + gpt-5.5 expansion + gemini B3 OLD
- **v3b/v3c**: те же лейблы, разные веса tier
- **v3d_promptfix**: Opus 4.5 + gemini-flash на NEW promptfix prompt (nutriments + Nutri-Score derivation rules)
- **v3e_off_tier0**: v3d + Tier 0 deterministic OFF-derived labels для `nutri_score_grade`/`protein_class`/`fat_class` (вес 8.0). 56k deterministic меток вместо LLM-разметки производных атрибутов

См. `notebooks/01_phase3_session_log.ipynb` и `datasets/processed/eval_5way_clean_opus_truth.parquet` для подробного сравнения.
