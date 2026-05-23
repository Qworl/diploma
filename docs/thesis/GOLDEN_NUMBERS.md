# GOLDEN_NUMBERS — числовые якоря ВКР

Наполнено в этапе -1 из `_baseline.docx`. Каждая правка, меняющая число, должна синхронно обновить эту таблицу. `check_regression.py` нормализует пробелы (`\s+ → ' '`), поэтому NBSP в DOCX и обычный пробел в этом файле эквивалентны.

## Правила

- Запись — в той форме, в которой число встречается в DOCX/`.md`.
- Если число встречается в нескольких разделах с разным форматированием — каждое представление = отдельная строка.
- Колонки: `Число | Файл | Раздел/Контекст | Происхождение (parquet/json)`.

## Числа реферата (00_titul_referat.md)

| Число | Файл | Раздел/Контекст | Происхождение |
|---|---|---|---|
| `91 страница` | `00_titul_referat.md` | объём работы | `count_pages.py` |
| `9 рисунк` | `00_titul_referat.md` | объём работы | ручной подсчёт |
| `22 таблиц` | `00_titul_referat.md` | объём работы | ручной подсчёт |
| `52 источник` | `00_titul_referat.md` | объём работы | `06_references.md` |

## Headline точности

| Число | Файл | Раздел/Контекст | Происхождение |
|---|---|---|---|
| `92,8 %` | `00_titul_referat.md`, `00_introduction.md`, `05_conclusion.md`, `04_chapter4_results.md` | сквозная точность каскад + gemini-flash | `grand_acc_summary_after_fix.parquet` |
| `92,78 %` | `03_chapter3_implementation.md` | детальная сквозная точность (§3.3.2) | `headline_v3e_after_fix.parquet` |
| `92,3 %` | `00_titul_referat.md`, `05_conclusion.md` | каскад + gpt-oss-120b | `headline_v3e_after_fix.parquet` |
| `83,79 %` | `03_chapter3_implementation.md` | all-sonnet baseline | `headline_v3e_after_fix.parquet` |
| `90,5 %` | `00_titul_referat.md`, `00_introduction.md`, `05_conclusion.md` | средняя точность каскада без LLM | `grand_acc_summary_after_fix.parquet` |
| `96,7 %` | `00_titul_referat.md`, `04_chapter4_results.md`, `05_conclusion.md` | покрытие каскада без LLM | `grand_acc_summary_after_fix.parquet` |
| `69,3 %` | `00_titul_referat.md` | прямое использование gpt-oss-120b | `headline_v3e_after_fix.parquet` |

## Размер выборки, бренды, k-NN

| Число | Файл | Раздел/Контекст | Происхождение |
|---|---|---|---|
| `n = 1539` | `03_chapter3_implementation.md` | brand-disjoint test §3.3.5 | `consensus_gold_v2_expanded.parquet` |
| `4350` | `00_titul_referat.md`, `00_introduction.md`, `05_conclusion.md` | тестовая выборка (ячеек) | `headline_v3e_after_fix.parquet` |
| `k = 5` | `03_chapter3_implementation.md` | knn-distance ablation §3.3.7.3 | `knn_distance_ablation.parquet` |
| `47/47` | `03_chapter3_implementation.md` | precision Слоя 1 на сырах | `lexicon_regex_comparison.parquet` |

## Приросты и кросс-доменная репликация

| Число | Файл | Раздел/Контекст | Происхождение |
|---|---|---|---|
| `+9,0 п. п.` | `00_titul_referat.md`, `00_introduction.md`, `04_chapter4_results.md`, `05_conclusion.md` | прирост точности vs Sonnet 4.5 | `grand_acc_summary_after_fix.parquet` |
| `+23,0 п. п.` | `00_titul_referat.md`, `05_conclusion.md` | прирост vs прямой gpt-oss-120b | `headline_v3e_after_fix.parquet` |
| `+18,2 п. п.` | `05_conclusion.md` | цикл аудита, реплика 1 | `cascade_preds_*_after_fix.parquet` |
| `+16,3 п. п.` | `05_conclusion.md` | цикл аудита, реплика 2 | `cascade_preds_*_after_fix.parquet` |
| `+14,4 п. п.` | `05_conclusion.md` | цикл аудита, реплика 3 | `cascade_preds_*_after_fix.parquet` |

## Стоимость и трафик LLM

| Число | Файл | Раздел/Контекст | Происхождение |
|---|---|---|---|
| `3,3 %` | `00_titul_referat.md`, `04_chapter4_results.md`, `05_conclusion.md` | доля обращений к LLM | `tier_breakdown.parquet` |
| `720` | `00_titul_referat.md`, `04_chapter4_results.md`, `05_conclusion.md` | кратность снижения стоимости | производное от cost-matrix |
| `0,14 %` | `04_chapter4_results.md` | стоимость каскад+gemini vs Sonnet 4.5 | `grand_acc_summary_after_fix.parquet` |

## Чек-листы

- [x] этап -1 завершён, наполнено по `_baseline.docx` (пример удалён)
- [x] `check_regression.py` зелёный на этих якорях (true positive)
- [x] для каждого числа указан источник в parquet/json/процедуре
