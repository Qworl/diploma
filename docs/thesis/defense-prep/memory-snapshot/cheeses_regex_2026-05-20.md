---
name: Layer 1 расширен на сыры
description: 2026-05-20 — в RegexExtractor добавлены multilingual паттерны для cheeses (milk_source / is_pdo / is_ultra_processed). Загружать когда трогаем extractor.py, регенерируем cascade, или обновляем §5/§6.13 ноутбука и слайд «послойный вклад каскада».
type: project
originSessionId: e4355493-787d-41ac-a3a3-de828031510b
---
В Layer 1 добавлены три атрибута для категории `cheeses` (`src/pipeline/regex/extractor.py`, ветка `category=='cheeses'` в `extract_all`):

- **milk_source** (3 паттерна): goat (chèvre/cabra/Ziege/capra/caprino), sheep (brebis/oveja/Schaf/pecora/pecorino), buffalo (bufala/Büffel). Cow намеренно не извлекается — слово частотное, ML точнее. Применяется к name_text (без ingredients_text — иначе «lait» из состава путает источники).
- **is_pdo**: один regex `\b(?:AOP|AOC|DOP|PDO|IGP)\b` на full_text.
- **is_ultra_processed**: узкие маркеры `fondu / processed cheese / Schmelzkäse / formaggio fuso / queso fundido`. **Намеренно исключены** «slices / tranches / triangles» — после аудита false-positive показал, что это формы нарезки натурального сыра (Gouda Slices, Double Gloucester slice → gold=False), а не маркеры NOVA 4. Аудит сделан на cascade_preds_cheeses_after_fix.parquet.

**Why:**
- Слайд «Послойный вклад каскада» исторически показывал 0 % regex на сырах — это была дыра в `extract_all` (ветки `cheeses` просто не существовало), а не отсутствие пригодных паттернов.
- Cheese regex даёт 47/1673 = 2.8 % ячеек на gold-eval, precision **100 %** (47/47) — все три атрибута 100 % accuracy: milk_source 31/31, is_pdo 14/14, is_ultra_processed 2/2.
- Headline: cascade-only acc 87.13 % → **90.51 %** (+3.38 п.п.); cascade + gemini25flash 91.49 % → **92.78 %** (+1.29 п.п.). Большая часть прироста — от threshold-overrides на cheese ML-атрибутах (texture / is_organic / is_ultra_processed), но regex даёт критическую долю на абстейны и закрывает дыру в нарративе слайда.
- Скрипт регенерации: `src/experiments/regen_all_categories_after_fix.py` — пересоздаёт `cascade_preds_{cat}_after_fix.parquet`, `headline_v3e_after_fix.parquet`, `grand_acc_summary_after_fix.parquet`.

**How to apply:**
- Тесты (10 новых в `tests/test_regex.py::TestCheeses`) фиксируют как positive matches, так и **отрицательные** кейсы (Gouda Slices → is_ultra_processed=None, Cheddar mature → milk_source=None) — это защита от регрессии к шумным паттернам.
- При обновлении слайда «Послойный вклад» новые значения: **Паста 18/74/8, Шоколад 23/72/5, Сыры 3/94/3** (regex / ml / abstain%). Колонка abstain в слайде маппится в «LLM-fallback».
- В ноутбуке §5 (cells 28–30) уже указывает на `cascade_preds_{cat}_after_fix.parquet`. Если когда-нибудь делать новый full-cascade re-run, обновлять путь к артефакту в cell 29 (load) и cell 28 (markdown narrative).

**Известные ограничения регекса на сырах:**
- texture, country_of_origin, fat_class, is_organic — обрабатываются ML. Country по бренду рискованно (без categories_tags), texture требует огромного списка famous-names, is_organic шумен (`bio` ≠ организ для brand-токена «Bio C'Bon»).
- milk_source на Manchego/Roquefort/Pecorino-style продуктах ловится только когда явное слово (`pecorino`, `oveja`) в названии — иначе передаётся в ML. Гольд показывает, что ML это компенсирует (cheeses/milk_source ML acc ≈ 99.5 %).
