#!/bin/bash
# Сборка ВКР в .docx через skill claude-skill-docx + post-process под ГОСТ.
#
# Скрипт лежит в docs/thesis/docx_build/, а md-исходники и итоговый docx —
# в docs/thesis/ (родительская папка). THESIS_DIR определяется автоматически.
#
# Usage:
#   bash docs/thesis/docx_build/build_docx.sh
# Output:
#   docs/thesis/VKR_Frolov_2026.docx

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
THESIS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$THESIS_DIR"

OUTPUT="VKR_Frolov_2026.docx"
SKILL_DIR="$HOME/.claude/skills/claude-skill-docx"

# Порядок сборки (титул → реферат → введение → главы → заключение → литература).
INPUTS=(
    "00_titul_referat.md"
    "00_introduction.md"
    "01_chapter1_analysis.md"
    "02_chapter2_theory.md"
    "03_chapter3_implementation.md"
    "04_chapter4_results.md"
    "05_conclusion.md"
    "06_references.md"
)

# Проверка наличия всех входных файлов
for f in "${INPUTS[@]}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f не найден"
        exit 1
    fi
done

# Шаг 1: склеить все md в один временный файл для skill (он принимает один input).
# Каждый файл закрываем двойным переносом строки — без этого H1 первого заголовка
# следующего файла прилипает к концу предыдущего текста и пандок теряет уровень.
COMBINED="$SCRIPT_DIR/_combined.md"
: > "$COMBINED"
for f in "${INPUTS[@]}"; do
    cat "$f" >> "$COMBINED"
    printf '\n\n' >> "$COMBINED"
done

# Пути к картинкам в md заданы относительно docs/thesis/ (figures/pptx/...).
# После склейки combined.md лежит в docs/thesis/docx_build/, поэтому переписываем
# относительные пути на ../figures/, чтобы md_to_docx_py их нашёл.
sed -i '' 's|](figures/|](../figures/|g' "$COMBINED"

# Шаг 1.5: рендеринг LaTeX-формул в PNG через codecogs.com
python "$SCRIPT_DIR/render_formulas.py" "$COMBINED"

# Шаг 2: md → docx через md_to_docx_py.py (без template — собираем с нуля)
echo "Сборка $OUTPUT из ${#INPUTS[@]} md-файлов через skill..."
python "$SKILL_DIR/scripts/md_to_docx_py.py" \
    "$COMBINED" "$OUTPUT" \
    --font-body "Times New Roman" \
    --font-heading "Times New Roman" \
    --font-size 14 \
    --color-heading "000000" \
    --color-body "000000" \
    --no-banded-rows \
    --table-header-bg "FFFFFF" \
    --table-header-text "000000" \
    --table-alt-row "FFFFFF" \
    --table-border "000000" \
    --table-font-size 12

# Шаги 3..7 — пост-обработка docx под ГОСТ 7.32-2017 (обиходно «7.32-2018»):
echo "Применяю fix_docx_styles.py (body: красная строка 1.25 см + justify)..."
python "$SCRIPT_DIR/fix_docx_styles.py" "$OUTPUT" "$OUTPUT"

echo "Применяю fix_headings_gost.py (заголовки: чёрные, H1 по центру прописными, H2/H3 слева жирные)..."
python "$SCRIPT_DIR/fix_headings_gost.py" "$OUTPUT"

echo "Применяю fix_title_page.py (титульный лист)..."
python "$SCRIPT_DIR/fix_title_page.py" "$OUTPUT"

echo "Применяю fix_tables_gost.py (подписи таблиц и ячейки)..."
python "$SCRIPT_DIR/fix_tables_gost.py" "$OUTPUT"

echo "Применяю fix_tables_repeat_header.py (повтор шапки при переносе)..."
python "$SCRIPT_DIR/fix_tables_repeat_header.py" "$OUTPUT"

echo "Применяю fix_list_numbering.py (локальный рестарт нумерации списков)..."
python "$SCRIPT_DIR/fix_list_numbering.py" "$OUTPUT"

echo "Применяю fix_page_setup.py (поля 30/15/20/20 + нумерация страниц)..."
python "$SCRIPT_DIR/fix_page_setup.py" "$OUTPUT"

echo "Применяю fix_typography_docx.py (NBSP для чисел и единиц)..."
python "$SCRIPT_DIR/fix_typography_docx.py" "$OUTPUT"

echo "Применяю insert_toc.py (Word TOC field перед ВВЕДЕНИЕМ)..."
python "$SCRIPT_DIR/insert_toc.py" "$OUTPUT"

# Валидация
echo "Валидация:"
python "$SKILL_DIR/scripts/docx_validate.py" "$OUTPUT" 2>&1 | tail -10

rm -f "$COMBINED"

# Шаг 8 (опционально): автоподсчёт страниц основной части и пересборка.
# Запускается, если в 00_titul_referat.md есть плейсхолдер {{MAIN_PAGES}}
# или если задана переменная окружения COUNT_PAGES=1.
if grep -q "{{MAIN_PAGES}}" 00_titul_referat.md 2>/dev/null || [ "${COUNT_PAGES:-0}" = "1" ]; then
    echo "Автоподсчёт страниц основной части (требуется Microsoft Word)..."
    python "$SCRIPT_DIR/count_pages.py" "$OUTPUT" || echo "WARN: автоподсчёт страниц не выполнен"
    if ! grep -q "{{MAIN_PAGES}}" 00_titul_referat.md 2>/dev/null; then
        echo "Пересборка docx с обновлённым числом страниц..."
        : > "$COMBINED"
        for f in "${INPUTS[@]}"; do
            cat "$f" >> "$COMBINED"
            printf '\n\n' >> "$COMBINED"
        done
        sed -i '' 's|](figures/|](../figures/|g' "$COMBINED"
        python "$SCRIPT_DIR/render_formulas.py" "$COMBINED"
        python "$SKILL_DIR/scripts/md_to_docx_py.py" \
            "$COMBINED" "$OUTPUT" \
            --font-body "Times New Roman" --font-heading "Times New Roman" \
            --font-size 14 --color-heading "000000" --color-body "000000" \
            --no-banded-rows --table-header-bg "FFFFFF" --table-header-text "000000" \
            --table-alt-row "FFFFFF" --table-border "000000" --table-font-size 12
        python "$SCRIPT_DIR/fix_docx_styles.py" "$OUTPUT" "$OUTPUT"
        python "$SCRIPT_DIR/fix_headings_gost.py" "$OUTPUT"
        python "$SCRIPT_DIR/fix_title_page.py" "$OUTPUT"
        python "$SCRIPT_DIR/fix_tables_gost.py" "$OUTPUT"
        python "$SCRIPT_DIR/fix_page_setup.py" "$OUTPUT"
        python "$SCRIPT_DIR/fix_typography_docx.py" "$OUTPUT"
        python "$SCRIPT_DIR/insert_toc.py" "$OUTPUT"
        rm -f "$COMBINED"
    fi
fi

echo "Готово. Размер файла:"
ls -lh "$OUTPUT" | awk '{print $5, $9}'
