#!/bin/bash
# Доустановка LaTeX-пакетов поверх BasicTeX (TeX Live 2026basic).
# Запуск:  sudo bash docs/thesis/latex/scripts/install_packages.sh
# Idempotent: tlmgr пропускает уже установленные пакеты.

set -eu

# tlmgr и собрат живут в этой папке после basictex 2026.
export PATH="/usr/local/texlive/2026basic/bin/universal-darwin:$PATH"

# 1) Обновить сам tlmgr (мелкая операция, помогает с потенциальными несовместимостями).
echo "==> Обновляю tlmgr…"
tlmgr update --self || echo "WARN: tlmgr self-update не прошёл (продолжаем)"

# 2) Полный список пакетов. Имена даны в виде TeX Live package, а НЕ имени .sty —
# многие .sty-файлы упакованы в более крупные пакеты (tools, koma-script, ...).
PACKAGES=(
    # Сборщики / инструменты команд.
    latexmk
    # БИБЛИОГРАФИЯ.
    biblatex biber biblatex-gost csquotes
    # Русский babel (включает hyphen-russian).
    collection-langcyrillic
    # Шрифтовые и языковые мелочи.
    anyfontsize cm-super lh
    # koma-script: содержит scrextend (его не существует как самостоятельного пакета).
    koma-script
    # tools: содержит hhline, indentfirst, longtable, tabularx, multirow, xpatch и др.
    tools
    # Заголовки, нумерация (titletoc внутри titlesec).
    titlesec chngcntr
    # Списки и нумерация.
    enumitem refcount totcount xassoccnt
    # Геометрия и колонтитулы.
    lastpage
    # Подписи и доп. таблицы.
    caption subcaption tabto-ltx xltabular ltablex
    # Рисунки.
    float
    # Глоссарий и термины.
    glossaries-extra glossaries mfirstuc xfor
    # Алгоритмы и листинги (transitive deps: relsize, needspace, environ, trimspaces).
    algorithm2e relsize needspace environ trimspaces
    # titlesec deps.
    ifoddpage
    # Прочие визуальные.
    nowidow stackengine ulem
    # Графика.
    pgf graphicx-psmin
    # Утилиты строк (требуется glossaries-extra и пр.).
    xstring
)

echo "==> Устанавливаю ${#PACKAGES[@]} пакетов…"
tlmgr install "${PACKAGES[@]}"

# 3) Симлинк для удобства: чтобы xelatex/latexmk были в /Library/TeX/texbin/.
echo "==> Настраиваю /Library/TeX/texbin симлинки (если ещё не созданы)…"
/usr/local/texlive/2026basic/bin/universal-darwin/tlmgr path add 2>&1 | tail -3 || true

# 4) Финальная проверка.
echo
echo "==> Проверка:"
for tool in xelatex latexmk biber; do
    if command -v "$tool" >/dev/null 2>&1; then
        echo "  ✓ $tool: $(command -v "$tool")"
    else
        echo "  ✘ $tool: НЕ НАЙДЕН"
    fi
done
