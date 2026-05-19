"""Пост-обработка собранного pandoc-ом docx под академический стиль МАИ.

Что делает:
  1. Открывает docx как zip-архив (обходит баг python-docx с embedded-шрифтами).
  2. Правит определения стилей в word/styles.xml для body-text-стилей pandoc
     (Compact, FirstParagraph, ac) — добавляет:
        - красную строку 1.25 см (firstLine=709 twips)
        - убирает space-before/space-after между абзацами
        - межстрочный интервал 1.5 (line=360 twips, lineRule=auto)
        - выравнивание по ширине (justify)
        - Times New Roman 14 pt (28 half-points)
  3. Записывает результат поверх исходного файла.

Заголовки (стили 1/2/3), таблицы и SourceCode не трогаем.

Usage:
    python fix_docx_styles.py [input.docx] [output.docx]

По умолчанию input = VKR_Frolov_2026.docx, output = VKR_Frolov_2026.docx (overwrite).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

# Стили pandoc, которые соответствуют body-text-абзацам и должны иметь
# красную строку + единый интервал.
BODY_STYLES = {"Compact", "FirstParagraph", "BodyText", "ac"}

# Параметры академического стиля
FIRST_LINE_TWIPS = 709   # 1.25 см
LINE_TWIPS = 360         # 1.5 интервал
FONT_HALF_PT = 28        # 14 pt
FONT_NAME = "Times New Roman"


def _build_ppr_xml() -> str:
    """Сформировать pPr-блок с нужными настройками абзаца."""
    return (
        f'<w:pPr xmlns:w="{NS_W}">'
        f'  <w:spacing w:before="0" w:after="0" w:line="{LINE_TWIPS}" w:lineRule="auto"/>'
        f'  <w:ind w:firstLine="{FIRST_LINE_TWIPS}"/>'
        f'  <w:jc w:val="both"/>'
        f'</w:pPr>'
    )


def _build_rpr_xml() -> str:
    return (
        f'<w:rPr xmlns:w="{NS_W}">'
        f'  <w:rFonts w:ascii="{FONT_NAME}" w:hAnsi="{FONT_NAME}" w:cs="{FONT_NAME}"/>'
        f'  <w:sz w:val="{FONT_HALF_PT}"/>'
        f'  <w:szCs w:val="{FONT_HALF_PT}"/>'
        f'</w:rPr>'
    )


def _build_style_definition(style_id: str, based_on: str = "Normal") -> str:
    """Сгенерировать полное определение стиля абзаца с body-text-параметрами."""
    ppr = _build_ppr_xml().replace(f' xmlns:w="{NS_W}"', "")
    rpr = _build_rpr_xml().replace(f' xmlns:w="{NS_W}"', "")
    return (
        f'<w:style w:type="paragraph" w:customStyle="1" w:styleId="{style_id}">'
        f'<w:name w:val="{style_id}"/>'
        f'<w:basedOn w:val="{based_on}"/>'
        f'{ppr}{rpr}'
        f'</w:style>'
    )


def fix_styles_xml(styles_xml: str) -> str:
    """Применить body-text-параметры к стилям BODY_STYLES.

    Если определение стиля есть — заменяет pPr/rPr;
    если определения нет — добавляет новое определение перед </w:styles>.
    """
    new_ppr = _build_ppr_xml().replace(f' xmlns:w="{NS_W}"', "")
    new_rpr = _build_rpr_xml().replace(f' xmlns:w="{NS_W}"', "")

    additions: list[str] = []

    for style_id in BODY_STYLES:
        pat_start = re.compile(
            rf'(<w:style[^>]*w:styleId="{re.escape(style_id)}"[^>]*>)',
            re.DOTALL,
        )
        m = pat_start.search(styles_xml)
        if not m:
            # Определения нет — нужно добавить новое
            additions.append(_build_style_definition(style_id))
            print(f"  стиль {style_id}: добавлено новое определение")
            continue
        close_idx = styles_xml.find("</w:style>", m.end())
        if close_idx < 0:
            continue
        block = styles_xml[m.start():close_idx + len("</w:style>")]
        block_new = re.sub(r"<w:pPr[^/]*?>.*?</w:pPr>", "", block, flags=re.DOTALL)
        block_new = re.sub(r"<w:pPr\s*/>", "", block_new)
        block_new = re.sub(r"<w:rPr[^/]*?>.*?</w:rPr>", "", block_new, flags=re.DOTALL)
        block_new = re.sub(r"<w:rPr\s*/>", "", block_new)
        block_new = block_new.replace(
            "</w:style>", f"{new_ppr}{new_rpr}</w:style>", 1,
        )
        styles_xml = styles_xml[:m.start()] + block_new + styles_xml[close_idx + len("</w:style>"):]
        print(f"  стиль {style_id}: обновлён")

    if additions:
        styles_xml = styles_xml.replace(
            "</w:styles>", "".join(additions) + "</w:styles>", 1,
        )

    # Также правим Normal — он подложка для всех body-style
    pat = re.compile(r'(<w:style[^>]*w:styleId="Normal"[^>]*>)(.*?)(</w:style>)', re.DOTALL)
    m = pat.search(styles_xml)
    if m:
        inner = m.group(2)
        inner_new = re.sub(r"<w:pPr[^/]*?>.*?</w:pPr>", "", inner, flags=re.DOTALL)
        inner_new = re.sub(r"<w:pPr\s*/>", "", inner_new)
        inner_new = re.sub(r"<w:rPr[^/]*?>.*?</w:rPr>", "", inner_new, flags=re.DOTALL)
        inner_new = re.sub(r"<w:rPr\s*/>", "", inner_new)
        # Добавим pPr и rPr в начало
        inner_new = new_ppr + new_rpr + inner_new
        styles_xml = styles_xml[:m.start()] + m.group(1) + inner_new + m.group(3) + styles_xml[m.end():]
        print("  стиль Normal: обновлён")

    return styles_xml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", default="VKR_Frolov_2026.docx")
    ap.add_argument("output", nargs="?", default=None)
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output) if args.output else src
    if not src.exists():
        print(f"ERROR: {src} не найден", file=sys.stderr)
        sys.exit(1)

    # Работаем через временный файл, чтобы не повредить исходный при ошибке
    tmp = src.with_suffix(".docx.tmp")
    shutil.copy2(src, tmp)

    # Прочитать и заменить styles.xml
    with zipfile.ZipFile(tmp, "r") as zin:
        names = zin.namelist()
        if "word/styles.xml" not in names:
            print("ERROR: word/styles.xml не найден в архиве", file=sys.stderr)
            sys.exit(1)
        styles_xml = zin.read("word/styles.xml").decode("utf-8")

    print("Правлю определения стилей в word/styles.xml:")
    styles_xml_new = fix_styles_xml(styles_xml)

    # Пересоберём zip с новым styles.xml (zipfile не умеет заменять in-place)
    out_tmp = src.with_suffix(".docx.new")
    with zipfile.ZipFile(tmp, "r") as zin, zipfile.ZipFile(out_tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for name in zin.namelist():
            data = zin.read(name)
            if name == "word/styles.xml":
                data = styles_xml_new.encode("utf-8")
            zout.writestr(name, data)

    os.replace(out_tmp, dst)
    os.remove(tmp)
    print(f"Готово: {dst}")


if __name__ == "__main__":
    main()
