"""Привести стили заголовков собранного docx к требованиям ГОСТ 7.32-2018.

Что правит в word/styles.xml:

  Heading 1 (главы)         — Times New Roman 16 pt, прописные, по центру,
                              чёрный, разрыв страницы перед, без отступа красной строки
  Heading 2 (§X.Y)          — Times New Roman 14 pt, жирный, слева,
                              отступ красной строки 1.25 см, чёрный
  Heading 3 (§X.Y.Z)        — Times New Roman 14 pt, курсив, слева,
                              отступ красной строки 1.25 см, чёрный

Запускать после md_to_docx_py.py:
    python fix_headings_gost.py [docx_path]
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

# Стили pandoc → как мы их хотим оформить
HEADING_OVERRIDES = {
    "Heading1": {
        "size_pt": 16,
        "bold": True,
        "italic": False,
        "uppercase": True,
        "align": "center",
        "color": "000000",
        "first_line": 0,
        "page_break_before": True,
        "space_before": 0,
        "space_after": 240,  # 12 pt
    },
    "Heading2": {
        "size_pt": 14,
        "bold": True,
        "italic": False,
        "uppercase": False,
        "align": "left",
        "color": "000000",
        "first_line": 720,  # 1.25 см — отступ красной строки по ГОСТ 7.32-2018
        "page_break_before": False,
        "space_before": 480,  # 24 pt — две пустые строки сверху, чтобы заголовок не сливался с body
        "space_after": 240,
    },
    "Heading3": {
        "size_pt": 14,
        "bold": True,
        "italic": False,
        "uppercase": False,
        "align": "left",
        "color": "000000",
        "first_line": 720,  # 1.25 см — отступ красной строки по ГОСТ 7.32-2018
        "page_break_before": False,
        "space_before": 360,  # 18 pt
        "space_after": 180,
    },
}


def _build_ppr(cfg: dict) -> str:
    elems: list[str] = []
    if cfg.get("page_break_before"):
        elems.append('<w:pageBreakBefore/>')
    elems.append(
        f'<w:spacing w:before="{cfg["space_before"]}" '
        f'w:after="{cfg["space_after"]}" '
        f'w:line="360" w:lineRule="auto"/>'
    )
    elems.append(f'<w:ind w:firstLine="{cfg["first_line"]}"/>')
    elems.append(f'<w:jc w:val="{cfg["align"]}"/>')
    return f'<w:pPr>{"".join(elems)}</w:pPr>'


def _build_rpr(cfg: dict) -> str:
    half_pt = int(cfg["size_pt"] * 2)
    elems = [
        f'<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>',
        f'<w:color w:val="{cfg["color"]}"/>',
        f'<w:sz w:val="{half_pt}"/>',
        f'<w:szCs w:val="{half_pt}"/>',
    ]
    if cfg.get("bold"):
        elems.append('<w:b/><w:bCs/>')
    if cfg.get("italic"):
        elems.append('<w:i/><w:iCs/>')
    if cfg.get("uppercase"):
        elems.append('<w:caps/>')
    return f'<w:rPr>{"".join(elems)}</w:rPr>'


def fix_styles_xml(styles_xml: str) -> str:
    for style_id, cfg in HEADING_OVERRIDES.items():
        pat_start = re.compile(
            rf'(<w:style[^>]*w:styleId="{re.escape(style_id)}"[^>]*>)',
            re.DOTALL,
        )
        m = pat_start.search(styles_xml)
        if not m:
            print(f"  стиль {style_id}: НЕ найден, пропуск")
            continue
        close_idx = styles_xml.find("</w:style>", m.end())
        if close_idx < 0:
            continue
        block = styles_xml[m.start():close_idx + len("</w:style>")]

        # Удаляем существующие pPr / rPr
        block_new = re.sub(r"<w:pPr[^/]*?>.*?</w:pPr>", "", block, flags=re.DOTALL)
        block_new = re.sub(r"<w:pPr\s*/>", "", block_new)
        block_new = re.sub(r"<w:rPr[^/]*?>.*?</w:rPr>", "", block_new, flags=re.DOTALL)
        block_new = re.sub(r"<w:rPr\s*/>", "", block_new)

        # Вставляем наши перед </w:style>
        block_new = block_new.replace(
            "</w:style>", f"{_build_ppr(cfg)}{_build_rpr(cfg)}</w:style>", 1,
        )

        styles_xml = styles_xml[:m.start()] + block_new + styles_xml[close_idx + len("</w:style>"):]
        print(f"  стиль {style_id}: обновлён")
    return styles_xml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("docx", nargs="?", default="VKR_Frolov_2026.docx")
    args = ap.parse_args()

    src = Path(args.docx)
    if not src.exists():
        print(f"ERROR: {src} не найден", file=sys.stderr)
        sys.exit(1)

    print(f"Правлю стили заголовков под ГОСТ в {src}:")

    tmp = src.with_suffix(".docx.tmp")
    shutil.copy2(src, tmp)
    with zipfile.ZipFile(tmp, "r") as zin:
        styles_xml = zin.read("word/styles.xml").decode("utf-8")

    styles_xml_new = fix_styles_xml(styles_xml)

    out_tmp = src.with_suffix(".docx.new")
    with zipfile.ZipFile(tmp, "r") as zin, zipfile.ZipFile(out_tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for name in zin.namelist():
            data = zin.read(name)
            if name == "word/styles.xml":
                data = styles_xml_new.encode("utf-8")
            zout.writestr(name, data)

    os.replace(out_tmp, src)
    os.remove(tmp)
    print("Готово.")


if __name__ == "__main__":
    main()
