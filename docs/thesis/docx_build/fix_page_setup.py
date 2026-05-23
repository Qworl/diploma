"""Поля страницы и нумерация — без template'а.

Что делает:
  1. word/document.xml → sectPr → поля 30 / 15 / 20 / 20 мм (ГОСТ 7.32-2017 п. 6.1.1; left / right / top / bottom)
     и формат A4 (11906 × 16838 twips)
  2. word/footer1.xml → нумерация страниц по центру внизу (PAGE field)
     + регистрация footer в content types и relationships
  3. Hyperlink/PageNumber стили — если отсутствуют, добавляем

Запускается ПОСЛЕ md_to_docx_py (без template) и до открытия в Word.
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
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_CT = "http://schemas.openxmlformats.org/package/2006/content-types"

# ГОСТ 7.32-2017 п. 6.1.1: левое 30, правое 15, верхнее 20, нижнее 20 мм.
# 1 мм ≈ 56.7 twips. Используем 1 cm = 567 twips.
MARGIN_LEFT = 1701   # 30 мм
MARGIN_RIGHT = 850   # 15 мм (ГОСТ 7.32-2017; обиходно называется «7.32-2018» по году ввода 01.07.2018)
MARGIN_TOP = 1134    # 20 мм
MARGIN_BOTTOM = 1134 # 20 мм
PAGE_W = 11906       # A4 ширина в twips
PAGE_H = 16838       # A4 высота в twips


def _fix_sect_pr(doc_xml: str, footer_rel_id: str | None,
                  first_footer_rel_id: str | None) -> str:
    """Заменить или добавить <w:sectPr> с правильными полями.

    footer_rel_id — id отношения footer'а по умолчанию (например rId9).
    first_footer_rel_id — id отношения для first-page footer (пустой, чтобы
    на титульном листе не появлялся номер страницы). Если задан, добавляем
    также <w:titlePg/> в конец sectPr (включает first-page header/footer).
    """
    # ВАЖНО: порядок дочерних элементов sectPr строго фиксирован OOXML-схемой:
    # headerReference, footerReference (в т.ч. type="first"), pgSz, pgMar,
    # cols, docGrid, titlePg в конце.
    footer_refs = []
    if footer_rel_id:
        footer_refs.append(
            f'<w:footerReference xmlns:r="{NS_R}" '
            f'w:type="default" r:id="{footer_rel_id}"/>'
        )
    if first_footer_rel_id:
        footer_refs.append(
            f'<w:footerReference xmlns:r="{NS_R}" '
            f'w:type="first" r:id="{first_footer_rel_id}"/>'
        )
    footer_block = "".join(footer_refs)
    title_pg = '<w:titlePg/>' if first_footer_rel_id else ''
    new_sect = (
        f'<w:sectPr xmlns:w="{NS_W}">'
        f'{footer_block}'
        f'<w:pgSz w:w="{PAGE_W}" w:h="{PAGE_H}"/>'
        f'<w:pgMar w:top="{MARGIN_TOP}" w:right="{MARGIN_RIGHT}" '
        f'w:bottom="{MARGIN_BOTTOM}" w:left="{MARGIN_LEFT}" '
        f'w:header="708" w:footer="708" w:gutter="0"/>'
        f'<w:cols w:space="708"/>'
        f'<w:docGrid w:linePitch="360"/>'
        f'{title_pg}'
        f'</w:sectPr>'
    )

    if "<w:sectPr" in doc_xml:
        doc_xml = re.sub(r"<w:sectPr[^/]*?>.*?</w:sectPr>", new_sect, doc_xml, flags=re.DOTALL)
        doc_xml = re.sub(r"<w:sectPr\s*/>", new_sect, doc_xml)
    else:
        doc_xml = doc_xml.replace("</w:body>", new_sect + "</w:body>")
    return doc_xml


def _find_or_create_footer_rel(rels_xml: str, target: str) -> tuple[str, str]:
    """Найти существующий footer-rel для target или создать новый. Возвращает (новый_rels_xml, rel_id)."""
    pattern = (
        r'<Relationship\s+Id="([^"]+)"[^>]*'
        r'Type="[^"]*relationships/footer"[^>]*Target="' + re.escape(target) + r'"[^>]*/>'
    )
    m = re.search(pattern, rels_xml)
    if m:
        return rels_xml, m.group(1)
    ids = re.findall(r'Id="rId(\d+)"', rels_xml)
    next_id = max((int(x) for x in ids), default=0) + 1
    rel_id = f"rId{next_id}"
    new_rel = (
        f'<Relationship Id="{rel_id}" '
        f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" '
        f'Target="{target}"/>'
    )
    rels_xml = rels_xml.replace("</Relationships>", new_rel + "</Relationships>")
    return rels_xml, rel_id


def _make_footer_xml() -> str:
    """Footer с центрированным полем PAGE для нумерации страниц."""
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:ftr xmlns:w="{NS_W}">
  <w:p>
    <w:pPr>
      <w:pStyle w:val="Footer"/>
      <w:jc w:val="center"/>
      <w:rPr>
        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>
        <w:sz w:val="28"/>
      </w:rPr>
    </w:pPr>
    <w:r>
      <w:rPr>
        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>
        <w:sz w:val="28"/>
      </w:rPr>
      <w:fldChar w:fldCharType="begin"/>
    </w:r>
    <w:r>
      <w:rPr>
        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>
        <w:sz w:val="28"/>
      </w:rPr>
      <w:instrText xml:space="preserve">PAGE</w:instrText>
    </w:r>
    <w:r>
      <w:rPr>
        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>
        <w:sz w:val="28"/>
      </w:rPr>
      <w:fldChar w:fldCharType="end"/>
    </w:r>
  </w:p>
</w:ftr>'''


def _make_empty_footer_xml() -> str:
    """Пустой footer (для титульного листа — без номера страницы)."""
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:ftr xmlns:w="{NS_W}">
  <w:p>
    <w:pPr>
      <w:pStyle w:val="Footer"/>
    </w:pPr>
  </w:p>
</w:ftr>'''


def _update_content_types(ct_xml: str) -> str:
    """Зарегистрировать footer1.xml и footer2.xml в [Content_Types].xml."""
    for fname in ("footer1.xml", "footer2.xml"):
        if f"/word/{fname}" in ct_xml:
            continue
        override = (
            f'<Override PartName="/word/{fname}" '
            f'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>'
        )
        ct_xml = ct_xml.replace("</Types>", override + "</Types>")
    return ct_xml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("docx", nargs="?", default="VKR_Frolov_2026.docx")
    args = ap.parse_args()

    src = Path(args.docx)
    if not src.exists():
        print(f"ERROR: {src} не найден", file=sys.stderr)
        sys.exit(1)

    print(f"Настраиваю поля и footer в {src}")

    tmp = src.with_suffix(".docx.tmp")
    shutil.copy2(src, tmp)

    with zipfile.ZipFile(tmp, "r") as zin:
        files = {name: zin.read(name) for name in zin.namelist()}

    # 1. Сначала обновляем rels — узнаём footer rel_id (или создаём)
    footer_rel_id = None
    first_footer_rel_id = None
    if "word/_rels/document.xml.rels" in files:
        rels = files["word/_rels/document.xml.rels"].decode("utf-8")
        rels, footer_rel_id = _find_or_create_footer_rel(rels, "footer1.xml")
        rels, first_footer_rel_id = _find_or_create_footer_rel(rels, "footer2.xml")
        files["word/_rels/document.xml.rels"] = rels.encode("utf-8")
        print(f"  default footer: {footer_rel_id}, first-page footer: {first_footer_rel_id}")

    # 2. Поля страницы + footerReference + titlePg (без номера на первой стр.)
    doc_xml = files["word/document.xml"].decode("utf-8")
    doc_xml = _fix_sect_pr(doc_xml, footer_rel_id, first_footer_rel_id)
    files["word/document.xml"] = doc_xml.encode("utf-8")
    print("  поля страницы: 30/15/20/20 мм + A4, titlePg=true")

    # 3. Footer XML
    files["word/footer1.xml"] = _make_footer_xml().encode("utf-8")
    files["word/footer2.xml"] = _make_empty_footer_xml().encode("utf-8")
    print("  footer1: PAGE field по центру, footer2: пустой (для титульной)")

    # 4. Content types
    if "[Content_Types].xml" in files:
        ct = files["[Content_Types].xml"].decode("utf-8")
        ct = _update_content_types(ct)
        files["[Content_Types].xml"] = ct.encode("utf-8")
        print("  content types: footer1 + footer2 зарегистрированы")

    out_tmp = src.with_suffix(".docx.new")
    with zipfile.ZipFile(out_tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in files.items():
            zout.writestr(name, data)

    os.replace(out_tmp, src)
    os.remove(tmp)
    print("Готово.")


if __name__ == "__main__":
    main()
