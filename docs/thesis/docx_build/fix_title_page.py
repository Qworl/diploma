"""Привести оформление титульного листа к академическому виду МАИ.

Использует lxml (а НЕ xml.etree.ElementTree), потому что lxml сохраняет все
namespace-декларации, объявленные в корне документа. ElementTree выкидывает
все namespaces, которые не появляются в дереве явно, что ломает
mc:Ignorable="w14 wp14" и валидацию Word.

Запускается после md_to_docx_py + fix_headings_gost.
Правит первые параграфы документа (до первого Heading1):

  - параграфы с ключевыми словами шапки → центрирование + bold:
       МИНИСТЕРСТВО, ФЕДЕРАЛЬНОЕ, УТВЕРЖДАЮ, ВЫПУСКНАЯ, Система автоматизированного
  - параграфы «Кафедра 806», «Заведующий…», «по теме:», «Москва 2026» — центрирование без bold
  - параграфы «Студент:», «Группа:», «Научный руководитель:» — слева, без bold
  - «(подпись) (инициалы, фамилия)» — мелким (12 pt), центрировано
  - убираем красную строку у всех параграфов титульника

Usage:
    python fix_title_page.py [docx_path]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

from lxml import etree

NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W_NS = f"{{{NS_W}}}"
NSMAP_W = {"w": NS_W}

CENTER_BOLD = ["МИНИСТЕРСТВО", "ФЕДЕРАЛЬНОЕ", "ВЫПУСКНАЯ"]
CENTER_PLAIN = [
    "Институт",
    "Группа М8О",
    "Профиль",
    "Квалификация",
    "на тему",
    "Автор ВКРМ",
    "Руководитель",
    "К защите допустить",
    "Заведующий кафедрой",
    "«____»",
    "Москва",
]
LEFT_BLOCKS: list[str] = []


def _para_text(p_elem) -> str:
    return "".join(t.text or "" for t in p_elem.iter(f"{W_NS}t"))


def _get_or_create_first(parent, tag: str):
    """Найти первый child с тегом или создать новый и вставить в начало."""
    existing = parent.find(tag)
    if existing is not None:
        return existing
    new_el = etree.SubElement(parent, tag)
    # Переставить в начало
    parent.remove(new_el)
    parent.insert(0, new_el)
    return new_el


def _set_para_alignment(p_elem, jc: str):
    pPr = _get_or_create_first(p_elem, f"{W_NS}pPr")
    for existing in pPr.findall(f"{W_NS}jc"):
        pPr.remove(existing)
    jc_elem = etree.SubElement(pPr, f"{W_NS}jc")
    jc_elem.set(f"{W_NS}val", jc)


def _remove_first_line_indent(p_elem):
    pPr = _get_or_create_first(p_elem, f"{W_NS}pPr")
    for existing in pPr.findall(f"{W_NS}ind"):
        pPr.remove(existing)
    ind = etree.SubElement(pPr, f"{W_NS}ind")
    ind.set(f"{W_NS}firstLine", "0")


def _set_bold(p_elem, bold: bool = True):
    for r in p_elem.iter(f"{W_NS}r"):
        rPr = _get_or_create_first(r, f"{W_NS}rPr")
        for existing in rPr.findall(f"{W_NS}b"):
            rPr.remove(existing)
        if bold:
            etree.SubElement(rPr, f"{W_NS}b")


def _set_size(p_elem, half_pt: int):
    for r in p_elem.iter(f"{W_NS}r"):
        rPr = _get_or_create_first(r, f"{W_NS}rPr")
        for sz in rPr.findall(f"{W_NS}sz"):
            rPr.remove(sz)
        for szCs in rPr.findall(f"{W_NS}szCs"):
            rPr.remove(szCs)
        sz_el = etree.SubElement(rPr, f"{W_NS}sz")
        sz_el.set(f"{W_NS}val", str(half_pt))
        szCs_el = etree.SubElement(rPr, f"{W_NS}szCs")
        szCs_el.set(f"{W_NS}val", str(half_pt))


def fix_document_xml(doc_xml: bytes) -> bytes:
    parser = etree.XMLParser(remove_blank_text=False)
    root = etree.fromstring(doc_xml, parser)
    body = root.find(f"{W_NS}body")
    if body is None:
        print("ERROR: <w:body> не найден")
        return doc_xml

    paragraphs = body.findall(f"{W_NS}p")

    changes = 0
    for p in paragraphs:
        pPr = p.find(f"{W_NS}pPr")
        if pPr is not None:
            pStyle = pPr.find(f"{W_NS}pStyle")
            if pStyle is not None and pStyle.get(f"{W_NS}val", "").startswith("Heading"):
                # Достигли первого заголовка — титульник закончился
                break

        text = _para_text(p).strip()
        if not text:
            continue

        if text.startswith("(подпись)"):
            _set_para_alignment(p, "center")
            _remove_first_line_indent(p)
            _set_size(p, 24)  # 12pt
            changes += 1
            continue

        matched = False
        for kw in CENTER_BOLD:
            if text.startswith(kw) or kw in text[:25]:
                _set_para_alignment(p, "center")
                _remove_first_line_indent(p)
                _set_bold(p, True)
                changes += 1
                matched = True
                break
        if matched:
            continue
        for kw in CENTER_PLAIN:
            if text.startswith(kw):
                _set_para_alignment(p, "center")
                _remove_first_line_indent(p)
                _set_bold(p, False)
                changes += 1
                matched = True
                break
        if matched:
            continue
        for kw in LEFT_BLOCKS:
            if text.startswith(kw):
                _set_para_alignment(p, "left")
                _remove_first_line_indent(p)
                _set_bold(p, False)
                changes += 1
                matched = True
                break
        if matched:
            continue
        _set_para_alignment(p, "center")
        _remove_first_line_indent(p)
        changes += 1

    print(f"  отформатировано параграфов титульника: {changes}")
    return etree.tostring(
        root, xml_declaration=True, encoding="UTF-8", standalone=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("docx", nargs="?", default="VKR_Frolov_2026.docx")
    args = ap.parse_args()

    src = Path(args.docx)
    if not src.exists():
        print(f"ERROR: {src} не найден", file=sys.stderr)
        sys.exit(1)

    print(f"Правлю титульный лист в {src}:")

    tmp = src.with_suffix(".docx.tmp")
    shutil.copy2(src, tmp)
    with zipfile.ZipFile(tmp, "r") as zin:
        doc_xml = zin.read("word/document.xml")

    doc_xml_new = fix_document_xml(doc_xml)

    out_tmp = src.with_suffix(".docx.new")
    with zipfile.ZipFile(tmp, "r") as zin, zipfile.ZipFile(out_tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for name in zin.namelist():
            data = zin.read(name)
            if name == "word/document.xml":
                data = doc_xml_new
            zout.writestr(name, data)

    os.replace(out_tmp, src)
    os.remove(tmp)
    print("Готово.")


if __name__ == "__main__":
    main()
