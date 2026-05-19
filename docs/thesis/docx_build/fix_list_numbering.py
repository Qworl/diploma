"""Привести нумерацию ordered-списков к локальной (рестарт с 1 в каждом списке).

Проблема: md_to_docx_py.py навешивает на все ordered-li один стиль ListNumber
без <w:numPr> и без сброса numId. Word воспринимает их как один сквозной
список, поэтому в §5.5 пункты получают номера вроде 25, 26 вместо 1, 2.

Решение: пройти параграфы по порядку; найти серии подряд идущих
ListNumber-параграфов; для каждой серии добавить явную текстовую нумерацию
«N. » в начало run-а и снять pStyle=ListNumber (заменить на BodyText), чтобы
Word не пытался автонумеровать.

Usage:
    python fix_list_numbering.py path/to/VKR.docx
"""
from __future__ import annotations

import os
import shutil
import sys
import zipfile
from pathlib import Path

from lxml import etree

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = f"{{{W_NS}}}"
NSMAP = {"w": W_NS}


def _has_list_number_style(p) -> bool:
    pPr = p.find(f"{W}pPr")
    if pPr is None:
        return False
    pStyle = pPr.find(f"{W}pStyle")
    if pStyle is None:
        return False
    return pStyle.get(f"{W}val") == "ListNumber"


def _set_style(p, style_id: str) -> None:
    pPr = p.find(f"{W}pPr")
    if pPr is None:
        pPr = etree.SubElement(p, f"{W}pPr")
        p.insert(0, pPr)
    pStyle = pPr.find(f"{W}pStyle")
    if pStyle is None:
        pStyle = etree.SubElement(pPr, f"{W}pStyle")
        pPr.insert(0, pStyle)
    pStyle.set(f"{W}val", style_id)


def _first_text_run(p):
    """Find first <w:r> with text, return (run, t_element)."""
    for r in p.findall(f"{W}r"):
        t = r.find(f"{W}t")
        if t is not None:
            return r, t
    return None, None


def _prepend_number(p, n: int) -> bool:
    """Вставить «N. » перед первым текстовым ранком."""
    r, t = _first_text_run(p)
    if t is None:
        return False
    existing = t.text or ""
    # xml:space="preserve", чтобы ведущий пробел не потерялся
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = f"{n}. {existing}"
    return True


def fix_document_xml(doc_xml: bytes) -> tuple[bytes, int, int]:
    root = etree.fromstring(doc_xml)
    body = root.find(f"{W}body")
    if body is None:
        return doc_xml, 0, 0

    # Идём строго по верхнеуровневым параграфам body
    paragraphs = body.findall(f"{W}p")
    series_count = 0
    item_count = 0
    cur_num = 0
    prev_was_list = False
    for p in paragraphs:
        if _has_list_number_style(p):
            if not prev_was_list:
                # начало нового списка
                cur_num = 1
                series_count += 1
            else:
                cur_num += 1
            if _prepend_number(p, cur_num):
                item_count += 1
            _set_style(p, "BodyText")
            prev_was_list = True
        else:
            prev_was_list = False

    return (
        etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True),
        series_count,
        item_count,
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_list_numbering.py path/to/VKR.docx", file=sys.stderr)
        sys.exit(1)

    src = Path(sys.argv[1])
    if not src.exists():
        print(f"ERROR: {src} не найден", file=sys.stderr)
        sys.exit(1)

    print(f"Локальная нумерация ordered-списков в {src}:")

    tmp = src.with_suffix(".docx.numfix.tmp")
    out_tmp = src.with_suffix(".docx.numfix.new")
    shutil.copy2(src, tmp)
    with zipfile.ZipFile(tmp, "r") as zin:
        doc_xml = zin.read("word/document.xml")

    new_xml, n_lists, n_items = fix_document_xml(doc_xml)

    with zipfile.ZipFile(tmp, "r") as zin, zipfile.ZipFile(out_tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for name in zin.namelist():
            data = zin.read(name)
            if name == "word/document.xml":
                data = new_xml
            zout.writestr(name, data)

    os.replace(out_tmp, src)
    os.remove(tmp)
    print(f"  обработано списков: {n_lists}, пунктов: {n_items}")
    print("Готово.")


if __name__ == "__main__":
    main()
