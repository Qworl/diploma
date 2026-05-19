"""Включить автоматическое повторение шапки таблицы при переносе на следующую страницу.

ГОСТ 7.32-2018 §6.5.7: «При переносе части таблицы на другую страницу
наименование помещают только над первой частью таблицы.» На практике
шапку таблицы (заголовки колонок) принято повторять на каждой странице.

Word делает это автоматически, если на первой строке стоит флаг
<w:tblHeader/> в <w:trPr>. Скрипт ставит этот флаг на первую строку
каждой таблицы документа.

Usage:
    python fix_tables_repeat_header.py path/to/VKR.docx
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


def _get_or_create(parent, tag):
    elem = parent.find(tag)
    if elem is not None:
        return elem
    elem = etree.SubElement(parent, tag)
    parent.remove(elem)
    parent.insert(0, elem)
    return elem


def fix(doc_xml: bytes) -> tuple[bytes, int]:
    root = etree.fromstring(doc_xml)
    body = root.find(f"{W}body")
    if body is None:
        return doc_xml, 0

    n = 0
    for tbl in body.iter(f"{W}tbl"):
        first_tr = tbl.find(f"{W}tr")
        if first_tr is None:
            continue
        trPr = _get_or_create(first_tr, f"{W}trPr")
        # Снять старый tblHeader (если стоял) и добавить заново
        for old in trPr.findall(f"{W}tblHeader"):
            trPr.remove(old)
        etree.SubElement(trPr, f"{W}tblHeader")
        n += 1

    return (
        etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True),
        n,
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_tables_repeat_header.py path/to/VKR.docx", file=sys.stderr)
        sys.exit(1)

    src = Path(sys.argv[1])
    if not src.exists():
        print(f"ERROR: {src} не найден", file=sys.stderr)
        sys.exit(1)

    print(f"Повторение шапки таблиц при переносе в {src}:")

    tmp = src.with_suffix(".docx.repeat.tmp")
    out_tmp = src.with_suffix(".docx.repeat.new")
    shutil.copy2(src, tmp)
    with zipfile.ZipFile(tmp, "r") as zin:
        doc_xml = zin.read("word/document.xml")

    new_xml, n = fix(doc_xml)

    with zipfile.ZipFile(tmp, "r") as zin, zipfile.ZipFile(out_tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for name in zin.namelist():
            data = zin.read(name)
            if name == "word/document.xml":
                data = new_xml
            zout.writestr(name, data)

    os.replace(out_tmp, src)
    os.remove(tmp)
    print(f"  таблиц обработано: {n}")
    print("Готово.")


if __name__ == "__main__":
    main()
