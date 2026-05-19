"""Привести оформление таблиц к требованиям ГОСТ 7.32-2018.

1. Подписи таблиц («Таблица X.Y — Название», «Рисунок X.Y — Название»)
   — слева, без красной строки, без выравнивания по ширине.
2. Текст в ячейках таблицы — без красной строки.

Использует lxml — сохраняет все namespace-декларации, объявленные в корне
документа (иначе Word ругается «unreadable content»).

Usage:
    python fix_tables_gost.py [docx_path]
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

from lxml import etree

NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W_NS = f"{{{NS_W}}}"

# Паттерны подписей, требующих оформления «слева без красной строки»
CAPTION_PATTERNS = [
    re.compile(r"^\s*(Таблица|Рисунок|Рис\.?)\s+\d+", re.IGNORECASE),
    # Markdown-табличные подписи иногда идут как параграф с **жирным** — учитываем
    re.compile(r"^\s*\*\*Таблица\s+\d", re.IGNORECASE),
]


def _para_text(p_elem) -> str:
    return "".join(t.text or "" for t in p_elem.iter(f"{W_NS}t"))


def _get_or_create_first(parent, tag: str):
    existing = parent.find(tag)
    if existing is not None:
        return existing
    new_el = etree.SubElement(parent, tag)
    parent.remove(new_el)
    parent.insert(0, new_el)
    return new_el


def _format_caption(p_elem):
    """Слева, без отступа красной строки, без justify."""
    pPr = _get_or_create_first(p_elem, f"{W_NS}pPr")
    # Удалить старые jc и ind
    for existing in pPr.findall(f"{W_NS}jc"):
        pPr.remove(existing)
    for existing in pPr.findall(f"{W_NS}ind"):
        pPr.remove(existing)
    jc = etree.SubElement(pPr, f"{W_NS}jc")
    jc.set(f"{W_NS}val", "left")
    ind = etree.SubElement(pPr, f"{W_NS}ind")
    ind.set(f"{W_NS}firstLine", "0")


def _format_table_cell_paragraph(p_elem):
    """Параграф внутри <w:tc>: без красной строки (но сохранить justify, если есть)."""
    pPr = _get_or_create_first(p_elem, f"{W_NS}pPr")
    for existing in pPr.findall(f"{W_NS}ind"):
        pPr.remove(existing)
    ind = etree.SubElement(pPr, f"{W_NS}ind")
    ind.set(f"{W_NS}firstLine", "0")


def fix_document_xml(doc_xml: bytes) -> tuple[bytes, int, int]:
    parser = etree.XMLParser(remove_blank_text=False)
    root = etree.fromstring(doc_xml, parser)
    body = root.find(f"{W_NS}body")
    if body is None:
        print("ERROR: <w:body> не найден")
        return doc_xml, 0, 0

    n_captions = 0
    n_cells = 0

    # 1. Подписи таблиц и рисунков (параграфы верхнего уровня в body)
    for p in body.findall(f"{W_NS}p"):
        text = _para_text(p).strip()
        if not text:
            continue
        for pat in CAPTION_PATTERNS:
            if pat.match(text):
                _format_caption(p)
                n_captions += 1
                break

    # 2. Параграфы внутри ячеек таблиц
    for tbl in body.iter(f"{W_NS}tbl"):
        for tc in tbl.iter(f"{W_NS}tc"):
            for p in tc.findall(f"{W_NS}p"):
                _format_table_cell_paragraph(p)
                n_cells += 1

    print(f"  подписей таблиц/рисунков: {n_captions}")
    print(f"  параграфов в ячейках таблиц: {n_cells}")
    return etree.tostring(
        root, xml_declaration=True, encoding="UTF-8", standalone=True,
    ), n_captions, n_cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("docx", nargs="?", default="VKR_Frolov_2026.docx")
    args = ap.parse_args()

    src = Path(args.docx)
    if not src.exists():
        print(f"ERROR: {src} не найден", file=sys.stderr)
        sys.exit(1)

    print(f"Правлю оформление таблиц в {src}:")

    tmp = src.with_suffix(".docx.tmp")
    shutil.copy2(src, tmp)
    with zipfile.ZipFile(tmp, "r") as zin:
        doc_xml = zin.read("word/document.xml")

    doc_xml_new, _, _ = fix_document_xml(doc_xml)

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
