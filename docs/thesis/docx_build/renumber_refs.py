"""Перенумеровать ссылки на литературу по порядку первого упоминания в тексте.

ГОСТ 7.32-2018 допускает три способа упорядочивания списка источников:
по алфавиту, по порядку упоминания, тематически. Этот скрипт реализует
вариант «по порядку упоминания», который считается классическим:
номер источника в [квадратных скобках] совпадает с порядком его первого
появления в тексте работы.

Что делает:
  1. Сканирует .md файлы (главы) в установленном порядке, находит все [N] и [N, M, ...].
  2. Строит словарь старый_номер → новый_номер (по порядку первого появления).
  3. Перенумеровывает все ссылки в .md файлах глав.
  4. Перенумеровывает 06_references.md, сохраняя порядок согласно
     новой нумерации и оригинальный текст библиографических записей.

Запускать ИЗ docs/thesis/:
    python docx_build/renumber_refs.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

THESIS_DIR = Path(__file__).resolve().parent.parent

# Файлы в порядке появления в финальном документе
CHAPTERS = [
    "00_titul_referat.md",
    "00_introduction.md",
    "01_chapter1_analysis.md",
    "02_chapter2_theory.md",
    "03_chapter3_implementation.md",
    "04_chapter4_results.md",
    "05_conclusion.md",
]
REFS_FILE = "06_references.md"

# Паттерн ссылки: [N] или [N, M, K] — допускаются пробелы и запятые
CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def extract_citations_in_order(text: str) -> list[int]:
    """Вернуть список номеров источников в порядке их появления в тексте.
    Дубликаты разрешены (но первое появление определит новый номер)."""
    out = []
    for m in CITE_RE.finditer(text):
        nums = [int(n.strip()) for n in m.group(1).split(",")]
        out.extend(nums)
    return out


def build_mapping(all_texts: list[str]) -> dict[int, int]:
    """Построить old_num → new_num по порядку первого появления."""
    seen: dict[int, int] = {}
    next_num = 1
    for text in all_texts:
        for n in extract_citations_in_order(text):
            if n not in seen:
                seen[n] = next_num
                next_num += 1
    return seen


def renumber_text(text: str, mapping: dict[int, int]) -> str:
    """Заменить все [N, M, ...] в тексте согласно mapping."""
    def repl(m):
        nums = [int(n.strip()) for n in m.group(1).split(",")]
        new = [str(mapping.get(n, n)) for n in nums]
        return "[" + ", ".join(new) + "]"
    return CITE_RE.sub(repl, text)


def parse_references_file(text: str) -> tuple[str, dict[int, str], str]:
    """Распарсить 06_references.md.

    Возвращает (header_text, old_num → entry_text, footer_text).
    entry_text включает только тело записи (без «1. » префикса).
    """
    lines = text.split("\n")
    entries: dict[int, str] = {}
    header_lines: list[str] = []
    footer_lines: list[str] = []
    current_num: int | None = None
    current_buf: list[str] = []
    state = "header"  # header → entries → footer

    entry_re = re.compile(r"^(\d+)\.\s+(.*)$")

    def flush():
        nonlocal current_num, current_buf
        if current_num is not None:
            entries[current_num] = "\n".join(current_buf).rstrip()
            current_num = None
            current_buf = []

    for ln in lines:
        m = entry_re.match(ln)
        if m:
            if state == "header":
                state = "entries"
            flush()
            current_num = int(m.group(1))
            current_buf = [m.group(2)]
        elif state == "entries":
            current_buf.append(ln)
        elif state == "header":
            header_lines.append(ln)
        else:
            footer_lines.append(ln)
    flush()
    header = "\n".join(header_lines)
    footer = "\n".join(footer_lines)
    return header, entries, footer


def main():
    # 1. Собрать тексты глав
    texts: list[tuple[str, str]] = []
    for fname in CHAPTERS:
        p = THESIS_DIR / fname
        if not p.exists():
            print(f"WARNING: {p} не найден", file=sys.stderr)
            continue
        texts.append((fname, p.read_text(encoding="utf-8")))

    # 2. Построить mapping старый → новый
    mapping = build_mapping([t for _, t in texts])
    print(f"Найдено уникальных источников в тексте: {len(mapping)}")

    # 3. Прочитать references
    refs_path = THESIS_DIR / REFS_FILE
    refs_text = refs_path.read_text(encoding="utf-8")
    header, old_entries, footer = parse_references_file(refs_text)
    print(f"В 06_references.md найдено записей: {len(old_entries)}")

    # Источники в списке, но не упомянутые в тексте — добавим в конец
    cited_set = set(mapping.keys())
    uncited = [n for n in sorted(old_entries) if n not in cited_set]
    if uncited:
        print(f"  не упомянуты в тексте, оставлены в конце: {uncited}")
        for n in uncited:
            mapping[n] = max(mapping.values()) + 1

    # 4. Применить mapping к текстам глав
    for fname, text in texts:
        new_text = renumber_text(text, mapping)
        if new_text != text:
            (THESIS_DIR / fname).write_text(new_text, encoding="utf-8")
            print(f"  обновлён: {fname}")

    # 5. Пересобрать references в новом порядке
    new_lines = [header.rstrip()] if header.strip() else []
    new_lines.append("")
    # Создаём обратный mapping: новый → старый
    inv_mapping = {new: old for old, new in mapping.items()}
    for new_num in sorted(inv_mapping):
        old_num = inv_mapping[new_num]
        if old_num not in old_entries:
            print(f"WARNING: source [{old_num}] упомянут в тексте, но не в 06_references.md")
            continue
        entry = old_entries[old_num]
        new_lines.append(f"{new_num}. {entry}")
        new_lines.append("")
    if footer.strip():
        new_lines.append(footer.rstrip())

    new_refs = "\n".join(new_lines).rstrip() + "\n"
    refs_path.write_text(new_refs, encoding="utf-8")
    print(f"  обновлён: {REFS_FILE}")
    print("Готово.")


if __name__ == "__main__":
    main()
