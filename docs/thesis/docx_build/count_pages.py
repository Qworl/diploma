"""Автоподсчёт страниц основной части ВКР.

Открывает собранный .docx в Microsoft Word через AppleScript,
находит страницы заголовков «Введение» и «Список использованных источников»,
подставляет число страниц основной части в плейсхолдер {{MAIN_PAGES}}
(или в шаблон «≈NN страниц основной части») в 00_titul_referat.md.

Usage:
    python count_pages.py path/to/VKR.docx
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
THESIS_DIR = SCRIPT_DIR.parent
REFERAT_MD = THESIS_DIR / "00_titul_referat.md"
APPLESCRIPT = SCRIPT_DIR / "_build_cache" / "count_pages.applescript"


def count_pages_via_word(docx_path: Path) -> tuple[int, int, int]:
    if not APPLESCRIPT.exists():
        raise FileNotFoundError(f"AppleScript не найден: {APPLESCRIPT}")
    result = subprocess.run(
        ["osascript", str(APPLESCRIPT), str(docx_path)],
        capture_output=True, text=True, timeout=180,
    )
    if result.returncode != 0:
        raise RuntimeError(f"AppleScript ошибка: {result.stderr.strip()}")
    parts = result.stdout.strip().split()
    if len(parts) != 3:
        raise RuntimeError(f"Неожиданный вывод AppleScript: {result.stdout!r}")
    intro, end, main = (int(x) for x in parts)
    return intro, end, main


def update_referat(main_pages: int) -> bool:
    text = REFERAT_MD.read_text(encoding="utf-8")
    new_text, n = re.subn(r"\{\{MAIN_PAGES\}\}", str(main_pages), text)
    if n == 0:
        new_text, n = re.subn(
            r"(?:≈)?\d+\s+страниц(?:ы)?\s+основной\s+части",
            f"{main_pages} страниц основной части",
            text,
        )
    if n == 0:
        print("WARN: плейсхолдер не найден в 00_titul_referat.md", file=sys.stderr)
        return False
    REFERAT_MD.write_text(new_text, encoding="utf-8")
    return True


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    docx_path = Path(sys.argv[1]).resolve()
    if not docx_path.exists():
        print(f"Файл не найден: {docx_path}", file=sys.stderr)
        return 1
    print(f"[1/2] Открытие {docx_path.name} в Microsoft Word, поиск границ основной части…")
    intro, end, main = count_pages_via_word(docx_path)
    print(f"     «Введение» — стр. {intro}; «Список использованных источников» — стр. {end}; основная часть: {main} стр.")
    print(f"[2/2] Обновление {REFERAT_MD.name} → ≈{main} страниц основной части")
    update_referat(main)
    print("Готово. Если был активен плейсхолдер {{MAIN_PAGES}}, пересоберите docx.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
