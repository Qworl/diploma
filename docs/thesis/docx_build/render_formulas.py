"""Препроцессор: LaTeX-формулы в combined.md → PNG-картинки.

Сканирует входной .md файл, находит:
  - блочные формулы $$...$$ (могут быть многострочными)
  - инлайн-формулы $...$

Каждой уникальной формуле:
  1. вычисляется хеш SHA-1
  2. если PNG нет в кэше (_build_cache/formulas/HASH.png), скачивается с
     latex.codecogs.com/png.image (бесплатный публичный сервис)
  3. заменяется в исходном тексте на ссылку markdown-image:
       - блочная: ![](path.png) на отдельной строке
       - инлайн: ![](path.png)  (тот же абзац)

Usage:
    python render_formulas.py combined.md
"""
from __future__ import annotations

import hashlib
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / "_build_cache" / "formulas"
CODECOGS = "https://latex.codecogs.com/png.image?\\dpi{{200}}\\bg{{white}}{body}"
INLINE_PREFIX = "\\inline "  # inline mode for codecogs (без рамки для коротких формул)

BLOCK_RE = re.compile(r"\$\$\s*(.+?)\s*\$\$", re.DOTALL)
INLINE_RE = re.compile(r"(?<!\$)\$([^\$\n]+?)\$(?!\$)")


def render(latex: str, inline: bool) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # v5: ключ кэша включает версию пост-обработки DPI, чтобы старые кэшированные
    # PNG не использовались после смены логики.
    key = "v5" + ("i" if inline else "b") + latex.strip()
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    png = CACHE_DIR / f"{h}.png"
    if png.exists() and png.stat().st_size > 0:
        return png
    body = (INLINE_PREFIX if inline else "") + latex
    encoded = urllib.parse.quote(body, safe="")
    url = CODECOGS.format(body=encoded)
    req = urllib.request.Request(url, headers={"User-Agent": "thesis-build/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = resp.read()
            if not data:
                raise RuntimeError("пустой ответ")
            png.write_bytes(data)
            # CodeCogs рендерит при \dpi{200}; для inline переписываем DPI на 220
            # — md_to_docx_py.py делит пиксели на DPI и получает физический
            # размер в дюймах. Чуть повышенный DPI = чуть меньше формула,
            # без потери читаемости.
            if inline:
                _rewrite_dpi(png, 250)
            else:
                _rewrite_dpi(png, 200)
            return png
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(1.0 + attempt)
    raise RuntimeError("не удалось отрендерить формулу")


def _rewrite_dpi(png_path: Path, dpi: int) -> None:
    """Переписать DPI-метаданные PNG."""
    try:
        from PIL import Image
    except ImportError:
        return
    try:
        with Image.open(png_path) as img:
            img.load()
            img.save(png_path, dpi=(dpi, dpi))
    except Exception:
        pass


def process(md_path: Path) -> int:
    text = md_path.read_text(encoding="utf-8")
    n_block = 0
    n_inline = 0

    def block_sub(m: re.Match[str]) -> str:
        nonlocal n_block
        latex = m.group(1)
        png = render(latex, inline=False)
        rel = png  # абсолютный путь — скрипт md_to_docx_py.py принимает
        n_block += 1
        return f"\n\n![]({rel.absolute().as_posix()})\n\n"

    text = BLOCK_RE.sub(block_sub, text)

    def inline_sub(m: re.Match[str]) -> str:
        nonlocal n_inline
        latex = m.group(1)
        png = render(latex, inline=True)
        rel = png.relative_to(SCRIPT_DIR.parent)
        n_inline += 1
        return f"![]({rel.absolute().as_posix()})"

    text = INLINE_RE.sub(inline_sub, text)
    md_path.write_text(text, encoding="utf-8")
    return n_block + n_inline


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    md = Path(sys.argv[1])
    if not md.exists():
        print(f"Не найден: {md}", file=sys.stderr)
        return 1
    print(f"Рендеринг LaTeX-формул в {md.name}…")
    n = process(md)
    print(f"  обработано формул: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
