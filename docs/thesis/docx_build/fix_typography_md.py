#!/usr/bin/env python3
"""Apply Russian typography to thesis .md files (out of code blocks).

Rules:
- ASCII pairs of " ... " are replaced with «...» (ёлочки).
- ' - ' (hyphen between spaces) is replaced with ' — ' (em-dash).
- Code fences (``` ... ```), inline code (`...`) and URLs are skipped.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

CODE_FENCE = re.compile(r'```.*?```', re.S)
INLINE_CODE = re.compile(r'`[^`]+`')
# Skip URLs (http://, https://, ftp://) entirely.
URL_RE = re.compile(r'(?:https?|ftp)://\S+')


def fix_quotes(text: str) -> str:
    """Replace pairs of ASCII " ... " with «...» (ёлочки)."""
    out, opening = [], True
    for ch in text:
        if ch == '"':
            out.append('«' if opening else '»')
            opening = not opening
        else:
            out.append(ch)
    return ''.join(out)


def fix_dashes(text: str) -> str:
    """Replace ' - ' (hyphen between spaces) with ' — '.

    Negative lookbehind for digit avoids touching numbers like '5 - 3'? Actually
    we want ' - ' between spaces regardless. Range '5 - 10' becoming '5 — 10'
    is acceptable as it's still in prose context.
    """
    return re.sub(r' - ', ' — ', text)


def process_chunk(text: str) -> str:
    """Process plain text (not code): apply quotes + dashes, skipping URLs."""
    # Skip URLs by splitting on them.
    parts, last = [], 0
    for m in URL_RE.finditer(text):
        parts.append(_apply_typography(text[last:m.start()]))
        parts.append(m.group())
        last = m.end()
    parts.append(_apply_typography(text[last:]))
    return ''.join(parts)


def _apply_typography(text: str) -> str:
    text = fix_quotes(text)
    text = fix_dashes(text)
    return text


def _skip_inline(text: str) -> str:
    parts, last = [], 0
    for m in INLINE_CODE.finditer(text):
        parts.append(process_chunk(text[last:m.start()]))
        parts.append(m.group())
        last = m.end()
    parts.append(process_chunk(text[last:]))
    return ''.join(parts)


def process(text: str) -> str:
    """Skip code fences first, then inline code, then apply typography."""
    parts = []
    last = 0
    for m in CODE_FENCE.finditer(text):
        parts.append(_skip_inline(text[last:m.start()]))
        parts.append(m.group())
        last = m.end()
    parts.append(_skip_inline(text[last:]))
    return ''.join(parts)


def main() -> int:
    for arg in sys.argv[1:]:
        p = Path(arg)
        src = p.read_text(encoding='utf-8')
        out = process(src)
        if out != src:
            p.write_text(out, encoding='utf-8')
            print(f'fixed: {p}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
