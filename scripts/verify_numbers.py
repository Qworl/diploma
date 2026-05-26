"""Verify thesis + slides against CANONICAL.md stop-list.

Usage:
    python scripts/verify_numbers.py                    # check all
    python scripts/verify_numbers.py --strict           # exit 1 on any match
    python scripts/verify_numbers.py --paths report/contents/0-abstract.tex slides/main.tex

Returns exit code:
    0 — clean (no stale strings found)
    1 — stale strings detected (with --strict) OR file read error
    2 — usage error

Stop-list source: docs/thesis/CANONICAL.md §11.
Whitelist source: known-OK contexts where a stop-list match is intentional
(e.g., §Ограничения mentioning "brand-disjoint обобщение не заявляется").
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Stop-list — устаревшие числа и формулировки из CANONICAL.md §11.
# Каждый паттерн — regex, ищется в .tex / .md / .ipynb.
STOP_PATTERNS: list[tuple[str, str]] = [
    # (pattern, why-stale)
    (r"92,8\s*\\?%", "92,8 % — устаревший headline; используй 91,1 % E2E или 94,8 % cascade-only"),
    (r"720\s*раз", "720× — устаревший cost factor; используй 333× combined или 14× architectural"),
    (r"\b4350\s*ячеек", "4350 cells — устарело; используй 3257 (LLM-consensus) или 615 (human gold)"),
    (r"3,3\s*\\?%\s+LLM", "3,3 % LLM — устарело; используй 7,1 %"),
    (r"\b9,0\s+п\.\s*п\.", "9,0 п. п. — устарело; используй 7,3 п. п. (E2E)"),
    (r"\b23,0\s+п\.\s*п\.", "23,0 п. п. — устарело; используй 21,3 п. п. (gpt-oss vs cascade)"),
    (r"24\s+поатрибутны[хй]\s+XGBoost", "24 XGBoost — устарело; используй 21 (8/6/7)"),
    (r"\bна\s+22\s+пар", "22 пары — устарело; 20 в headline-таблице, 21 в production-схеме"),
    (r"по\s+22\s+пар", "22 пары — устарело"),
    (r"\b4\s*НФР\b", "4 НФР — устарело; ТЗ содержит 5 НФР (НФ-1..НФ-5)"),
    (r"четыр[её]м\s+архитектурным", "4 архитектурных — устарело; ТЗ содержит 3 А (А-1..А-3)"),
    (r"четыр[её]х\s+архитектурных", "4 архитектурных — устарело; ТЗ содержит 3 А"),
    (r"\b96,7\s*\\?%", "96,7 % — устаревшее coverage; используй 96,0 %"),
    (r"без\s+пересечения\s+брендов", "brand-disjoint claim снят; реально code-disjoint (brand overlap 82-85%)"),
    (r"непересекающи(ми|х)ся\s+брендами?", "brand-disjoint в слайдах — реально code-disjoint"),
    (r"\bbrand-disjoint\b", "brand-disjoint — устарело; реально code-disjoint"),
]

# Whitelist: контексты, где упоминание stop-list ЛЕГИТИМНО (disclaimer-абзацы и т. п.).
# Pattern matches на той же строке — игнорируется.
WHITELIST_PATTERNS: list[str] = [
    r"brand-disjoint\s+(обобщение|claim)\s+(в\s+работе\s+)?не\s+заявля",
    r"без\s+пересечения\s+брендов\s+(невозможн|снят)",
    r"в\s+работе\s+не\s+заявляется",  # general disclaimer marker
    r"устаревш",  # explicit "deprecated" mention in stop-list itself
    r"стоп-лист",  # CANONICAL §11 itself
    r"\\textbf\{",  # safety: skip headline strings if in bold (manual flag)
]

# Файлы, которые проверяются по умолчанию.
DEFAULT_PATHS: list[str] = [
    "report/contents/*.tex",
    "slides/main.tex",
]

# Файлы-исключения (ones that intentionally contain stop-list strings).
SKIP_FILES: list[str] = [
    "docs/thesis/CANONICAL.md",  # contains stop-list itself
    "docs/thesis/data_methodology.md",  # source-of-truth allows historical refs
    "scripts/verify_numbers.py",  # this file
]


class Hit(NamedTuple):
    path: Path
    line_no: int
    line_text: str
    pattern: str
    why: str


def is_whitelisted(line: str) -> bool:
    """Return True if line matches any whitelist pattern (legitimate context)."""
    for wp in WHITELIST_PATTERNS:
        if re.search(wp, line, flags=re.IGNORECASE):
            return True
    return False


def check_file(path: Path) -> list[Hit]:
    hits: list[Hit] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as e:
        print(f"  WARN: cannot read {path}: {e}", file=sys.stderr)
        return hits

    for line_no, line in enumerate(text.splitlines(), start=1):
        # Skip lines obviously inside the stop-list metadata itself
        if is_whitelisted(line):
            continue
        for pattern, why in STOP_PATTERNS:
            if re.search(pattern, line):
                hits.append(Hit(path, line_no, line.strip()[:200], pattern, why))
    return hits


def expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pat in patterns:
        if "*" in pat:
            paths.extend(sorted(PROJECT_ROOT.glob(pat)))
        else:
            p = PROJECT_ROOT / pat
            if p.exists():
                paths.append(p)
    # Filter skip-list
    skip_resolved = {(PROJECT_ROOT / s).resolve() for s in SKIP_FILES}
    return [p for p in paths if p.resolve() not in skip_resolved]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify thesis + slides against CANONICAL.md stop-list.")
    parser.add_argument(
        "--paths",
        nargs="*",
        default=DEFAULT_PATHS,
        help="Globs (relative to repo root). Default: report/contents/*.tex slides/main.tex",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 on any hit (CI mode). Default: print and exit 0 unless --strict.",
    )
    args = parser.parse_args()

    paths = expand_paths(args.paths)
    if not paths:
        print(f"No files matched: {args.paths}", file=sys.stderr)
        return 2

    print(f"Checking {len(paths)} files against {len(STOP_PATTERNS)} stop patterns...")
    all_hits: list[Hit] = []
    for p in paths:
        hits = check_file(p)
        all_hits.extend(hits)

    if not all_hits:
        print(f"✓ clean — 0 stop-list matches across {len(paths)} files")
        return 0

    # Group by file
    by_file: dict[Path, list[Hit]] = {}
    for h in all_hits:
        by_file.setdefault(h.path, []).append(h)

    print(f"\n✗ found {len(all_hits)} stop-list matches in {len(by_file)} files:\n")
    for path, hits in by_file.items():
        rel = path.relative_to(PROJECT_ROOT)
        print(f"  {rel} ({len(hits)} hits)")
        for h in hits:
            print(f"    L{h.line_no}: {h.line_text[:150]}...")
            print(f"         → pattern '{h.pattern}': {h.why}")
        print()

    return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
