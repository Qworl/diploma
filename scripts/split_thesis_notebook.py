#!/usr/bin/env python3
"""Split notebooks/00_thesis_main.ipynb into per-section notebooks.

Each output notebook gets a verbatim copy of the setup cell (cell 2) so it
can be opened and run standalone. Cell execution counts are reset.

Usage:
    python scripts/split_thesis_notebook.py            # dry-run (lists actions)
    python scripts/split_thesis_notebook.py --write    # actually write files
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_NB = ROOT / "notebooks" / "00_thesis_main.ipynb"
OUT_DIR = ROOT / "notebooks"

SPLITS = [
    (
        "03_headline_and_conditions.ipynb",
        [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        "Headline + conditions of verification + single-prediction demo (§3.3.1)",
    ),
    (
        "04_cost_quality_and_layers.ipynb",
        [23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
        "Cost-quality matrix + per-layer contributions + ECE calibration (§3.3.2–3.3.3)",
    ),
    (
        "05_bayes_and_audit_retrain.ipynb",
        [33, 34, 35, 36, 37, 38, 39, 44, 45, 46, 47],
        "Bayes validator + Audit-and-retrain loop (§3.3.4 + §3.3.6)",
    ),
    (
        "06_routing_and_h1_negative.ipynb",
        [15, 16, 17, 18, 19, 20, 21, 22, 40, 41, 42, 43],
        "Layer 0 router + OOD + Plan C + learned-router H1 FAIL (§3.1.1 + §3.3.5)",
    ),
    (
        "07_robustness_and_chapter4.ipynb",
        [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58],
        "Robustness/language/kNN-distance/baseline + Chapter 4 summary (§3.3.7 + Ch.4)",
    ),
]

SETUP_CELL_IDX = 2


def load_source_nb() -> dict:
    with SRC_NB.open(encoding="utf-8") as f:
        return json.load(f)


def make_header_cell(title: str) -> dict:
    text = (
        f"# {title}\n\n"
        "Часть исполняемого приложения к главе 3 ВКР. Полный набор разделов "
        "см. в соседних ноутбуках `03_*` … `07_*`. Исходный монолит — "
        "`00_thesis_main.ipynb` (сохранён в истории git до коммита split).\n"
    )
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def make_setup_cell(orig_setup: dict) -> dict:
    cell = copy.deepcopy(orig_setup)
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def build_notebook(src_nb: dict, cell_indices: list[int], title: str) -> dict:
    setup_cell = make_setup_cell(src_nb["cells"][SETUP_CELL_IDX])
    header_cell = make_header_cell(title)

    body_cells = []
    for idx in cell_indices:
        cell = copy.deepcopy(src_nb["cells"][idx])
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        body_cells.append(cell)

    return {
        "cells": [header_cell, setup_cell, *body_cells],
        "metadata": src_nb.get("metadata", {}),
        "nbformat": src_nb.get("nbformat", 4),
        "nbformat_minor": src_nb.get("nbformat_minor", 5),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="actually write output notebooks (default: dry-run)")
    args = ap.parse_args()

    if not SRC_NB.exists():
        print(f"ERROR: source notebook missing: {SRC_NB}", file=sys.stderr)
        return 1

    src_nb = load_source_nb()
    total_cells = len(src_nb["cells"])
    covered: set[int] = {SETUP_CELL_IDX}

    for fname, cells, title in SPLITS:
        covered.update(cells)
        out_path = OUT_DIR / fname
        new_nb = build_notebook(src_nb, cells, title)
        n = len(new_nb["cells"])
        print(f"{'WRITE' if args.write else 'PLAN '}  {fname:55s} "
              f"src_cells={cells[0]:>2}..{cells[-1]:<2} "
              f"out_cells={n:>2}")
        if args.write:
            if out_path.exists():
                bak = out_path.with_suffix(".ipynb.bak")
                out_path.rename(bak)
                print(f"        backed up existing -> {bak.name}")
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(new_nb, f, ensure_ascii=False, indent=1)

    missing = sorted(set(range(total_cells)) - covered)
    if missing:
        print(f"\nWARNING: cells not assigned to any output: {missing}",
              file=sys.stderr)
        return 2

    print(f"\nAll {total_cells} cells accounted for "
          f"({len(SPLITS)} new notebooks + setup cell reused).")
    if not args.write:
        print("Dry-run only. Re-run with --write to materialise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
