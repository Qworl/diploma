"""Migrate manual-label CSVs to the pre-fill + override workflow.

Adds manual_<attr>_mode columns, re-derives statuses from existing data,
pre-fills empty cells where silver is set, tags origin (blind for human-
typed cells, prefill for auto-filled cells). Idempotent.

Run on production CSVs from the repo root:

    python -m src.manual_label.migrate_to_prefill \\
        --csv datasets/manual_label/pasta_gold_250.csv
"""
from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

from src.manual_label.schemas_loader import load_pasta_attrs
from src.manual_label.status import derive_status


def _insert_mode_columns(fieldnames: list[str], attrs: list[str]) -> list[str]:
    """Insert manual_<attr>_mode immediately after manual_<attr>_at."""
    out: list[str] = []
    for col in fieldnames:
        out.append(col)
        if col.startswith("manual_") and col.endswith("_at"):
            attr = col[len("manual_"):-len("_at")]
            if attr in attrs and f"manual_{attr}_mode" not in fieldnames:
                out.append(f"manual_{attr}_mode")
    return out


def migrate_csv(path: Path, *, attrs: list[str] | None = None) -> dict:
    """Migrate one CSV in place. Returns summary dict."""
    if attrs is None:
        attrs = list(load_pasta_attrs())

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    new_fieldnames = _insert_mode_columns(fieldnames, attrs)
    attrs_with_at = {
        col[len("manual_"):-len("_at")]
        for col in fieldnames
        if col.startswith("manual_") and col.endswith("_at")
    }
    migratable_attrs = [a for a in attrs if a in attrs_with_at]
    for row in rows:
        for attr in migratable_attrs:
            row.setdefault(f"manual_{attr}_mode", "")

    summary: Counter[tuple[str, str]] = Counter()
    for row in rows:
        for attr in migratable_attrs:
            status = (row.get(f"manual_{attr}_status") or "empty").strip() or "empty"
            silver = (row.get(f"silver_{attr}") or "").strip()
            manual = (row.get(f"manual_{attr}") or "").strip()
            mode = (row.get(f"manual_{attr}_mode") or "").strip()

            if status == "confident":
                new_status = derive_status(silver, manual, "confident")
                row[f"manual_{attr}_status"] = new_status
                if mode == "":
                    row[f"manual_{attr}_mode"] = "blind"
            elif status == "unsure":
                if mode == "":
                    row[f"manual_{attr}_mode"] = "blind"
            elif status == "empty" and manual == "" and silver != "":
                row[f"manual_{attr}"] = silver
                row[f"manual_{attr}_status"] = "auto"
                row[f"manual_{attr}_mode"] = "prefill"
                # _at intentionally left untouched: pre-fill is not a human action.
            elif status in {"auto", "confirmed", "override", "manual_only"}:
                # Already migrated; idempotent — no change.
                pass
            else:
                # status==empty and (manual nonempty OR silver empty): no migration.
                pass

            final_status = (row.get(f"manual_{attr}_status") or "empty").strip() or "empty"
            final_mode = (row.get(f"manual_{attr}_mode") or "").strip()
            summary[(final_status, final_mode)] += 1

    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    import os
    os.replace(tmp_path, path)

    return {
        "rows": len(rows),
        "attrs": len(attrs),
        "status_mode_crosstab": dict(summary),
        "backup": str(backup),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, type=Path)
    args = p.parse_args()
    res = migrate_csv(args.csv)
    print(f"Migrated {res['rows']} rows × {res['attrs']} attrs")
    print(f"Backup: {res['backup']}")
    print("Status × mode crosstab:")
    for (st, mo), n in sorted(res["status_mode_crosstab"].items()):
        print(f"  {st:14s} × {mo or '(none)':8s}: {n}")
    print("OK to resume editing.")


if __name__ == "__main__":
    main()
