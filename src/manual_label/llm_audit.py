"""Apply LLM (Opus) audit decisions to a manual-label gold CSV.

Reads a JSON file of decisions produced by an LLM (e.g. Opus 4.6) and
writes them into the gold CSV, setting `mode=llm` for all touched cells.
Human-audited cells (mode=blind or status already confirmed/override/
manual_only/unsure with a mode set) are protected from overwrite.

Input JSON shape
----------------
{
  "<product_code>": {
    "<attr>": {
      "value": "<string or null>",
      "status_hint": null | "unsure",
      "reasoning": "<optional free text>"
    },
    ...
  },
  ...
}

If `value` is null  → cell is skipped (LLM couldn't decide).
If `status_hint == "unsure"` → status forced to `unsure` regardless of value.
If value is not in schema's allowed values → cell is skipped with a warning.

Run from repo root::

    python -m src.manual_label.llm_audit \\
        --csv datasets/manual_label/pasta_gold_250.csv \\
        --decisions /path/to/opus_decisions.json
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import os
import shutil
from collections import Counter
from pathlib import Path

from src.manual_label.schemas_loader import load_domain_attrs, load_pasta_attrs
from src.manual_label.status import derive_status

logger = logging.getLogger(__name__)

# Statuses that indicate the cell was already reviewed by a human and must
# not be overwritten by the LLM pipeline.
_HUMAN_STATUSES = {"confirmed", "override", "manual_only", "unsure", "confident"}
# Statuses that are eligible for LLM fill-in.
_FILLABLE_STATUSES = {"auto", "empty", ""}


def _allowed_values(attrs: dict[str, dict], attr: str) -> set[str] | None:
    """Return the set of valid non-null values for `attr`, or None if unrestricted."""
    spec = attrs.get(attr)
    if spec is None:
        return None
    vals = spec.get("values")
    if not vals:
        return None
    allowed = set(vals)
    if spec.get("nullable"):
        allowed.add("")
    return allowed


def apply_llm_decisions(
    csv_path: Path,
    decisions_path: Path,
    *,
    attrs: list[str] | None = None,
    domain: str = "pasta",
) -> dict:
    """Apply Opus decisions to the gold CSV. Returns summary dict.

    Only touches cells where current status is `auto` or `empty`. Never
    overwrites human-audited cells (mode in {blind} or status in
    {confirmed, override, manual_only, unsure} with mode set).

    `domain` selects the schema used for value validation; defaults to
    "pasta" for backward compatibility.
    """
    domain_attrs = load_domain_attrs(domain)
    if attrs is None:
        attrs = list(domain_attrs)

    with decisions_path.open(encoding="utf-8") as f:
        decisions: dict[str, dict] = json.load(f)

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    # Build a code → row-index lookup for O(1) access.
    code_index: dict[str, int] = {
        r.get("code", "").strip(): i for i, r in enumerate(rows)
    }

    now_iso = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")

    summary: Counter[tuple[str, str]] = Counter()

    for code, attr_decisions in decisions.items():
        row_idx = code_index.get(str(code).strip())
        if row_idx is None:
            logger.warning("Code %s not found in CSV — skipping.", code)
            summary[("not_found", "")] += 1
            continue

        row = rows[row_idx]

        for attr, decision in attr_decisions.items():
            if attr not in attrs:
                continue

            # --- Protection check (short-circuit) ---
            cur_status = (row.get(f"manual_{attr}_status") or "").strip()
            cur_mode = (row.get(f"manual_{attr}_mode") or "").strip()

            # A cell is human-protected if:
            #   1. status is a "done" status AND mode is set (was human-touched), OR
            #   2. mode == "blind" regardless of status (human explicitly typed it).
            is_human = (
                cur_mode == "blind"
                or (cur_status in _HUMAN_STATUSES and cur_mode != "")
            )
            if is_human:
                summary[("human_protected", attr)] += 1
                logger.debug(
                    "Skipping %s/%s: human-protected (status=%s, mode=%s)",
                    code, attr, cur_status, cur_mode,
                )
                continue

            # --- Read decision fields ---
            value = decision.get("value")
            status_hint = (decision.get("status_hint") or "").strip()

            # Null value → LLM couldn't decide
            if value is None:
                summary[("llm_skipped", attr)] += 1
                continue

            value = str(value).strip()

            # --- Schema validation ---
            allowed = _allowed_values(domain_attrs, attr)
            if allowed is not None and value not in allowed:
                logger.warning(
                    "Invalid value %r for attr %r (code=%s) — allowed: %s. Skipping.",
                    value, attr, code, sorted(allowed),
                )
                summary[("invalid_value", attr)] += 1
                continue

            # --- Derive new status ---
            silver = (row.get(f"silver_{attr}") or "").strip()
            if status_hint == "unsure":
                new_status = "unsure"
            else:
                new_status = derive_status(silver, value, cur_status)

            # --- Write cell ---
            col = f"manual_{attr}"
            st_col = f"manual_{attr}_status"
            at_col = f"manual_{attr}_at"
            mode_col = f"manual_{attr}_mode"

            if col in fieldnames:
                row[col] = value
            if st_col in fieldnames:
                row[st_col] = new_status
            if at_col in fieldnames:
                row[at_col] = now_iso
            if mode_col in fieldnames:
                row[mode_col] = "llm"

            summary[(new_status, "llm")] += 1

    # Atomic write: backup → .tmp → os.replace
    backup = csv_path.with_suffix(csv_path.suffix + ".bak")
    shutil.copy2(csv_path, backup)
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, csv_path)

    return {
        "rows_in_csv": len(rows),
        "codes_in_json": len(decisions),
        "backup": str(backup),
        "summary": dict(summary),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description="Apply Opus LLM audit decisions to a gold CSV."
    )
    p.add_argument("--csv", required=True, type=Path,
                   help="Path to domain gold CSV (e.g. chocolate_gold_239.csv).")
    p.add_argument("--decisions", required=True, type=Path,
                   help="Path to JSON file with Opus decisions.")
    p.add_argument("--domain", default="pasta",
                   help="Schema domain for value validation "
                        "(pasta/chocolate/cheeses/beverages/cereals/cosmetics).")
    p.add_argument("--attrs", nargs="*",
                   help="Restrict to specific attributes (default: all domain attrs).")
    args = p.parse_args()

    result = apply_llm_decisions(
        args.csv,
        args.decisions,
        attrs=args.attrs or None,
        domain=args.domain,
    )

    print(f"Rows in CSV: {result['rows_in_csv']}")
    print(f"Codes in decisions JSON: {result['codes_in_json']}")
    print(f"Backup written to: {result['backup']}")
    print()
    print("Summary (status × mode / counter):")

    # Separate the special counters from the (status, mode) update counters.
    special_keys = {"human_protected", "llm_skipped", "invalid_value", "not_found"}
    updates: Counter = Counter()
    specials: Counter = Counter()
    for (k, v), n in result["summary"].items():
        if k in special_keys:
            specials[(k, v)] += n
        else:
            updates[(k, v)] += n

    if updates:
        print("  Cells written:")
        for (st, mo), n in sorted(updates.items()):
            print(f"    {st:14s} × {mo:8s}: {n}")
    if specials:
        print("  Cells skipped:")
        for (reason, attr), n in sorted(specials.items()):
            print(f"    {reason:18s} ({attr or 'all'}): {n}")

    total_written = sum(updates.values())
    total_skipped = sum(specials.values())
    print()
    print(f"Total written: {total_written}, total skipped: {total_skipped}")


if __name__ == "__main__":
    main()
