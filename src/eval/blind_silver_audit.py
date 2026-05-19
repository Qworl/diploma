"""Audit silver-extractor errors using blind-mode manual annotations.

Runs on `datasets/manual_label/pasta_gold_250.csv` after the pre-fill
pivot migration. Reports per-attribute `override_rate(blind)`,
`manual_only_rate(blind)`, and the top silver→manual correction
patterns. These numbers are the methodologically-clean evidence of
silver-extractor noise, separated from any pre-fill anchoring bias.

Run:
    python -m src.eval.blind_silver_audit \\
        --csv datasets/manual_label/pasta_gold_250.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from src.manual_label.schemas_loader import load_domain_attrs, load_pasta_attrs


AUDITED_STATUSES = {"confirmed", "override", "manual_only"}


def audit(csv_path: Path, attrs: list[str], *, modes: set[str] | None = None) -> dict:
    if modes is None:
        modes = {"blind"}
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    per_attr: dict[str, dict] = {}
    for attr in attrs:
        n_audited = n_override = n_manual_only = n_confirmed = 0
        override_pairs: Counter[tuple[str, str]] = Counter()
        manual_only_values: Counter[str] = Counter()
        per_source = defaultdict(lambda: {"n": 0, "override": 0, "manual_only": 0})
        for r in rows:
            mode = (r.get(f"manual_{attr}_mode") or "").strip()
            if mode not in modes:
                continue
            status = (r.get(f"manual_{attr}_status") or "").strip()
            if status not in AUDITED_STATUSES:
                continue
            silver = (r.get(f"silver_{attr}") or "").strip()
            manual = (r.get(f"manual_{attr}") or "").strip()
            source = (r.get("source") or "").strip() or "unknown"
            n_audited += 1
            per_source[source]["n"] += 1
            if status == "override":
                n_override += 1
                override_pairs[(silver, manual)] += 1
                per_source[source]["override"] += 1
            elif status == "manual_only":
                n_manual_only += 1
                manual_only_values[manual] += 1
                per_source[source]["manual_only"] += 1
            else:
                n_confirmed += 1
        per_attr[attr] = {
            "n_audited": n_audited,
            "n_confirmed": n_confirmed,
            "n_override": n_override,
            "n_manual_only": n_manual_only,
            "override_rate": (n_override / n_audited) if n_audited else None,
            "manual_only_rate": (n_manual_only / n_audited) if n_audited else None,
            "top_override_pairs": override_pairs.most_common(5),
            "top_manual_only_values": manual_only_values.most_common(5),
            "by_source": dict(per_source),
        }
    return {
        "rows": len(rows),
        "per_attr": per_attr,
    }


def _print(report: dict) -> None:
    print(f"Total rows in CSV: {report['rows']}")
    print()
    print(f"{'attribute':22s} {'n_aud':>6s} {'conf':>5s} {'ovr':>4s} {'m_only':>7s} {'override_rate':>14s} {'manual_only_rate':>18s}")
    print("-" * 90)
    for attr, st in report["per_attr"].items():
        ovr_pct = f"{st['override_rate'] * 100:>6.1f}%" if st["override_rate"] is not None else "  n/a "
        mo_pct = f"{st['manual_only_rate'] * 100:>6.1f}%" if st["manual_only_rate"] is not None else "  n/a "
        print(f"{attr:22s} {st['n_audited']:>6d} {st['n_confirmed']:>5d} {st['n_override']:>4d} {st['n_manual_only']:>7d} {ovr_pct:>14s} {mo_pct:>18s}")
    print()
    print("Top silver→manual corrections (override pairs):")
    for attr, st in report["per_attr"].items():
        if st["top_override_pairs"]:
            print(f"  [{attr}]")
            for (silver, manual), n in st["top_override_pairs"]:
                s = silver if silver else "(empty)"
                print(f"    {s!r:25s} → {manual!r:25s}  ×{n}")
    print()
    print("Top manual_only fillings (silver was empty):")
    for attr, st in report["per_attr"].items():
        if st["top_manual_only_values"]:
            print(f"  [{attr}]")
            for value, n in st["top_manual_only_values"]:
                print(f"    {value!r:25s}  ×{n}")
    print()
    print("Per-source breakdown (n / override / manual_only):")
    for attr, st in report["per_attr"].items():
        if st["by_source"]:
            print(f"  [{attr}]")
            for src, counts in sorted(st["by_source"].items()):
                print(f"    {src:25s} n={counts['n']:>3d} override={counts['override']:>2d} manual_only={counts['manual_only']:>2d}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--domain", default="pasta",
                   help="Schema domain for attribute list (pasta/chocolate/cheeses/...)")
    p.add_argument("--modes", default="blind",
                   help="Comma-separated modes to include (e.g. 'blind' or 'llm' or 'blind,llm')")
    args = p.parse_args()
    if args.domain == "pasta":
        attrs = list(load_pasta_attrs())
    else:
        attrs = list(load_domain_attrs(args.domain))
    modes = {m.strip() for m in args.modes.split(",") if m.strip()}
    report = audit(args.csv, attrs, modes=modes)
    _print(report)


if __name__ == "__main__":
    main()
