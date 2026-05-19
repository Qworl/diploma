"""Cross-domain audit summary for Trek E (§6.18).

Aggregates per-domain `cascade_vs_audited_gold_<domain>.json` artefacts
into a single 3-row table (pasta / chocolate / cheeses) and emits both
JSON and markdown. The summary captures the published headline ("82.1%
on consensus_gold") and the cleaner audit-cell headline per domain.

Inputs (one per domain; missing files are reported, never abort):
    datasets/processed/cascade_vs_audited_gold_pasta.json
    datasets/processed/cascade_vs_audited_gold_chocolate.json
    datasets/processed/cascade_vs_audited_gold_cheeses.json

For each domain, also reads the gold CSV (datasets/manual_label/<...>) to
derive `override_rate(mode=llm)` and `override_rate(mode=blind)` totals
— the same numbers `blind_silver_audit` prints, computed inline so the
summary is self-contained.

Run::

    python -m src.eval.cross_domain_summary \\
        --out-json datasets/processed/cross_domain_audit_summary.json \\
        --out-md  datasets/processed/cross_domain_audit_summary.md
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from src.common import PROCESSED_DIR
from src.manual_label.schemas_loader import load_domain_attrs

logger = logging.getLogger(__name__)

AUDITED_STATUSES = {"confirmed", "override", "manual_only"}

# (domain, default gold CSV path) — defaults consistent with sample_domain_gold.
_DOMAIN_GOLD = {
    "pasta": "datasets/manual_label/pasta_gold_250.csv",
    "chocolate": "datasets/manual_label/chocolate_gold_239.csv",
    "cheeses": "datasets/manual_label/cheeses_gold_239.csv",
}


def _override_rates(csv_path: Path, attrs: list[str]) -> dict:
    """Compute per-attr and overall override_rate split by mode."""
    if not csv_path.exists():
        return {"missing_csv": str(csv_path)}
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    per_mode: dict[str, dict] = {}
    for mode in ("blind", "llm"):
        n_aud = n_ovr = n_mo = n_conf = 0
        per_attr: dict[str, dict] = {}
        for attr in attrs:
            a_n = a_ovr = a_mo = a_conf = 0
            for r in rows:
                if (r.get(f"manual_{attr}_mode") or "").strip() != mode:
                    continue
                status = (r.get(f"manual_{attr}_status") or "").strip()
                if status not in AUDITED_STATUSES:
                    continue
                a_n += 1
                if status == "override":
                    a_ovr += 1
                elif status == "manual_only":
                    a_mo += 1
                else:
                    a_conf += 1
            per_attr[attr] = {
                "n_audited": a_n,
                "n_override": a_ovr,
                "n_manual_only": a_mo,
                "n_confirmed": a_conf,
                "override_rate": (a_ovr / a_n) if a_n else None,
            }
            n_aud += a_n
            n_ovr += a_ovr
            n_mo += a_mo
            n_conf += a_conf
        per_mode[mode] = {
            "n_audited": n_aud,
            "n_override": n_ovr,
            "n_manual_only": n_mo,
            "n_confirmed": n_conf,
            "override_rate": (n_ovr / n_aud) if n_aud else None,
            "per_attr": per_attr,
        }
    return per_mode


def _load_cascade_metrics(domain: str, processed_dir: str) -> dict | None:
    path = Path(processed_dir) / f"cascade_vs_audited_gold_{domain}.json"
    if not path.exists():
        return None
    with path.open() as f:
        payload = json.load(f)
    return payload


def _summary_row(domain: str, *, processed_dir: str) -> dict:
    cascade = _load_cascade_metrics(domain, processed_dir)
    if cascade is None:
        return {"domain": domain, "status": "missing"}

    attrs = list(load_domain_attrs(domain))
    gold_csv = Path(_DOMAIN_GOLD[domain])
    rates = _override_rates(gold_csv, attrs)

    metrics = cascade["metrics"]
    overall = metrics["all_audited"]["overall"]
    override_or_mo = metrics["override_or_manual_only"]["overall"]
    confirmed = metrics["confirmed"]["overall"]

    return {
        "domain": domain,
        "status": "ok",
        "n_gold_products": cascade.get("n_gold_products"),
        "cascade_config": cascade.get("cascade_config"),
        "accuracy_overall_audited": overall["acc_on_audited"],
        "accuracy_overall_covered": overall["acc_on_covered"],
        "coverage_overall": overall["coverage"],
        "n_overall": overall["n"],
        "accuracy_on_confirmed": confirmed["acc_on_covered"],
        "n_confirmed": confirmed["n"],
        "accuracy_on_override_manual_only": override_or_mo["acc_on_covered"],
        "n_override_manual_only": override_or_mo["n"],
        "override_rate_llm": rates.get("llm", {}).get("override_rate"),
        "n_llm_audited": rates.get("llm", {}).get("n_audited"),
        "override_rate_blind": rates.get("blind", {}).get("override_rate"),
        "n_blind_audited": rates.get("blind", {}).get("n_audited"),
    }


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%"


def _markdown_table(rows: list[dict]) -> str:
    header = (
        "| Domain | n products | Acc (audited) | Acc (override∪manual_only) | "
        "n override∪manual_only | override_rate(llm) | override_rate(blind) |"
    )
    sep = "|" + "|".join(["---"] * 7) + "|"
    out = [header, sep]
    for r in rows:
        if r["status"] != "ok":
            out.append(f"| {r['domain']} | _missing artefact_ | | | | | |")
            continue
        out.append(
            f"| {r['domain']} | {r['n_gold_products']} | "
            f"{_fmt_pct(r['accuracy_overall_audited'])} | "
            f"{_fmt_pct(r['accuracy_on_override_manual_only'])} | "
            f"{r['n_override_manual_only']} | "
            f"{_fmt_pct(r['override_rate_llm'])} (n={r['n_llm_audited']}) | "
            f"{_fmt_pct(r['override_rate_blind'])} (n={r['n_blind_audited']}) |"
        )
    return "\n".join(out)


def build_summary(processed_dir: str = PROCESSED_DIR) -> dict:
    rows = [_summary_row(d, processed_dir=processed_dir)
            for d in ("pasta", "chocolate", "cheeses")]
    return {
        "domains": rows,
        "table_markdown": _markdown_table(rows),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", default=PROCESSED_DIR)
    p.add_argument("--out-json", type=Path,
                   default=Path(PROCESSED_DIR) / "cross_domain_audit_summary.json")
    p.add_argument("--out-md", type=Path,
                   default=Path(PROCESSED_DIR) / "cross_domain_audit_summary.md")
    args = p.parse_args()

    summary = build_summary(args.processed_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(summary, f, indent=2)
    with args.out_md.open("w") as f:
        f.write("# Cross-domain audit summary (§6.18)\n\n")
        f.write(summary["table_markdown"])
        f.write("\n")
    print(summary["table_markdown"])
    print()
    print(f"Wrote {args.out_json} + {args.out_md}")


if __name__ == "__main__":
    main()
