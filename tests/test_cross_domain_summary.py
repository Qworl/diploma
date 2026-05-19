"""Tests for src.eval.cross_domain_summary (Trek E §6.18 reporting)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.eval.cross_domain_summary import (
    _DOMAIN_GOLD,
    build_summary,
)


def _write_fake_cascade_json(path: Path, domain: str, *,
                              n_products: int, acc_aud: float,
                              acc_override: float, n_audited: int) -> None:
    payload = {
        "domain": domain,
        "n_gold_products": n_products,
        "cascade_config": f"regex_ml_bayes ({domain}_stratified models)",
        "metrics": {
            "all_audited": {"overall": {
                "n": n_audited, "n_covered": n_audited, "n_correct": int(n_audited * acc_aud),
                "coverage": 1.0, "acc_on_audited": acc_aud, "acc_on_covered": acc_aud,
            }},
            "override_or_manual_only": {"overall": {
                "n": 50, "n_covered": 50, "n_correct": int(50 * acc_override),
                "coverage": 1.0, "acc_on_audited": acc_override, "acc_on_covered": acc_override,
            }},
            "confirmed": {"overall": {
                "n": 100, "n_covered": 100, "n_correct": 95,
                "coverage": 1.0, "acc_on_audited": 0.95, "acc_on_covered": 0.95,
            }},
        },
    }
    path.write_text(json.dumps(payload))


def _write_fake_gold_csv(path: Path, attrs: list[str], *,
                          n_llm: int, n_blind: int, n_override_llm: int) -> None:
    fieldnames = ["code", "source"]
    for a in attrs:
        fieldnames += [f"silver_{a}", f"manual_{a}",
                       f"manual_{a}_status", f"manual_{a}_at",
                       f"manual_{a}_mode", f"manual_{a}_note"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Distribute n_llm cells across attrs and rows, marking some as override.
        cells_written = 0
        rid = 0
        while cells_written < n_llm:
            row = {k: "" for k in fieldnames}
            row["code"] = f"L{rid:04d}"
            rid += 1
            for a in attrs:
                if cells_written >= n_llm:
                    break
                status = "override" if cells_written < n_override_llm else "confirmed"
                row[f"silver_{a}"] = "x"
                row[f"manual_{a}"] = "y" if status == "override" else "x"
                row[f"manual_{a}_status"] = status
                row[f"manual_{a}_mode"] = "llm"
                cells_written += 1
            writer.writerow(row)
        # Blind cells
        cells_written = 0
        while cells_written < n_blind:
            row = {k: "" for k in fieldnames}
            row["code"] = f"B{rid:04d}"
            rid += 1
            for a in attrs:
                if cells_written >= n_blind:
                    break
                row[f"silver_{a}"] = "x"
                row[f"manual_{a}"] = "x"
                row[f"manual_{a}_status"] = "confirmed"
                row[f"manual_{a}_mode"] = "blind"
                cells_written += 1
            writer.writerow(row)


def test_build_summary_three_rows(tmp_path, monkeypatch):
    # Pasta uses real pasta gold path; copy it into a fake processed_dir
    # so we test the whole pipeline.
    processed = tmp_path / "processed"
    processed.mkdir()

    _write_fake_cascade_json(processed / "cascade_vs_audited_gold_pasta.json",
                              "pasta", n_products=239, acc_aud=0.86,
                              acc_override=0.55, n_audited=1841)
    _write_fake_cascade_json(processed / "cascade_vs_audited_gold_chocolate.json",
                              "chocolate", n_products=239, acc_aud=0.78,
                              acc_override=0.48, n_audited=1500)
    _write_fake_cascade_json(processed / "cascade_vs_audited_gold_cheeses.json",
                              "cheeses", n_products=239, acc_aud=0.72,
                              acc_override=0.40, n_audited=1400)

    # Redirect gold CSVs to temp fakes so we can control override_rate.
    fake_csv_pasta = tmp_path / "pasta_gold.csv"
    _write_fake_gold_csv(fake_csv_pasta,
                          ["grain_type", "pasta_shape"],
                          n_llm=100, n_blind=20, n_override_llm=20)
    fake_csv_choc = tmp_path / "chocolate_gold.csv"
    _write_fake_gold_csv(fake_csv_choc,
                          ["chocolate_type", "cocoa_percentage"],
                          n_llm=100, n_blind=20, n_override_llm=30)
    fake_csv_cheeses = tmp_path / "cheeses_gold.csv"
    _write_fake_gold_csv(fake_csv_cheeses,
                          ["milk_source", "texture"],
                          n_llm=100, n_blind=20, n_override_llm=15)
    monkeypatch.setitem(_DOMAIN_GOLD, "pasta", str(fake_csv_pasta))
    monkeypatch.setitem(_DOMAIN_GOLD, "chocolate", str(fake_csv_choc))
    monkeypatch.setitem(_DOMAIN_GOLD, "cheeses", str(fake_csv_cheeses))

    summary = build_summary(processed_dir=str(processed))
    assert len(summary["domains"]) == 3
    for r in summary["domains"]:
        assert r["status"] == "ok"

    by_domain = {r["domain"]: r for r in summary["domains"]}
    assert by_domain["pasta"]["accuracy_overall_audited"] == pytest.approx(0.86, abs=0.001)
    assert by_domain["chocolate"]["accuracy_overall_audited"] == pytest.approx(0.78, abs=0.001)
    assert by_domain["cheeses"]["accuracy_overall_audited"] == pytest.approx(0.72, abs=0.001)

    # Override rates
    assert by_domain["pasta"]["override_rate_llm"] == pytest.approx(0.20, abs=0.001)
    assert by_domain["chocolate"]["override_rate_llm"] == pytest.approx(0.30, abs=0.001)
    assert by_domain["cheeses"]["override_rate_llm"] == pytest.approx(0.15, abs=0.001)

    # Markdown table has the right shape
    md = summary["table_markdown"]
    assert "pasta" in md and "chocolate" in md and "cheeses" in md
    assert md.startswith("| Domain |")


def test_build_summary_missing_domain(tmp_path):
    # Only pasta cascade JSON exists — other domains should be reported as missing.
    processed = tmp_path / "processed"
    processed.mkdir()
    _write_fake_cascade_json(processed / "cascade_vs_audited_gold_pasta.json",
                              "pasta", n_products=239, acc_aud=0.86,
                              acc_override=0.55, n_audited=1841)
    summary = build_summary(processed_dir=str(processed))
    by_domain = {r["domain"]: r for r in summary["domains"]}
    assert by_domain["pasta"]["status"] == "ok"
    assert by_domain["chocolate"]["status"] == "missing"
    assert by_domain["cheeses"]["status"] == "missing"
