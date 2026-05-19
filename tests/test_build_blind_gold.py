"""Tests for src.eval.build_blind_gold."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.eval.build_blind_gold import build_gold_long


def _write_decisions(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_build_gold_long_format(tmp_path):
    decisions = {
        "code1": {"attr_a": {"value": "X"}, "attr_b": {"value": None}},
        "code2": {"attr_a": {"value": "Y"}, "attr_b": {"value": "Z"}},
    }
    _write_decisions(tmp_path / "pasta_decisions.json", decisions)

    df = build_gold_long(
        decisions_dir=tmp_path,
        categories=["pasta"],
        taxonomy_df=pd.DataFrame([
            {"category": "pasta", "attr": "attr_a", "signal_type": "tag_derived"},
            {"category": "pasta", "attr": "attr_b", "signal_type": "text_derived"},
        ]),
    )
    assert set(df.columns) == {
        "category", "code", "attr", "gold_value", "gold_is_null",
        "opus_reasoning", "signal_type",
    }
    # 2 codes × 2 attrs = 4 rows
    assert len(df) == 4
    null_row = df[(df["code"] == "code1") & (df["attr"] == "attr_b")].iloc[0]
    assert null_row["gold_is_null"] == True
    assert null_row["gold_value"] is None or pd.isna(null_row["gold_value"])
    val_row = df[(df["code"] == "code2") & (df["attr"] == "attr_a")].iloc[0]
    assert val_row["gold_value"] == "Y"
    assert val_row["gold_is_null"] == False


def test_build_gold_attaches_signal_type(tmp_path):
    _write_decisions(tmp_path / "pasta_decisions.json", {
        "c1": {"attr_a": {"value": "X"}},
    })
    df = build_gold_long(
        decisions_dir=tmp_path,
        categories=["pasta"],
        taxonomy_df=pd.DataFrame([
            {"category": "pasta", "attr": "attr_a", "signal_type": "nutri_derived"},
        ]),
    )
    assert df["signal_type"].iloc[0] == "nutri_derived"


def test_build_gold_handles_multiple_categories(tmp_path):
    _write_decisions(tmp_path / "pasta_decisions.json", {
        "p1": {"a": {"value": "X"}},
    })
    _write_decisions(tmp_path / "chocolate_decisions.json", {
        "c1": {"b": {"value": "Y"}},
    })
    df = build_gold_long(
        decisions_dir=tmp_path,
        categories=["pasta", "chocolate"],
        taxonomy_df=pd.DataFrame([
            {"category": "pasta", "attr": "a", "signal_type": "tag_derived"},
            {"category": "chocolate", "attr": "b", "signal_type": "tag_derived"},
        ]),
    )
    assert set(df["category"].unique()) == {"pasta", "chocolate"}
