"""Build long-format v2 OFF-grounded blind gold parquet.

Reads per-category Opus decisions JSON files written by
src.manual_label.opus_off_grounded_audit and merges them with the attribute
signal taxonomy.

Output columns:
  category, code, attr, gold_value (str|None), gold_is_null (bool),
  opus_reasoning (str|None), signal_type (str)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def build_gold_long(
    decisions_dir: Path,
    categories: list[str],
    taxonomy_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the long-format v2 gold dataframe.

    decisions_dir: directory containing {category}_decisions.json files.
    categories: list of category names to load.
    taxonomy_df: must have columns [category, attr, signal_type].
    """
    decisions_dir = Path(decisions_dir)
    tax_lookup = taxonomy_df.set_index(["category", "attr"])["signal_type"].to_dict()

    rows = []
    for cat in categories:
        path = decisions_dir / f"{cat}_decisions.json"
        if not path.exists():
            logger.warning("Missing decisions file: %s", path)
            continue
        with path.open(encoding="utf-8") as f:
            decisions = json.load(f)
        for code, attrs in decisions.items():
            for attr, cell in attrs.items():
                value = cell.get("value") if isinstance(cell, dict) else None
                reasoning = cell.get("reasoning") if isinstance(cell, dict) else None
                rows.append({
                    "category": cat,
                    "code": str(code),
                    "attr": attr,
                    "gold_value": value,
                    "gold_is_null": value is None,
                    "opus_reasoning": reasoning,
                    "signal_type": tax_lookup.get((cat, attr), "text_derived"),
                })

    return pd.DataFrame(rows)


def main(
    decisions_dir: str = "datasets/manual_label/opus_batches/blind_v2",
    out_path: str = "datasets/processed/consensus_gold_v2_off_grounded.parquet",
    taxonomy_path: str = "datasets/processed/attribute_signal_taxonomy.parquet",
    categories: tuple[str, ...] = ("pasta", "chocolate", "cheeses"),
) -> None:
    """CLI: build and save v2 gold parquet."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    taxonomy = pd.read_parquet(taxonomy_path)
    df = build_gold_long(
        decisions_dir=Path(decisions_dir),
        categories=list(categories),
        taxonomy_df=taxonomy,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logging.info("Wrote %s rows to %s", len(df), out_path)
    logging.info("Per-category counts:\n%s", df["category"].value_counts().to_string())
    logging.info("Null cells: %d / %d (%.1f%%)",
                 df["gold_is_null"].sum(), len(df),
                 100.0 * df["gold_is_null"].mean())


if __name__ == "__main__":
    main()
