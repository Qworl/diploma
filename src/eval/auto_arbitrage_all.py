"""
Generate arbitrage CSV + apply rule-based classifier for ALL silver_strong attrs.

Объединяет:
- build_arbitrage_csv.py (CSV с silver + 3 LLM votes + status)
- coverage_extension.CLASSIFIERS (rule-based per-attr classifier)

Output per (cat, attr):
    datasets/manual_label/arbitrage_{cat}_{attr}.csv
        columns: code, product_name, brands, generic_name, ingredients_text,
                 silver, gpt4omini, gptoss, llama3b, status, enum_choices,
                 auto_arbitrage, auto_reason, your_arbitrage, note

Usage:
    python -m src.eval.auto_arbitrage_all              # все 8 silver_strong attrs
    python -m src.eval.auto_arbitrage_all --pairs cheeses:milk_source
"""
from __future__ import annotations

import argparse
import logging
import os

import pandas as pd

from src.common import setup_logging
from src.eval.build_arbitrage_csv import build
from src.eval.coverage_extension import CLASSIFIERS
from src.eval.validation_sources import VALIDATION_SOURCE, get_tier

logger = logging.getLogger(__name__)


def apply_classifier_to_csv(cat: str, attr: str, csv_path: str) -> int:
    """Apply rule-based classifier, fill auto_arbitrage + auto_reason.

    Preserve user manual edits (your_arbitrage != auto_arbitrage prev).
    Pre-fill your_arbitrage из auto, если пусто.
    """
    if (cat, attr) not in CLASSIFIERS:
        logger.warning("No classifier registered for %s/%s", cat, attr)
        return 0
    cls_fn = CLASSIFIERS[(cat, attr)]

    df = pd.read_csv(csv_path, dtype=str)
    for col in ("your_arbitrage", "note", "auto_arbitrage", "auto_reason"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    # Identify true manual edits (your_arbitrage != prev_auto, not blank)
    is_manual = (df["your_arbitrage"].str.strip() != "") & \
                 (df["your_arbitrage"] != df["auto_arbitrage"])

    n_classified = 0
    for idx, row in df.iterrows():
        try:
            lab, reason = cls_fn(row.to_dict())
        except Exception as e:
            lab, reason = None, f"err:{e}"
        df.at[idx, "auto_arbitrage"] = lab or ""
        df.at[idx, "auto_reason"] = reason
        if lab and not is_manual.iloc[idx]:
            df.at[idx, "your_arbitrage"] = lab
        if lab:
            n_classified += 1

    df.to_csv(csv_path, index=False, encoding="utf-8")
    return n_classified


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", nargs="*",
                   help="Список (cat:attr) пар. По умолчанию — все silver_strong.")
    p.add_argument("--force-rebuild", action="store_true",
                   help="Перестроить CSV из silver_standard с нуля (новые колонки и т.п.).")
    args = p.parse_args()

    if args.pairs:
        pairs = [tuple(p.split(":")) for p in args.pairs]
    else:
        pairs = [(c, a) for (c, a) in VALIDATION_SOURCE
                 if get_tier(c, a).value == "silver_strong"]

    print(f"{'category':<12} {'attr':<22} {'CSV_rows':>10} {'auto_filled':>12}")
    print("-" * 60)
    for cat, attr in pairs:
        out_dir = "datasets/manual_label"
        os.makedirs(out_dir, exist_ok=True)
        csv_path = f"{out_dir}/arbitrage_{cat}_{attr}.csv"

        if not os.path.exists(csv_path) or args.force_rebuild:
            # Preserve user manual edits before rebuild
            preserved = []
            if os.path.exists(csv_path):
                old = pd.read_csv(csv_path, dtype=str).fillna("")
                if "auto_arbitrage" in old.columns and "your_arbitrage" in old.columns:
                    mn = old[(old["your_arbitrage"].str.strip() != "") &
                              (old["your_arbitrage"] != old["auto_arbitrage"])]
                    preserved = mn[["code", "your_arbitrage", "note"]].to_dict("records")
                    if preserved:
                        logger.info("[%s/%s] preserving %d manual edits", cat, attr, len(preserved))
            df = build(cat, attr)
            df["auto_arbitrage"] = ""
            df["auto_reason"] = ""
            if "your_arbitrage" not in df.columns:
                df["your_arbitrage"] = ""
            if "note" not in df.columns:
                df["note"] = ""
            # Restore manual edits
            for e in preserved:
                mask = df["code"].astype(str) == str(e["code"])
                if mask.any():
                    df.loc[mask, "your_arbitrage"] = e["your_arbitrage"]
                    df.loc[mask, "note"] = e.get("note", "") or ""
            df.to_csv(csv_path, index=False, encoding="utf-8")

        n = apply_classifier_to_csv(cat, attr, csv_path)
        total = len(pd.read_csv(csv_path))
        print(f"{cat:<12} {attr:<22} {total:>10d} {n:>12d}")


if __name__ == "__main__":
    main()
