"""Cohen's κ for human (manual_*) vs proxy-LLM (proxy_*) per attribute.

Drops rows where either annotator is empty. Computes per-attribute κ +
agreement rate + n.

Usage:
    python -m src.eval.manual_label_iaa \\
        --gold datasets/manual_label/pasta_gold_250.csv \\
        --proxy datasets/manual_label/pasta_gold_250_proxy.csv \\
        --out datasets/processed/pasta_gold_iaa.parquet
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.manual_label.schemas_loader import load_pasta_attrs


def _normalise(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def kappa_for_pair(a: pd.Series, b: pd.Series) -> float:
    aa = _normalise(a)
    bb = _normalise(b)
    mask = (aa != "") & (bb != "")
    if mask.sum() < 2:
        return float("nan")
    return float(cohen_kappa_score(aa[mask], bb[mask]))


def compute_kappa_table(
    gold: pd.DataFrame, proxy: pd.DataFrame, attrs: list[str]
) -> pd.DataFrame:
    merged = gold.merge(proxy, on="code", how="inner")
    rows = []
    for attr in attrs:
        g = merged[f"manual_{attr}"]
        p = merged[f"proxy_{attr}"]
        g_n = _normalise(g)
        p_n = _normalise(p)
        mask = (g_n != "") & (p_n != "")
        n = int(mask.sum())
        if n < 2:
            rows.append({"attribute": attr, "n": n, "kappa": float("nan"), "agreement": float("nan")})
            continue
        agreement = float((g_n[mask] == p_n[mask]).mean())
        kappa = float(cohen_kappa_score(g_n[mask], p_n[mask]))
        rows.append({"attribute": attr, "n": n, "kappa": kappa, "agreement": agreement})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--proxy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    gold = pd.read_csv(args.gold, dtype=str).fillna("")
    proxy = pd.read_csv(args.proxy, dtype=str).fillna("")
    attrs = list(load_pasta_attrs().keys())
    table = compute_kappa_table(gold, proxy, attrs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
