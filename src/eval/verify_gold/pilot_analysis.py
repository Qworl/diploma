"""Pilot analysis: consensus rate на 200 продуктов (pasta + beverages), 3 strong LLMs.

Gate 1 of router rescue plan: consensus_rate (unanimous + majority_2of3) MUST be ≥ 70%
for each category; if < 70% → escalate (simplify class spaces or scope-cut). If ≥ 95% →
can use smaller A.3 sample (500/category instead of 1000).
"""
from __future__ import annotations
import pandas as pd
from src.common import PROCESSED_DIR
from src.eval.validation_sources import VALIDATION_SOURCE, get_tier

LLM_SUFFIXES = ["sonnet45_pilot", "gpt4o_pilot", "gemini25flash_pilot"]


def _normalize(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().lower()
    return None if s in ("", "none", "nan", "null") else s


def analyze(cat: str) -> pd.DataFrame:
    """Long format consensus result on silver_strong attrs for one category."""
    dfs = {}
    for s in LLM_SUFFIXES:
        p = f"{PROCESSED_DIR}/direct_llm_eval_{cat}_{s}.parquet"
        df = pd.read_parquet(p)[["code", "attr", "pred"]]
        dfs[s] = df.rename(columns={"pred": s})
    m = dfs[LLM_SUFFIXES[0]]
    for s in LLM_SUFFIXES[1:]:
        m = m.merge(dfs[s], on=["code", "attr"], how="outer")

    cat_base = cat.replace("_stratified", "")
    ss_attrs = {a for (c, a) in VALIDATION_SOURCE
                if c == cat_base and get_tier(c, a).value == "silver_strong"}
    m = m[m["attr"].isin(ss_attrs)].copy()

    def vote(row):
        votes = [_normalize(row[s]) for s in LLM_SUFFIXES]
        non_null = [v for v in votes if v is not None]
        if len(non_null) < 2:
            return "no_majority"
        if len(set(non_null)) == 1:
            return "unanimous"
        from collections import Counter
        top, n = Counter(non_null).most_common(1)[0]
        return "majority_2of3" if n >= 2 else "no_majority"

    m["vote_outcome"] = m.apply(vote, axis=1)
    return m


def main():
    rows = []
    per_attr_rows = []
    for cat in ("pasta_stratified", "beverages_stratified"):
        m = analyze(cat)
        if len(m) == 0:
            print(f"WARN: no silver_strong rows for {cat}")
            continue
        vc = m["vote_outcome"].value_counts(normalize=True)
        consensus = vc.get("unanimous", 0) + vc.get("majority_2of3", 0)
        rows.append({
            "category": cat,
            "n_pairs": len(m),
            "unanimous_pct": round(vc.get("unanimous", 0) * 100, 1),
            "majority_2of3_pct": round(vc.get("majority_2of3", 0) * 100, 1),
            "no_majority_pct": round(vc.get("no_majority", 0) * 100, 1),
            "consensus_rate_pct": round(consensus * 100, 1),
        })
        # Per-attr breakdown for diagnostics
        for attr in sorted(m["attr"].unique()):
            sub = m[m["attr"] == attr]
            vc_sub = sub["vote_outcome"].value_counts(normalize=True)
            consensus_sub = vc_sub.get("unanimous", 0) + vc_sub.get("majority_2of3", 0)
            per_attr_rows.append({
                "category": cat,
                "attr": attr,
                "n": len(sub),
                "unanimous_pct": round(vc_sub.get("unanimous", 0) * 100, 1),
                "majority_2of3_pct": round(vc_sub.get("majority_2of3", 0) * 100, 1),
                "no_majority_pct": round(vc_sub.get("no_majority", 0) * 100, 1),
                "consensus_rate_pct": round(consensus_sub * 100, 1),
            })
    summary = pd.DataFrame(rows)
    per_attr = pd.DataFrame(per_attr_rows)
    print("=== Per-category summary ===")
    print(summary.to_string(index=False))
    print()
    print("=== Per-attribute breakdown ===")
    print(per_attr.to_string(index=False))
    print()

    # Gate 1 verdict
    print("=== Gate 1 verdict ===")
    all_pass = True
    all_high = True
    for _, r in summary.iterrows():
        cat = r["category"]
        rate = r["consensus_rate_pct"]
        if rate < 70:
            print(f"  {cat}: {rate:.1f}% — FAIL (<70%)")
            all_pass = False
            all_high = False
        elif rate >= 95:
            print(f"  {cat}: {rate:.1f}% — PASS+ (≥95%, can shrink A.3 sample)")
        else:
            print(f"  {cat}: {rate:.1f}% — PASS (≥70%)")
            all_high = False

    if not all_pass:
        print("\n[Gate 1 FAIL] Consensus rate below 70% on at least one category.")
        print("→ STOP. Discuss with controller: simplify class spaces or scope-cut.")
    elif all_high:
        print("\n[Gate 1 PASS+] All categories ≥95% consensus.")
        print("→ Can shrink A.3 sample to 500/category (MVP path).")
    else:
        print("\n[Gate 1 PASS] Continue to A.3 full run on 1000/category.")

    summary.to_parquet(f"{PROCESSED_DIR}/pilot_consensus_rate.parquet", index=False)
    per_attr.to_parquet(f"{PROCESSED_DIR}/pilot_consensus_rate_per_attr.parquet", index=False)
    return summary, per_attr


if __name__ == "__main__":
    main()
