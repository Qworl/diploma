"""One-shot status snapshot for Phase 2 background work."""
import glob
import os
import subprocess

import pandas as pd


def safe_read(path):
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def count_procs(pattern):
    out = subprocess.run(["pgrep", "-fl", pattern], capture_output=True, text=True)
    return len([l for l in out.stdout.split("\n") if l.strip()])


def main():
    fetchers = count_procs("off_fetcher")
    llm = count_procs("direct_llm_v2 ")
    train = count_procs("train_hybrid")
    eval80 = count_procs("eval_v2_expanded") + count_procs("build_expanded_gold")

    print(f"alive: OFF_fetcher={fetchers}  LLM={llm}  train_hybrid={train}  honest_eval={eval80}")
    print()

    cache_count = len(glob.glob("datasets/manual_label/off_cache/*.json"))
    print(f"OFF cache: {cache_count} JSONs (need ~1950 + 717 existing = ~2667 target)")

    jsonl = "/tmp/openfoodfacts-products.jsonl.gz"
    if os.path.exists(jsonl):
        sz = os.path.getsize(jsonl)
        target = 12_125_584_520
        pct = 100 * sz / target
        print(f"JSONL download: {sz/1e9:.2f} GB / 12.13 GB  ({pct:.1f}%)")

    # OFF fetcher progress (both legacy off_fetch_* and refetch_*)
    for log in sorted(glob.glob("/tmp/off_fetch_*.log") + glob.glob("/tmp/refetch_*.log")):
        last = ""
        try:
            with open(log) as f:
                lines = f.readlines()
                if lines:
                    last = lines[-1].strip()[:90]
        except Exception:
            pass
        if last:
            print(f"  {os.path.basename(log):25s} {last}")

    print()
    print("=== gpt55_gold annotation chunks ===")
    total_n = 0
    total_cost = 0.0
    for p in sorted(glob.glob("datasets/processed/gpt55_gold/*.parquet")):
        df = safe_read(p)
        if df is None:
            print(f"  {os.path.basename(p):40s} (read err)")
            continue
        empty = (df["parsed_json"] == "{}").sum() if len(df) else 0
        cost = df["cost_usd"].sum() if len(df) else 0.0
        total_n += len(df)
        total_cost += cost
        print(f"  {os.path.basename(p):40s} n={len(df):>3d} empty={empty} ${cost:.2f}")
    print(f"  {'TOTAL':40s} n={total_n} ${total_cost:.2f} / target ~1950")

    # Annotation logs tail
    print()
    print("=== annotation logs ===")
    for log in sorted(glob.glob("/tmp/bscale_*.log")):
        last = ""
        try:
            with open(log) as f:
                lines = f.readlines()
                if lines:
                    last = lines[-1].strip()[:100]
        except Exception:
            pass
        if last:
            print(f"  {os.path.basename(log):25s} {last}")


if __name__ == "__main__":
    main()
