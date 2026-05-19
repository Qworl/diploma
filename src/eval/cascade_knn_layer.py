"""k-NN retrieval layer: top-K nearest neighbours by sentence embedding,
vote on attr value. Honest 80/20 split (seed=42) per (cat, attr).

Compared as standalone layer vs current cascade v3 stacked ensemble baseline.

Output: datasets/processed/cascade_knn_eval.parquet
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout, force=True,
)
logger = logging.getLogger(__name__)

WORKTREE_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = WORKTREE_ROOT / "datasets" / "processed"
OUT_PATH = PROCESSED_DIR / "cascade_knn_eval.parquet"

CATEGORIES = ["pasta", "chocolate", "cheeses"]
K_VALUES = [1, 3, 5, 10]
RANDOM_STATE = 42
TEST_SIZE = 0.2


def cosine_top_k(query: np.ndarray, train: np.ndarray, k: int) -> np.ndarray:
    """Returns indices of top-k most similar (cosine) train rows for each query."""
    # Normalize
    q_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-9)
    t_norm = train / (np.linalg.norm(train, axis=1, keepdims=True) + 1e-9)
    sims = q_norm @ t_norm.T  # (n_query, n_train)
    return np.argsort(-sims, axis=1)[:, :k]


def majority_vote(values: list[str]) -> tuple[str, float]:
    """Returns (most_common_value, fraction_agree)."""
    if not values:
        return "", 0.0
    cnt = Counter(values)
    top_val, top_count = cnt.most_common(1)[0]
    return top_val, top_count / len(values)


def eval_attr(cat: str, attr: str, emb: np.ndarray, codes: list[str],
              gold_map: dict[str, str]) -> list[dict]:
    """Evaluate kNN for one (cat, attr). Returns rows for K_VALUES results."""
    # Filter to codes with gold for this attr
    valid_codes = [c for c in codes if c in gold_map]
    if len(valid_codes) < 20:
        return []
    code_to_idx = {c: i for i, c in enumerate(codes)}
    valid_idx = np.array([code_to_idx[c] for c in valid_codes])
    valid_emb = emb[valid_idx]
    y = np.array([gold_map[c] for c in valid_codes])

    try:
        tr_idx, te_idx = train_test_split(
            np.arange(len(valid_codes)), test_size=TEST_SIZE,
            random_state=RANDOM_STATE, stratify=y,
        )
    except ValueError:
        tr_idx, te_idx = train_test_split(
            np.arange(len(valid_codes)), test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

    X_tr, X_te = valid_emb[tr_idx], valid_emb[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    rows = []
    for k in K_VALUES:
        topk_indices = cosine_top_k(X_te, X_tr, k)
        preds = []
        confidences = []
        for row_idx_arr in topk_indices:
            neighbour_vals = [y_tr[i] for i in row_idx_arr]
            pred, conf = majority_vote(neighbour_vals)
            preds.append(pred)
            confidences.append(conf)
        preds = np.array(preds)
        acc = float(np.mean(preds == y_te))
        rows.append({
            "category": cat, "attr": attr, "k": k,
            "n_test": len(te_idx), "n_train": len(tr_idx),
            "acc_knn": acc, "mean_neighbour_agree": float(np.mean(confidences)),
        })
    return rows


def main() -> None:
    gold = pd.read_parquet(PROCESSED_DIR / "consensus_gold_v2_expanded.parquet")
    gold = gold[~gold["gold_is_null"]].copy()
    gold["code"] = gold["code"].astype(str)
    # Normalize gold values
    gold["gold_str"] = gold["gold_value"].astype(str).str.strip()

    all_rows = []
    for cat in CATEGORIES:
        emb_path = PROCESSED_DIR / f"{cat}_stratified_embeddings.npy"
        silver_path = PROCESSED_DIR / f"{cat}_stratified_silver_standard.parquet"
        if not emb_path.exists() or not silver_path.exists():
            logger.warning("Missing data for %s", cat)
            continue
        emb = np.load(emb_path)
        silver = pd.read_parquet(silver_path)
        silver["code"] = silver["code"].astype(str)
        codes = silver["code"].tolist()  # order matches embeddings
        logger.info("%s: %d codes, emb shape %s", cat, len(codes), emb.shape)

        cat_gold = gold[gold["category"] == cat]
        for attr, grp in cat_gold.groupby("attr"):
            gold_map = dict(zip(grp["code"], grp["gold_str"]))
            rows = eval_attr(cat, attr, emb, codes, gold_map)
            all_rows.extend(rows)

    out = pd.DataFrame(all_rows)
    out.to_parquet(OUT_PATH, index=False)
    logger.info("Saved %d rows to %s", len(out), OUT_PATH)

    # Summary
    print(f"\n{'='*72}")
    print("k-NN retrieval baseline on v2 gold (honest 80/20 split, seed=42)")
    print(f"{'='*72}")
    print(f"\n{'Cat':<12} {'Attrs':>6} {'k=1':>8} {'k=3':>8} {'k=5':>8} "
          f"{'k=10':>8}")
    print("-" * 60)
    for cat in CATEGORIES:
        sub = out[out["category"] == cat]
        line = f"{cat:<12} {sub['attr'].nunique():>6}"
        for k in K_VALUES:
            v = sub[sub["k"] == k]["acc_knn"].mean() * 100
            line += f" {v:>7.2f}%"
        print(line)
    print("-" * 60)
    line = f"{'GRAND':<12} {out['attr'].nunique() * 3:>6}"
    for k in K_VALUES:
        v = out[out["k"] == k]["acc_knn"].mean() * 100
        line += f" {v:>7.2f}%"
    print(line)

    # Compare with v3 cascade (acc_v3_stacked from EXP13 results)
    v3_files = [PROCESSED_DIR / f"cascade_v3_eval_{c}.parquet" for c in CATEGORIES]
    if all(p.exists() for p in v3_files):
        v3 = pd.concat([pd.read_parquet(p) for p in v3_files], ignore_index=True)
        v3_mean = v3["acc_v3_stacked"].mean() * 100
        best_k = K_VALUES[np.argmax([out[out["k"] == k]["acc_knn"].mean() for k in K_VALUES])]
        knn_best = out[out["k"] == best_k]["acc_knn"].mean() * 100
        print(f"\nBaseline cascade v3 stacked: {v3_mean:.2f}%")
        print(f"Best kNN (k={best_k}):           {knn_best:.2f}%")
        print(f"Delta: {knn_best - v3_mean:+.2f}pp")


if __name__ == "__main__":
    main()
