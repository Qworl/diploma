"""E11 — k-NN distance ablation на SBERT-эмбеддингах.

Цель: количественно проверить, что brand-disjoint test действительно semantically
disjoint от train, или брэнды-разные-но-семантически-похожие маскируют утечку.

Метод:
1. Загрузить SBERT-эмбеддинги per category (`{cat}_stratified_embeddings.npy`).
2. Разделить на train/test по `{cat}_gold_split.parquet`.
3. Для каждого тестового товара найти k=5 ближайших train-товаров по cosine distance.
4. Сгруппировать тестовые товары по quartile median-NN-distance.
5. Для каждой quartile посчитать cascade accuracy.
6. Если accuracy не падает на дальних NN — это сильный pro-аргумент о semantic-disjointness.

Output: datasets/processed/knn_distance_ablation.parquet
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
PROCESSED = ROOT / "datasets" / "processed"
OUT = PROCESSED / "knn_distance_ablation.parquet"
CATEGORIES = ["pasta", "chocolate", "cheeses"]
K = 5
N_QUARTILES = 4


def normalize(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().lower()
    if s in ("none", "null", "nan", ""):
        return None
    return s


def main():
    gold = pd.read_parquet(PROCESSED / "consensus_gold_v2_expanded.parquet")
    gold = gold[~gold.gold_is_null].copy()
    gold["gold_norm"] = gold.gold_value.map(normalize)
    gold["code"] = gold["code"].astype(str)

    per_cell_rows: list[dict] = []
    per_quartile_rows: list[dict] = []

    for cat in CATEGORIES:
        logger.info("=== %s ===", cat)
        emb = np.load(PROCESSED / f"{cat}_stratified_embeddings.npy")
        sil = pd.read_parquet(PROCESSED / f"{cat}_stratified_silver_standard.parquet",
                              columns=["code"])
        sil["code"] = sil["code"].astype(str)
        split = pd.read_parquet(PROCESSED / f"{cat}_gold_split.parquet")
        split["code"] = split["code"].astype(str)

        # Build code → embedding row index map
        sil = sil.reset_index().rename(columns={"index": "emb_idx"})
        sil["emb_idx"] = sil["emb_idx"].astype(int)

        sp = split.merge(sil, on="code", how="inner")
        train_codes = sp[sp.split == "train"]["code"].tolist()
        test_codes = sp[sp.split == "test"]["code"].tolist()
        train_idx = sp[sp.split == "train"]["emb_idx"].values
        test_idx = sp[sp.split == "test"]["emb_idx"].values

        logger.info("train: %d codes, test: %d codes (after merge with silver)",
                    len(train_codes), len(test_codes))

        # L2-normalize embeddings for cosine similarity
        train_emb = emb[train_idx]
        test_emb = emb[test_idx]
        train_norm = train_emb / (np.linalg.norm(train_emb, axis=1, keepdims=True) + 1e-12)
        test_norm = test_emb / (np.linalg.norm(test_emb, axis=1, keepdims=True) + 1e-12)

        # cosine similarity: test × train ; distance = 1 - sim
        sim = test_norm @ train_norm.T
        # for each test row, find top-K largest similarities
        topk_idx = np.argpartition(-sim, K, axis=1)[:, :K]
        topk_sim = np.take_along_axis(sim, topk_idx, axis=1)
        # cosine distance = 1 - sim; median distance over k=5
        topk_dist = 1.0 - topk_sim
        median_nn_dist = np.median(topk_dist, axis=1)  # per test row
        mean_nn_dist = np.mean(topk_dist, axis=1)
        min_nn_dist = np.min(topk_dist, axis=1)

        test_dist = pd.DataFrame({
            "category": cat,
            "code": test_codes,
            "median_nn_dist": median_nn_dist,
            "mean_nn_dist": mean_nn_dist,
            "min_nn_dist": min_nn_dist,
        })

        # Cascade predictions on test cells
        casc = pd.read_parquet(PROCESSED / f"cascade_preds_{cat}_v2_gold_hybrid_v3_fixed.parquet")
        casc["code"] = casc["code"].astype(str)
        casc["cascade_pred_norm"] = casc.predicted.map(normalize)
        casc["cascade_abstain"] = (casc.layer == "abstain")

        g_cat = gold[gold.category == cat][["code", "attr", "gold_norm"]]
        m = (casc[["code", "attr", "cascade_pred_norm", "cascade_abstain"]]
             .merge(g_cat, on=["code", "attr"], how="inner")
             .merge(test_dist[["code", "median_nn_dist"]], on="code", how="inner"))
        m["category"] = cat
        m["correct_e2e"] = (~m.cascade_abstain) & (m.cascade_pred_norm == m.gold_norm)
        logger.info("%s test cells matched to gold + distance: %d", cat, len(m))

        # Quartile by median_nn_dist (товар-уровень)
        prod_dist = test_dist.set_index("code")["median_nn_dist"]
        prod_dist = prod_dist.loc[m.code.unique()]
        quartile_edges = np.quantile(prod_dist.values, np.linspace(0, 1, N_QUARTILES + 1))
        prod_q = pd.cut(prod_dist, bins=quartile_edges, labels=[f"Q{i+1}" for i in range(N_QUARTILES)],
                        include_lowest=True)
        m = m.merge(prod_q.rename("quartile").to_frame().reset_index(), on="code", how="left")

        per_cell_rows.append(m[["category", "code", "attr", "median_nn_dist", "quartile", "correct_e2e"]])

        for q in [f"Q{i+1}" for i in range(N_QUARTILES)]:
            sub = m[m.quartile == q]
            if not len(sub):
                continue
            acc = float(sub.correct_e2e.mean())
            per_quartile_rows.append({
                "category": cat, "quartile": q,
                "median_nn_dist_min": float(sub.median_nn_dist.min()),
                "median_nn_dist_max": float(sub.median_nn_dist.max()),
                "median_nn_dist_mean": float(sub.median_nn_dist.mean()),
                "n_test_products": int(sub.code.nunique()),
                "n_test_cells": len(sub),
                "acc_e2e": acc,
            })

    out_quart = pd.DataFrame(per_quartile_rows)
    # Add global (across cats) quartile aggregation
    all_cells = pd.concat(per_cell_rows, ignore_index=True)
    # Re-bucket globally
    prod_global = all_cells.groupby("code")["median_nn_dist"].first()
    g_edges = np.quantile(prod_global.values, np.linspace(0, 1, N_QUARTILES + 1))
    prod_q_glob = pd.cut(prod_global, bins=g_edges,
                        labels=[f"Q{i+1}" for i in range(N_QUARTILES)],
                        include_lowest=True)
    all_cells["q_global"] = all_cells.code.map(prod_q_glob.to_dict())
    for q in [f"Q{i+1}" for i in range(N_QUARTILES)]:
        sub = all_cells[all_cells.q_global == q]
        if not len(sub):
            continue
        out_quart = pd.concat([out_quart, pd.DataFrame([{
            "category": "global", "quartile": q,
            "median_nn_dist_min": float(sub.median_nn_dist.min()),
            "median_nn_dist_max": float(sub.median_nn_dist.max()),
            "median_nn_dist_mean": float(sub.median_nn_dist.mean()),
            "n_test_products": int(sub.code.nunique()),
            "n_test_cells": len(sub),
            "acc_e2e": float(sub.correct_e2e.mean()),
        }])], ignore_index=True)

    out_quart.to_parquet(OUT, index=False)
    logger.info("Saved %d rows → %s", len(out_quart), OUT)

    print("\n" + "=" * 92)
    print("E11 — k-NN distance ablation: точность каскада по квартилям близости к train")
    print("(k=5 nearest train products per test product, cosine distance в SBERT-пространстве)")
    print("=" * 92)
    print(f"{'scope':<10} {'q':<5} {'dist range':<20} {'n_prod':>8} {'n_cells':>8} {'acc':>10}")
    print("-" * 92)
    order = {"pasta": 0, "chocolate": 1, "cheeses": 2, "global": 3}
    out_quart["__o"] = out_quart.category.map(order)
    out_quart["__q"] = out_quart.quartile.str[1:].astype(int)
    out_sorted = out_quart.sort_values(["__o", "__q"]).drop(columns=["__o", "__q"])
    for _, r in out_sorted.iterrows():
        rng = f"[{r.median_nn_dist_min:.3f}; {r.median_nn_dist_max:.3f}]"
        print(f"{r.category:<10} {r.quartile:<5} {rng:<20} {r.n_test_products:>8} {r.n_test_cells:>8} "
              f"{r.acc_e2e*100:>8.2f}%")
    print("=" * 92)
    print("\nИНТЕРПРЕТАЦИЯ: если acc_e2e примерно одинаков по Q1→Q4 (дальние NN не "
          "роняют точность), это означает, что brand-disjoint тест действительно "
          "семантически disjoint от train, и наблюдаемая точность каскада не "
          "обеспечена близостью test-семантики к train-семантике.")


if __name__ == "__main__":
    main()
