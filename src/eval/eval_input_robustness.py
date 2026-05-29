"""Robustness каскада к шуму во входных партнёрских данных.

Тикет: 2026-05-28-input-robustness.

Симулирует три типа реального партнёрского шума на тестовом срезе
gold (cascade_preds_{cat}_gold.parquet × {cat}_stratified_silver_standard.parquet)
и измеряет деградацию точности слоёв Layer 1 (regex/правила) + Layer 2 (ML).
Layer 4 (LLM) намеренно не вызывается — измеряется доля ячеек, которые
система должна была бы эскалировать.

Виды шума (исходный текст НЕ мутируется на gold-уровне; шум применяется
к product_name / brands / ingredients_text на копии входа перед инференсом):

  1. typo: random char-swap в product_name (1, 5, 10 % символов).
  2. missing_brands: brands → None для X % ячеек (50, 100 %).
  3. missing_ingredients: ingredients_text → None для X % ячеек (50, 100 %).
  4. combined (опционально): 5 % typo + 50 % missing brands.

Выход: datasets/processed/input_robustness_results.parquet со столбцами
  noise_type, noise_level, category, attr, n, accuracy, llm_share,
  accuracy_delta_vs_clean.

Запуск:
  source .venv/bin/activate && OMP_NUM_THREADS=1 \
      python src/eval/eval_input_robustness.py \
      [--sample 500] [--seed 42]
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common import PARTNER_TEXT_FIELDS
from src.pipeline.regex.extractor import RegexExtractor

# Категории, для которых есть полный набор v4 моделей + gold
CATEGORIES = ["pasta", "chocolate", "cheeses"]
MODEL_PREFIX_FMT = "{cat}_v4"  # без mpnet/hybrid — соответствует cascade_preds_{cat}_gold
# v4 модели обучены на MiniLM (384d). MPNet 768d не подходит — feature mismatch.
# Это сознательно: cascade_preds_{cat}_gold.parquet — артефакт V4 cascade, см.
# memory/v4_FINAL_HONEST_2026-05-25 — производственная конфигурация в этом скрипте.
EMBEDDING_MODEL_LOCAL = "paraphrase-multilingual-MiniLM-L12-v2"

# Шумовые конфигурации (название → (тип, уровень))
NOISE_CONFIGS = [
    ("clean", 0.0),
    ("typo", 0.01),
    ("typo", 0.05),
    ("typo", 0.10),
    ("missing_brands", 0.50),
    ("missing_brands", 1.00),
    ("missing_ingredients", 0.50),
    ("missing_ingredients", 1.00),
    ("combined", 0.0),  # 5 % typo + 50 % missing brands
]


def apply_typo(text: str | None, frac: float, rng: np.random.Generator) -> str | None:
    """Поменять местами frac символов с соседом (random adjacent swap)."""
    if text is None or not isinstance(text, str) or len(text) < 2:
        return text
    chars = list(text)
    n_swaps = max(1, int(round(len(chars) * frac)))
    positions = rng.choice(len(chars) - 1, size=min(n_swaps, len(chars) - 1),
                           replace=False)
    for p in positions:
        chars[p], chars[p + 1] = chars[p + 1], chars[p]
    return "".join(chars)


def apply_noise_to_inputs(inputs: pd.DataFrame, noise_type: str, level: float,
                          seed: int = 42) -> pd.DataFrame:
    """Вернуть копию inputs с применённым шумом ко входным полям."""
    out = inputs.copy()
    rng = np.random.default_rng(seed)
    n = len(out)

    if noise_type == "clean":
        return out

    if noise_type == "typo":
        out["product_name"] = out["product_name"].apply(
            lambda t: apply_typo(t, level, rng))
        return out

    if noise_type == "missing_brands":
        mask = rng.random(n) < level
        out.loc[mask, "brands"] = None
        return out

    if noise_type == "missing_ingredients":
        mask = rng.random(n) < level
        out.loc[mask, "ingredients_text"] = None
        return out

    if noise_type == "combined":
        # 5 % typo + 50 % missing brands
        out["product_name"] = out["product_name"].apply(
            lambda t: apply_typo(t, 0.05, rng))
        mask = rng.random(n) < 0.50
        out.loc[mask, "brands"] = None
        return out

    raise ValueError(f"Unknown noise_type: {noise_type}")


def build_texts(df: pd.DataFrame) -> list[str]:
    """Собрать тексты для embeddings — конкатенация партнёр-доступных полей."""
    texts = []
    for _, row in df.iterrows():
        parts = [str(row[c]) for c in PARTNER_TEXT_FIELDS
                 if c in row.index and pd.notna(row[c])]
        texts.append(" ".join(parts))
    return texts


def load_models(prefix: str) -> dict:
    """Загрузить все XGB + LE + thresholds для category prefix."""
    mdir = PROJECT_ROOT / "models"
    thr_path = mdir / f"{prefix}_thresholds.pkl"
    with open(thr_path, "rb") as f:
        thresholds = pickle.load(f)

    models = {}
    for attr in thresholds.keys():
        clf_path = mdir / f"{prefix}_{attr}_xgb.pkl"
        if not clf_path.exists():
            continue
        with open(clf_path, "rb") as f:
            clf = pickle.load(f)
        le_path = mdir / f"{prefix}_{attr}_le.pkl"
        le = None
        if le_path.exists():
            with open(le_path, "rb") as f:
                le = pickle.load(f)
        models[attr] = {"clf": clf, "le": le, "thr": float(thresholds[attr])}
    return models


def predict_ml_layer(embeddings: np.ndarray, models: dict, codes: list[str]) -> pd.DataFrame:
    """Прогон XGBoost-классификаторов по embeddings.

    Возвращает long-DataFrame: code, attr, ml_pred, ml_conf, ml_fired.
    """
    rows = []
    for attr, m in models.items():
        clf, le, thr = m["clf"], m["le"], m["thr"]
        proba = clf.predict_proba(embeddings)
        if le is None:
            # binary
            confs = np.maximum(proba[:, 1], 1 - proba[:, 1])
            preds = (proba[:, 1] >= 0.5).astype(int)
            labels = [bool(p) for p in preds]
        else:
            confs = proba.max(axis=1)
            preds = proba.argmax(axis=1)
            labels = [str(le.classes_[p]) for p in preds]
        for i, code in enumerate(codes):
            rows.append({
                "code": code, "attr": attr,
                "ml_pred": labels[i],
                "ml_conf": float(confs[i]),
                "ml_fired": bool(confs[i] >= thr),
            })
    return pd.DataFrame(rows)


def predict_regex_layer(inputs: pd.DataFrame, category: str) -> pd.DataFrame:
    """Layer 1: regex extraction (только для category).

    Возвращает long-DataFrame: code, attr, regex_pred (None если не сработал).
    """
    ext = RegexExtractor()
    rows = []
    for _, row in inputs.iterrows():
        code = str(row["code"])
        results = ext.extract_all(
            product_name=str(row.get("product_name") or ""),
            brands=str(row.get("brands") or ""),
            ingredients_text=str(row.get("ingredients_text") or ""),
            quantity=str(row.get("quantity") or ""),
            category=category,
        )
        for attr, res in results.items():
            if res.value is not None:
                rows.append({"code": code, "attr": attr,
                             "regex_pred": str(res.value).lower(),
                             "regex_conf": float(res.confidence)})
    return pd.DataFrame(rows, columns=["code", "attr", "regex_pred", "regex_conf"])


def norm_value(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    s = str(v).strip().lower()
    if s in ("", "none", "null", "nan"):
        return None
    return s


def cascade_predict(regex_df: pd.DataFrame, ml_df: pd.DataFrame,
                    gold_layers: pd.DataFrame) -> pd.DataFrame:
    """Применить cascade policy: rule_h (regex hit) → ML (confident) → fallback.

    gold_layers: cascade_preds_{cat}_gold.parquet — содержит исходный
    `cascade_layer` для каждой (code,attr) ячейки; используется как референс
    для определения того, какой слой В ПРИНЦИПЕ работает на этом атрибуте
    (rule_h vs ml vs both).

    Возвращает DataFrame с колонками: code, attr, layer, pred.
    """
    # Long → wide для merge
    g_keys = gold_layers[["code", "attr"]].drop_duplicates().copy()
    g_keys["code"] = g_keys["code"].astype(str)

    if regex_df.empty:
        regex_df = pd.DataFrame(columns=["code", "attr", "regex_pred", "regex_conf"])
    regex_df["code"] = regex_df["code"].astype(str)
    ml_df["code"] = ml_df["code"].astype(str)

    merged = g_keys.merge(regex_df, on=["code", "attr"], how="left") \
                   .merge(ml_df, on=["code", "attr"], how="left")

    # Достанем «допустимые» слои из gold_layers — какие слои реально
    # фигурируют для каждого attr (sanity check; не блокируем cascade).
    rule_h_attrs = set(
        gold_layers[gold_layers.cascade_layer == "rule_h"]["attr"].unique()
    )

    out_rows = []
    for _, row in merged.iterrows():
        attr = row["attr"]
        has_regex = pd.notna(row.get("regex_pred"))
        ml_fired = bool(row.get("ml_fired"))
        # rule_h приоритет — только для атрибутов, где он вообще участвует
        if has_regex and attr in rule_h_attrs:
            out_rows.append({"code": row["code"], "attr": attr,
                             "layer": "rule_h", "pred": row["regex_pred"]})
        elif ml_fired:
            out_rows.append({"code": row["code"], "attr": attr,
                             "layer": "ml",
                             "pred": str(row["ml_pred"]).lower()})
        else:
            out_rows.append({"code": row["code"], "attr": attr,
                             "layer": "fallback", "pred": None})
    return pd.DataFrame(out_rows)


def score(cascade: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Подсчитать accuracy и llm_share для каждого (cat,attr).

    accuracy = correct / answered (ячейки, где layer != fallback и pred совпал
    с gold_value после нормализации).
    llm_share = fallback / total — доля ячеек, которые ушли бы в Layer 4.

    Возвращает DataFrame: attr, n_total, n_answered, n_correct, accuracy,
    llm_share.
    """
    cascade["code"] = cascade["code"].astype(str)
    gold["code"] = gold["code"].astype(str)
    g = gold[["code", "attr", "gold_value"]].merge(
        cascade[["code", "attr", "layer", "pred"]],
        on=["code", "attr"], how="inner",
    )
    rows = []
    for attr in sorted(g.attr.unique()):
        sub = g[g.attr == attr]
        total = len(sub)
        answered = (sub.layer != "fallback").sum()
        g_norm = sub.gold_value.apply(norm_value)
        p_norm = sub.pred.apply(norm_value)
        valid = (sub.layer != "fallback") & g_norm.notna() & p_norm.notna()
        correct = (g_norm == p_norm)[valid].sum()
        acc = correct / valid.sum() if valid.sum() > 0 else float("nan")
        llm_share = (sub.layer == "fallback").sum() / total if total > 0 else 0.0
        rows.append({
            "attr": attr,
            "n_total": int(total),
            "n_answered": int(answered),
            "n_correct": int(correct),
            "accuracy": float(acc),
            "llm_share": float(llm_share),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=500,
                        help="Sample N codes per category (None = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str,
                        default="datasets/processed/input_robustness_results.parquet")
    args = parser.parse_args()

    # Lazy import — sentence-transformers тяжёлый
    from sentence_transformers import SentenceTransformer
    print(f"[info] Loading embedding model {EMBEDDING_MODEL_LOCAL} (это ~30 сек)...")
    t0 = time.time()
    st_model = SentenceTransformer(EMBEDDING_MODEL_LOCAL)
    print(f"[info] Loaded in {time.time()-t0:.1f}s")

    rng = np.random.default_rng(args.seed)
    all_results = []

    for cat in CATEGORIES:
        print(f"\n=== {cat.upper()} ===")
        gold_path = PROJECT_ROOT / f"datasets/processed/cascade_preds_{cat}_gold.parquet"
        silver_path = PROJECT_ROOT / f"datasets/processed/{cat}_stratified_silver_standard.parquet"
        gold = pd.read_parquet(gold_path)
        gold["code"] = gold["code"].astype(str)
        silver = pd.read_parquet(silver_path)
        silver["code"] = silver["code"].astype(str)

        gold_codes = set(gold["code"].unique())
        if args.sample and len(gold_codes) > args.sample:
            sampled = rng.choice(list(gold_codes), size=args.sample, replace=False)
            gold_codes = set(sampled.tolist())
        gold_use = gold[gold["code"].isin(gold_codes)].copy()
        print(f"  gold cells: {len(gold_use)}, unique codes: {len(gold_codes)}")

        # Build inputs from silver
        inputs = silver[silver["code"].isin(gold_codes)] \
            .drop_duplicates(subset=["code"]) \
            .reset_index(drop=True)
        # ensure all 4 partner fields present
        for c in PARTNER_TEXT_FIELDS:
            if c not in inputs.columns:
                inputs[c] = None
        print(f"  input rows: {len(inputs)}")

        prefix = MODEL_PREFIX_FMT.format(cat=cat)
        print(f"  loading models prefix={prefix}...")
        models = load_models(prefix)
        print(f"  loaded {len(models)} classifiers: {sorted(models.keys())}")

        # Clean baseline + each noise variant
        for noise_type, level in NOISE_CONFIGS:
            t1 = time.time()
            noisy = apply_noise_to_inputs(inputs, noise_type, level, seed=args.seed)
            texts = build_texts(noisy)
            embeddings = st_model.encode(texts, show_progress_bar=False,
                                         batch_size=64).astype(np.float32)
            codes = noisy["code"].astype(str).tolist()
            ml_df = predict_ml_layer(embeddings, models, codes)
            regex_df = predict_regex_layer(noisy, cat)
            cascade = cascade_predict(regex_df, ml_df, gold_use)
            attr_scores = score(cascade, gold_use)
            attr_scores["category"] = cat
            attr_scores["noise_type"] = noise_type
            attr_scores["noise_level"] = level
            all_results.append(attr_scores)
            overall_acc = (attr_scores["n_correct"].sum()
                           / max(attr_scores.apply(
                               lambda r: r["n_answered"] if pd.notna(r["accuracy"]) else 0,
                               axis=1).sum(), 1))
            overall_llm = (attr_scores.assign(
                fb=lambda d: d["n_total"] - d["n_answered"]
            )["fb"].sum() / max(attr_scores["n_total"].sum(), 1))
            print(f"  [{noise_type:>20s} @ {level:.2f}] "
                  f"acc={overall_acc*100:5.1f}% llm_share={overall_llm*100:5.1f}% "
                  f"({time.time()-t1:.1f}s)")

    df = pd.concat(all_results, ignore_index=True)

    # Compute accuracy_delta_vs_clean per (category, attr)
    clean_idx = (df.noise_type == "clean")
    clean = df[clean_idx][["category", "attr", "accuracy"]] \
        .rename(columns={"accuracy": "accuracy_clean"})
    df = df.merge(clean, on=["category", "attr"], how="left")
    df["accuracy_delta_vs_clean"] = df["accuracy"] - df["accuracy_clean"]
    df = df[[
        "noise_type", "noise_level", "category", "attr",
        "n_total", "n_answered", "n_correct",
        "accuracy", "llm_share", "accuracy_clean", "accuracy_delta_vs_clean",
    ]]
    df = df.rename(columns={"n_total": "n"})

    out_path = PROJECT_ROOT / args.out
    df.to_parquet(out_path, index=False)
    print(f"\n[done] saved {len(df)} rows → {out_path}")

    # Summary table by (noise_type, noise_level)
    print("\n=== SUMMARY: weighted overall accuracy per noise config ===")
    summary_rows = []
    for (nt, lvl), grp in df.groupby(["noise_type", "noise_level"]):
        n_ans = grp["n_answered"].sum()
        n_cor = grp["n_correct"].sum()
        n_tot = grp["n"].sum()
        n_fb = n_tot - n_ans
        acc = n_cor / n_ans if n_ans > 0 else float("nan")
        llm = n_fb / n_tot if n_tot > 0 else 0.0
        summary_rows.append({
            "noise_type": nt, "noise_level": lvl,
            "overall_acc": acc, "overall_llm_share": llm,
            "n_total": int(n_tot),
        })
    summary = pd.DataFrame(summary_rows).sort_values(["noise_type", "noise_level"])
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
