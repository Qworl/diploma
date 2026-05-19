"""EXP11 Phase C+D+E: annotate missing AL/random groups, then eval.

Usage:
  python -m src.experiments.al_annotate_and_eval --phase annotate [--cats pasta chocolate cheeses]
  python -m src.experiments.al_annotate_and_eval --phase eval

Phase annotate: for each cat, runs LLM annotation (gpt-oss-120b, off_grounded) on:
  - al_gpt55_uncertain/{cat}_gold.parquet  (from al_codes_{cat}.csv)
  - al_gpt55_random/{cat}_gold.parquet     (from al_control_codes_{cat}.csv)
  Resume-safe: skips codes already in output parquet.

Phase eval: re-trains R_ml (XGB + LGBM) in 3 variants and compares on 20% held-out.
  Outputs: datasets/processed/active_learning_pilot.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.common import PROCESSED_DIR, setup_logging
from src.eval.direct_llm_v2 import run_llm_on_products
from src.llm.client import call_openrouter
from src.manual_label.schemas_loader import load_domain_attrs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WORKTREE_ROOT = Path(__file__).parent.parent.parent
OFF_PATH = WORKTREE_ROOT / "datasets" / "raw" / "en.openfoodfacts.org.products.parquet"
OFF_CACHE_DIR = WORKTREE_ROOT / "datasets" / "manual_label" / "off_cache"
MANUAL_LABEL_DIR = WORKTREE_ROOT / "datasets" / "manual_label"
PROCESSED_PATH = Path(PROCESSED_DIR)

GOLD_PATH = PROCESSED_PATH / "consensus_gold_v2_expanded.parquet"
OUT_PATH = PROCESSED_PATH / "active_learning_pilot.parquet"
AL_GOLD_DIR = PROCESSED_PATH / "al_gpt55_uncertain"
RAND_GOLD_DIR = PROCESSED_PATH / "al_gpt55_random"

CATEGORIES = ["pasta", "chocolate", "cheeses"]
MODEL_NAME = "openai/gpt-oss-120b"   # cheaper than gpt-5.5, suitable for annotation
MAX_COST_PER_RUN = 10.0              # USD per (cat, group) annotation run
SEED = 42
TEST_SIZE = 0.2
GOLD_WEIGHT = 5.0

META_COLS = {
    "code", "product_name", "brands", "ingredients_text", "quantity",
    "categories_tags", "labels_tags", "ingredients_analysis_tags",
    "traces_tags", "countries_tags",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_off_products_by_code(codes: list[str]) -> pd.DataFrame:
    """Load OFF product rows for given codes from parquet."""
    logger.info("Loading OFF parquet for %d codes...", len(codes))
    cols = ["code", "product_name", "brands", "ingredients_text", "quantity", "categories_tags"]
    off = pd.read_parquet(OFF_PATH, columns=cols)
    off["code"] = off["code"].astype(str)
    result = off[off["code"].isin(set(codes))].copy().reset_index(drop=True)
    logger.info("Found %d/%d codes in OFF parquet", len(result), len(codes))
    return result


def _build_text(row: pd.Series) -> str:
    parts = []
    for col in ["product_name", "brands", "ingredients_text", "quantity"]:
        v = row.get(col)
        if pd.notna(v) and str(v).strip():
            parts.append(str(v).strip())
    return " ".join(parts)


def _annotations_to_long(df_wide: pd.DataFrame, cat: str) -> pd.DataFrame:
    """Convert gpt wide-format annotation parquet to long gold format."""
    sig_path = PROCESSED_PATH / "attribute_signal_taxonomy.parquet"
    sig = pd.read_parquet(sig_path) if sig_path.exists() else pd.DataFrame()
    sig_map = {}
    if len(sig) > 0:
        sig_map = {(r["category"], r["attr"]): r["signal_type"] for _, r in sig.iterrows()}

    rows = []
    for _, row in df_wide.iterrows():
        code = str(row["code"])
        try:
            parsed = json.loads(row["parsed_json"]) if row.get("parsed_json") else {}
        except (json.JSONDecodeError, TypeError):
            parsed = {}
        for attr, val in parsed.items():
            rows.append({
                "category": cat,
                "code": code,
                "attr": attr,
                "gold_value": str(val) if val is not None else None,
                "gold_is_null": val is None,
                "signal_type": sig_map.get((cat, attr), "text_derived"),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["category", "code", "attr", "gold_value", "gold_is_null", "signal_type"]
    )


# ---------------------------------------------------------------------------
# Phase annotate
# ---------------------------------------------------------------------------

def annotate_group(
    cat: str,
    codes: list[str],
    out_path: Path,
    api_key: str,
    max_cost: float,
) -> pd.DataFrame:
    """Annotate codes in `codes` for `cat`, writing results to `out_path` (resume-safe)."""
    products = _load_off_products_by_code(codes)
    if len(products) == 0:
        logger.warning("[%s] No products found for annotation — check OFF parquet.", cat)
        return pd.DataFrame()

    logger.info("[%s] Annotating %d products with %s (off_grounded)...", cat, len(products), MODEL_NAME)
    df = run_llm_on_products(
        products,
        domain=cat,
        model=MODEL_NAME,
        api_key=api_key,
        out_path=out_path,
        context_mode="off_grounded",
        off_cache_dir=OFF_CACHE_DIR,
        max_cost_usd=max_cost,
        sleep_between=0.1,
        call_fn=call_openrouter,
    )
    cost = float(df["cost_usd"].sum()) if len(df) > 0 and "cost_usd" in df.columns else 0.0
    logger.info("[%s] Done: %d rows, cost=$%.3f", cat, len(df), cost)
    return df


def run_annotate(cats: list[str], api_key: str) -> None:
    AL_GOLD_DIR.mkdir(parents=True, exist_ok=True)
    RAND_GOLD_DIR.mkdir(parents=True, exist_ok=True)

    total_cost = 0.0
    for cat in cats:
        al_csv = MANUAL_LABEL_DIR / f"al_codes_{cat}.csv"
        ctrl_csv = MANUAL_LABEL_DIR / f"al_control_codes_{cat}.csv"

        if not al_csv.exists() or not ctrl_csv.exists():
            logger.error("[%s] Missing AL CSV files — run Phase A first", cat)
            continue

        al_codes = pd.read_csv(al_csv)["code"].astype(str).tolist()
        ctrl_codes = pd.read_csv(ctrl_csv)["code"].astype(str).tolist()
        logger.info("[%s] AL codes: %d, ctrl codes: %d", cat, len(al_codes), len(ctrl_codes))

        # AL uncertain annotation
        al_out = AL_GOLD_DIR / f"{cat}_gold.parquet"
        al_df = annotate_group(cat, al_codes, al_out, api_key, MAX_COST_PER_RUN)
        if "cost_usd" in al_df.columns:
            total_cost += float(al_df["cost_usd"].sum())

        # Random control annotation
        ctrl_out = RAND_GOLD_DIR / f"{cat}_gold.parquet"
        ctrl_df = annotate_group(cat, ctrl_codes, ctrl_out, api_key, MAX_COST_PER_RUN)
        if "cost_usd" in ctrl_df.columns:
            total_cost += float(ctrl_df["cost_usd"].sum())

    logger.info("Total annotation cost: $%.3f", total_cost)
    print(f"\nAnnotation done. Total cost: ${total_cost:.3f}")


# ---------------------------------------------------------------------------
# Phase eval: R_ml train + compare
# ---------------------------------------------------------------------------

def _train_ensemble(
    X_tr_emb: np.ndarray,
    y_tr: list[str],
    tr_texts: list[str],
    w_tr: np.ndarray,
    X_te_emb: np.ndarray,
    te_texts: list[str],
    y_te: list[str],
) -> float:
    """Train XGB + LGBM ensemble and return accuracy on test set."""
    classes = sorted(set(y_tr))
    if len(classes) < 2 or len(y_tr) < 10:
        return float("nan")

    le = LabelEncoder()
    le.fit(classes)
    y_enc = le.transform(y_tr)
    n_classes = len(classes)

    # XGB
    common_xgb = dict(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
        tree_method="hist", verbosity=0,
    )
    if n_classes == 2:
        pos = int((y_enc == 1).sum())
        neg = int((y_enc == 0).sum())
        spw = max(neg / max(pos, 1), 0.5)
        clf_xgb = xgb.XGBClassifier(scale_pos_weight=spw, **common_xgb)
    else:
        clf_xgb = xgb.XGBClassifier(objective="multi:softmax", num_class=n_classes, **common_xgb)
    clf_xgb.fit(X_tr_emb, y_enc, sample_weight=w_tr)
    p_xgb = clf_xgb.predict_proba(X_te_emb)

    # LGBM
    p_lgbm = None
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=10_000)
        X_tfidf_tr = vec.fit_transform(tr_texts)
        X_tfidf_te = vec.transform(te_texts)
        obj = "binary" if n_classes == 2 else "multiclass"
        lgbm_kw: dict = dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                              num_leaves=31, min_child_samples=5,
                              objective=obj, verbose=-1)
        if n_classes > 2:
            lgbm_kw["num_class"] = n_classes
        clf_lgbm = lgb.LGBMClassifier(**lgbm_kw)
        clf_lgbm.fit(X_tfidf_tr, y_enc, sample_weight=w_tr)
        p_lgbm = clf_lgbm.predict_proba(X_tfidf_te)
    except Exception as ex:
        logger.debug("LGBM failed: %s", ex)

    if p_lgbm is not None:
        p_avg = (p_xgb + p_lgbm) / 2.0
    else:
        p_avg = p_xgb

    pred_enc = np.argmax(p_avg, axis=1)
    pred_labels = le.inverse_transform(pred_enc)
    acc = float(sum(p == g for p, g in zip(pred_labels, y_te)) / len(y_te))
    return acc


def run_eval(cats: list[str]) -> None:
    gold = pd.read_parquet(GOLD_PATH)
    gold["code"] = gold["code"].astype(str)
    logger.info("Gold: %d rows, %d codes", len(gold), gold["code"].nunique())

    all_rows: list[dict] = []
    total_cost_usd = 0.0

    for cat in cats:
        logger.info("=== Eval category: %s ===", cat)
        silver_path = PROCESSED_PATH / f"{cat}_stratified_silver_standard.parquet"
        emb_path = PROCESSED_PATH / f"{cat}_stratified_embeddings.npy"

        if not silver_path.exists() or not emb_path.exists():
            logger.warning("[%s] Missing silver/emb — skipping eval", cat)
            continue

        silver = pd.read_parquet(silver_path)
        silver["code"] = silver["code"].astype(str)
        emb_all = np.load(emb_path)
        code_to_idx = {c: i for i, c in enumerate(silver["code"].tolist())}

        # Load annotation files
        al_out = AL_GOLD_DIR / f"{cat}_gold.parquet"
        ctrl_out = RAND_GOLD_DIR / f"{cat}_gold.parquet"

        al_wide = pd.read_parquet(al_out) if al_out.exists() else pd.DataFrame()
        ctrl_wide = pd.read_parquet(ctrl_out) if ctrl_out.exists() else pd.DataFrame()

        for df_w in [al_wide, ctrl_wide]:
            if len(df_w) > 0 and "cost_usd" in df_w.columns:
                total_cost_usd += float(df_w["cost_usd"].sum())

        al_long = _annotations_to_long(al_wide, cat) if len(al_wide) > 0 else pd.DataFrame()
        ctrl_long = _annotations_to_long(ctrl_wide, cat) if len(ctrl_wide) > 0 else pd.DataFrame()

        logger.info("[%s] AL long: %d rows, ctrl long: %d rows", cat, len(al_long), len(ctrl_long))

        # Compute embeddings for extra codes (from OFF parquet)
        all_new_codes: list[str] = []
        if len(al_long) > 0:
            all_new_codes += al_long["code"].unique().tolist()
        if len(ctrl_long) > 0:
            all_new_codes += ctrl_long["code"].unique().tolist()
        all_new_codes = list(set(all_new_codes) - set(code_to_idx.keys()))

        extra_emb: Optional[np.ndarray] = None
        extra_code_to_idx: dict[str, int] = {}
        extra_df: Optional[pd.DataFrame] = None

        if all_new_codes:
            logger.info("[%s] Loading %d new code texts from OFF parquet...", cat, len(all_new_codes))
            extra_df = _load_off_products_by_code(all_new_codes)
            if len(extra_df) > 0:
                # Compute embeddings using cached sentence-transformers
                # We'll use the same silver emb approach: TF-IDF only
                # (loading sentence-transformers would require torch — skip embeddings,
                # use text only for LGBM part)
                extra_code_to_idx = {c: i for i, c in enumerate(extra_df["code"].tolist())}
                # Use zeros as dummy embeddings for extra codes (XGB won't have signal but LGBM will)
                extra_emb = np.zeros((len(extra_df), emb_all.shape[1]), dtype=np.float32)
                logger.info("[%s] Extra emb (zeros): shape=%s", cat, extra_emb.shape)

        # 80/20 split on base gold
        cat_gold = gold[gold["category"] == cat].copy()
        unique_codes = sorted(cat_gold["code"].unique().tolist())
        train_codes, test_codes = train_test_split(unique_codes, test_size=TEST_SIZE, random_state=SEED)
        train_set = set(train_codes)
        test_set = set(test_codes)
        logger.info("[%s] Gold split: %d train, %d test", cat, len(train_codes), len(test_codes))

        attrs = sorted(cat_gold["attr"].unique().tolist())

        silver_indexed = silver.groupby("code").first()

        for attr in attrs:
            # Get gold for this attr
            ag = cat_gold[(cat_gold["attr"] == attr) & ~cat_gold["gold_is_null"]].copy()
            ag["code"] = ag["code"].astype(str)
            ag = ag[ag["code"].isin(code_to_idx)]
            train_g = ag[ag["code"].isin(train_set)]
            test_g = ag[ag["code"].isin(test_set)]

            if len(train_g) < 5 or len(test_g) < 3:
                continue

            test_codes_list = test_g["code"].tolist()
            y_te = test_g["gold_value"].astype(str).values.tolist()
            X_te_emb = emb_all[np.array([code_to_idx[c] for c in test_codes_list])]
            te_texts = [
                _build_text(silver_indexed.loc[c]) if c in silver_indexed.index else ""
                for c in test_codes_list
            ]

            # Silver data (exclude test codes)
            sil_attr = None
            if attr in silver.columns:
                sil_attr = silver[silver[attr].notna()].copy()
                sil_attr["code"] = sil_attr["code"].astype(str)
                sil_attr = sil_attr[~sil_attr["code"].isin(test_set)]
                sil_attr = sil_attr[~sil_attr["code"].isin(train_set)]
                sil_attr = sil_attr[sil_attr["code"].isin(code_to_idx)]

            def build_train_inputs(extra_long: Optional[pd.DataFrame]) -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
                """Build combined training data."""
                X_parts = []
                y_parts: list[str] = []
                texts: list[str] = []
                w_parts = []

                # Silver
                if sil_attr is not None and len(sil_attr) > 0:
                    s_idx = np.array([code_to_idx[c] for c in sil_attr["code"]])
                    X_parts.append(emb_all[s_idx])
                    y_parts += sil_attr[attr].astype(str).values.tolist()
                    texts += [
                        _build_text(silver_indexed.loc[c]) if c in silver_indexed.index else ""
                        for c in sil_attr["code"]
                    ]
                    w_parts.append(np.ones(len(sil_attr)))

                # Gold train
                tr_idx = np.array([code_to_idx[c] for c in train_g["code"]])
                X_parts.append(emb_all[tr_idx])
                y_parts += train_g["gold_value"].astype(str).values.tolist()
                texts += [
                    _build_text(silver_indexed.loc[c]) if c in silver_indexed.index else ""
                    for c in train_g["code"]
                ]
                w_parts.append(GOLD_WEIGHT * np.ones(len(train_g)))

                # Extra (AL or random)
                if extra_long is not None and len(extra_long) > 0 and extra_emb is not None:
                    eg = extra_long[(extra_long["attr"] == attr) & ~extra_long["gold_is_null"]].copy()
                    eg["code"] = eg["code"].astype(str)
                    eg = eg[eg["code"].isin(extra_code_to_idx)]
                    if len(eg) > 0:
                        eg_idx = np.array([extra_code_to_idx[c] for c in eg["code"]])
                        X_parts.append(extra_emb[eg_idx])
                        y_parts += eg["gold_value"].astype(str).values.tolist()
                        if extra_df is not None:
                            extra_indexed = extra_df.groupby("code").first()
                            texts += [
                                _build_text(extra_indexed.loc[c]) if c in extra_indexed.index else ""
                                for c in eg["code"]
                            ]
                        else:
                            texts += [""] * len(eg)
                        w_parts.append(GOLD_WEIGHT * np.ones(len(eg)))

                if not X_parts:
                    return np.empty((0, emb_all.shape[1])), [], [], np.array([])
                X = np.vstack(X_parts)
                w = np.concatenate(w_parts)
                return X, y_parts, texts, w

            # Baseline
            X_tr, y_tr, tr_texts, w_tr = build_train_inputs(None)
            if len(set(y_tr)) < 2:
                continue
            acc_base = _train_ensemble(X_tr, y_tr, tr_texts, w_tr, X_te_emb, te_texts, y_te)

            # AL treatment
            X_tr_al, y_tr_al, tr_texts_al, w_tr_al = build_train_inputs(al_long if len(al_long) > 0 else None)
            acc_al = _train_ensemble(X_tr_al, y_tr_al, tr_texts_al, w_tr_al, X_te_emb, te_texts, y_te) if len(set(y_tr_al)) >= 2 else float("nan")

            # Random control
            X_tr_rnd, y_tr_rnd, tr_texts_rnd, w_tr_rnd = build_train_inputs(ctrl_long if len(ctrl_long) > 0 else None)
            acc_rand = _train_ensemble(X_tr_rnd, y_tr_rnd, tr_texts_rnd, w_tr_rnd, X_te_emb, te_texts, y_te) if len(set(y_tr_rnd)) >= 2 else float("nan")

            logger.info("[%s/%s] baseline=%.3f AL=%.3f random=%.3f",
                        cat, attr, acc_base, acc_al, acc_rand)

            all_rows.append({
                "category": cat,
                "attr": attr,
                "baseline_R_ml": acc_base,
                "AL_R_ml": acc_al,
                "random_R_ml": acc_rand,
                "AL_lift_pp": (acc_al - acc_base) * 100 if not (np.isnan(acc_al) or np.isnan(acc_base)) else float("nan"),
                "random_lift_pp": (acc_rand - acc_base) * 100 if not (np.isnan(acc_rand) or np.isnan(acc_base)) else float("nan"),
                "AL_vs_random_pp": (acc_al - acc_rand) * 100 if not (np.isnan(acc_al) or np.isnan(acc_rand)) else float("nan"),
            })

    result_df = pd.DataFrame(all_rows)
    result_df.to_parquet(OUT_PATH, index=False)
    logger.info("Wrote %d rows to %s", len(result_df), OUT_PATH)

    # Summary
    print(f"\n{'='*72}")
    print("P2-EXP11: Active Learning Pilot — Results")
    print(f"{'='*72}")
    print(f"Total annotation cost (cumulative): ${total_cost_usd:.3f}")
    print()

    if len(result_df) == 0:
        print("No results.")
        return

    grand = result_df[["baseline_R_ml", "AL_R_ml", "random_R_ml",
                        "AL_lift_pp", "random_lift_pp", "AL_vs_random_pp"]].mean()
    print(f"  baseline_R_ml     : {grand['baseline_R_ml']*100:.2f}%")
    print(f"  AL_R_ml           : {grand['AL_R_ml']*100:.2f}%  ({grand['AL_lift_pp']:+.2f} pp vs baseline)")
    print(f"  random_R_ml       : {grand['random_R_ml']*100:.2f}%  ({grand['random_lift_pp']:+.2f} pp vs baseline)")
    print(f"  AL vs random      : {grand['AL_vs_random_pp']:+.2f} pp")
    print()
    verdict = (
        "AL WINS" if grand["AL_vs_random_pp"] > 0.5
        else "MARGINAL" if grand["AL_vs_random_pp"] > 0
        else "RANDOM WINS"
    )
    print(f"  VERDICT: {verdict}")
    print()

    print("Per-category:")
    for cat in cats:
        cat_rows = result_df[result_df["category"] == cat]
        if len(cat_rows) == 0:
            continue
        m = cat_rows[["baseline_R_ml", "AL_R_ml", "random_R_ml", "AL_vs_random_pp"]].mean()
        print(f"  [{cat}] base={m['baseline_R_ml']*100:.2f}% AL={m['AL_R_ml']*100:.2f}% "
              f"random={m['random_R_ml']*100:.2f}% AL_vs_rand={m['AL_vs_random_pp']:+.2f}pp")

    print(f"\n{'='*72}")
    print("Per-(category, attr):")
    print(result_df.set_index(["category", "attr"])[
        ["baseline_R_ml", "AL_R_ml", "random_R_ml", "AL_vs_random_pp"]
    ].round(4).to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["annotate", "eval", "all"], default="all")
    ap.add_argument("--cats", nargs="+", default=CATEGORIES)
    args = ap.parse_args()

    if args.phase in ("annotate", "all"):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY not set")
        run_annotate(args.cats, api_key)

    if args.phase in ("eval", "all"):
        run_eval(args.cats)


if __name__ == "__main__":
    main()
