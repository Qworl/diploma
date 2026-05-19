"""P2-EXP10: CLIP image features as 3rd ensemble component.

Architecture:
  R_ml     = EXP9 ensemble (LightGBM[TF-IDF] + XGB[embeddings]) / 2  -- baseline
  C_clip   = LightGBM on CLIP 512-dim image embeddings per (cat, attr)
  R_clip_ensemble = (R_ml + C_clip) / 2 soft-vote where image available
  R_visual = C_clip on visual attrs only; R_ml elsewhere

Target visual attrs:
  pasta/pasta_shape, cheeses/texture, chocolate/chocolate_type, (any)/is_organic

Pipeline:
  1. Load/cache CLIP embeddings per category.
  2. For each (cat, attr): 80/20 split (seed=42, same as EXP9).
  3. Train LightGBM on CLIP train-set; predict on test.
  4. Build R_ml baseline from cascade_vs_blind_gold_{cat}_v2_ensemble.parquet.
  5. Soft-average where images available.

Output:
  datasets/processed/clip_embeddings_{cat}.npy   — 512-dim CLIP vectors per code
  datasets/processed/clip_code_index_{cat}.json  — code→row index mapping
  datasets/processed/clip_ensemble_eval.parquet  — per-(cat,attr) accuracy
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKTREE_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = WORKTREE_ROOT / "datasets" / "processed"
IMAGES_DIR = WORKTREE_ROOT / "datasets" / "raw" / "off_images"

GOLD_PATH = PROCESSED_DIR / "consensus_gold_v2_expanded.parquet"
OUT_EVAL_PATH = PROCESSED_DIR / "clip_ensemble_eval.parquet"

CATEGORIES = ["pasta", "chocolate", "cheeses"]
RANDOM_STATE = 42
TEST_FRACTION = 0.2
MIN_GOLD = 20  # minimum non-null gold rows to attempt training

VISUAL_ATTRS = {
    "pasta": ["pasta_shape"],
    "chocolate": ["chocolate_type"],
    "cheeses": ["texture"],
}


# ---------------------------------------------------------------------------
# CLIP model (lazy singleton)
# ---------------------------------------------------------------------------
_clip_model = None
_clip_processor = None
_clip_device = None


def _get_clip() -> tuple:
    """Lazy-load CLIP model. Returns (model, processor, device)."""
    global _clip_model, _clip_processor, _clip_device
    if _clip_model is not None:
        return _clip_model, _clip_processor, _clip_device

    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info("Loading CLIP openai/clip-vit-base-patch32 on %s …", device)
    t0 = time.time()
    _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    _clip_model.eval()
    _clip_device = device
    logger.info("CLIP loaded in %.1fs", time.time() - t0)
    return _clip_model, _clip_processor, _clip_device


def _embed_image(img_path: Path) -> Optional[np.ndarray]:
    """Return 512-dim CLIP image embedding or None if image unreadable."""
    import torch
    from PIL import Image

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        logger.debug("Cannot open image %s: %s", img_path, e)
        return None

    model, processor, device = _get_clip()
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.get_image_features(**inputs)
    # In newer transformers, get_image_features returns BaseModelOutputWithPooling
    if hasattr(out, "pooler_output"):
        emb_tensor = out.pooler_output  # (1, 512)
    else:
        emb_tensor = out  # plain tensor in older versions
    return emb_tensor.cpu().numpy()[0].astype(np.float32)


# ---------------------------------------------------------------------------
# CLIP embeddings cache per category
# ---------------------------------------------------------------------------

def _emb_path(cat: str) -> Path:
    return PROCESSED_DIR / f"clip_embeddings_{cat}.npy"


def _idx_path(cat: str) -> Path:
    return PROCESSED_DIR / f"clip_code_index_{cat}.json"


def load_or_build_clip_embeddings(
    cat: str,
    codes: list[str],
    force_rebuild: bool = False,
) -> tuple[np.ndarray, dict[str, int]]:
    """Load cached CLIP embeddings or compute and cache them.

    Returns:
        emb_array: (N, 512) float32 — only codes with valid images
        code_to_idx: code → row index in emb_array
    """
    ep = _emb_path(cat)
    ip = _idx_path(cat)

    if not force_rebuild and ep.exists() and ip.exists():
        logger.info("[%s] Loading cached CLIP embeddings from %s", cat, ep)
        emb = np.load(str(ep))
        with open(ip, encoding="utf-8") as f:
            code_to_idx: dict[str, int] = json.load(f)
        logger.info("[%s] Loaded %d CLIP embeddings", cat, len(code_to_idx))
        return emb, code_to_idx

    logger.info("[%s] Building CLIP embeddings for %d codes …", cat, len(codes))
    t0 = time.time()
    rows: list[np.ndarray] = []
    code_to_idx = {}
    missing = 0

    for code in codes:
        img_path = IMAGES_DIR / f"{code}.jpg"
        if not img_path.exists():
            missing += 1
            continue
        emb = _embed_image(img_path)
        if emb is None:
            missing += 1
            continue
        code_to_idx[code] = len(rows)
        rows.append(emb)

    if not rows:
        logger.warning("[%s] No CLIP embeddings — all images missing!", cat)
        return np.empty((0, 512), dtype=np.float32), {}

    emb_array = np.vstack(rows).astype(np.float32)
    np.save(str(ep), emb_array)
    with open(ip, "w", encoding="utf-8") as f:
        json.dump(code_to_idx, f)

    elapsed = time.time() - t0
    logger.info(
        "[%s] Built %d CLIP embeddings (%d missing) in %.1fs",
        cat, len(rows), missing, elapsed,
    )
    return emb_array, code_to_idx


# ---------------------------------------------------------------------------
# LightGBM trainer on CLIP features
# ---------------------------------------------------------------------------

def _train_lgbm_clip(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[Optional[lgb.LGBMClassifier], Optional[LabelEncoder]]:
    all_classes = sorted(set(y_train.tolist()))
    if len(all_classes) < 2 or len(X_train) < 10:
        return None, None

    le = LabelEncoder()
    le.fit(all_classes)
    y_enc = le.transform(y_train)

    n_classes = len(all_classes)
    if n_classes == 2:
        clf = lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            num_leaves=31, min_child_samples=5, objective="binary", verbose=-1,
        )
    else:
        clf = lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            num_leaves=31, min_child_samples=5,
            objective="multiclass", num_class=n_classes, verbose=-1,
        )

    clf.fit(X_train, y_enc)
    return clf, le


# ---------------------------------------------------------------------------
# Probability alignment helpers
# ---------------------------------------------------------------------------

def _align_probas(
    p1: np.ndarray,
    le1: LabelEncoder,
    p2: np.ndarray,
    le2: LabelEncoder,
) -> tuple[np.ndarray, list[str]]:
    """Average two probability matrices over their union of class labels."""
    classes1 = list(le1.classes_)
    classes2 = list(le2.classes_)
    all_cls = sorted(set(classes1) | set(classes2))
    n = p1.shape[0]
    k = len(all_cls)

    full1 = np.zeros((n, k))
    for j, c in enumerate(classes1):
        full1[:, all_cls.index(c)] = p1[:, j]

    full2 = np.zeros((n, k))
    for j, c in enumerate(classes2):
        full2[:, all_cls.index(c)] = p2[:, j]

    return (0.5 * full1 + 0.5 * full2), all_cls


# ---------------------------------------------------------------------------
# EXP9 R_ml baseline — from cascade ensemble parquets
# ---------------------------------------------------------------------------

def _load_rml_probas(cat: str) -> Optional[pd.DataFrame]:
    """Load EXP9 predictions (cascade_vs_blind_gold_{cat}_v2_ensemble.parquet).

    Returns DataFrame with acc_v2_ensemble per attr, used as reference only.
    Full per-code probabilities are not stored — we retrain R_ml fresh below.
    """
    p = PROCESSED_DIR / f"cascade_vs_blind_gold_{cat}_v2_ensemble.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


# ---------------------------------------------------------------------------
# R_ml retrainer (XGB[emb] + LightGBM[TF-IDF] fresh 80/20 — mirrors EXP9)
# ---------------------------------------------------------------------------

def _retrain_rml_attr(
    cat: str,
    attr: str,
    gold_long: pd.DataFrame,
    silver: pd.DataFrame,
    silver_emb: np.ndarray,
    silver_code_to_idx: dict[str, int],
    train_codes_set: set[str],
    test_codes_set: set[str],
) -> tuple[Optional[np.ndarray], Optional[list[str]]]:
    """Retrain EXP9-style R_ml and return (probas_test, classes) or (None, None)."""
    import xgboost as xgb
    from sklearn.feature_extraction.text import TfidfVectorizer

    # ---- XGB on silver embeddings ----
    gold_attr_long = gold_long[
        (gold_long["category"] == cat)
        & (gold_long["attr"] == attr)
        & ~gold_long["gold_is_null"]
    ].copy()
    gold_attr_long["code"] = gold_attr_long["code"].astype(str)

    # Silver rows (not in test, not in gold train set)
    if attr in silver.columns:
        silver_attr = silver[silver[attr].notna()].copy()
        silver_attr["code"] = silver_attr["code"].astype(str)
        silver_attr = silver_attr[
            ~silver_attr["code"].isin(test_codes_set)
            & ~silver_attr["code"].isin(train_codes_set)
            & silver_attr["code"].isin(silver_code_to_idx)
        ]
        X_s = silver_emb[[silver_code_to_idx[c] for c in silver_attr["code"]]]
        y_s = silver_attr[attr].astype(str).values
    else:
        X_s = np.empty((0, silver_emb.shape[1]))
        y_s = np.array([], dtype=str)

    gold_train = gold_attr_long[gold_attr_long["code"].isin(train_codes_set)]
    gold_train = gold_train[gold_train["code"].isin(silver_code_to_idx)]
    if len(gold_train) == 0:
        return None, None

    X_g = silver_emb[[silver_code_to_idx[c] for c in gold_train["code"]]]
    y_g = gold_train["gold_value"].astype(str).values

    if len(X_s) > 0:
        X_combo = np.vstack([X_s, X_g])
        y_combo = np.concatenate([y_s, y_g])
        w_combo = np.concatenate([np.ones(len(y_s)), 5.0 * np.ones(len(y_g))])
    else:
        X_combo = X_g
        y_combo = y_g
        w_combo = 5.0 * np.ones(len(y_g))

    all_cls = sorted(set(y_combo.tolist()))
    if len(all_cls) < 2:
        return None, None

    le_xgb = LabelEncoder()
    le_xgb.fit(all_cls)
    y_enc = le_xgb.transform(y_combo)

    n_cls = len(all_cls)
    common = dict(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
        tree_method="hist", verbosity=0,
    )
    if n_cls == 2:
        pos = int((y_enc == 1).sum()); neg = int((y_enc == 0).sum())
        spw = max(neg / max(pos, 1), 0.5)
        clf_xgb = xgb.XGBClassifier(scale_pos_weight=spw, **common)
    else:
        clf_xgb = xgb.XGBClassifier(objective="multi:softmax", num_class=n_cls, **common)
    clf_xgb.fit(X_combo, y_enc, sample_weight=w_combo)

    # ---- Test embeddings ----
    gold_test = gold_attr_long[gold_attr_long["code"].isin(test_codes_set)]
    gold_test = gold_test[gold_test["code"].isin(silver_code_to_idx)]
    if len(gold_test) == 0:
        return None, None

    test_idx = [silver_code_to_idx[c] for c in gold_test["code"]]
    X_te_emb = silver_emb[test_idx]
    probas_xgb = clf_xgb.predict_proba(X_te_emb)

    # ---- LightGBM on TF-IDF ----
    def _build_text(row: pd.Series) -> str:
        parts = []
        for col in ["product_name", "ingredients_text", "brands"]:
            v = row.get(col)
            if pd.notna(v) and str(v).strip():
                parts.append(str(v).strip())
        return " ".join(parts)

    gold_train_texts_df = gold_train.merge(
        silver[["code", "product_name", "ingredients_text", "brands"]],
        on="code", how="left",
    )
    train_texts = [_build_text(r) for _, r in gold_train_texts_df.iterrows()]
    y_train_cls = gold_train["gold_value"].astype(str).tolist()

    silver_idx_map = silver.set_index("code")
    test_texts = []
    for c in gold_test["code"]:
        if c in silver_idx_map.index:
            test_texts.append(_build_text(silver_idx_map.loc[c]))
        else:
            test_texts.append("")

    from sklearn.feature_extraction.text import TfidfVectorizer
    if len(set(y_train_cls)) < 2 or len(train_texts) < MIN_GOLD:
        # LightGBM can't train — use XGB only
        all_cls_str = list(le_xgb.classes_)
        return probas_xgb, all_cls_str

    le_lgbm = LabelEncoder()
    le_lgbm.fit(sorted(set(y_train_cls)))
    y_lgbm_enc = le_lgbm.transform(y_train_cls)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=10000)
    X_tfidf_tr = vec.fit_transform(train_texts)
    X_tfidf_te = vec.transform(test_texts)

    n_lgbm_cls = len(le_lgbm.classes_)
    if n_lgbm_cls == 2:
        lgbm_clf = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=5, objective="binary", verbose=-1,
        )
    else:
        lgbm_clf = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=5,
            objective="multiclass", num_class=n_lgbm_cls, verbose=-1,
        )
    lgbm_clf.fit(X_tfidf_tr, y_lgbm_enc)
    probas_lgbm = lgbm_clf.predict_proba(X_tfidf_te)

    # Align and average
    avg_probas, merged_cls = _align_probas(probas_lgbm, le_lgbm, probas_xgb, le_xgb)
    return avg_probas, merged_cls


# ---------------------------------------------------------------------------
# Per-(cat, attr) evaluation
# ---------------------------------------------------------------------------

def run_one_attr(
    cat: str,
    attr: str,
    gold_long: pd.DataFrame,
    silver: pd.DataFrame,
    silver_emb: np.ndarray,
    silver_code_to_idx: dict[str, int],
    clip_emb: np.ndarray,
    clip_code_to_idx: dict[str, int],
) -> list[dict]:
    """Evaluate 4 variants for one (cat, attr).

    Returns list of row dicts: category, attr, variant, accuracy, n_test,
    n_with_image, pct_with_image.
    """
    attr_gold = gold_long[
        (gold_long["category"] == cat)
        & (gold_long["attr"] == attr)
        & ~gold_long["gold_is_null"]
    ].copy()
    attr_gold["code"] = attr_gold["code"].astype(str)

    # Only keep codes that have silver embeddings (same filter as EXP9)
    attr_gold = attr_gold[attr_gold["code"].isin(silver_code_to_idx)]

    if len(attr_gold) < MIN_GOLD:
        logger.info("[%s/%s] only %d gold rows — skipping", cat, attr, len(attr_gold))
        return []

    # ---- Consistent 80/20 split (same seed=42 as EXP9) ----
    all_codes = attr_gold["code"].tolist()
    train_codes, test_codes = train_test_split(
        all_codes, test_size=TEST_FRACTION, random_state=RANDOM_STATE
    )
    train_codes_set = set(train_codes)
    test_codes_set = set(test_codes)

    test_gold = attr_gold[attr_gold["code"].isin(test_codes_set)].copy()
    y_te = test_gold["gold_value"].astype(str).values
    test_code_list = test_gold["code"].tolist()
    n_test = len(test_gold)

    # ---- Codes with CLIP embeddings in test set ----
    test_with_clip = [c for c in test_code_list if c in clip_code_to_idx]
    n_with_image = len(test_with_clip)
    pct_with_image = 100.0 * n_with_image / n_test if n_test > 0 else 0.0

    rows: list[dict] = []
    base = {
        "category": cat,
        "attr": attr,
        "n_test": n_test,
        "n_with_image": n_with_image,
        "pct_with_image": round(pct_with_image, 1),
        "is_visual_attr": attr in VISUAL_ATTRS.get(cat, []),
    }

    # ---- Variant 1: R_ml (EXP9 baseline retrained fresh) ----
    rml_probas, rml_classes = _retrain_rml_attr(
        cat, attr, gold_long, silver,
        silver_emb, silver_code_to_idx,
        train_codes_set, test_codes_set,
    )
    if rml_probas is None:
        logger.info("[%s/%s] R_ml training failed — skipping all variants", cat, attr)
        return []

    rml_pred = [rml_classes[i] for i in np.argmax(rml_probas, axis=1)]
    # rml_probas are in order of test_code_list (only codes in silver_code_to_idx)
    # We need to re-derive the test gold in that order
    test_gold_ordered = test_gold.set_index("code").loc[
        [c for c in test_code_list if c in silver_code_to_idx]
    ]
    y_te_for_rml = test_gold_ordered["gold_value"].astype(str).values

    rml_acc = float(accuracy_score(y_te_for_rml, rml_pred))
    rows.append({**base, "variant": "R_ml", "accuracy": rml_acc})
    logger.info("[%s/%s] R_ml acc=%.3f (n=%d)", cat, attr, rml_acc, len(y_te_for_rml))

    # ---- Build CLIP train/test sets ----
    # Only codes that have both gold label AND CLIP embedding
    train_gold_clip = attr_gold[
        attr_gold["code"].isin(train_codes_set) & attr_gold["code"].isin(clip_code_to_idx)
    ]
    test_gold_clip = attr_gold[
        attr_gold["code"].isin(test_codes_set) & attr_gold["code"].isin(clip_code_to_idx)
    ]

    if len(train_gold_clip) < 10 or len(test_gold_clip) < 5:
        logger.info(
            "[%s/%s] Not enough CLIP-covered gold: train=%d test=%d — skipping CLIP variants",
            cat, attr, len(train_gold_clip), len(test_gold_clip),
        )
        return rows  # return R_ml only

    X_clip_train = clip_emb[[clip_code_to_idx[c] for c in train_gold_clip["code"]]]
    y_clip_train = train_gold_clip["gold_value"].astype(str).values

    X_clip_test = clip_emb[[clip_code_to_idx[c] for c in test_gold_clip["code"]]]
    y_clip_test = test_gold_clip["gold_value"].astype(str).values

    # ---- Variant 2: C_clip (CLIP-only LightGBM) ----
    clf_clip, le_clip = _train_lgbm_clip(X_clip_train, y_clip_train)
    if clf_clip is None or le_clip is None:
        logger.info("[%s/%s] C_clip training failed (degenerate)", cat, attr)
        return rows

    clip_probas_test = clf_clip.predict_proba(X_clip_test)
    clip_pred = le_clip.inverse_transform(np.argmax(clip_probas_test, axis=1))
    clip_acc = float(accuracy_score(y_clip_test, clip_pred))
    n_clip_test = len(test_gold_clip)
    rows.append({
        **base,
        "variant": "C_clip",
        "accuracy": clip_acc,
        "n_test": n_clip_test,  # only on codes with images
    })
    logger.info("[%s/%s] C_clip acc=%.3f (n=%d)", cat, attr, clip_acc, n_clip_test)

    # ---- Variant 3: R_clip_ensemble (R_ml + C_clip avg where image available) ----
    # For codes with images: build R_ml probas subset and avg with C_clip
    # rml_probas is aligned to test_gold_ordered (silver-covered codes in test set)
    test_rml_codes = [c for c in test_code_list if c in silver_code_to_idx]
    rml_code_to_row: dict[str, int] = {c: i for i, c in enumerate(test_rml_codes)}

    ensemble_preds: list[str] = []
    ensemble_true: list[str] = []

    for code, true_label in zip(test_rml_codes, y_te_for_rml):
        if code in clip_code_to_idx:
            # Both modalities available — average probas
            clip_row_idx = clip_code_to_idx[code]
            # We need to predict on this single test image
            X_single = clip_emb[clip_row_idx : clip_row_idx + 1]
            clip_p_single = clf_clip.predict_proba(X_single)
            rml_row = rml_code_to_row[code]
            rml_p_single = rml_probas[rml_row : rml_row + 1]
            avg_p, merged_cls = _align_probas(clip_p_single, le_clip, rml_p_single, rml_classes_le(rml_classes))
            pred = merged_cls[int(np.argmax(avg_p[0]))]
        else:
            # No image — use R_ml only
            rml_row = rml_code_to_row[code]
            pred = rml_pred[rml_row]
        ensemble_preds.append(pred)
        ensemble_true.append(true_label)

    ens_acc = float(accuracy_score(ensemble_true, ensemble_preds))
    rows.append({**base, "variant": "R_clip_ensemble", "accuracy": ens_acc, "n_test": len(ensemble_true)})
    logger.info("[%s/%s] R_clip_ensemble acc=%.3f (n=%d)", cat, attr, ens_acc, len(ensemble_true))

    # ---- Variant 4: Visual-specialized ----
    # Use C_clip only for visual attrs; R_ml elsewhere (on CLIP-covered test subset)
    if attr in VISUAL_ATTRS.get(cat, []):
        # Use C_clip predictions on clip-test subset
        visual_acc = clip_acc
        visual_n = n_clip_test
        visual_note = "clip_only"
    else:
        # Use R_ml on the standard test set
        visual_acc = rml_acc
        visual_n = len(y_te_for_rml)
        visual_note = "rml_only"
    rows.append({
        **base,
        "variant": "visual_specialized",
        "accuracy": visual_acc,
        "n_test": visual_n,
        "visual_note": visual_note,
    })
    logger.info("[%s/%s] visual_specialized acc=%.3f (n=%d) [%s]",
                cat, attr, visual_acc, visual_n, visual_note)

    return rows


def rml_classes_le(classes: list[str]) -> LabelEncoder:
    """Wrap a list of class strings into a LabelEncoder for _align_probas."""
    le = LabelEncoder()
    le.fit(classes)
    return le


# ---------------------------------------------------------------------------
# CLIP embedding loader (Phase 2: load pre-built .npy files, no torch)
# ---------------------------------------------------------------------------

def load_clip_embeddings_cached(cat: str) -> tuple[np.ndarray, dict[str, int]]:
    """Load pre-built CLIP embeddings from Phase 1. Raises if not found."""
    ep = _emb_path(cat)
    ip = _idx_path(cat)
    if not ep.exists() or not ip.exists():
        raise FileNotFoundError(
            f"CLIP embeddings not found for {cat}. "
            f"Run Phase 1 first: python -m src.experiments.clip_embed_only"
        )
    emb = np.load(str(ep))
    with open(ip, encoding="utf-8") as f:
        code_to_idx: dict[str, int] = json.load(f)
    logger.info("[%s] Loaded cached CLIP embeddings: %d rows, %d codes", cat, len(emb), len(code_to_idx))
    return emb, code_to_idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="P2-EXP10 CLIP ensemble evaluation")
    parser.add_argument(
        "--phase",
        choices=["all", "embed", "eval"],
        default="all",
        help=(
            "all = embed + eval in one process (may segfault on macOS). "
            "embed = Phase 1 only (calls clip_embed_only). "
            "eval = Phase 2 only (loads pre-built embeddings, no torch)."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.phase == "embed":
        import subprocess, sys
        logger.info("Delegating to clip_embed_only for Phase 1 …")
        ret = subprocess.run(
            [sys.executable, "-m", "src.experiments.clip_embed_only"],
            check=False,
        )
        if ret.returncode != 0:
            logger.error("clip_embed_only failed with code %d", ret.returncode)
        return

    import pandas as pd

    gold = pd.read_parquet(GOLD_PATH)
    gold["code"] = gold["code"].astype(str)
    logger.info("Gold dataset: %d rows, %d unique codes", len(gold), gold["code"].nunique())

    all_rows: list[dict] = []
    clip_stats: dict[str, dict] = {}

    for cat in CATEGORIES:
        logger.info("=== Category: %s ===", cat)

        # Silver data + embeddings
        silver = pd.read_parquet(PROCESSED_DIR / f"{cat}_stratified_silver_standard.parquet")
        silver["code"] = silver["code"].astype(str)
        silver_emb = np.load(PROCESSED_DIR / f"{cat}_stratified_embeddings.npy")
        silver_code_to_idx: dict[str, int] = {
            c: i for i, c in enumerate(silver["code"].tolist())
        }

        # Gold codes for this category
        cat_gold = gold[gold["category"] == cat]
        gold_codes = sorted(cat_gold["code"].unique().tolist())
        logger.info("  %d unique gold codes", len(gold_codes))

        # CLIP embeddings for this category
        t0 = time.time()
        if args.phase == "eval":
            # Phase 2: load pre-built embeddings (no torch in this process)
            clip_emb, clip_code_to_idx = load_clip_embeddings_cached(cat)
        else:
            clip_emb, clip_code_to_idx = load_or_build_clip_embeddings(cat, gold_codes)
        clip_elapsed = time.time() - t0
        n_with_image = len(clip_code_to_idx)
        coverage = 100.0 * n_with_image / len(gold_codes) if gold_codes else 0.0
        clip_stats[cat] = {
            "n_codes": len(gold_codes),
            "n_with_image": n_with_image,
            "coverage_pct": round(coverage, 1),
            "embed_time_s": round(clip_elapsed, 1),
        }
        logger.info(
            "  CLIP coverage: %d/%d (%.1f%%) in %.1fs",
            n_with_image, len(gold_codes), coverage, clip_elapsed,
        )

        # Per-attr evaluation
        attrs = sorted(cat_gold["attr"].unique().tolist())
        logger.info("  Attrs: %s", attrs)

        for attr in attrs:
            attr_rows = run_one_attr(
                cat=cat,
                attr=attr,
                gold_long=gold,
                silver=silver,
                silver_emb=silver_emb,
                silver_code_to_idx=silver_code_to_idx,
                clip_emb=clip_emb,
                clip_code_to_idx=clip_code_to_idx,
            )
            all_rows.extend(attr_rows)

    # ---- Save results ----
    result_df = pd.DataFrame(all_rows)
    result_df.to_parquet(OUT_EVAL_PATH, index=False)
    logger.info("Saved eval results to %s", OUT_EVAL_PATH)

    # ---- Summary ----
    print("\n" + "=" * 75)
    print("P2-EXP10: CLIP IMAGE FEATURES — SUMMARY")
    print("=" * 75)

    for cat in CATEGORIES:
        stats = clip_stats.get(cat, {})
        print(f"\n[{cat.upper()}]  CLIP coverage: "
              f"{stats.get('n_with_image','?')}/{stats.get('n_codes','?')} "
              f"({stats.get('coverage_pct','?')}%)")

        cat_df = result_df[result_df["category"] == cat]
        if cat_df.empty:
            continue

        variants = ["R_ml", "C_clip", "R_clip_ensemble", "visual_specialized"]
        for v in variants:
            sub = cat_df[cat_df["variant"] == v]
            if sub.empty:
                continue
            # weight by n_test
            total_n = sub["n_test"].sum()
            w_acc = (sub["accuracy"] * sub["n_test"]).sum() / total_n if total_n > 0 else float("nan")
            print(f"  {v:<22}: weighted_acc={w_acc:.4f}  n_attrs={len(sub)}")

    print("\n--- Visual attr deep-dive ---")
    visual_pairs = [(cat, attr) for cat, attrs in VISUAL_ATTRS.items() for attr in attrs]
    for cat, attr in visual_pairs:
        sub = result_df[(result_df["category"] == cat) & (result_df["attr"] == attr)]
        if sub.empty:
            print(f"  {cat}/{attr}: no data")
            continue
        for _, row in sub.iterrows():
            print(f"  {cat}/{attr} | {row['variant']:<22}: acc={row['accuracy']:.3f}  n={row['n_test']}")

    print("\n--- Overall (all cats, all attrs) ---")
    for v in ["R_ml", "C_clip", "R_clip_ensemble", "visual_specialized"]:
        sub = result_df[result_df["variant"] == v]
        if sub.empty:
            continue
        total_n = sub["n_test"].sum()
        w_acc = (sub["accuracy"] * sub["n_test"]).sum() / total_n if total_n > 0 else float("nan")
        print(f"  {v:<22}: weighted_acc={w_acc:.4f}  n_attr_rows={len(sub)}")

    print("\nCLIP embedding stats:")
    for cat, s in clip_stats.items():
        print(f"  {cat}: {s}")

    print(f"\nEval saved: {OUT_EVAL_PATH}")


if __name__ == "__main__":
    main()
