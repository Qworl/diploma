"""P2-EXP10 Phase 1: Generate CLIP embeddings ONLY (no lgb/xgb to avoid libomp conflict).

Run this first:
    export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=2
    python -m src.experiments.clip_embed_only

Then run Phase 2 (classifier eval, no torch):
    python -m src.experiments.clip_features --phase eval
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WORKTREE_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = WORKTREE_ROOT / "datasets" / "processed"
IMAGES_DIR = WORKTREE_ROOT / "datasets" / "raw" / "off_images"
GOLD_PATH = PROCESSED_DIR / "consensus_gold_v2_expanded.parquet"

CATEGORIES = ["pasta", "chocolate", "cheeses"]


def _embed_batch(img_paths: list[Path], model, processor, device) -> list[tuple[str, "np.ndarray"]]:
    """Embed a batch of images; return (code, 512-vec) pairs for successes."""
    import torch
    from PIL import Image

    results = []
    for img_path in img_paths:
        code = img_path.stem
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.debug("Cannot open %s: %s", img_path, e)
            continue
        try:
            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.get_image_features(**inputs)
            if hasattr(out, "pooler_output"):
                emb = out.pooler_output.cpu().numpy()[0]
            else:
                emb = out.cpu().numpy()[0]
            results.append((code, emb.astype(np.float32)))
        except Exception as e:
            logger.debug("Failed embedding %s: %s", code, e)
    return results


def build_embeddings_for_cat(
    cat: str,
    codes: list[str],
    model,
    processor,
    device,
    force_rebuild: bool = False,
) -> tuple[np.ndarray, dict[str, int]]:
    ep = PROCESSED_DIR / f"clip_embeddings_{cat}.npy"
    ip = PROCESSED_DIR / f"clip_code_index_{cat}.json"

    if not force_rebuild and ep.exists() and ip.exists():
        logger.info("[%s] Loading cached CLIP embeddings from %s", cat, ep)
        emb = np.load(str(ep))
        with open(ip, encoding="utf-8") as f:
            code_to_idx: dict[str, int] = json.load(f)
        logger.info("[%s] Loaded %d embeddings", cat, len(code_to_idx))
        return emb, code_to_idx

    logger.info("[%s] Building CLIP embeddings for %d codes …", cat, len(codes))
    t0 = time.time()
    rows: list[np.ndarray] = []
    code_to_idx = {}
    missing = 0

    BATCH = 32
    for i in range(0, len(codes), BATCH):
        batch_codes = codes[i : i + BATCH]
        batch_paths = []
        for c in batch_codes:
            p = IMAGES_DIR / f"{c}.jpg"
            if p.exists():
                batch_paths.append(p)
            else:
                missing += 1

        results = _embed_batch(batch_paths, model, processor, device)
        for code, emb in results:
            code_to_idx[code] = len(rows)
            rows.append(emb)

        if (i // BATCH) % 10 == 0:
            logger.info(
                "[%s] Progress: %d/%d codes, %d embedded so far",
                cat, min(i + BATCH, len(codes)), len(codes), len(rows),
            )

    if not rows:
        logger.warning("[%s] No CLIP embeddings built — all images missing!", cat)
        return np.empty((0, 512), dtype=np.float32), {}

    emb_array = np.vstack(rows).astype(np.float32)
    np.save(str(ep), emb_array)
    with open(ip, "w", encoding="utf-8") as f:
        json.dump(code_to_idx, f)

    elapsed = time.time() - t0
    logger.info(
        "[%s] Built %d CLIP embeddings (%d missing images) in %.1fs (%.1f img/s)",
        cat, len(rows), missing, elapsed, len(rows) / max(elapsed, 1),
    )
    return emb_array, code_to_idx


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info("Loading CLIP on device=%s …", device)
    t_load = time.time()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    logger.info("CLIP loaded in %.1fs", time.time() - t_load)

    gold = pd.read_parquet(GOLD_PATH)
    gold["code"] = gold["code"].astype(str)

    t_total = time.time()
    stats = {}
    for cat in CATEGORIES:
        cat_gold = gold[gold["category"] == cat]
        gold_codes = sorted(cat_gold["code"].unique().tolist())
        t0 = time.time()
        emb, code_to_idx = build_embeddings_for_cat(cat, gold_codes, model, processor, device)
        elapsed = time.time() - t0
        n_total = len(gold_codes)
        n_embedded = len(code_to_idx)
        coverage = 100.0 * n_embedded / n_total if n_total else 0.0
        stats[cat] = {
            "n_codes": n_total,
            "n_embedded": n_embedded,
            "coverage_pct": round(coverage, 1),
            "time_s": round(elapsed, 1),
        }

    print("\n===== CLIP Embedding Phase Complete =====")
    for cat, s in stats.items():
        print(f"  {cat}: {s['n_embedded']}/{s['n_codes']} ({s['coverage_pct']}%) in {s['time_s']}s")
    print(f"  Total wall clock: {time.time() - t_total:.1f}s")
    print("Ready for Phase 2: python -m src.experiments.clip_features --phase eval")


if __name__ == "__main__":
    main()
