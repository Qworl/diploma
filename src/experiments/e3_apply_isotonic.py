"""E3: apply isotonic regression to 8 ECE-drifted attrs after silver-fix."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from src.common import PROCESSED_DIR, MODELS_DIR, RANDOM_STATE, TEST_SIZE

ATTRS_TO_FIX = [
    ("pasta", "nutri_score_grade"),
    ("chocolate", "cocoa_percentage"),
    ("cheeses", "fat_class"),
    ("cheeses", "milk_source"),
    ("beverages", "nutri_score_grade"),
    ("cosmetics", "product_type"),
    ("cereals", "nova_class"),
    ("pasta", "grain_type"),
]


def compute_ece(probs_max, correct, n_bins=10):
    """Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs_max)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (probs_max >= bin_edges[i]) & (probs_max <= bin_edges[i + 1])
        else:
            mask = (probs_max >= bin_edges[i]) & (probs_max < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = probs_max[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def recalibrate_one(cat, attr):
    silver_path = Path(PROCESSED_DIR) / f"{cat}_stratified_silver_standard.parquet"
    emb_path = Path(PROCESSED_DIR) / f"{cat}_stratified_embeddings.npy"
    xgb_path = Path(MODELS_DIR) / f"{cat}_stratified_{attr}_xgb.pkl"
    le_path = Path(MODELS_DIR) / f"{cat}_stratified_{attr}_le.pkl"

    if not (silver_path.exists() and emb_path.exists() and xgb_path.exists() and le_path.exists()):
        missing = [str(p) for p in [silver_path, emb_path, xgb_path, le_path] if not p.exists()]
        return {"category": cat, "attr": attr, "status": "missing_files", "missing": missing}

    df = pd.read_parquet(silver_path).reset_index(drop=True)
    emb = np.load(emb_path)
    if attr not in df.columns:
        return {"category": cat, "attr": attr, "status": "attr_not_in_silver"}

    with open(xgb_path, "rb") as f:
        clf = pickle.load(f)
    with open(le_path, "rb") as f:
        le = pickle.load(f)

    # Replicate train_test_split from main training (same seed)
    mask_labeled = df[attr].notna()
    idx_labeled = df.index[mask_labeled].to_numpy()
    if len(idx_labeled) < 50:
        return {"category": cat, "attr": attr, "status": "too_few_labeled"}

    train_idx, test_idx = train_test_split(
        idx_labeled, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_test = emb[test_idx]
    y_test_raw = df.loc[test_idx, attr].astype(str).values
    known = np.isin(y_test_raw, le.classes_)
    X_test = X_test[known]
    y_test = le.transform(y_test_raw[known])

    if len(y_test) < 30:
        return {"category": cat, "attr": attr, "status": "too_few_test", "n_test": int(len(y_test))}

    # Get current predict_proba (clf may be a calibrated wrapper already)
    probs = clf.predict_proba(X_test)
    probs_max = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == y_test).astype(int)
    ece_before = compute_ece(probs_max, correct)

    # Fit isotonic on (max-prob, correct) — recalibrate the confidence-of-prediction
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs_max, correct)
    probs_max_calibrated = iso.predict(probs_max)
    ece_after = compute_ece(probs_max_calibrated, correct)

    # Save isotonic params + extend calibration JSON
    iso_path = Path(MODELS_DIR) / f"{cat}_stratified_{attr}_isotonic.pkl"
    with open(iso_path, "wb") as f:
        pickle.dump(iso, f)

    cal_json_path = Path(MODELS_DIR) / f"{cat}_stratified_{attr}_calibration.json"
    if cal_json_path.exists():
        with open(cal_json_path) as f:
            cal_data = json.load(f)
    else:
        cal_data = {}
    cal_data["ece_isotonic"] = float(ece_after)
    cal_data["isotonic_applied"] = True
    cal_data["isotonic_pkl"] = iso_path.name
    cal_data["n_test_used_for_isotonic"] = int(len(y_test))
    with open(cal_json_path, "w") as f:
        json.dump(cal_data, f, indent=2)

    return {
        "category": cat,
        "attr": attr,
        "status": "ok",
        "ece_before": float(ece_before),
        "ece_after_isotonic": float(ece_after),
        "delta_ece": float(ece_after - ece_before),
        "n_test_used": int(len(y_test)),
    }


def main():
    results = []
    for cat, attr in ATTRS_TO_FIX:
        print(f"Processing {cat}/{attr}...")
        r = recalibrate_one(cat, attr)
        print(f"  -> status={r.get('status')}, "
              f"ece_before={r.get('ece_before')}, "
              f"ece_after={r.get('ece_after_isotonic')}")
        results.append(r)

    df = pd.DataFrame(results)
    print("\n=== Summary ===")
    print(df.to_string(index=False))

    n_ok = sum(1 for r in results if r.get("status") == "ok")
    n_improved = sum(1 for r in results
                     if r.get("status") == "ok"
                     and r.get("delta_ece", 0) < 0)

    out_path = Path(PROCESSED_DIR) / "isotonic_recalibration_summary.json"
    with open(out_path, "w") as f:
        json.dump({
            "results": results,
            "n_attrs_processed": len(results),
            "n_attrs_ok": n_ok,
            "n_attrs_improved": n_improved,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"OK: {n_ok}/{len(results)}; ECE improved: {n_improved}/{n_ok}")


if __name__ == "__main__":
    main()
