"""Split pasta_gold_250 cells into 80% train_audit / 20% holdout_audit, brand-disjoint."""
import csv
import json
from pathlib import Path

import numpy as np

from src.common import PROCESSED_DIR

GOLD_CSV = Path("datasets/manual_label/pasta_gold_250.csv")
SEED = 42

with GOLD_CSV.open() as f:
    rows = list(csv.DictReader(f))

# Group products by brand
brand_to_codes = {}
for r in rows:
    brand = (r.get("brands") or "").strip()
    code = r["code"]
    brand_to_codes.setdefault(brand, []).append(code)

brands = sorted(brand_to_codes.keys())
rng = np.random.default_rng(SEED)
rng.shuffle(brands)

total_products = len(rows)
target_holdout = int(0.2 * total_products)

holdout_codes = set()
train_codes = set()
for brand in brands:
    if len(holdout_codes) < target_holdout:
        holdout_codes.update(brand_to_codes[brand])
    else:
        train_codes.update(brand_to_codes[brand])

print(f"Total products: {total_products}")
print(f"Train_audit: {len(train_codes)} products")
print(f"Holdout_audit: {len(holdout_codes)} products")
overlap = set(train_codes) & set(holdout_codes)
print(f"Overlap (must be 0): {len(overlap)}")
assert len(overlap) == 0, f"Brand-disjoint split failed: {len(overlap)} overlap"

out_train = Path(PROCESSED_DIR) / "e2_train_audit_codes.json"
out_holdout = Path(PROCESSED_DIR) / "e2_holdout_audit_codes.json"
with open(out_train, "w") as f:
    json.dump(sorted(train_codes), f)
with open(out_holdout, "w") as f:
    json.dump(sorted(holdout_codes), f)
print(f"Split saved: {out_train.name} + {out_holdout.name}")
