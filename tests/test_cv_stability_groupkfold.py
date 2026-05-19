"""GroupKFold must keep brands disjoint across train/test folds."""
import numpy as np
from src.diagnostics.ml.cv_stability_groupkfold import iter_group_folds


def test_no_brand_overlap_across_folds():
    brands = np.array(["A"] * 10 + ["B"] * 10 + ["C"] * 10 + ["D"] * 10)
    for tr_idx, te_idx in iter_group_folds(brands, n_splits=4, seed=0):
        tr_brands = set(brands[tr_idx])
        te_brands = set(brands[te_idx])
        assert tr_brands.isdisjoint(te_brands), \
            f"brand leak: train={tr_brands} test={te_brands}"


def test_seed_changes_fold_assignment():
    """Different seeds → different fold assignments (else seed dimension is fake)."""
    brands = np.array(["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5 + ["E"] * 5 + ["F"] * 5)
    folds_seed0 = [set(brands[te]) for _, te in iter_group_folds(brands, n_splits=3, seed=0)]
    folds_seed1 = [set(brands[te]) for _, te in iter_group_folds(brands, n_splits=3, seed=1)]
    assert folds_seed0 != folds_seed1, "iter_group_folds ignores seed"
