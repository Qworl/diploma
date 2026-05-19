"""Tests for Layer 0 — category router."""
from __future__ import annotations


def test_constants_exposed():
    from src.pipeline.category_router.constants import (
        ROUTER_CLASSES,
        DEMO_SUPPORTED_CATEGORIES,
        ROUTER_INPUT_FIELDS,
        ARTIFACT_XGB,
        ARTIFACT_LE,
        ARTIFACT_THRESHOLD,
        ARTIFACT_META,
        ARTIFACT_LOCO,
        TRAIN_PARQUET,
        TEST_PARQUET,
        EMBEDDINGS_NPY,
    )
    # 7 known классов в детерминированном порядке.
    assert ROUTER_CLASSES == (
        "pasta", "chocolate", "beverages",
        "cheeses", "cereals", "cosmetics", "electronics",
    )
    assert DEMO_SUPPORTED_CATEGORIES == frozenset(
        {"pasta", "chocolate", "cheeses"}
    )
    assert ROUTER_INPUT_FIELDS == (
        "product_name", "brands", "ingredients_text", "quantity",
    )
    assert ARTIFACT_XGB.endswith("category_router_xgb.pkl")
    assert ARTIFACT_LOCO.endswith("category_router_loco.parquet")
    assert TRAIN_PARQUET.endswith("category_router_train.parquet")


import pandas as pd
import pytest


@pytest.fixture
def fake_stratified_dir(tmp_path):
    """Creates fake {cat}_stratified_raw.parquet for 6 food/cosmetics
    and electronics_silver_standard.parquet."""
    from src.pipeline.category_router.constants import ROUTER_CLASSES
    for cat in ROUTER_CLASSES:
        if cat == "electronics":
            path = tmp_path / "electronics_silver_standard.parquet"
        else:
            path = tmp_path / f"{cat}_stratified_raw.parquet"
        df = pd.DataFrame({
            "product_name": [f"{cat} prod {i}" for i in range(50)],
            "brands":      [f"brand-{cat}-{i % 5}" for i in range(50)],
            "ingredients_text": [""] * 50,
            "quantity":    ["100g"] * 50,
        })
        df.to_parquet(path)
    return tmp_path


def test_load_positive_balances_classes(fake_stratified_dir):
    from src.pipeline.category_router.data import load_positive
    df = load_positive(
        processed_dir=str(fake_stratified_dir),
        n_per_class=30,
        seed=42,
    )
    assert len(df) == 7 * 30
    counts = df["category_label"].value_counts()
    assert set(counts.index) == set([
        "pasta", "chocolate", "beverages",
        "cheeses", "cereals", "cosmetics", "electronics",
    ])
    assert (counts == 30).all()
    assert list(df.columns) == [
        "product_name", "brands", "ingredients_text", "quantity",
        "category_label", "brand",
    ]


def test_load_positive_truncates_when_class_smaller(fake_stratified_dir):
    """Если N_per_class < n_per_class, берём всё что есть."""
    from src.pipeline.category_router.data import load_positive
    df = load_positive(
        processed_dir=str(fake_stratified_dir),
        n_per_class=200,
        seed=42,
    )
    assert len(df) == 7 * 50


def test_load_positive_balances_to_global_min(tmp_path):
    """When classes have different sizes, balance to the smallest."""
    from src.pipeline.category_router.constants import ROUTER_CLASSES
    from src.pipeline.category_router.data import load_positive

    # cosmetics is smallest at 30 rows; others have 100. Electronics has 5000.
    sizes = {"pasta": 100, "chocolate": 100, "beverages": 100,
             "cheeses": 100, "cereals": 100, "cosmetics": 30,
             "electronics": 5000}
    for cat, n in sizes.items():
        if cat == "electronics":
            path = tmp_path / "electronics_silver_standard.parquet"
        else:
            path = tmp_path / f"{cat}_stratified_raw.parquet"
        df = pd.DataFrame({
            "product_name": [f"{cat} {i}" for i in range(n)],
            "brands": [""] * n,
            "ingredients_text": [""] * n,
            "quantity": ["1"] * n,
        })
        df.to_parquet(path)

    df_out = load_positive(processed_dir=str(tmp_path), n_per_class=1000, seed=42)
    # 7 classes × 30 = 210 (balanced to smallest = cosmetics).
    assert len(df_out) == 7 * 30
    counts = df_out["category_label"].value_counts()
    assert (counts == 30).all(), counts


def test_load_positive_deterministic(fake_stratified_dir):
    from src.pipeline.category_router.data import load_positive
    df1 = load_positive(processed_dir=str(fake_stratified_dir), n_per_class=20, seed=42)
    df2 = load_positive(processed_dir=str(fake_stratified_dir), n_per_class=20, seed=42)
    pd.testing.assert_frame_equal(df1.reset_index(drop=True),
                                  df2.reset_index(drop=True))


@pytest.fixture
def fake_off_parquet(tmp_path):
    """Synthetic OFF: 100 rows. Half tagged pasta/chocolate, half tagged
    en:snacks/en:dairies — should appear in OOD sample."""
    rows = []
    for i in range(50):
        rows.append({
            "product_name": f"pasta prod {i}",
            "brands": "BarillaSynth",
            "ingredients_text": "",
            "quantity": "500g",
            "categories_tags": "en:pastas,en:dried-pastas",
        })
    for i in range(50):
        rows.append({
            "product_name": f"snack prod {i}",
            "brands": "SnackBrand",
            "ingredients_text": "",
            "quantity": "200g",
            "categories_tags": "en:snacks,en:salty-snacks",
        })
    df = pd.DataFrame(rows)
    path = tmp_path / "en.openfoodfacts.org.products.parquet"
    df.to_parquet(path)
    return path


def test_sample_ood_excludes_known_tags(fake_off_parquet):
    from src.pipeline.category_router.ood_sampler import sample_ood
    df = sample_ood(off_parquet=str(fake_off_parquet), n=30, seed=42)
    assert len(df) == 30
    forbidden = {"en:pastas", "en:dried-pastas"}
    for tags in df["categories_tags_raw"]:
        assert not (set(tags.split(",")) & forbidden), tags
    assert list(df.columns) == [
        "product_name", "brands", "ingredients_text", "quantity",
        "category_label", "brand", "categories_tags_raw",
    ]
    assert (df["category_label"] == "unknown").all()


def test_sample_ood_smaller_than_requested(fake_off_parquet):
    """When fewer OOD rows exist than requested, return what we have."""
    from src.pipeline.category_router.ood_sampler import sample_ood
    df = sample_ood(off_parquet=str(fake_off_parquet), n=1000, seed=42)
    assert len(df) == 50


def test_brand_disjoint_split_no_brand_leakage():
    from src.pipeline.category_router.split import brand_disjoint_split
    df = pd.DataFrame({
        "product_name": [f"p{i}" for i in range(60)],
        "brands":       ["",] * 60,
        "ingredients_text": [""] * 60,
        "quantity":     ["100g"] * 60,
        "category_label": (["pasta"] * 30 + ["chocolate"] * 30),
        "brand": ([f"brand-{i % 5}" for i in range(30)]
                  + [f"brand-{i % 7}" for i in range(30)]),
    })
    train, test = brand_disjoint_split(df, test_size=0.2, seed=42)
    assert 0.7 < len(train) / (len(train) + len(test)) < 0.9
    train_brands = set(train["brand"].unique()) - {""}
    test_brands = set(test["brand"].unique()) - {""}
    assert train_brands.isdisjoint(test_brands), \
        f"brand leakage: {train_brands & test_brands}"


def test_brand_disjoint_split_empty_brand_random():
    """Товары без бренда распределяются независимым random 80/20."""
    from src.pipeline.category_router.split import brand_disjoint_split
    df = pd.DataFrame({
        "product_name": [f"p{i}" for i in range(100)],
        "brands": [""] * 100,
        "ingredients_text": [""] * 100,
        "quantity": ["100g"] * 100,
        "category_label": ["unknown"] * 100,
        "brand": [""] * 100,
    })
    train, test = brand_disjoint_split(df, test_size=0.2, seed=42)
    assert 15 <= len(test) <= 25
    assert len(train) + len(test) == 100


def test_category_router_predict_with_fake_artefacts(tmp_path):
    """Сохраняем мини-модель в tmp_path/models, грузим, проверяем shape."""
    import json
    import pickle

    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier

    from src.pipeline.category_router.constants import ROUTER_CLASSES
    from src.pipeline.category_router.infer import CategoryRouter

    class FakeEmbedder:
        def encode(self, texts, show_progress_bar=False):
            out = []
            for t in texts:
                rng = np.random.default_rng(abs(hash(t)) % (2**32))
                out.append(rng.standard_normal(384))
            return np.array(out)

    rng = np.random.default_rng(0)
    X = rng.standard_normal((140, 384))
    y_str = np.array([ROUTER_CLASSES[i % 7] for i in range(140)])
    le = LabelEncoder().fit(list(ROUTER_CLASSES))
    y = le.transform(y_str)
    clf = XGBClassifier(n_estimators=10, max_depth=2, eval_metric="mlogloss")
    clf.fit(X, y)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    with open(models_dir / "category_router_xgb.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(models_dir / "category_router_le.pkl", "wb") as f:
        pickle.dump(le, f)
    with open(models_dir / "category_router_threshold.json", "w") as f:
        json.dump({"threshold": 0.5}, f)

    router = CategoryRouter.load(
        models_dir=str(models_dir), embedder=FakeEmbedder(),
    )
    out = router.predict({
        "product_name": "spaghetti barilla",
        "brands": "Barilla",
        "ingredients_text": "",
        "quantity": "500g",
    })
    assert set(out.keys()) == {"predicted", "confidence", "alternatives", "is_ood"}
    assert out["predicted"] in set(ROUTER_CLASSES) | {"unknown"}
    assert 0.0 <= out["confidence"] <= 1.0
    assert len(out["alternatives"]) == 3
    assert isinstance(out["is_ood"], bool)


def test_train_router_end_to_end(tmp_path, fake_stratified_dir, fake_off_parquet):
    """Run full train pipeline with FakeEmbedder; verify artefacts on disk."""
    import json

    import numpy as np

    from src.pipeline.category_router.train import train_router

    class FakeEmbedder:
        def encode(self, texts, show_progress_bar=False, batch_size=None):
            out = []
            for t in texts:
                rng = np.random.default_rng(abs(hash(t)) % (2**32))
                out.append(rng.standard_normal(384))
            return np.array(out)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    train_parquet = tmp_path / "train.parquet"
    test_parquet = tmp_path / "test.parquet"
    emb_npy = tmp_path / "emb.npy"

    meta = train_router(
        processed_dir=str(fake_stratified_dir),
        off_parquet=str(fake_off_parquet),
        embedder=FakeEmbedder(),
        n_per_class=30,
        ood_fraction=1.0,
        seed=42,
        models_dir=str(models_dir),
        train_parquet=str(train_parquet),
        test_parquet=str(test_parquet),
        embeddings_npy=str(emb_npy),
        target_fpr_known=0.05,
    )
    assert (models_dir / "category_router_xgb.pkl").exists()
    assert (models_dir / "category_router_le.pkl").exists()
    assert (models_dir / "category_router_threshold.json").exists()
    assert (models_dir / "category_router_meta.json").exists()
    assert train_parquet.exists()
    assert test_parquet.exists()
    import pandas as pd
    train_loaded = pd.read_parquet(train_parquet)
    assert "is_ood" in train_loaded.columns
    test_loaded = pd.read_parquet(test_parquet)
    assert "is_ood" in test_loaded.columns
    assert emb_npy.exists()
    assert "test_accuracy" in meta
    assert "test_f1_macro" in meta
    assert "ood_auroc" in meta
    assert "threshold" in meta
    thr_data = json.loads((models_dir / "category_router_threshold.json").read_text())
    assert 0.0 <= thr_data["threshold"] <= 1.0
    assert "fpr_on_known" in thr_data


def test_category_router_marks_low_confidence_as_ood(tmp_path):
    import json
    import pickle

    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier

    from src.pipeline.category_router.constants import ROUTER_CLASSES
    from src.pipeline.category_router.infer import CategoryRouter

    class FakeEmbedder:
        def encode(self, texts, show_progress_bar=False):
            return np.zeros((len(texts), 384))

    rng = np.random.default_rng(1)
    X = rng.standard_normal((140, 384))
    y = np.array([i % 7 for i in range(140)])
    le = LabelEncoder().fit(list(ROUTER_CLASSES))
    clf = XGBClassifier(n_estimators=5, max_depth=2, eval_metric="mlogloss")
    clf.fit(X, y)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    pickle.dump(clf, open(models_dir / "category_router_xgb.pkl", "wb"))
    pickle.dump(le, open(models_dir / "category_router_le.pkl", "wb"))
    json.dump({"threshold": 0.999},
              open(models_dir / "category_router_threshold.json", "w"))

    router = CategoryRouter.load(
        models_dir=str(models_dir), embedder=FakeEmbedder(),
    )
    out = router.predict({
        "product_name": "x", "brands": "", "ingredients_text": "", "quantity": "",
    })
    assert out["is_ood"] is True
    assert out["predicted"] == "unknown"


def test_cascade_auto_routes_to_supported_category():
    """When category is None and router predicts a supported class, cascade runs."""
    from demo.ml_service.cascade import CascadePipeline, DEMO_SUPPORTED_CATEGORIES

    pipeline = CascadePipeline.__new__(CascadePipeline)
    pipeline.rx = type("RX", (), {
        "extract_all": lambda self, *a, **kw: {}
    })()
    pipeline._embedder = None
    pipeline.ml_models = {f"{c}_stratified": {} for c in DEMO_SUPPORTED_CATEGORIES}
    pipeline.thresholds = {f"{c}_stratified": {} for c in DEMO_SUPPORTED_CATEGORIES}
    pipeline.validator = type("V", (), {
        "validate_value": lambda self, *a, **kw: None,
        "bucketize_value": lambda self, *a, **kw: None,
        "brand_status": lambda self, *a, **kw: "n/a",
    })()
    pipeline.router = type("R", (), {
        "predict": lambda self, prod: {
            "predicted": "chocolate", "confidence": 0.9,
            "alternatives": [("chocolate", 0.9), ("pasta", 0.05), ("beverages", 0.05)],
            "is_ood": False,
        }
    })()

    # Stub _embed to avoid SBERT.
    import numpy as np
    pipeline._embed = lambda product: np.zeros((1, 384))

    out = pipeline.predict(
        public_category=None,
        product={"product_name": "Lindt 70%", "brands": "Lindt",
                 "ingredients_text": "", "quantity": "100g"},
    )
    assert out["category"] == "chocolate"
    assert out["category_inference"]["predicted"] == "chocolate"
    assert out.get("is_ood", False) is False
    assert out.get("is_known_but_unsupported", False) is False


def test_cascade_returns_ood_block_when_router_says_ood():
    from demo.ml_service.cascade import CascadePipeline

    pipeline = CascadePipeline.__new__(CascadePipeline)
    pipeline.router = type("R", (), {
        "predict": lambda self, prod: {
            "predicted": "unknown", "confidence": 0.1,
            "alternatives": [("pasta", 0.1), ("beverages", 0.09), ("chocolate", 0.08)],
            "is_ood": True,
        }
    })()
    out = pipeline.predict(
        public_category=None,
        product={"product_name": "wrench", "brands": "Stanley",
                 "ingredients_text": "", "quantity": ""},
    )
    assert out["is_ood"] is True
    assert out["category"] is None
    assert out["predictions"] == {}


def test_cascade_returns_unsupported_when_router_picks_disabled_category():
    from demo.ml_service.cascade import CascadePipeline

    pipeline = CascadePipeline.__new__(CascadePipeline)
    pipeline.router = type("R", (), {
        "predict": lambda self, prod: {
            "predicted": "cheeses", "confidence": 0.92,
            "alternatives": [("cheeses", 0.92), ("chocolate", 0.05), ("pasta", 0.03)],
            "is_ood": False,
        }
    })()
    out = pipeline.predict(
        public_category=None,
        product={"product_name": "Camembert", "brands": "President",
                 "ingredients_text": "", "quantity": "200g"},
    )
    assert out["is_known_but_unsupported"] is True
    assert out["category"] == "cheeses"
    assert out["predictions"] == {}


def test_cascade_manual_override_skips_router():
    """When public_category is provided, router.predict must NOT be called."""
    from demo.ml_service.cascade import CascadePipeline, DEMO_SUPPORTED_CATEGORIES

    called = {"v": False}
    class _Router:
        def predict(self, prod):
            called["v"] = True
            return {"predicted": "x", "confidence": 1.0,
                    "alternatives": [], "is_ood": False}

    pipeline = CascadePipeline.__new__(CascadePipeline)
    pipeline.router = _Router()
    pipeline.rx = type("RX", (), {"extract_all": lambda self, *a, **kw: {}})()
    pipeline._embedder = None
    pipeline.ml_models = {f"{c}_stratified": {} for c in DEMO_SUPPORTED_CATEGORIES}
    pipeline.thresholds = {f"{c}_stratified": {} for c in DEMO_SUPPORTED_CATEGORIES}
    pipeline.validator = type("V", (), {
        "validate_value": lambda self, *a, **kw: None,
        "bucketize_value": lambda self, *a, **kw: None,
        "brand_status": lambda self, *a, **kw: "n/a",
    })()
    import numpy as np
    pipeline._embed = lambda product: np.zeros((1, 384))

    out = pipeline.predict(
        public_category="pasta",
        product={"product_name": "Spaghetti", "brands": "Barilla",
                 "ingredients_text": "", "quantity": "500g"},
    )
    assert called["v"] is False
    assert out["category"] == "pasta"
    assert out["category_inference"] is None


def test_loco_eval_writes_table(tmp_path, fake_stratified_dir, fake_off_parquet):
    import numpy as np

    from src.eval.router_category import run_loco

    class FakeEmbedder:
        def encode(self, texts, show_progress_bar=False, batch_size=None):
            out = []
            for t in texts:
                rng = np.random.default_rng(abs(hash(t)) % (2**32))
                out.append(rng.standard_normal(384))
            return np.array(out)

    out_path = tmp_path / "loco.parquet"
    loco_df = run_loco(
        processed_dir=str(fake_stratified_dir),
        off_parquet=str(fake_off_parquet),
        embedder=FakeEmbedder(),
        n_per_class=30,
        seed=42,
        output_path=str(out_path),
        target_fpr_known=0.05,
    )
    assert out_path.exists()
    assert len(loco_df) == 7
    assert set(loco_df.columns) >= {
        "leave_out_category", "ood_recall",
        "mean_confidence_on_loco", "n_loco_examples",
    }
    assert ((loco_df["ood_recall"] >= 0) & (loco_df["ood_recall"] <= 1)).all()


def test_api_enrich_accepts_missing_category():
    from fastapi.testclient import TestClient
    import demo.ml_service.main as m

    fake_out = {
        "category": "chocolate", "internal_category": "chocolate_stratified",
        "n_attrs_total": 7, "n_covered": 5, "n_llm_fallback": 2,
        "predictions": {}, "expected": {},
        "validation_summary": {
            "n_flagged_predictions": 0, "n_flagged_expected": 0,
            "brand_status": "known", "mode": "warn",
        },
        "category_inference": {
            "predicted": "chocolate", "confidence": 0.9,
            "alternatives": [["chocolate", 0.9]],
            "is_ood": False,
        },
        "is_ood": False, "is_known_but_unsupported": False,
        "pending_llm_fallback": False,
    }
    class FakePipeline:
        validator = type("V", (), {"ready": staticmethod(lambda: True),
                                      "models": {}, "inferences": {}})()
        router = type("R", (), {"predict": lambda self, p: {}})()
        def predict(self, **kw):
            return fake_out

    m.pipeline = FakePipeline()
    client = TestClient(m.app)
    resp = client.post("/api/enrich", json={
        "product_name": "Lindt 70%",
        "brands": "Lindt",
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["category_inference"]["predicted"] == "chocolate"


def test_api_enrich_rejects_unsupported_category():
    from fastapi.testclient import TestClient
    import demo.ml_service.main as m

    client = TestClient(m.app)
    resp = client.post("/api/enrich", json={
        "category": "cheeses",
        "product_name": "Camembert",
    })
    assert resp.status_code == 422


def test_mahalanobis_ood_separates_synthetic_gaussians():
    """Two well-separated Gaussians; LOCO on one of them should give high recall."""
    import numpy as np
    from src.pipeline.category_router.mahalanobis_ood import (
        fit_mahalanobis, distance_to_nearest_centroid, loco_recall,
    )
    rng = np.random.default_rng(0)
    D = 32
    n_per = 200
    # Class A and B — well separated; class C — held out, even further
    X_a = rng.standard_normal((n_per, D)) * 0.5
    X_b = rng.standard_normal((n_per, D)) * 0.5 + 5.0
    X_c = rng.standard_normal((n_per, D)) * 0.5 + 15.0

    fit = fit_mahalanobis(
        np.vstack([X_a, X_b]),
        np.array(['a'] * n_per + ['b'] * n_per),
    )
    assert set(fit.classes) == {'a', 'b'}
    assert fit.inv_cov.shape == (D, D)
    # Distance from class-A points to nearest centroid should be small
    d_a = distance_to_nearest_centroid(fit, X_a)
    assert d_a.mean() < 10.0, d_a.mean()
    # Distance from far-away C points should be much larger
    d_c = distance_to_nearest_centroid(fit, X_c)
    assert d_c.mean() > d_a.mean() * 3, (d_a.mean(), d_c.mean())
    # LOCO recall: C as held-out, A+B as known.
    X_known_test = np.vstack([X_a[:100], X_b[:100]])
    result = loco_recall(fit, X_known_test, X_c, target_fpr=0.05)
    assert result['ood_recall_loco'] > 0.9, result
    # FPR on known should be close to target.
    assert 0.0 <= result['fpr_on_known'] <= 0.15, result


def test_mahalanobis_distance_min_over_classes():
    """Distance to nearest centroid = min over per-class Mahalanobis."""
    import numpy as np
    from src.pipeline.category_router.mahalanobis_ood import (
        fit_mahalanobis, distance_to_nearest_centroid,
    )
    rng = np.random.default_rng(1)
    D = 16
    # 3 classes at (-5, 0, +5) along first axis
    X = np.concatenate([
        rng.standard_normal((50, D)) - np.eye(D)[0] * 5,
        rng.standard_normal((50, D)),
        rng.standard_normal((50, D)) + np.eye(D)[0] * 5,
    ])
    y = np.array(['neg'] * 50 + ['zero'] * 50 + ['pos'] * 50)
    fit = fit_mahalanobis(X, y)
    # Points centered at 0 should be near 'zero' centroid → small distance
    test_zero = rng.standard_normal((20, D))
    d = distance_to_nearest_centroid(fit, test_zero)
    # And the distance should be smaller than to 'neg' centroid alone.
    d_neg = np.sqrt(np.einsum(
        'ij,jk,ik->i',
        test_zero - fit.centroids['neg'],
        fit.inv_cov,
        test_zero - fit.centroids['neg'],
    ))
    assert d.mean() < d_neg.mean()
