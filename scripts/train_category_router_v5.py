"""Train Layer-0 category router on MPNet embeddings (3 cats + OOD).

Data:
  - In-class (pasta/chocolate/cheeses): up to 5000 codes per cat from v5_relabel
  - OOD: 5000 random codes from food.parquet not tagged as any of those 3 cats
  - Embedding: paraphrase-multilingual-mpnet-base-v2 (768d, matches new ML cascade)
  - Classifier: XGBoost multi-class with 4 labels {pasta, chocolate, cheeses, ood}

Output:
  - models/category_router_v5.pkl       — XGBClassifier
  - models/category_router_v5_le.pkl    — LabelEncoder
  - models/category_router_v5_meta.json — train stats (n per class, F1, conf-mtx)
"""
from __future__ import annotations
import sys, os, json, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd

for root in ['/home/miafrolov/Desktop/diploma',
             '/Users/miafrolov/Desktop/stuff/ai_attributes']:
    if Path(root).exists():
        sys.path.insert(0, root)
        PROJECT_ROOT = Path(root)
        break

from src.common import EMBEDDING_MODEL, build_text

import duckdb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix

CATS = ['pasta', 'chocolate', 'cheeses']
PER_CAT_N = 5000
OOD_N = 5000
RNG = np.random.default_rng(42)


def load_in_class(cat: str, off_dir: Path) -> pd.DataFrame:
    v5 = pd.read_parquet(PROJECT_ROOT / f'datasets/processed/v5_relabel/{cat}_relabel_v5.parquet')
    v5 = v5[v5.parse_status == True].copy()
    v5['code'] = v5['code'].astype(str)
    print(f'  {cat}: v5 has {len(v5)}')
    if len(v5) > PER_CAT_N:
        v5 = v5.sample(n=PER_CAT_N, random_state=42).reset_index(drop=True)
    codes = set(v5.code)
    # Join with OFF dump for partner-fields (product_name, brands, ingredients_text, quantity)
    code_sql = ','.join(f"'{c}'" for c in codes)
    con = duckdb.connect()
    inputs = con.execute(f"""
      SELECT CAST(code AS VARCHAR) AS code, product_name, brands, ingredients_text, quantity
      FROM '{off_dir / f"{cat}_off_full.parquet"}'
      WHERE CAST(code AS VARCHAR) IN ({code_sql})
    """).fetchdf()
    inputs['label'] = cat
    print(f'  {cat}: joined {len(inputs)} with OFF dump')
    return inputs


def load_ood(off_dir: Path, exclude_tags: set, target_n: int) -> pd.DataFrame:
    """Sample OOD rows from food.parquet. product_name/brands/ingredients_text are STRUCT[];
    flatten via the existing helper in scripts.build_gold_v4_wide.
    """
    from scripts.build_gold_v4_wide import _pick_text, _to_str, _safe_list
    food_path = off_dir / 'food.parquet'
    con = duckdb.connect()
    df = con.execute(f"""
      SELECT CAST(code AS VARCHAR) AS code, product_name, brands, ingredients_text,
             quantity, categories_tags
      FROM '{food_path}'
      USING SAMPLE {target_n * 8}
    """).fetchdf()
    df['product_name'] = df['product_name'].apply(_pick_text)
    df['ingredients_text'] = df['ingredients_text'].apply(_pick_text)
    df['brands'] = df['brands'].apply(_to_str)
    df['quantity'] = df['quantity'].apply(_to_str)
    df['categories_tags'] = df['categories_tags'].apply(lambda v: ','.join(_safe_list(v) or []) or None)
    df = df[df['product_name'].notna() & (df['product_name'].str.len() > 2)]
    def is_ood(s):
        if not isinstance(s, str): return True
        sl = s.lower()
        return not any(t in sl for t in exclude_tags)
    df = df[df['categories_tags'].apply(is_ood)]
    print(f'  ood pool after filter: {len(df)}')
    if len(df) > target_n:
        df = df.sample(n=target_n, random_state=42).reset_index(drop=True)
    df['label'] = 'ood'
    return df[['code', 'product_name', 'brands', 'ingredients_text', 'quantity', 'label']]


def main():
    off_dir = Path.home() / 'off_work'
    print('=== Loading in-class data ===')
    parts = []
    for cat in CATS:
        parts.append(load_in_class(cat, off_dir))

    print('\n=== Loading OOD data ===')
    cat_tags = {
        'en:pastas', 'en:noodles', 'en:asian-noodles', 'en:gnocchi',
        'en:chocolates', 'en:chocolate-bars', 'en:chocolate-confectionery',
        'en:cheeses', 'en:fresh-cheeses', 'en:hard-cheeses', 'en:soft-cheeses',
    }
    parts.append(load_ood(off_dir, cat_tags, OOD_N))

    df = pd.concat(parts, ignore_index=True)
    print(f'\n=== Total: {len(df)} ===')
    print(df['label'].value_counts())

    # Defensive: cat-specific OFF parquets may still have STRUCT[]/list types for brands;
    # flatten everything to plain str so build_text doesn't trip on pd.notna(array).
    from scripts.build_gold_v4_wide import _to_str, _pick_text
    for col in ('product_name', 'ingredients_text'):
        df[col] = df[col].apply(lambda v: _pick_text(v) if not isinstance(v, str) else v)
    for col in ('brands', 'quantity'):
        df[col] = df[col].apply(lambda v: _to_str(v) if not isinstance(v, str) else v)
    # also drop rows where product_name is empty after flattening
    df = df[df['product_name'].fillna('').astype(str).str.len() > 1].reset_index(drop=True)
    print(f'  after flattening + filter: {len(df)}')

    print('\n=== Embedding (MPNet) ===')
    texts = build_text(df)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    t0 = time.time()
    X = model.encode(texts, show_progress_bar=True, batch_size=128).astype(np.float32)
    print(f'  encoded {X.shape} in {time.time()-t0:.1f}s')

    le = LabelEncoder()
    y = le.fit_transform(df['label'].values)
    print(f'  classes: {le.classes_.tolist()}')

    # Brand-stratified split when possible
    brands = df['brands'].fillna('').astype(str)
    X_tr, X_te, y_tr, y_te, b_tr, b_te = train_test_split(
        X, y, brands, test_size=0.2, random_state=42, stratify=y,
    )
    print(f'  train={len(X_tr)}, test={len(X_te)}')

    print('\n=== Training XGBoost ===')
    from xgboost import XGBClassifier
    clf = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective='multi:softprob', num_class=len(le.classes_),
        eval_metric='mlogloss',
        n_jobs=int(os.environ.get('XGB_N_JOBS', 4)),
        early_stopping_rounds=20, random_state=42, verbosity=1,
    )
    t0 = time.time()
    clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    print(f'  fit done in {time.time()-t0:.1f}s, best_iter={clf.best_iteration}')

    print('\n=== Eval ===')
    y_pred = clf.predict(X_te)
    f1m = f1_score(y_te, y_pred, average='macro')
    f1w = f1_score(y_te, y_pred, average='weighted')
    print(f'  F1_macro={f1m:.4f}  F1_weighted={f1w:.4f}')
    print('  classification report:')
    print(classification_report(y_te, y_pred, target_names=le.classes_, digits=3))
    print('  confusion matrix (rows=true, cols=pred):')
    cm = confusion_matrix(y_te, y_pred)
    print(f'  classes: {le.classes_.tolist()}')
    print(cm)

    models_dir = PROJECT_ROOT / 'models'
    models_dir.mkdir(exist_ok=True)
    out_clf = models_dir / 'category_router_v5.pkl'
    out_le = models_dir / 'category_router_v5_le.pkl'
    out_meta = models_dir / 'category_router_v5_meta.json'
    with open(out_clf, 'wb') as f: pickle.dump(clf, f)
    with open(out_le, 'wb') as f: pickle.dump(le, f)
    meta = {
        'embedding_model': EMBEDDING_MODEL,
        'classes': le.classes_.tolist(),
        'n_per_class': df['label'].value_counts().to_dict(),
        'n_train': len(X_tr), 'n_test': len(X_te),
        'f1_macro': float(f1m), 'f1_weighted': float(f1w),
        'confusion_matrix': cm.tolist(),
        'best_iteration': int(clf.best_iteration),
    }
    with open(out_meta, 'w') as f: json.dump(meta, f, indent=2)
    print(f'\n  saved: {out_clf}, {out_le}, {out_meta}')


if __name__ == '__main__':
    main()
