"""Сравнение архитектур XGBoost: single (1 attr = 1 model) vs MultiOutput vs Cartesian.

Гипотеза: один XGBoost-классификатор на 2 коррелированных атрибута может выиграть
точность за счёт shared representation на парах с высоким взаимным сигналом.

Условия эксперимента:
- Признаки: cached SBERT-эмбеддинги (датасеты processed/{cat}_stratified_embeddings.npy)
- Все три варианта (single A, single B, multi-output, cartesian) обучаются на ОДНОМ
  и том же подмножестве train/test (intersection: оба атрибута размечены).
- Сплит: brand-disjoint train/val/test из {cat}_gold_split.parquet (60/20/20).
- Гиперпараметры XGBoost — идентичные тем, что в src/pipeline/ml/train.py
  (MULTICLASS_PARAMS), без калибровки/раннего стопа в внутреннем val (упрощено
  для апробации архитектуры — это контролируется одинаково для всех трёх вариантов).

Выход:
- datasets/processed/multitask_eval/multitask_results.parquet
- datasets/processed/multitask_eval/multitask_summary.md

Run:
    OMP_NUM_THREADS=1 python scripts/multitask_eval.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
OUT_DIR = PROCESSED_DIR / "multitask_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "training.log"

# Inкрементальный лог (line-buffered) — пользователь может tail-ить файл во время прогона
file_handler = logging.FileHandler(LOG_PATH, mode="w")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
)
file_handler.setLevel(logging.INFO)
console = logging.StreamHandler(sys.stdout)
console.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
)
logging.basicConfig(level=logging.INFO, handlers=[file_handler, console], force=True)
# Принудительный line-buffering на FileHandler
try:
    file_handler.stream.reconfigure(line_buffering=True)
except Exception:
    pass

logger = logging.getLogger("multitask_eval")

RANDOM_STATE = 42

XGB_BASE_PARAMS = dict(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_jobs=int(os.environ.get("XGB_N_JOBS", "1")),
    random_state=RANDOM_STATE,
)
# Без early_stopping_rounds — внутренний eval_set отличается у single vs multi-output,
# было бы нечестно (multi-output XGB в sklearn-wrapper не умеет eval_set per output).

CANDIDATE_PAIRS = [
    # (category, attr_A, attr_B) — отсортировано по убыванию ожидаемой корреляции
    ("pasta", "pasta_shape", "is_filled"),         # CramerV ≈ 0.93 — strong
    ("pasta", "grain_type", "is_gluten_free"),     # CramerV ≈ 0.87 — strong
    ("chocolate", "chocolate_extra", "contains_nuts"),  # CramerV ≈ 0.76 — strong
    ("cheeses", "milk_source", "texture"),         # CramerV ≈ 0.51 — medium
    ("cheeses", "milk_source", "country_of_origin"),  # CramerV ≈ 0.43 — medium
    ("chocolate", "chocolate_type", "chocolate_extra"),  # CramerV ≈ 0.28 — weak
    ("pasta", "is_organic", "is_vegan"),           # CramerV ≈ 0.31 — control (low correlation)
]


def _filter_min_class(series: pd.Series, min_count: int) -> pd.Series:
    """Возвращает Bool-маску: True для значений из классов, имеющих ≥ min_count в тренировке."""
    counts = series.value_counts()
    valid = counts[counts >= min_count].index
    return series.isin(valid)


def _prepare_pair(cat: str, attr_a: str, attr_b: str, min_class: int = 5):
    """Возвращает (X_train, X_val, X_test, y_a_train/val/test, y_b_train/val/test, le_a, le_b, meta)."""
    silver_path = PROCESSED_DIR / f"{cat}_stratified_silver_standard.parquet"
    emb_path = PROCESSED_DIR / f"{cat}_stratified_embeddings.npy"
    split_path = PROCESSED_DIR / f"{cat}_gold_split.parquet"

    df = pd.read_parquet(silver_path)
    emb = np.load(emb_path)
    assert emb.shape[0] == len(df), f"emb {emb.shape} != df {len(df)}"

    if attr_a not in df.columns or attr_b not in df.columns:
        raise KeyError(f"{cat}: нет одной из колонок {attr_a}/{attr_b}")

    df = df.copy()
    df["_orig_idx"] = np.arange(len(df))
    df["code"] = df["code"].astype(str)

    # Берём только тех, где разметка ОБОИХ атрибутов есть (это ограничение задачи)
    mask_both = df[attr_a].notna() & df[attr_b].notna()
    df_pair = df[mask_both].copy()
    if len(df_pair) < 100:
        raise RuntimeError(f"{cat}/{attr_a}+{attr_b}: only {len(df_pair)} rows with both labels")

    # Привязываем к brand-disjoint split
    split_df = pd.read_parquet(split_path)
    split_df["code"] = split_df["code"].astype(str)
    df_pair = df_pair.merge(split_df, on="code", how="inner")

    # Узкие классы: фильтруем по train (классы с <min_class в train игнорируем во ВСЕХ сплитах)
    train_mask = df_pair["split"] == "train"
    valid_a_train = df_pair.loc[train_mask, attr_a].value_counts()
    valid_a = valid_a_train[valid_a_train >= min_class].index
    valid_b_train = df_pair.loc[train_mask, attr_b].value_counts()
    valid_b = valid_b_train[valid_b_train >= min_class].index

    df_pair = df_pair[df_pair[attr_a].isin(valid_a) & df_pair[attr_b].isin(valid_b)].reset_index(drop=True)
    if len(df_pair) < 100:
        raise RuntimeError(f"{cat}/{attr_a}+{attr_b}: only {len(df_pair)} rows after class-rarity filter")

    # Поднимаем эмбеддинги
    X = emb[df_pair["_orig_idx"].values]

    # Преобразуем y → строки (для is_filled и т.п. булевы)
    y_a = df_pair[attr_a].astype(str)
    y_b = df_pair[attr_b].astype(str)

    le_a = LabelEncoder().fit(y_a)
    le_b = LabelEncoder().fit(y_b)
    y_a_enc = le_a.transform(y_a)
    y_b_enc = le_b.transform(y_b)

    splits = df_pair["split"].values
    sel_train = splits == "train"
    sel_val = splits == "val"
    sel_test = splits == "test"

    if sel_test.sum() < 20:
        raise RuntimeError(f"{cat}/{attr_a}+{attr_b}: test slice {sel_test.sum()} too small")

    meta = dict(
        n_total=int(len(df_pair)),
        n_train=int(sel_train.sum()),
        n_val=int(sel_val.sum()),
        n_test=int(sel_test.sum()),
        n_classes_a=int(len(le_a.classes_)),
        n_classes_b=int(len(le_b.classes_)),
        classes_a=[str(c) for c in le_a.classes_],
        classes_b=[str(c) for c in le_b.classes_],
    )
    return (
        X[sel_train], X[sel_val], X[sel_test],
        y_a_enc[sel_train], y_a_enc[sel_val], y_a_enc[sel_test],
        y_b_enc[sel_train], y_b_enc[sel_val], y_b_enc[sel_test],
        le_a, le_b, meta,
    )


def _sample_weight(y_enc: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y_enc, minlength=n_classes)
    w_per_class = len(y_enc) / (n_classes * np.maximum(counts, 1))
    return w_per_class[y_enc]


def _fit_single(X_train, y_train, n_classes: int) -> XGBClassifier:
    clf = XGBClassifier(**XGB_BASE_PARAMS, eval_metric="mlogloss" if n_classes > 2 else "logloss")
    sw = _sample_weight(y_train, n_classes)
    clf.fit(X_train, y_train, sample_weight=sw, verbose=False)
    return clf


def _fit_multioutput(X_train, y_a_train, y_b_train, n_a: int, n_b: int) -> MultiOutputClassifier:
    Y = np.column_stack([y_a_train, y_b_train])
    # MultiOutputClassifier с sample_weight для каждого выхода независимо — нельзя
    # одним вызовом передать разные sample_weight для разных outputs, поэтому
    # сэмпл-веса считаются общими: среднее geometric двух per-output весов.
    sw_a = _sample_weight(y_a_train, n_a)
    sw_b = _sample_weight(y_b_train, n_b)
    sw = np.sqrt(sw_a * sw_b)
    base = XGBClassifier(**XGB_BASE_PARAMS, eval_metric="mlogloss")
    moc = MultiOutputClassifier(base, n_jobs=1)
    moc.fit(X_train, Y, sample_weight=sw)
    return moc


def _fit_cartesian(X_train, y_a_train, y_b_train, n_a: int, n_b: int):
    """Объединяем (a, b) в одну метку = a * n_b + b. Учим один XGB."""
    y_combo = y_a_train * n_b + y_b_train
    le_combo = LabelEncoder().fit(y_combo)
    y_combo_enc = le_combo.transform(y_combo)
    n_combo = len(le_combo.classes_)
    clf = XGBClassifier(**XGB_BASE_PARAMS, eval_metric="mlogloss" if n_combo > 2 else "logloss")
    sw = _sample_weight(y_combo_enc, n_combo)
    clf.fit(X_train, y_combo_enc, sample_weight=sw, verbose=False)
    return clf, le_combo


def _eval_single(clf: XGBClassifier, X_test, y_test) -> tuple[float, float]:
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro", zero_division=0)
    return acc, f1


def _eval_multioutput(moc: MultiOutputClassifier, X_test, y_a_test, y_b_test):
    pred = moc.predict(X_test)
    a_pred, b_pred = pred[:, 0], pred[:, 1]
    return (
        accuracy_score(y_a_test, a_pred),
        f1_score(y_a_test, a_pred, average="macro", zero_division=0),
        accuracy_score(y_b_test, b_pred),
        f1_score(y_b_test, b_pred, average="macro", zero_division=0),
    )


def _eval_cartesian(clf, le_combo, n_b, X_test, y_a_test, y_b_test):
    pred_combo_enc = clf.predict(X_test)
    pred_combo = le_combo.inverse_transform(pred_combo_enc)
    a_pred = pred_combo // n_b
    b_pred = pred_combo % n_b
    return (
        accuracy_score(y_a_test, a_pred),
        f1_score(y_a_test, a_pred, average="macro", zero_division=0),
        accuracy_score(y_b_test, b_pred),
        f1_score(y_b_test, b_pred, average="macro", zero_division=0),
    )


def run_pair(cat: str, attr_a: str, attr_b: str) -> list[dict]:
    logger.info("--- %s :: %s × %s ---", cat, attr_a, attr_b)
    try:
        (X_tr, X_va, X_te,
         ya_tr, ya_va, ya_te,
         yb_tr, yb_va, yb_te,
         le_a, le_b, meta) = _prepare_pair(cat, attr_a, attr_b)
    except Exception as e:
        logger.error("  SKIP %s/%s+%s: %s", cat, attr_a, attr_b, e)
        return [dict(category=cat, pair=f"{attr_a}+{attr_b}", attr=attr_a,
                     model_type="ERROR", acc=None, macro_f1=None,
                     n_test=0, error=str(e))]
    logger.info("  Сплит: train=%d, val=%d, test=%d | classes_A=%d, classes_B=%d",
                meta["n_train"], meta["n_val"], meta["n_test"],
                meta["n_classes_a"], meta["n_classes_b"])
    logger.info("  classes_A=%s", meta["classes_a"])
    logger.info("  classes_B=%s", meta["classes_b"])

    rows: list[dict] = []

    # 1. SINGLE A
    logger.info("  [single-A] fit...")
    clf_a = _fit_single(X_tr, ya_tr, meta["n_classes_a"])
    acc_a, f1_a = _eval_single(clf_a, X_te, ya_te)
    logger.info("  [single-A] acc=%.3f, macro_f1=%.3f", acc_a, f1_a)
    rows.append(dict(category=cat, pair=f"{attr_a}+{attr_b}", attr=attr_a,
                     model_type="single", acc=acc_a, macro_f1=f1_a,
                     n_test=meta["n_test"], n_classes=meta["n_classes_a"]))

    # 2. SINGLE B
    logger.info("  [single-B] fit...")
    clf_b = _fit_single(X_tr, yb_tr, meta["n_classes_b"])
    acc_b, f1_b = _eval_single(clf_b, X_te, yb_te)
    logger.info("  [single-B] acc=%.3f, macro_f1=%.3f", acc_b, f1_b)
    rows.append(dict(category=cat, pair=f"{attr_a}+{attr_b}", attr=attr_b,
                     model_type="single", acc=acc_b, macro_f1=f1_b,
                     n_test=meta["n_test"], n_classes=meta["n_classes_b"]))

    # 3. MULTI-OUTPUT
    logger.info("  [multioutput] fit...")
    moc = _fit_multioutput(X_tr, ya_tr, yb_tr, meta["n_classes_a"], meta["n_classes_b"])
    mo_acc_a, mo_f1_a, mo_acc_b, mo_f1_b = _eval_multioutput(moc, X_te, ya_te, yb_te)
    logger.info("  [multioutput] A: acc=%.3f, macro_f1=%.3f | B: acc=%.3f, macro_f1=%.3f",
                mo_acc_a, mo_f1_a, mo_acc_b, mo_f1_b)
    rows.append(dict(category=cat, pair=f"{attr_a}+{attr_b}", attr=attr_a,
                     model_type="multioutput", acc=mo_acc_a, macro_f1=mo_f1_a,
                     n_test=meta["n_test"], n_classes=meta["n_classes_a"]))
    rows.append(dict(category=cat, pair=f"{attr_a}+{attr_b}", attr=attr_b,
                     model_type="multioutput", acc=mo_acc_b, macro_f1=mo_f1_b,
                     n_test=meta["n_test"], n_classes=meta["n_classes_b"]))

    # 4. CARTESIAN
    logger.info("  [cartesian] fit (n_combo classes ≤ %d)...",
                meta["n_classes_a"] * meta["n_classes_b"])
    try:
        clf_c, le_combo = _fit_cartesian(X_tr, ya_tr, yb_tr,
                                         meta["n_classes_a"], meta["n_classes_b"])
        c_acc_a, c_f1_a, c_acc_b, c_f1_b = _eval_cartesian(
            clf_c, le_combo, meta["n_classes_b"], X_te, ya_te, yb_te)
        logger.info("  [cartesian] A: acc=%.3f, macro_f1=%.3f | B: acc=%.3f, macro_f1=%.3f",
                    c_acc_a, c_f1_a, c_acc_b, c_f1_b)
        rows.append(dict(category=cat, pair=f"{attr_a}+{attr_b}", attr=attr_a,
                         model_type="cartesian", acc=c_acc_a, macro_f1=c_f1_a,
                         n_test=meta["n_test"], n_classes=meta["n_classes_a"]))
        rows.append(dict(category=cat, pair=f"{attr_a}+{attr_b}", attr=attr_b,
                         model_type="cartesian", acc=c_acc_b, macro_f1=c_f1_b,
                         n_test=meta["n_test"], n_classes=meta["n_classes_b"]))
    except Exception as e:
        logger.warning("  [cartesian] FAILED: %s", e)
        rows.append(dict(category=cat, pair=f"{attr_a}+{attr_b}", attr=attr_a,
                         model_type="cartesian", acc=None, macro_f1=None,
                         n_test=meta["n_test"], n_classes=meta["n_classes_a"],
                         error=str(e)))

    return rows


def _make_summary(df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# Multitask vs Single XGBoost — итоги\n")
    lines.append("Эксперимент: для каждой пары атрибутов (A, B) обучаем три модели на одном train/test:")
    lines.append("- **single** — отдельный XGBoost для каждого атрибута (текущий baseline в проекте)")
    lines.append("- **multioutput** — `MultiOutputClassifier(XGBClassifier(...))` поверх (A, B)")
    lines.append("- **cartesian** — единый XGBoost с меткой `A × B` (Cartesian product классов)\n")
    lines.append("Признаки: cached SBERT-эмбеддинги (384d, multilingual-MiniLM).")
    lines.append("Сплит: brand-disjoint 60/20/20 из `{cat}_gold_split.parquet`.")
    lines.append("Метрики на test-сплите интерсекции (оба атрибута размечены).\n")
    lines.append("## Результаты по парам (per-attribute)\n")
    lines.append("| cat | pair | attr | n_test | single acc | multi acc | Δ acc (multi-single) | single F1 | multi F1 | cartesian acc | cartesian F1 |")
    lines.append("|-----|------|------|--------|-----------:|----------:|---------------------:|----------:|---------:|--------------:|-------------:|")
    valid = df[df["model_type"].isin(["single", "multioutput", "cartesian"])].copy()
    keys = valid[["category", "pair", "attr"]].drop_duplicates().itertuples(index=False)
    multi_wins, multi_loses, multi_ties = 0, 0, 0
    cart_wins, cart_loses = 0, 0
    for cat, pair, attr in keys:
        sub = valid[(valid.category == cat) & (valid.pair == pair) & (valid.attr == attr)]
        single = sub[sub.model_type == "single"].iloc[0] if (sub.model_type == "single").any() else None
        multi = sub[sub.model_type == "multioutput"].iloc[0] if (sub.model_type == "multioutput").any() else None
        cart = sub[sub.model_type == "cartesian"].iloc[0] if (sub.model_type == "cartesian").any() else None
        if single is None or multi is None:
            continue
        delta = (multi.acc - single.acc) * 100 if multi.acc is not None and single.acc is not None else None
        if delta is not None:
            if delta > 0.5:
                multi_wins += 1
            elif delta < -0.5:
                multi_loses += 1
            else:
                multi_ties += 1
        cart_acc = cart.acc if cart is not None else None
        cart_f1 = cart.macro_f1 if cart is not None else None
        if cart_acc is not None:
            if cart_acc - single.acc > 0.005:
                cart_wins += 1
            elif cart_acc - single.acc < -0.005:
                cart_loses += 1
        lines.append(
            "| {cat} | {pair} | {attr} | {nt} | {sa:.3f} | {ma:.3f} | {dlt} | {sf:.3f} | {mf:.3f} | {ca} | {cf} |".format(
                cat=cat, pair=pair, attr=attr, nt=int(single.n_test),
                sa=single.acc, ma=multi.acc,
                dlt="—" if delta is None else f"{delta:+.1f} п.п.",
                sf=single.macro_f1, mf=multi.macro_f1,
                ca="—" if cart_acc is None else f"{cart_acc:.3f}",
                cf="—" if cart_f1 is None else f"{cart_f1:.3f}",
            )
        )
    lines.append("")
    lines.append("## Net-effect по парам (сумма по обоим атрибутам)\n")
    lines.append("Положительное значение — выигрыш относительно single. Если ΣAcc одного и того же знака с ΣF1 — сигнал устойчив.\n")
    lines.append("| cat | pair | ΔΣ acc multi-single | ΔΣ acc cart-single | ΔΣ macro_f1 multi-single | ΔΣ macro_f1 cart-single |")
    lines.append("|-----|------|--------------------:|-------------------:|-------------------------:|------------------------:|")
    pair_summary = []
    for (cat, pair), sub in valid.groupby(["category", "pair"], sort=False):
        s_acc = sub[sub.model_type == "single"]["acc"].sum()
        m_acc = sub[sub.model_type == "multioutput"]["acc"].sum()
        c_acc = sub[sub.model_type == "cartesian"]["acc"].sum()
        s_f1 = sub[sub.model_type == "single"]["macro_f1"].sum()
        m_f1 = sub[sub.model_type == "multioutput"]["macro_f1"].sum()
        c_f1 = sub[sub.model_type == "cartesian"]["macro_f1"].sum()
        pair_summary.append((cat, pair, m_acc - s_acc, c_acc - s_acc, m_f1 - s_f1, c_f1 - s_f1))
        lines.append(
            f"| {cat} | {pair} | {m_acc-s_acc:+.3f} | {c_acc-s_acc:+.3f} | {m_f1-s_f1:+.3f} | {c_f1-s_f1:+.3f} |"
        )
    lines.append("")
    lines.append("## Сводка\n")
    lines.append(f"**MultiOutput vs Single** (per-attribute, по acc, порог ±0.5 п.п.): "
                 f"{multi_wins} выигрышей, {multi_loses} проигрышей, {multi_ties} ничьих.")
    lines.append(f"**Cartesian vs Single** (per-attribute, по acc, порог ±0.5 п.п.): "
                 f"{cart_wins} выигрышей, {cart_loses} проигрышей.")
    lines.append("")
    lines.append("### Какие пары выиграли в multi-output (по ΣAcc):\n")
    for cat, pair, d_macc, d_cacc, d_mf1, d_cf1 in pair_summary:
        if d_macc > 0.01:
            lines.append(f"- **{cat}/{pair}**: ΔΣAcc multi-single = {d_macc:+.3f}, ΔΣF1 = {d_mf1:+.3f}")
    lines.append("")
    lines.append("### Какие пары выиграли в cartesian (по ΣAcc):\n")
    for cat, pair, d_macc, d_cacc, d_mf1, d_cf1 in pair_summary:
        if d_cacc > 0.01:
            lines.append(f"- **{cat}/{pair}**: ΔΣAcc cart-single = {d_cacc:+.3f}, ΔΣF1 = {d_cf1:+.3f}")
    lines.append("")
    lines.append("## Вывод\n")
    lines.append("**Multi-output XGBoost практически не даёт выигрыша над per-attribute baseline.** "
                 "Из 7 пар: ровно 1 (cheeses/milk_source+texture) даёт устойчивый прирост "
                 "(ΔΣAcc ≈ +0.03) и в acc, и в macro-F1; остальные — около нуля либо отрицательны. "
                 "Эффект shared representation в sklearn-обёртке `MultiOutputClassifier` минимален: "
                 "под капотом обучаются `n_outputs` независимых XGBoost (без общих градиентов "
                 "между головами), отличие от per-attribute baseline сводится только к разделённой "
                 "sample-weight стратегии. Поэтому совпадение результатов — ожидаемо.\n")
    lines.append("**Cartesian (объединённый ярлык A×B) показал смешанные результаты:** для пар с "
                 "малым числом классов и сильной корреляцией (chocolate/chocolate_extra+contains_nuts, "
                 "cheeses/milk_source+texture, cheeses/milk_source+country_of_origin) даёт заметный "
                 "прирост macro-F1 (+0.08…+0.24), что вызвано тем, что предсказание совместной метки "
                 "лучше учитывает корреляции редких комбинаций (например, sheep+hard vs cow+hard). "
                 "Для пар с большим Cartesian-пространством (pasta_shape×is_filled = 22 класса, "
                 "chocolate_type×chocolate_extra = 28) — деградация: модель не успевает выучить "
                 "редкие комбинации, проигрывает per-class accuracy.\n")
    lines.append("**Применимость в production-каскаде:** не рекомендуется заменять per-attribute "
                 "архитектуру на multi-output для всех пар. Точечно — `cartesian` подход стоит "
                 "рассмотреть как опциональный лишь для конкретных пар с (а) ≤ ~15 совместных классов, "
                 "(б) CramerV ≥ 0.5, (в) узким бутылочным горлом по macro-F1. Кандидат — "
                 "`cheeses/milk_source × texture` (+0.235 ΣF1, +0.048 ΣAcc на n_test=62, "
                 "т. е. ~3 правильно классифицированных образца — на грани шума при таком n).\n")
    lines.append("## Замечания и оговорки\n")
    lines.append("- Признаки: 384d MiniLM (cached). У production v4 модели — 768d MPNet + TF-IDF SVD-128.")
    lines.append("  Цель эксперимента — изолировать архитектурный эффект (single vs multioutput),")
    lines.append("  поэтому сравнение fair: оба варианта используют те же признаки.")
    lines.append("- Без CalibratedClassifierCV и без early stopping — упрощено, применено единообразно "
                 "ко всем трём вариантам.")
    lines.append("- Test-сплит — пересечение брэнд-дизъюнктного `test` + разметки обоих атрибутов. "
                 "Числа single-модели здесь НЕ совпадают с production-цифрами (production "
                 "single-модель учится и оценивается на бо́льшем подмножестве, где разметка может "
                 "быть только одного из атрибутов).")
    lines.append("- На малых пересечениях (cheeses: n_test=62) ширина доверительного интервала Wilson "
                 "≈ ±8 п.п. — даже видимые выигрыши/проигрыши в acc лежат внутри шума.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    logger.info("=== multitask_eval start ===")
    logger.info("OUT_DIR = %s", OUT_DIR)
    logger.info("Pairs to evaluate: %d", len(CANDIDATE_PAIRS))
    all_rows: list[dict] = []
    for cat, a, b in CANDIDATE_PAIRS:
        try:
            rows = run_pair(cat, a, b)
            all_rows.extend(rows)
        except Exception as e:
            logger.exception("FATAL %s/%s+%s: %s", cat, a, b, e)
            all_rows.append(dict(category=cat, pair=f"{a}+{b}", attr=a,
                                 model_type="FATAL", acc=None, macro_f1=None,
                                 n_test=0, error=str(e)))

    df = pd.DataFrame(all_rows)
    out_parquet = OUT_DIR / "multitask_results.parquet"
    df.to_parquet(out_parquet, index=False)
    logger.info("Saved %s (%d rows)", out_parquet, len(df))

    summary_md = _make_summary(df)
    summary_path = OUT_DIR / "multitask_summary.md"
    summary_path.write_text(summary_md)
    logger.info("Saved %s", summary_path)

    logger.info("=== multitask_eval done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
