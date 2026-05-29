"""Failure taxonomy для каскада v4 (ticket 2026-05-28-failure-taxonomy).

Загружает per-category cascade predictions (`cascade_preds_{cat}_gold.parquet`),
отбирает ячейки где `cascade_pred != gold_value`, и классифицирует ошибки в
6 классов:

    1. regex-false-positive — Layer 1 (rule_h) вернул конкретное значение,
       но оно не совпадает с эталоном.
    2. layer4-miss — Layer 4 (fallback) дал ответ ≠ gold; редко встречается
       среди cascade-only ошибок, оставлен для полноты.
    3. null-as-other — gold пустой, prediction вернул «other» или конкретный
       класс (over-prediction).
    4. silver-noise — эталон сомнительный (disputed по голосованию 3-х LLM
       или agreement_ratio < 0.67 в `manual_gold_consensus.parquet`).
    5. class-confusion — ошибка между «близкими» классами по per-attr
       mapping (semi_soft↔soft, dark↔extra_dark и т.п.).
    6. layer2-class-shift — всё остальное (ML дал не-близкий класс).

Приоритет применения (первый совпавший выигрывает):
    regex-false-positive > layer4-miss > null-as-other >
    silver-noise > class-confusion > layer2-class-shift.

Сохраняет:
    datasets/processed/cascade_errors_taxonomy_v4.parquet
    report/contents/tables/failure_taxonomy.tex
    report/contents/tables/failure_examples.tex
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "datasets" / "processed"
TABLES_DIR = PROJECT_ROOT / "report" / "contents" / "tables"

CATEGORIES = ["pasta", "chocolate", "cheeses"]

# Класс-эвристики: пары значений, считающиеся «близкими» внутри атрибута.
# Симметрично применяются (a,b) ⇔ (b,a).
CLASS_CONFUSION_PAIRS: dict[str, list[tuple[str, str]]] = {
    # pasta
    "grain_type": [
        ("wheat", "spelt"),
        ("wheat", "mixed"),
        ("rice", "mixed"),
        ("legume", "mixed"),
        ("corn", "mixed"),
        ("buckwheat", "mixed"),
        ("oat", "mixed"),
    ],
    "pasta_shape": [
        ("fusilli", "rotini"),
        ("penne", "rigatoni"),
        ("linguine", "spaghetti"),
        ("linguine", "fettuccine"),
        ("tagliatelle", "fettuccine"),
        ("noodles", "spaghetti"),
        ("noodles", "linguine"),
        ("macaroni", "penne"),
        ("orzo", "rice"),
    ],
    "cuisine_origin": [
        ("italian", "other_regional"),
        ("german_alpine", "other_regional"),
        ("asian", "other_regional"),
        ("italian", "other"),
        ("asian", "other"),
    ],
    # chocolate
    "chocolate_type": [
        ("dark", "milk"),
        ("milk", "white"),
        ("dark", "filled"),
        ("milk", "filled"),
        ("other", "milk"),
        ("other", "dark"),
    ],
    "chocolate_extra": [
        ("plain", "other"),
        ("with_fruit", "other"),
        ("with_cookie", "filled"),
        ("with_caramel", "filled"),
        ("with_nuts", "with_fruit"),
        ("with_fruit", "with_caramel"),
        ("with_coffee", "intense_bitter"),
    ],
    "flavor_profile": [
        ("sweet_creamy", "nutty"),
        ("sweet_creamy", "fruity"),
        ("intense_bitter", "spiced"),
        ("salty_caramel", "sweet_creamy"),
        ("nutty", "other"),
        ("fruity", "floral"),
    ],
    # cheeses
    "texture": [
        ("soft", "cream"),
        ("soft", "fresh"),
        ("hard", "processed"),
        ("blue", "soft"),
        ("cream", "fresh"),
    ],
    "aging": [
        ("aged", "young"),
        ("young", "fresh"),
    ],
    "milk_source": [
        ("cow", "mixed"),
        ("sheep", "mixed"),
        ("goat", "mixed"),
        ("sheep", "goat"),
        ("cow", "other"),
    ],
    "country_of_origin": [
        ("france", "italy"),
        ("italy", "greece"),
        ("germany", "netherlands"),
        ("germany", "denmark"),
        ("italy", "cyprus"),
        ("france", "netherlands"),
    ],
}


def _is_close_pair(attr: str, a: str, b: str) -> bool:
    pairs = CLASS_CONFUSION_PAIRS.get(attr, [])
    a, b = str(a), str(b)
    return any({a, b} == {x, y} for x, y in pairs)


def _is_null_like(value) -> bool:
    if value is None:
        return True
    try:
        if isinstance(value, float) and math.isnan(value):
            return True
    except TypeError:
        pass
    s = str(value).strip().lower()
    return s in {"", "nan", "none", "null"}


def load_cascade_errors() -> pd.DataFrame:
    """Соберёт ошибки каскада (cascade_pred != gold_value) по 3 категориям."""
    rows = []
    for cat in CATEGORIES:
        df = pd.read_parquet(PROCESSED / f"cascade_preds_{cat}_gold.parquet")
        df = df[df["in_scope"] & df["cascade_pred"].notna()].copy()
        df["category"] = cat
        # Ошибка = pred отличается от gold (gold может быть None — null-as-other).
        # Считаем строки, где gold_value != cascade_pred (учёт None через !=).
        gold_str = df["gold_value"].astype("object")
        pred_str = df["cascade_pred"].astype("object")
        is_err = gold_str != pred_str
        err = df[is_err].copy()
        rows.append(err)
    return pd.concat(rows, ignore_index=True)


def attach_consensus_signal(errors: pd.DataFrame) -> pd.DataFrame:
    """Подключит agreement_ratio из manual_gold_consensus."""
    cons = pd.read_parquet(PROCESSED / "manual_gold_consensus.parquet")
    cons_sub = cons[["category", "code", "attr", "agreement_ratio", "disputed", "n_voters"]]
    merged = errors.merge(cons_sub, on=["category", "code", "attr"], how="left")
    return merged


def attach_product_names(errors: pd.DataFrame) -> pd.DataFrame:
    """Подтянет product_name из silver_standard.parquet по category+code."""
    pieces = []
    for cat in CATEGORIES:
        ss = pd.read_parquet(PROCESSED / f"{cat}_stratified_silver_standard.parquet")
        keep_cols = [c for c in ["code", "product_name", "brands"] if c in ss.columns]
        ss = ss[keep_cols].drop_duplicates(subset=["code"])
        ss["category"] = cat
        pieces.append(ss)
    names = pd.concat(pieces, ignore_index=True)
    out = errors.merge(names, on=["category", "code"], how="left")
    return out


def classify_error(row) -> str:
    """Приоритезированная классификация одной ошибки.

    Порядок: regex-FP > layer4-miss > null-as-other >
    silver-noise > class-confusion > layer2-class-shift.
    """
    layer = row["cascade_layer"]
    gold = row["gold_value"]
    pred = row["cascade_pred"]
    attr = row["attr"]
    agreement = row.get("agreement_ratio", None)
    disputed = row.get("disputed", None)

    # 1. Regex false positive — Layer 1 дал ответ, но он не совпадает.
    if layer in {"rule_h", "rule_l"} and not _is_null_like(pred):
        return "regex-false-positive"

    # 2. Layer 4 fallback miss — редкий случай в cascade-only.
    if layer == "fallback":
        return "layer4-miss"

    # 3. Null-as-other — gold None / null, pred вернул что-то конкретное.
    if _is_null_like(gold) and not _is_null_like(pred):
        return "null-as-other"

    # 4. Silver noise — низкое согласие LLM-разметчиков (или disputed).
    if agreement is not None and not pd.isna(agreement):
        if agreement < 0.67 or (disputed is True):
            return "silver-noise"

    # 5. Class-confusion — пара (gold, pred) в per-attr mapping.
    if (
        not _is_null_like(gold)
        and not _is_null_like(pred)
        and _is_close_pair(attr, gold, pred)
    ):
        return "class-confusion"

    # 6. Остальное — ML дал не-близкий класс.
    return "layer2-class-shift"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Двусторонний 95% Wilson confidence interval для биномиальной доли."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = max(0.0, centre - half)
    hi = min(1.0, centre + half)
    return (lo, hi)


def fmt_pct_comma(x: float, digits: int = 1) -> str:
    """Форматирует долю как процент с запятой (русский стиль)."""
    return f"{x * 100:.{digits}f}".replace(".", ",")


CLASS_LABEL_RU = {
    "regex-false-positive": "Ложное срабатывание Слоя 1 (regex-FP)",
    "layer4-miss": "Промах запасного слоя (Layer 4 miss)",
    "null-as-other": "Гиперпредсказание на null-эталоне",
    "silver-noise": "Шум разметки (silver-noise)",
    "class-confusion": "Спутывание близких классов",
    "layer2-class-shift": "ML-смещение в дальний класс",
}

CLASS_ORDER = [
    "regex-false-positive",
    "layer4-miss",
    "null-as-other",
    "silver-noise",
    "class-confusion",
    "layer2-class-shift",
]


def latex_escape(s: str) -> str:
    """Минимальный LaTeX-escape для текстового поля (кириллица не трогается)."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return "—"
    s = str(s)
    # порядок важен: сначала backslash
    repls = [
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ]
    for a, b in repls:
        s = s.replace(a, b)
    return s


def render_taxonomy_table(distrib: pd.DataFrame) -> str:
    """Сгенерирует TeX-таблицу долей классов с Wilson CI."""
    total = int(distrib["n"].sum())
    lines = []
    lines.append("% Auto-generated by src/eval/failure_taxonomy.py")
    lines.append("% Не редактировать вручную — перезапустить ноутбук-cell.")
    lines.append("\\begin{xltabular}{\\textwidth}{|X|r|r|r|}")
    lines.append(
        "\\caption{Таксономия ошибок каскада (LLM-consensus gold, "
        f"3 категории, всего ошибок {total}).}} \\\\ \\hline"
    )
    lines.append("    Класс ошибки & $n$ & Доля от ошибок & 95\\,\\% Wilson CI \\\\ \\hline")
    lines.append("\\endfirsthead")
    lines.append("\\multicolumn{4}{l}{\\textit{Продолжение таблицы}} \\\\ \\hline")
    lines.append("    Класс ошибки & $n$ & Доля от ошибок & 95\\,\\% Wilson CI \\\\ \\hline")
    lines.append("\\endhead")
    for _, r in distrib.iterrows():
        klass = CLASS_LABEL_RU.get(r["error_class"], r["error_class"])
        n = int(r["n"])
        share = r["share"]
        lo, hi = wilson_ci(n, total)
        lines.append(
            f"    {latex_escape(klass)} & {n} & {fmt_pct_comma(share)}\\,\\% "
            f"& [{fmt_pct_comma(lo)}; {fmt_pct_comma(hi)}]\\,\\% \\\\ \\hline"
        )
    lines.append(
        f"    \\textbf{{Всего}} & \\textbf{{{total}}} & \\textbf{{100,0\\,\\%}} & — \\\\ \\hline"
    )
    lines.append("\\end{xltabular}")
    return "\n".join(lines) + "\n"


def pick_examples(errors: pd.DataFrame, per_class: int = 2) -> pd.DataFrame:
    """Возьмёт по 1-2 примера каждого класса.

    Round-robin по category: для класса с per_class=2 берёт примеры из двух
    разных категорий по возможности; меняет стартовую категорию между
    классами, чтобы итоговая выборка покрывала все 3 категории.
    """
    rows = []
    for idx, klass in enumerate(CLASS_ORDER):
        sub = errors[errors["error_class"] == klass].copy()
        if sub.empty:
            continue
        sub = sub[sub["product_name"].notna() & (sub["product_name"].astype(str).str.len() > 3)]
        if sub.empty:
            continue
        # Сдвинутый порядок категорий: idx=0 → pasta,chocolate,cheeses;
        # idx=1 → chocolate,cheeses,pasta; и т. п. — равномерное покрытие.
        cat_order = CATEGORIES[idx % len(CATEGORIES):] + CATEGORIES[: idx % len(CATEGORIES)]
        picked_idx = []
        for cat in cat_order:
            cat_sub = sub[sub["category"] == cat].sort_values(["attr", "code"])
            for ridx in cat_sub.index:
                if len(picked_idx) >= per_class:
                    break
                picked_idx.append(ridx)
            if len(picked_idx) >= per_class:
                break
        # Если так и не набралось — добираем из остатка.
        if len(picked_idx) < per_class:
            remainder = sub[~sub.index.isin(picked_idx)].sort_values(
                ["category", "attr", "code"]
            )
            for ridx in remainder.index:
                if len(picked_idx) >= per_class:
                    break
                picked_idx.append(ridx)
        rows.append(sub.loc[picked_idx])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def render_examples_table(examples: pd.DataFrame) -> str:
    """Таблица с 5–10 примерами ошибок."""
    lines = []
    lines.append("% Auto-generated by src/eval/failure_taxonomy.py")
    lines.append("% Не редактировать вручную — перезапустить ноутбук-cell.")
    lines.append("\\begin{xltabular}{\\textwidth}{|p{0.20\\textwidth}|p{0.13\\textwidth}|p{0.13\\textwidth}|p{0.10\\textwidth}|p{0.10\\textwidth}|p{0.18\\textwidth}|}")
    lines.append(
        "\\caption{Иллюстративные примеры ошибок каскада по таксономии "
        "(LLM-consensus gold).} \\\\ \\hline"
    )
    lines.append(
        "    Класс ошибки & Категория, атрибут & Товар (фрагмент) & "
        "Эталон & Предсказание & Слой каскада \\\\ \\hline"
    )
    lines.append("\\endfirsthead")
    lines.append("\\multicolumn{6}{l}{\\textit{Продолжение таблицы}} \\\\ \\hline")
    lines.append(
        "    Класс ошибки & Категория, атрибут & Товар (фрагмент) & "
        "Эталон & Предсказание & Слой каскада \\\\ \\hline"
    )
    lines.append("\\endhead")
    for _, r in examples.iterrows():
        klass = CLASS_LABEL_RU.get(r["error_class"], r["error_class"])
        cat_attr = f"{r['category']}/{r['attr']}"
        pname = (str(r["product_name"]) if pd.notna(r["product_name"]) else "")[:60]
        gold = str(r["gold_value"]) if pd.notna(r["gold_value"]) else "—"
        pred = str(r["cascade_pred"]) if pd.notna(r["cascade_pred"]) else "—"
        layer = str(r["cascade_layer"])
        lines.append(
            f"    {latex_escape(klass)} & {latex_escape(cat_attr)} & "
            f"{latex_escape(pname)} & {latex_escape(gold)} & "
            f"{latex_escape(pred)} & {latex_escape(layer)} \\\\ \\hline"
        )
    lines.append("\\end{xltabular}")
    return "\n".join(lines) + "\n"


def main() -> None:
    print(f"[failure_taxonomy] PROCESSED = {PROCESSED}")
    print(f"[failure_taxonomy] TABLES_DIR = {TABLES_DIR}")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    errors = load_cascade_errors()
    print(f"[failure_taxonomy] cascade errors total: {len(errors)}")
    errors = attach_consensus_signal(errors)
    errors = attach_product_names(errors)

    errors["error_class"] = errors.apply(classify_error, axis=1)

    keep_cols = [
        "code", "category", "attr", "gold_value", "cascade_pred",
        "cascade_layer", "error_class", "product_name", "brands",
        "agreement_ratio", "disputed",
    ]
    keep_cols = [c for c in keep_cols if c in errors.columns]
    errors_out = errors[keep_cols].copy()
    errors_out.to_parquet(PROCESSED / "cascade_errors_taxonomy_v4.parquet", index=False)
    print(
        f"[failure_taxonomy] saved {PROCESSED / 'cascade_errors_taxonomy_v4.parquet'} "
        f"({len(errors_out)} rows)"
    )

    # Распределение классов
    distrib = (
        errors_out["error_class"]
        .value_counts()
        .rename_axis("error_class")
        .reset_index(name="n")
    )
    total = int(distrib["n"].sum())
    distrib["share"] = distrib["n"] / total
    distrib["error_class"] = pd.Categorical(
        distrib["error_class"], categories=CLASS_ORDER, ordered=True
    )
    distrib = distrib.sort_values("error_class").reset_index(drop=True)
    print("\n[failure_taxonomy] distribution:")
    for _, r in distrib.iterrows():
        lo, hi = wilson_ci(int(r["n"]), total)
        print(
            f"  {r['error_class']:24s}  n={int(r['n']):4d}  share={r['share']*100:5.2f}%  "
            f"CI=[{lo*100:5.2f}%;{hi*100:5.2f}%]"
        )

    # TeX таблица
    tex_dist = render_taxonomy_table(distrib)
    (TABLES_DIR / "failure_taxonomy.tex").write_text(tex_dist, encoding="utf-8")
    print(f"[failure_taxonomy] wrote {TABLES_DIR / 'failure_taxonomy.tex'}")

    # Примеры
    examples = pick_examples(errors_out, per_class=2)
    print(f"[failure_taxonomy] examples picked: {len(examples)}")
    tex_ex = render_examples_table(examples)
    (TABLES_DIR / "failure_examples.tex").write_text(tex_ex, encoding="utf-8")
    print(f"[failure_taxonomy] wrote {TABLES_DIR / 'failure_examples.tex'}")


if __name__ == "__main__":
    main()
