"""Регенерация Рисунка/слайда «Вклад слоёв каскада по атрибутам».

Источник: `cascade_preds_{cat}_gold.parquet` — per-attribute cascade layer
breakdown на LLM-consensus gold (n=3257).

ВАЖНО: фильтруем по CANONICAL V6 production schema (см.
`docs/thesis/CANONICAL.md` §4). Из локальных parquet удаляются deprecated
классы V4/V5 (chocolate.cocoa_percentage, *.nutri_score_grade,
cheeses.fat_class, chocolate.protein_class), которые не входят в финальную
схему 21 атрибута. Если локальные parquet не покрывают полный CANONICAL
(missing canonical attrs), скрипт печатает WARNING — для полной картины
нужен пересборка на VM:
    python -m src.eval.end_to_end --consensus
    rsync pull datasets/processed/cascade_preds_{cat}_gold.parquet

Запуск:
    OMP_NUM_THREADS=1 python -m src.figures.render_layer_per_attribute
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROCESSED = Path("datasets/processed")
OUT = Path("images/layer_per_attribute.png")
OUT_FILTERED = Path("images/layer_per_attribute_filtered.png")
# Если L1 покрывает >= L1_EXCLUDE_THRESHOLD % атрибута, считаем его тривиальным
# и не показываем на фильтрованном графике (захламляет визуализацию каскада).
L1_EXCLUDE_THRESHOLD = 99.0

# CANONICAL V6 production schema (docs/thesis/CANONICAL.md §4).
# 21 атрибут (20 в headline + опциональный pasta.protein_class).
CANON_V6 = {
    "pasta": [
        "grain_type", "pasta_shape", "is_filled", "is_gluten_free",
        "is_organic", "is_vegan", "cuisine_origin", "protein_class",
    ],
    "chocolate": [
        "chocolate_type", "is_filled", "chocolate_extra",
        "contains_nuts", "is_organic", "flavor_profile",
    ],
    "cheeses": [
        "milk_source", "texture", "country_of_origin", "aging",
        "is_pdo", "is_organic", "is_ultra_processed",
    ],
}

# Маппинг внутренних имён слоёв (parquet `cascade_layer`) → слой каскада.
# rule_h   → Layer 1 (regex high-precision)
# ml       → Layer 2 (ML XGBoost)
# rule_l   → Layer 3 (regex low-precision)
# fallback → Layer 4 (LLM fallback / abstain)
# Legacy aliases (cascade_preds от старого pipeline, до 2026-05-27).
LAYER_MAP = {
    "rule_h": "L1", "regex": "L1",
    "ml": "L2",
    "rule_l": "L3", "bayes": "L3",
    "fallback": "L4", "none": "L4",
}
LAYER_LABELS = {
    "L1": "L1 — regex",
    "L2": "L2 — ML (XGBoost)",
    "L3": "L3 — Bayes",
    "L4": "L4 — LLM fallback",
}
LAYER_COLORS = {
    "L1": "#2ca02c",   # зелёный (regex)
    "L2": "#1f77b4",   # синий (ML)
    "L3": "#9467bd",   # фиолетовый (Bayes)
    "L4": "#ff7f0e",   # оранжевый (LLM)
}


def load_per_attribute_layers() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Собирает long-DF (cat, attr, layer, share) по локальным parquet'ам.

    Возвращает (df, warnings_missing, warnings_deprecated).
    """
    rows = []
    missing: list[str] = []
    deprecated: list[str] = []

    for cat, canon_attrs in CANON_V6.items():
        path = PROCESSED / f"cascade_preds_{cat}_gold.parquet"
        if not path.exists():
            missing.append(f"{cat}: parquet {path.name} not found")
            continue
        df = pd.read_parquet(path)
        have = set(df["attr"].unique())
        canon_set = set(canon_attrs)
        # Лог deprecated/missing.
        for a in sorted(have - canon_set):
            deprecated.append(f"{cat}.{a}")
        for a in sorted(canon_set - have):
            missing.append(f"{cat}.{a}")
        # Только canonical attrs.
        df_c = df[df["attr"].isin(canon_set)].copy()
        if df_c.empty:
            continue
        df_c["layer_code"] = df_c["cascade_layer"].map(LAYER_MAP).fillna("L4")
        for attr, sub in df_c.groupby("attr"):
            total = len(sub)
            for lc in ("L1", "L2", "L3", "L4"):
                cnt = int((sub["layer_code"] == lc).sum())
                rows.append({
                    "cat": cat,
                    "attr": attr,
                    "layer": lc,
                    "share": cnt / total * 100 if total else 0.0,
                    "n": total,
                })
    df_out = pd.DataFrame(rows)
    return df_out, missing, deprecated


def _render(df: pd.DataFrame, ordered: list[tuple[str, str]], out_path: Path,
            trivial_note: str | None = None) -> None:
    """Рисует horizontal stacked bar по списку (cat, attr) → файл out_path."""
    n_attrs = len(ordered)
    fig_h = max(4.5, 0.32 * n_attrs + 1.2) + (0.45 if trivial_note else 0.0)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.set_facecolor("white")

    y_positions = list(range(n_attrs))
    y_labels = [f"{cat}.{attr}" for cat, attr in ordered]

    for i, (cat, attr) in enumerate(ordered):
        sub = df[(df["cat"] == cat) & (df["attr"] == attr)]
        left = 0.0
        for lc in ("L1", "L2", "L3", "L4"):
            v = float(sub[sub["layer"] == lc]["share"].sum())
            if v < 0.5:
                left += v
                continue
            ax.barh(i, v, left=left, height=0.72,
                    color=LAYER_COLORS[lc], edgecolor="white", linewidth=0.6)
            if v >= 6:
                ax.text(left + v / 2, i, f"{int(round(v))}%",
                        ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
            left += v

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Доля решений, %", fontsize=10)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="x", labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    # Легенда сверху.
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=LAYER_COLORS[lc]) for lc in ("L1", "L2", "L3", "L4")
    ]
    legend_labels = [LAYER_LABELS[lc] for lc in ("L1", "L2", "L3", "L4")]
    ax.legend(legend_handles, legend_labels,
              loc="lower center", ncol=4, frameon=False,
              fontsize=10, bbox_to_anchor=(0.5, 1.01))

    if trivial_note:
        # Добавляем подпись внизу для фильтрованной версии.
        fig.text(0.5, 0.005, trivial_note, ha="center", va="bottom",
                 fontsize=9, style="italic", color="black")
        plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                metadata={"Date": None, "Software": None, "Creator": None})
    plt.close(fig)


def main() -> int:
    df, missing, deprecated = load_per_attribute_layers()

    print(f"Loaded per-attribute layer breakdown: {len(df)} rows "
          f"({df['attr'].nunique()} unique attrs × 4 layers).")
    if deprecated:
        print()
        print("⚠️  WARNING: parquet содержит deprecated классы V4/V5, "
              "которых нет в CANONICAL V6 — они исключены из графика:")
        for d in deprecated:
            print(f"     - {d}")
    if missing:
        print()
        print("⚠️  WARNING: CANONICAL V6 атрибутов нет в локальных parquet — "
              "график неполный. Для полного breakdown нужно перезапустить на VM "
              "(python -m src.eval.end_to_end) и pull cascade_preds_*_gold.parquet:")
        for m in missing:
            print(f"     - {m}")

    # Полный список (cat, attr) в порядке CANON для устойчивого отображения.
    ordered = []
    for cat, attrs in CANON_V6.items():
        for a in attrs:
            if ((df["cat"] == cat) & (df["attr"] == a)).any():
                ordered.append((cat, a))
    n_attrs = len(ordered)
    print(f"\nИтого атрибутов на графике: {n_attrs} "
          f"(из {sum(len(v) for v in CANON_V6.values())} в CANONICAL V6).")

    if n_attrs == 0:
        print("ERROR: нет данных для рендера.")
        return 1

    # Полная версия (как было).
    _render(df, ordered, OUT, trivial_note=None)
    print(f"\nSaved {OUT}  ({n_attrs} attrs)")

    # Фильтрованная версия — без тривиальных (L1 >= L1_EXCLUDE_THRESHOLD).
    trivial: list[tuple[str, str]] = []
    non_trivial: list[tuple[str, str]] = []
    for cat, attr in ordered:
        sub = df[(df["cat"] == cat) & (df["attr"] == attr) & (df["layer"] == "L1")]
        l1_share = float(sub["share"].sum()) if len(sub) else 0.0
        if l1_share >= L1_EXCLUDE_THRESHOLD:
            trivial.append((cat, attr))
        else:
            non_trivial.append((cat, attr))

    if trivial:
        names = ", ".join(f"{c}.{a}" for c, a in trivial)
        note = f"Исключены {len(trivial)} тривиальных атрибута (100 % L1 regex): {names}"
        _render(df, non_trivial, OUT_FILTERED, trivial_note=note)
        print(f"Saved {OUT_FILTERED}  ({len(non_trivial)} attrs; "
              f"исключено {len(trivial)}: {names})")
    else:
        _render(df, non_trivial, OUT_FILTERED, trivial_note=None)
        print(f"Saved {OUT_FILTERED}  ({len(non_trivial)} attrs; "
              f"тривиальных не найдено)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
