"""Рендер `images/layer_per_attribute_electronics.png` — распределение
работы каскада по атрибутам смартфонов (cold-start, без LLM).

Использует те же цвета и формат подписи, что и
`render_layer_per_attribute.py` (food categories): L1/L2/L3/L4 легенда,
проценты внутри баров. Источник:
`datasets/processed/catalog_completion_log_electronics_no_llm.parquet`.

В конфигурации no_llm слой L4 (LLM fallback) не вызывается, но ячейки
с layer="none" в production-режиме эскалировались бы в L4 — поэтому
показываем их оранжевым L4-сегментом для consistency со стилем food-чарта.

Запуск:
    OMP_NUM_THREADS=1 python -m src.figures.render_layer_per_attribute_electronics
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROCESSED = Path("datasets/processed")
SRC = PROCESSED / "catalog_completion_log_electronics_no_llm.parquet"
OUT = Path("images/layer_per_attribute_electronics.png")

# Целевые атрибуты с русскими названиями (порядок — как в DAG: predictor → leaf).
TARGET_ATTRS = [
    ("brand",              "бренд"),
    ("os",                 "ОС"),
    ("form_factor",        "форм-фактор"),
    ("ram_class",          "ОЗУ"),
    ("storage_class",      "память"),
    ("screen_size_class",  "диагональ"),
    ("release_year_class", "год выпуска"),
]

# Цвета и подписи слоёв (синхронизированы с render_layer_per_attribute.py).
LAYER_MAP = {"ml": "L2", "bayes": "L3", "none": "L4"}
LAYER_LABELS = {
    "L1": "L1 — regex",
    "L2": "L2 — ML (XGBoost)",
    "L3": "L3 — Bayes",
    "L4": "L4 — без ответа (escalate→LLM)",
}
LAYER_COLORS = {
    "L1": "#2ca02c",
    "L2": "#1f77b4",
    "L3": "#9467bd",
    "L4": "#ff7f0e",
}


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: {SRC} not found", file=sys.stderr)
        return 1

    df = pd.read_parquet(SRC)
    df = df[df["attr"].isin(dict(TARGET_ATTRS).keys())].copy()
    df["layer_code"] = df["cascade_layer"].map(LAYER_MAP).fillna("L4")

    rows = []
    for attr_key, attr_ru in TARGET_ATTRS:
        sub = df[df["attr"] == attr_key]
        total = len(sub)
        if total == 0:
            continue
        for lc in ("L1", "L2", "L3", "L4"):
            cnt = int((sub["layer_code"] == lc).sum())
            rows.append({"attr": attr_ru, "layer": lc,
                         "share": cnt / total * 100, "n": total})

    bd = pd.DataFrame(rows)
    ordered = [ru for _, ru in TARGET_ATTRS
               if (bd["attr"] == ru).any()]
    n_attrs = len(ordered)

    fig_h = max(3.5, 0.45 * n_attrs + 1.0)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.set_facecolor("white")

    for i, attr_ru in enumerate(ordered):
        sub = bd[bd["attr"] == attr_ru]
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
                        color="white", fontsize=11, fontweight="bold")
            left += v

    ax.set_yticks(range(n_attrs))
    ax.set_yticklabels(ordered, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Доля решений, %", fontsize=11)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="x", labelsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=LAYER_COLORS[lc])
        for lc in ("L1", "L2", "L3", "L4")
    ]
    legend_labels = [LAYER_LABELS[lc] for lc in ("L1", "L2", "L3", "L4")]
    ax.legend(legend_handles, legend_labels,
              loc="lower center", ncol=4, frameon=False,
              fontsize=10, bbox_to_anchor=(0.5, 1.01))

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight",
                metadata={"Date": None, "Software": None, "Creator": None})
    plt.close(fig)
    print(f"Saved {OUT} ({n_attrs} attrs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
