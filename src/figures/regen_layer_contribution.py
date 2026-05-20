"""Регенерация Рисунка 3.4 — Послойный вклад каскада.

Источник чисел: `cascade_preds_{cat}_after_fix.parquet` (доля каждого слоя в
финальных предсказаниях, по полному набору ячеек cascade_preds, не только gold).

Цветовая схема и компоновка повторяют исходное изображение
`docs/thesis/figures/pptx/layer_contribution.png`.

Запуск:
    OMP_NUM_THREADS=1 python -m src.figures.regen_layer_contribution
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROCESSED = Path("datasets/processed")
OUT = Path("docs/thesis/figures/pptx/layer_contribution.png")

CATS = [
    ("паста", "pasta"),
    ("шоколад", "chocolate"),
    ("сыры", "cheeses"),
]

# Цвета из исходного PNG (приближённо).
COLOR_REGEX = "#2c4a6e"   # тёмно-синий
COLOR_ML = "#3d8278"      # бирюзовый/teal
COLOR_LLM = "#d68f3a"     # оранжевый
BG = "#f6efe3"            # фон, кремовый

LAYER_ORDER = ["regex", "ml", "abstain"]
LAYER_COLOR = {"regex": COLOR_REGEX, "ml": COLOR_ML, "abstain": COLOR_LLM}
LAYER_LABEL = {
    "regex": "Слой 1 — регулярные выражения",
    "ml": "Слой 2 — машинное обучение",
    "abstain": "Слой 4 — запасной слой на LLM",
}


def _round_to_sum_100(values: dict[str, float]) -> dict[str, int]:
    """Largest-remainder method: округление до int с гарантией Σ = 100."""
    raw = {k: v * 100 for k, v in values.items()}
    floors = {k: int(v) for k, v in raw.items()}
    remainders = {k: raw[k] - floors[k] for k in raw}
    diff = 100 - sum(floors.values())
    # Раздаём недостающие единицы тем сегментам, у которых дробная часть наибольшая.
    for k in sorted(remainders, key=lambda x: -remainders[x]):
        if diff <= 0:
            break
        floors[k] += 1
        diff -= 1
    return floors


def load_layer_shares() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for ru_label, cat in CATS:
        df = pd.read_parquet(PROCESSED / f"cascade_preds_{cat}_after_fix.parquet")
        vc = df["layer"].value_counts(normalize=True)
        out[ru_label] = {layer: float(vc.get(layer, 0.0)) for layer in LAYER_ORDER}
    return out


def main() -> int:
    shares = load_layer_shares()

    # Широкая горизонтальная компоновка: метки категорий слева, бары справа,
    # одна общая ось внизу. Один Axes с тремя горизонтальными барами.
    fig, ax = plt.subplots(figsize=(20, 7.5), facecolor=BG)
    ax.set_facecolor(BG)

    y_positions = [2, 1, 0]  # сверху вниз: паста, шоколад, сыры
    bar_height = 0.55

    for y, (ru_label, _) in zip(y_positions, CATS):
        s = shares[ru_label]
        labels_int = _round_to_sum_100(s)
        left = 0.0
        for layer in LAYER_ORDER:
            v = s[layer] * 100
            if v < 0.4:
                left += v
                continue
            ax.barh(y, v, left=left, height=bar_height,
                    color=LAYER_COLOR[layer], edgecolor="none")
            label = f"{labels_int[layer]} %"
            # Большой сегмент — подпись внутри белым. Узкий сегмент (<8 %) —
            # подпись над баром цветом сегмента, чтобы не вылезать за границы.
            if v >= 8:
                ax.text(left + v / 2, y, label, ha="center", va="center",
                        color="white", fontsize=28, fontweight="bold")
            else:
                ax.text(left + v / 2, y + bar_height / 2 + 0.08, label,
                        ha="center", va="bottom",
                        color=LAYER_COLOR[layer],
                        fontsize=16, fontweight="bold")
            left += v

    # Метки категорий слева — крупные, жирные.
    for y, (ru_label, _) in zip(y_positions, CATS):
        ax.text(-2.5, y, ru_label, ha="right", va="center",
                fontsize=32, fontweight="bold", color="#2a2a2a")
        ax.text(-2.5, y - 0.30, "КАТЕГОРИЯ", ha="right", va="center",
                fontsize=11, color="#888888", fontweight="normal",
                family="sans-serif")

    ax.set_xlim(-26, 102)
    ax.set_ylim(-0.7, 2.7)
    ax.set_yticks([])
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(["0", "20", "40", "60", "80", "100"],
                       color="#888888", fontsize=12)
    ax.tick_params(axis="x", colors="#888888", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("доля финальных предсказаний, %",
                  color="#888888", fontsize=12, labelpad=10)
    ax.xaxis.set_label_coords(0.55, -0.08)
    # Ограничить ось x только зоной баров (0–100), без захода в область меток
    ax.set_xticks([0, 20, 40, 60, 80, 100])

    # Маскировка тиков в области меток слева (отрицательные x для текста)
    ax.spines["bottom"].set_visible(False)
    # Тонкая линия-разделитель под осью только в зоне баров
    ax.plot([0, 100], [-0.55, -0.55], color="#cccccc", linewidth=0.8)

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=LAYER_COLOR[k])
                      for k in LAYER_ORDER]
    legend_labels = [LAYER_LABEL[k] for k in LAYER_ORDER]
    ax.legend(legend_handles, legend_labels,
              loc="lower center", ncol=3, frameon=False,
              fontsize=14, bbox_to_anchor=(0.55, -0.22))

    plt.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.18)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=BG, bbox_inches="tight")
    print(f"Saved {OUT}")
    print("Shares used:")
    for cat in shares:
        pretty = {k: f"{v*100:.1f}%" for k, v in shares[cat].items()}
        print(f"  {cat}: {pretty}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
