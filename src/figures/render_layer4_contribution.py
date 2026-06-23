"""Render images/layer4_contribution.png — слайд защиты «Слой 4: вклад в каскад».

Автономный аналог ячейки 31 (`5b`) нотбука notebooks/03_evaluate.ipynb, с двумя
исправлениями для презентации:
  1. per-attribute панель фильтруется по КАНОНИЧЕСКОЙ схеме 21 атрибута
     (отброшены TYPE_C `*.nutri_score_grade`, `cheeses.fat_class`, которых нет в
     финальной схеме — иначе на графике появляются «лишние» атрибуты);
  2. убран англоязычный suptitle «LLM fallback» (дублировал заголовок слайда),
     деанглизированы подписи панелей («router» / «Per-attribute»).

Источник: llm_fallback_eval_{cat}_stratified.parquet + cascade_plus_llm4_summary.parquet.

Запуск:
    OMP_NUM_THREADS=1 python -m src.figures.render_layer4_contribution
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score

PROCESSED = Path("datasets/processed")
IMAGES = Path("images")

# Каноническая схема V6 (21 атрибут) — docs/thesis/CANONICAL.md §4.
CANON_V6 = {
    "pasta": {"grain_type", "pasta_shape", "is_filled", "is_gluten_free",
              "is_organic", "is_vegan", "cuisine_origin", "protein_class"},
    "chocolate": {"chocolate_type", "is_filled", "chocolate_extra",
                  "contains_nuts", "is_organic", "flavor_profile"},
    "cheeses": {"milk_source", "texture", "country_of_origin", "aging",
                "is_pdo", "is_organic", "is_ultra_processed"},
}

# Палитра (приближённо к нотбучной PALETTE): хорошо/средне/плохо.
C_GOOD, C_MID, C_BAD = "#1f77b4", "#7e57c2", "#d68f3a"

MODEL_LABELS = {
    "gemini25flash": "Gemini 2.5 Flash",
    "sonnet45": "Claude Sonnet 4.5",
    "gpt4o": "GPT-4o",
    "gptoss": "gpt-oss-120b",
    "llama3b": "Llama-3.2-3B",
}


def main() -> int:
    cats = ["pasta", "chocolate", "cheeses"]
    fb = pd.concat(
        [pd.read_parquet(PROCESSED / f"llm_fallback_eval_{c}_stratified.parquet")
         for c in cats],
        ignore_index=True,
    )

    # per-attribute F1 Слоя 4 — только канонические атрибуты.
    per_attr = []
    dropped = []
    for (cat, attr), g_all in fb.groupby(["category", "attr"]):
        g = g_all[g_all["predicted_non_null"] == 1]
        if len(g) < 5:
            continue
        if attr not in CANON_V6.get(cat, set()):
            dropped.append(f"{cat}/{attr}")
            continue
        f1 = f1_score(g["gt"].astype(str), g["pred"].astype(str),
                      average="macro", zero_division=0)
        per_attr.append({"key": f"{cat}/{attr}", "n": len(g), "f1": f1})
    per_attr_df = (pd.DataFrame(per_attr)
                   .sort_values("f1", ascending=False).reset_index(drop=True))
    if dropped:
        print("Отброшены неканонические атрибуты:", ", ".join(sorted(dropped)))

    summ = (pd.read_parquet(PROCESSED / "cascade_plus_llm4_summary.parquet")
            .sort_values("delta_layer4_pp", ascending=False).reset_index(drop=True))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    # (a) Δ Слоя 4 по 5 БЯМ
    ax = axes[0]
    labels = [MODEL_LABELS.get(m, m) for m in summ["llm_model"]]
    deltas = summ["delta_layer4_pp"].values
    colors_bar = [C_GOOD if v >= 6 else (C_MID if v >= 5 else C_BAD) for v in deltas]
    bars = ax.barh(labels, deltas, color=colors_bar, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, deltas):
        ax.text(v + 0.12, b.get_y() + b.get_height() / 2,
                f"+{v:.2f} пп", va="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Δ сквозной точности от Слоя 4, п.п.", fontsize=11)
    ax.set_xlim(0, float(deltas.max()) * 1.20)
    ax.invert_yaxis()
    ax.tick_params(labelsize=11)
    ax.set_title(f"Вклад Слоя 4 в сквозную точность (n={int(summ['n_test_total'].iloc[0])})",
                 fontsize=11.5)

    # (b) Поатрибутный F1 Слоя 4
    ax = axes[1]
    labels2 = per_attr_df["key"].values
    f1s = per_attr_df["f1"].values * 100
    colors2 = [C_GOOD if v >= 70 else (C_MID if v >= 50 else C_BAD) for v in f1s]
    bars = ax.barh(labels2, f1s, color=colors2, edgecolor="black", linewidth=0.5)
    for b, v, n in zip(bars, f1s, per_attr_df["n"].values):
        ax.text(v + 0.8, b.get_y() + b.get_height() / 2,
                f"{v:.0f}% (n={n})", va="center", fontsize=10)
    ax.set_xlabel("F1-macro Слоя 4, %", fontsize=11)
    ax.set_xlim(0, 112)
    ax.invert_yaxis()
    ax.tick_params(labelsize=11)
    ax.set_title("Поатрибутный F1 Слоя 4 на отказных ячейках", fontsize=11.5)

    plt.tight_layout()
    out = IMAGES / "layer4_contribution.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}  ({len(per_attr_df)} канонических атрибутов)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
