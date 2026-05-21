"""Landscape variant of Рисунок 3.1 — for presentation slide.

Same ГОСТ 19.701-90 symbols but flow goes left → right (compact horizontal),
fits widescreen 16:9 slide without overlapping side text.

Usage:
    OMP_NUM_THREADS=1 python -m src.figures.render_algorithm_flowchart_landscape
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "thesis" / "figures"

rcParams["font.family"] = "Times New Roman"
rcParams["font.size"] = 10

STROKE = "#1f1f4d"
LW_THICK = 1.1
LW_THIN = 0.7


def _terminator(ax, cx, cy, w, h, label, *, font_size=10):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={h / 2}",
        linewidth=LW_THICK, edgecolor=STROKE, facecolor="white", zorder=2,
    )
    ax.add_patch(box)
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=font_size, color=STROKE, zorder=3)
    return (cx, cy, w, h)


def _process(ax, cx, cy, w, h, label, *, font_size=10):
    ax.add_patch(Rectangle(
        (cx - w / 2, cy - h / 2), w, h,
        linewidth=LW_THICK, edgecolor=STROKE, facecolor="white", zorder=2,
        joinstyle="miter",
    ))
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=font_size, color=STROKE, zorder=3)
    return (cx, cy, w, h)


def _decision(ax, cx, cy, w, h, label, *, font_size=9):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy),
           (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True,
                         linewidth=LW_THICK, edgecolor=STROKE,
                         facecolor="white", zorder=2, joinstyle="miter"))
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=font_size, color=STROKE, zorder=3)
    return (cx, cy, w, h)


def _io(ax, cx, cy, w, h, label, *, skew=0.20, font_size=10):
    s = skew
    pts = [(cx - w / 2 + s, cy + h / 2),
           (cx + w / 2,     cy + h / 2),
           (cx + w / 2 - s, cy - h / 2),
           (cx - w / 2,     cy - h / 2)]
    ax.add_patch(Polygon(pts, closed=True,
                         linewidth=LW_THICK, edgecolor=STROKE,
                         facecolor="white", zorder=2, joinstyle="miter"))
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=font_size, color=STROKE, zorder=3)
    return (cx, cy, w, h)


def _arrow(ax, p_from, p_to, *, label=None, label_offset=(0, 0)):
    arrow = FancyArrowPatch(
        p_from, p_to,
        arrowstyle="-|>,head_length=6,head_width=4",
        linewidth=LW_THIN, color=STROKE,
        connectionstyle="arc3,rad=0", zorder=2,
    )
    ax.add_patch(arrow)
    if label:
        mx = (p_from[0] + p_to[0]) / 2 + label_offset[0]
        my = (p_from[1] + p_to[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, fontsize=9, color=STROKE,
                ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5),
                zorder=4)


def _ortho_arrow(ax, p_from, p_to, *, label=None, label_pos=None, route="HV"):
    sx, sy = p_from
    tx, ty = p_to
    if route == "HV":
        corner = (tx, sy)
    elif route == "VH":
        corner = (sx, ty)
    else:
        raise ValueError(route)
    ax.plot([sx, corner[0]], [sy, corner[1]],
            color=STROKE, linewidth=LW_THIN, zorder=2, solid_joinstyle="miter")
    arrow = FancyArrowPatch(
        corner, p_to,
        arrowstyle="-|>,head_length=6,head_width=4",
        linewidth=LW_THIN, color=STROKE,
        connectionstyle="arc3,rad=0", zorder=2,
    )
    ax.add_patch(arrow)
    if label and label_pos:
        ax.text(label_pos[0], label_pos[1], label,
                fontsize=9, color=STROKE,
                ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5),
                zorder=4)


def render(out_name: str = "fig_3_1_algorithm_flowchart_landscape.png") -> Path:
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 14)
    ax.set_aspect("equal")
    ax.axis("off")

    Y_MAIN = 8.0
    Y_BRANCH_DOWN = 3.5

    PROC_W, PROC_H = 4.0, 1.6
    DEC_W, DEC_H = 3.8, 2.4
    TERM_W, TERM_H = 2.4, 1.4
    IO_W, IO_H = 3.6, 1.4

    # Horizontal positions along the main flow
    xs = [1.7, 5.0, 8.6, 12.5, 16.4, 20.3, 24.2, 28.4, 30.5]

    # Main chain
    _terminator(ax, xs[0], Y_MAIN, TERM_W, TERM_H, "Начало")
    _io(ax,         xs[1], Y_MAIN, IO_W, IO_H, "Запрос\nпартнёра")
    _process(ax,    xs[2], Y_MAIN, PROC_W, PROC_H, "Валидация\nзапроса")
    _decision(ax,   xs[3], Y_MAIN, DEC_W, DEC_H, "Запрос\nвалиден?")
    _process(ax,    xs[4], Y_MAIN, PROC_W, PROC_H, "Слои 0–3\n(категория →\nregex → ML → Bayes)", font_size=9)
    _decision(ax,   xs[5], Y_MAIN, DEC_W, DEC_H, "Все\nуверены?")
    _process(ax,    xs[6], Y_MAIN, PROC_W, PROC_H, "Сборка\nответа")
    _io(ax,         xs[7], Y_MAIN, IO_W, IO_H, "Атрибуты\nпартнёру")
    _terminator(ax, xs[8], Y_MAIN, TERM_W, TERM_H, "Конец")

    # Branch processes (below)
    _process(ax, xs[3], Y_BRANCH_DOWN, PROC_W, PROC_H, "Возврат\nошибки HTTP 400")
    _process(ax, xs[5], Y_BRANCH_DOWN, PROC_W + 0.6, PROC_H,
             "Слой 4 — Запасной слой\nбольшой языковой модели", font_size=9)

    # Main chain arrows
    pairs = [
        (xs[0], TERM_W, xs[1], IO_W),
        (xs[1], IO_W,   xs[2], PROC_W),
        (xs[2], PROC_W, xs[3], DEC_W),
    ]
    for x1, w1, x2, w2 in pairs:
        _arrow(ax, (x1 + w1 / 2, Y_MAIN), (x2 - w2 / 2, Y_MAIN))

    # Decision Валиден? — Да (right to Слои 0–3)
    _arrow(ax, (xs[3] + DEC_W / 2, Y_MAIN), (xs[4] - PROC_W / 2, Y_MAIN),
           label="Да", label_offset=(0, 0.35))

    # Слои → Все уверены?
    _arrow(ax, (xs[4] + PROC_W / 2, Y_MAIN), (xs[5] - DEC_W / 2, Y_MAIN))

    # Decision Уверены? — Да (right to Сборка)
    _arrow(ax, (xs[5] + DEC_W / 2, Y_MAIN), (xs[6] - PROC_W / 2, Y_MAIN),
           label="Да", label_offset=(0, 0.35))

    # Сборка → Атрибуты → Конец
    _arrow(ax, (xs[6] + PROC_W / 2, Y_MAIN), (xs[7] - IO_W / 2, Y_MAIN))
    _arrow(ax, (xs[7] + IO_W / 2, Y_MAIN), (xs[8] - TERM_W / 2, Y_MAIN))

    # Decision Валиден? — Нет (down to error)
    _arrow(ax, (xs[3], Y_MAIN - DEC_H / 2),
           (xs[3], Y_BRANCH_DOWN + PROC_H / 2),
           label="Нет", label_offset=(0.4, 0))

    # Error → Конец (down then right)
    _ortho_arrow(ax,
                 (xs[3], Y_BRANCH_DOWN - PROC_H / 2),
                 (xs[8] - TERM_W / 2, Y_MAIN),
                 route="HV")

    # Decision Уверены? — Нет (down to Layer 4)
    _arrow(ax, (xs[5], Y_MAIN - DEC_H / 2),
           (xs[5], Y_BRANCH_DOWN + PROC_H / 2),
           label="Нет", label_offset=(0.4, 0))

    # Layer 4 → Сборка (up to main flow at xs[6])
    _ortho_arrow(ax,
                 (xs[5] + (PROC_W + 0.6) / 2, Y_BRANCH_DOWN),
                 (xs[6], Y_MAIN - PROC_H / 2),
                 route="HV")

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / out_name
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.2, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    render()
