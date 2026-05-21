"""Render algorithm flowchart per ГОСТ 19.701-90 (ISO 5807) — Рисунок 3.1.

Symbols:
- Terminator (Start/End): stadium / rounded rectangle
- Process: plain rectangle
- Decision: rhombus (diamond)
- I/O (data input/output): parallelogram

Flow direction: top to bottom; "No" branches go right and merge back.

Usage:
    OMP_NUM_THREADS=1 python -m src.figures.render_algorithm_flowchart
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


# ─── primitives ──────────────────────────────────────────────────────────────

def _terminator(ax, cx, cy, w, h, label, *, font_size=10):
    """Start/End: rounded rectangle with stadium shape (radius = h/2)."""
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
    """Process: plain rectangle."""
    ax.add_patch(Rectangle(
        (cx - w / 2, cy - h / 2), w, h,
        linewidth=LW_THICK, edgecolor=STROKE, facecolor="white", zorder=2,
        joinstyle="miter",
    ))
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=font_size, color=STROKE, zorder=3)
    return (cx, cy, w, h)


def _decision(ax, cx, cy, w, h, label, *, font_size=9):
    """Decision: rhombus / diamond."""
    pts = [(cx, cy + h / 2), (cx + w / 2, cy),
           (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True,
                         linewidth=LW_THICK, edgecolor=STROKE,
                         facecolor="white", zorder=2, joinstyle="miter"))
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=font_size, color=STROKE, zorder=3)
    return (cx, cy, w, h)


def _io(ax, cx, cy, w, h, label, *, skew=0.35, font_size=10):
    """Input/Output: parallelogram (skewed rectangle)."""
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
    """Plain arrow with optional label."""
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
    """L-shape arrow: HV = horizontal then vertical; VH = vertical then horizontal."""
    sx, sy = p_from
    tx, ty = p_to
    if route == "HV":
        corner = (tx, sy)
    elif route == "VH":
        corner = (sx, ty)
    elif route == "HVH":  # 3-segment for joining back
        mx = (sx + tx) / 2
        ax.plot([sx, mx, mx, tx], [sy, sy, ty, ty],
                color=STROKE, linewidth=LW_THIN, zorder=2,
                solid_joinstyle="miter")
        # Add arrowhead on last segment
        arrow = FancyArrowPatch(
            (mx, ty), p_to,
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
        return
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


# ─── scene ───────────────────────────────────────────────────────────────────

def render(out_name: str = "fig_3_1_algorithm_flowchart.png") -> Path:
    fig, ax = plt.subplots(figsize=(11, 14))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(-1, 18)
    ax.set_aspect("equal")
    ax.axis("off")

    X_MAIN = 6.0
    X_BRANCH = 12.0
    PROC_W, PROC_H = 5.6, 1.0
    DEC_W, DEC_H = 5.0, 1.6
    TERM_W, TERM_H = 3.2, 0.9
    IO_W, IO_H = 5.6, 1.0

    # Flow positions (top to bottom)
    y_start = 17.0
    y_io_in = 15.5
    y_valid = 14.0
    y_dec_valid = 12.3
    y_l0 = 10.4
    y_l1 = 9.0
    y_l2 = 7.6
    y_l3 = 6.2
    y_dec_conf = 4.4
    y_join = 2.5
    y_assemble = 2.5
    y_io_out = 1.0
    y_end = -0.3

    # Blocks
    _terminator(ax, X_MAIN, y_start, TERM_W, TERM_H, "Начало")
    _io(ax, X_MAIN, y_io_in, IO_W, IO_H, "Запрос партнёра")
    _process(ax, X_MAIN, y_valid, PROC_W, PROC_H, "Валидация запроса")
    _decision(ax, X_MAIN, y_dec_valid, DEC_W, DEC_H, "Запрос валиден?")

    _process(ax, X_MAIN, y_l0, PROC_W, PROC_H,
             "Слой 0 — Маршрутизация по категории")
    _process(ax, X_MAIN, y_l1, PROC_W, PROC_H,
             "Слой 1 — Извлечение правилами")
    _process(ax, X_MAIN, y_l2, PROC_W, PROC_H,
             "Слой 2 — Классификация SBERT + XGBoost")
    _process(ax, X_MAIN, y_l3, PROC_W, PROC_H,
             "Слой 3 — Валидация байесовской сетью")

    _decision(ax, X_MAIN, y_dec_conf, DEC_W, DEC_H,
              "Все атрибуты уверены?", font_size=9)

    # Branch: Layer 4 LLM (right side)
    _process(ax, X_BRANCH, y_dec_conf, PROC_W - 0.5, PROC_H,
             "Слой 4 — Запасной слой\nбольшой языковой модели", font_size=9)

    # Branch: Error return (right side)
    _process(ax, X_BRANCH, y_dec_valid, PROC_W - 0.5, PROC_H,
             "Возврат ошибки\nHTTP 400")

    _process(ax, X_MAIN, y_assemble, PROC_W, PROC_H, "Сборка ответа")
    _io(ax, X_MAIN, y_io_out, IO_W, IO_H, "Атрибуты партнёру")
    _terminator(ax, X_MAIN, y_end, TERM_W, TERM_H, "Конец")

    # ─── arrows (vertical chain) ─────────────────────────────────────────────
    pairs_main = [
        (y_start, y_io_in, TERM_H, IO_H),
        (y_io_in, y_valid, IO_H, PROC_H),
        (y_valid, y_dec_valid, PROC_H, DEC_H),
    ]
    for y1, y2, h1, h2 in pairs_main:
        _arrow(ax, (X_MAIN, y1 - h1 / 2), (X_MAIN, y2 + h2 / 2))

    # Decision Валиден? — Да (down)
    _arrow(ax, (X_MAIN, y_dec_valid - DEC_H / 2),
           (X_MAIN, y_l0 + PROC_H / 2), label="Да", label_offset=(0.25, 0))

    # Decision Валиден? — Нет (right to error)
    _ortho_arrow(ax,
                 (X_MAIN + DEC_W / 2, y_dec_valid),
                 (X_BRANCH - (PROC_W - 0.5) / 2, y_dec_valid),
                 label="Нет", label_pos=(X_MAIN + DEC_W / 2 + 0.6, y_dec_valid + 0.25))

    # Error → End (right column down, then merge to End)
    _ortho_arrow(ax,
                 (X_BRANCH, y_dec_valid - PROC_H / 2),
                 (X_MAIN + TERM_W / 2, y_end),
                 route="VH")

    # Layers chain (L0 → L1 → L2 → L3)
    for y1, y2 in [(y_l0, y_l1), (y_l1, y_l2), (y_l2, y_l3)]:
        _arrow(ax, (X_MAIN, y1 - PROC_H / 2), (X_MAIN, y2 + PROC_H / 2))

    # L3 → Decision Уверены?
    _arrow(ax, (X_MAIN, y_l3 - PROC_H / 2),
           (X_MAIN, y_dec_conf + DEC_H / 2))

    # Decision Уверены? — Нет (right to Layer 4)
    _ortho_arrow(ax,
                 (X_MAIN + DEC_W / 2, y_dec_conf),
                 (X_BRANCH - (PROC_W - 0.5) / 2, y_dec_conf),
                 label="Нет", label_pos=(X_MAIN + DEC_W / 2 + 0.6, y_dec_conf + 0.25))

    # Decision Уверены? — Да (down to Сборка)
    _arrow(ax, (X_MAIN, y_dec_conf - DEC_H / 2),
           (X_MAIN, y_assemble + PROC_H / 2),
           label="Да", label_offset=(0.25, 0))

    # Layer 4 → Сборка (down then left)
    _ortho_arrow(ax,
                 (X_BRANCH, y_dec_conf - PROC_H / 2),
                 (X_MAIN + PROC_W / 2, y_assemble),
                 route="VH")

    # Сборка → I/O out → End
    _arrow(ax, (X_MAIN, y_assemble - PROC_H / 2),
           (X_MAIN, y_io_out + IO_H / 2))
    _arrow(ax, (X_MAIN, y_io_out - IO_H / 2),
           (X_MAIN, y_end + TERM_H / 2))

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / out_name
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.2, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    render()
