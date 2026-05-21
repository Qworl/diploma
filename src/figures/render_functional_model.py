"""Render functional model (DFD-style) — Рисунок 2.1.

Horizontal data-flow layout:
  [Партнёр] → [Шлюз] → [Слой 0 → 1 → 2 → 3 → 4] → [Витрина]
                                ↓     ↓     ↑
                              [XGB] [Bayes] [API LLM]

Usage:
    OMP_NUM_THREADS=1 python -m src.figures.render_functional_model
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import (
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    Rectangle,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "thesis" / "figures"

rcParams["font.family"] = "Times New Roman"
rcParams["font.size"] = 10

STROKE = "#1f1f4d"
LW_THICK = 1.1
LW_THIN = 0.7


# ─── primitives ──────────────────────────────────────────────────────────────

def _container(ax, x, y, w, h, title, *, font_size=11):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.25",
        linewidth=LW_THICK, edgecolor=STROKE, facecolor="white", zorder=1,
    )
    ax.add_patch(box)
    ax.text(x + 0.3, y + h - 0.32, title, fontsize=font_size,
            fontstyle="italic", color=STROKE, ha="left", va="top", zorder=4)


def _block(ax, x, y, w, h, label, *, font_size=10):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.12",
        linewidth=LW_THICK, edgecolor=STROKE, facecolor="white", zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=font_size, color=STROKE, zorder=3)
    return (x, y, w, h)


def _actor(ax, cx, cy, w, h, label, *, font_size=10):
    el = Ellipse((cx, cy), w, h, linewidth=LW_THICK, edgecolor=STROKE,
                 facecolor="white", zorder=2)
    ax.add_patch(el)
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=font_size, color=STROKE, zorder=3)
    return (cx, cy, w, h)


def _cylinder(ax, x, y, w, h, label, *, font_size=10):
    cap_h = min(0.36, h * 0.28)
    ax.add_patch(Rectangle((x, y + cap_h / 2), w, h - cap_h,
                            linewidth=0, facecolor="white", zorder=2))
    ax.plot([x, x], [y + cap_h / 2, y + h - cap_h / 2],
            color=STROKE, linewidth=LW_THICK, zorder=2)
    ax.plot([x + w, x + w], [y + cap_h / 2, y + h - cap_h / 2],
            color=STROKE, linewidth=LW_THICK, zorder=2)
    bot = Ellipse((x + w / 2, y + cap_h / 2), w, cap_h,
                   linewidth=LW_THICK, edgecolor=STROKE, facecolor="white", zorder=2)
    ax.add_patch(bot)
    top = Ellipse((x + w / 2, y + h - cap_h / 2), w, cap_h,
                   linewidth=LW_THICK, edgecolor=STROKE, facecolor="white", zorder=3)
    ax.add_patch(top)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=font_size, color=STROKE, zorder=4)
    return (x, y, w, h)


def _arrow(ax, p_from, p_to, label="", *, rad=0.0, route="straight",
           label_offset=(0.0, 0.0)):
    """Arrow from p_from to p_to.

    route:
      'straight' — direct (curved if rad != 0)
      'HV'       — horizontal then vertical L-shape
      'VH'       — vertical then horizontal L-shape
    """
    sx, sy = p_from
    tx, ty = p_to
    if route == "straight":
        arrow = FancyArrowPatch(
            p_from, p_to,
            arrowstyle="-|>,head_length=6,head_width=4",
            linewidth=LW_THIN, color=STROKE,
            connectionstyle=f"arc3,rad={rad}",
            zorder=2,
        )
        ax.add_patch(arrow)
        mx = (sx + tx) / 2
        my = (sy + ty) / 2
        if rad != 0.0:
            dx, dy = tx - sx, ty - sy
            length = (dx * dx + dy * dy) ** 0.5
            if length > 1e-6:
                nx, ny = -dy / length, dx / length
                mx += rad * length * nx * 0.5
                my += rad * length * ny * 0.5
    else:
        # Ortho L-shape: draw stem as plain line, arrow head as last segment
        if route == "HV":
            corner = (tx, sy)
        elif route == "VH":
            corner = (sx, ty)
        else:
            raise ValueError(route)
        # First segment (no arrowhead)
        ax.plot([sx, corner[0]], [sy, corner[1]],
                color=STROKE, linewidth=LW_THIN, zorder=2,
                solid_joinstyle="miter", solid_capstyle="butt")
        # Second segment with arrowhead
        arrow = FancyArrowPatch(
            corner, p_to,
            arrowstyle="-|>,head_length=6,head_width=4",
            linewidth=LW_THIN, color=STROKE,
            connectionstyle="arc3,rad=0",
            zorder=2,
        )
        ax.add_patch(arrow)
        mx = corner[0]
        my = corner[1]

    if label:
        mx += label_offset[0]
        my += label_offset[1]
        ax.text(mx, my, label, fontsize=8.5, color=STROKE,
                ha="center", va="center", fontstyle="italic",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.3),
                zorder=4)


def _ellipse_edge(actor, y, side="right"):
    """Compute x coordinate where horizontal line at y crosses ellipse edge."""
    cx, cy, w, h = actor
    a, b = w / 2, h / 2
    dy = y - cy
    if abs(dy) >= b:
        return cx + (a if side == "right" else -a)
    dx = a * (1 - (dy / b) ** 2) ** 0.5
    return cx + (dx if side == "right" else -dx)


# ─── scene (horizontal flow) ─────────────────────────────────────────────────

def render(out_name: str = "fig_2_1_functional_model.png") -> Path:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 13.0)  # aspect ≈ 16:9
    ax.set_aspect("equal")
    ax.axis("off")

    # ─── External entities ───────────────────────────────────────────────────
    # Centered on the horizontal flow line (Слой 0/4 cy = cs_y+0.15+bh/2 = 4.525)
    partner = _actor(ax, 1.3, 4.525, 2.2, 1.3, "Партнёр\n(продавец)")
    showcase = _actor(ax, 21.7, 4.525, 2.2, 1.3, "Витрина\nкаталога")


    # ─── Main system container ───────────────────────────────────────────────
    main_x, main_y, main_w, main_h = 4.2, 0.6, 14.4, 7.4
    _container(ax, main_x, main_y, main_w, main_h,
               "Функции системы обогащения товарных данных", font_size=10)

    # Шлюз API — single block aligned with cascade flow line (cy = 4.525)
    gw_w, gw_h = 2.2, 1.6
    gw_x = main_x + 0.4
    gw_y = 4.525 - gw_h / 2
    _block(ax, gw_x, gw_y, gw_w, gw_h,
           "Шлюз API (Go)\n\nВалидация и\nмаршрутизация\nзапроса", font_size=9)

    # ML-сервис container (gap to Шлюз = 0.7)
    ml_x, ml_y, ml_w, ml_h = 7.4, 1.0, 11.0, 6.0
    _container(ax, ml_x, ml_y, ml_w, ml_h, "ML-сервис (Python / FastAPI)", font_size=9.5)

    # Каскад слоёв container (horizontal) — pushed up to leave room for cylinders + labels below
    cs_x, cs_y, cs_w, cs_h = ml_x + 0.3, ml_y + 2.4, 10.4, 3.0
    _container(ax, cs_x, cs_y, cs_w, cs_h, "Каскад слоёв", font_size=9)

    # 5 layer blocks, horizontal
    layer_labels = [
        "Слой 0\nМаршрутизация\nпо категории",
        "Слой 1\nИзвлечение\nправилами",
        "Слой 2\nКлассификация\nSBERT + XGBoost",
        "Слой 3\nВалидация\nбайесовской\nсетью",
        "Слой 4\nЗапасной слой\nбольшой\nязыковой модели",
    ]
    n = len(layer_labels)
    bw = 1.85
    bh = 1.95
    gap = (cs_w - 0.4 - n * bw) / (n - 1)
    layer_rects = []
    for i, lbl in enumerate(layer_labels):
        x = cs_x + 0.2 + i * (bw + gap)
        y = cs_y + 0.15
        _block(ax, x, y, bw, bh, lbl, font_size=8.5)
        layer_rects.append((x, y, bw, bh))

    # Cylinders below cascade (data stores) — inside ML-сервис container
    cyl_xgb = _cylinder(ax, layer_rects[2][0] - 0.05, ml_y + 0.15, 2.0, 1.4,
                        "Модели XGBoost", font_size=9)
    cyl_bn = _cylinder(ax, layer_rects[3][0] - 0.05, ml_y + 0.15, 2.0, 1.4,
                       "Байесовские\nсети", font_size=9)

    # API LLM cylinder — centered directly above Слой 4
    l4_cx = layer_rects[4][0] + layer_rects[4][2] / 2
    llm_w = 2.7
    llm_cyl = _cylinder(ax, l4_cx - llm_w / 2, 8.8, llm_w, 1.7,
                        "API внешней\nбольшой языковой\nмодели", font_size=9)

    # ─── Arrows ──────────────────────────────────────────────────────────────

    # 1. Партнёр → Шлюз (horizontal on flow line cy=4.525)
    flow_cy = 4.525
    _arrow(ax,
           (_ellipse_edge(partner, flow_cy, "right"), flow_cy),
           (gw_x + 0.05, flow_cy),
           "поля партнёра", route="straight",
           label_offset=(0.0, 0.3))

    # 3. Шлюз → Слой 0 (horizontal on flow line)
    l0 = layer_rects[0]
    _arrow(ax,
           (gw_x + gw_w, flow_cy),
           (l0[0] + 0.05, l0[1] + l0[3] / 2),
           "товар +\nкатегория", route="straight",
           label_offset=(0.0, 0.3))

    # 4. Layer chain L0 → L1 → … → L4
    for i in range(n - 1):
        r1, r2 = layer_rects[i], layer_rects[i + 1]
        _arrow(ax,
               (r1[0] + r1[2], r1[1] + r1[3] / 2),
               (r2[0], r2[1] + r2[3] / 2), rad=0.0)

    # 5. Слой 2 ↔ Модели XGBoost (parallel down-up arrows, label between them)
    l2 = layer_rects[2]
    cx_xgb = cyl_xgb[0] + cyl_xgb[2] / 2
    _arrow(ax,
           (l2[0] + l2[2] / 2 - 0.3, l2[1]),
           (cx_xgb - 0.3, cyl_xgb[1] + cyl_xgb[3]), rad=0.0)
    _arrow(ax,
           (cx_xgb + 0.3, cyl_xgb[1] + cyl_xgb[3]),
           (l2[0] + l2[2] / 2 + 0.3, l2[1]),
           "запрос /\nмодель", rad=0.0, label_offset=(0.55, 0))

    # 6. Слой 3 ↔ Байесовские сети
    l3 = layer_rects[3]
    cx_bn = cyl_bn[0] + cyl_bn[2] / 2
    _arrow(ax,
           (l3[0] + l3[2] / 2 - 0.3, l3[1]),
           (cx_bn - 0.3, cyl_bn[1] + cyl_bn[3]), rad=0.0)
    _arrow(ax,
           (cx_bn + 0.3, cyl_bn[1] + cyl_bn[3]),
           (l3[0] + l3[2] / 2 + 0.3, l3[1]),
           "контекст /\nплотность", rad=0.0, label_offset=(0.65, 0))

    # 7. Слой 4 ↔ API LLM (vertical arrows; labels in clear strip above main container)
    l4 = layer_rects[4]
    cx_llm = llm_cyl[0] + llm_cyl[2] / 2
    main_top = main_y + main_h        # 8.0
    llm_bottom = llm_cyl[1]           # 8.8
    label_band_cy = (main_top + llm_bottom) / 2  # 8.4 — clear strip
    arrow_mid_y = (l4[1] + l4[3] + llm_bottom) / 2
    _arrow(ax,
           (l4[0] + l4[2] / 2 - 0.3, l4[1] + l4[3]),
           (cx_llm - 0.3, llm_bottom),
           "промпт-запрос", rad=0.0,
           label_offset=(-0.85, label_band_cy - arrow_mid_y))
    _arrow(ax,
           (cx_llm + 0.3, llm_bottom),
           (l4[0] + l4[2] / 2 + 0.3, l4[1] + l4[3]),
           "ответ модели", rad=0.0,
           label_offset=(0.85, label_band_cy - arrow_mid_y))

    # 8. Слой 4 → Витрина (horizontal, aligned with L4 center)
    l4_cy = l4[1] + l4[3] / 2
    _arrow(ax,
           (l4[0] + l4[2], l4_cy),
           (_ellipse_edge(showcase, l4_cy, "left"), l4_cy),
           "обогащённые\nатрибуты", route="straight",
           label_offset=(0.55, 0.45))

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / out_name
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.2, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    render()
