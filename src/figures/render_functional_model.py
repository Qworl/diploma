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
OUT = ROOT / "images"

rcParams["font.family"] = "Times New Roman"
rcParams["font.size"] = 10

STROKE = "#1f1f4d"
LW_THICK = 1.1
LW_THIN = 0.7


# ─── primitives ──────────────────────────────────────────────────────────────

def _container(ax, x, y, w, h, title, *, font_size=11, facecolor="white"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.25",
        linewidth=LW_THICK, edgecolor=STROKE, facecolor=facecolor, zorder=1,
    )
    ax.add_patch(box)
    if title:
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


def _predef(ax, x, y, w, h, label, *, font_size=10):
    """Внешняя система / предопределённый процесс (ГОСТ 19.701-90):
    прямоугольник с двумя вертикальными гранями по краям."""
    ax.add_patch(Rectangle((x, y), w, h, linewidth=LW_THICK,
                           edgecolor=STROKE, facecolor="white", zorder=2))
    inset = 0.16
    ax.plot([x + inset, x + inset], [y, y + h], color=STROKE, lw=LW_THIN, zorder=3)
    ax.plot([x + w - inset, x + w - inset], [y, y + h], color=STROKE, lw=LW_THIN, zorder=3)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=font_size, color=STROKE, zorder=4)
    return (x, y, w, h)


def _legend(ax):
    """Условные обозначения (пояснение символов схемы), одна строка."""
    y = -0.9
    ax.text(0.4, 0.15, "Условные обозначения:", fontsize=11,
            fontstyle="italic", color=STROKE, ha="left", va="center")
    rows = [
        # (icon_kind, x_icon, text)
        ("ellipse", 0.4,  "внешний актор"),
        ("rect",    5.8,  "функциональный блок"),
        ("cyl",    11.6,  "хранилище данных"),
        ("predef", 16.8,  "внешняя система"),
    ]
    for kind, xi, text in rows:
        if kind == "ellipse":
            ax.add_patch(Ellipse((xi + 0.55, y), 1.1, 0.6, linewidth=LW_THIN,
                                 edgecolor=STROKE, facecolor="white", zorder=2))
        elif kind == "rect":
            ax.add_patch(FancyBboxPatch((xi, y - 0.3), 1.1, 0.6,
                         boxstyle="round,pad=0,rounding_size=0.1", linewidth=LW_THIN,
                         edgecolor=STROKE, facecolor="white", zorder=2))
        elif kind == "cyl":
            _cylinder(ax, xi, y - 0.35, 1.0, 0.7, "", font_size=1)
        elif kind == "predef":
            _predef(ax, xi, y - 0.3, 1.1, 0.6, "", font_size=1)
        ax.text(xi + 1.4, y, text, fontsize=11, color=STROKE, ha="left", va="center")


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
        ax.text(mx, my, label, fontsize=12, color=STROKE,
                ha="center", va="center", fontstyle="italic",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6),
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
    # Компактнее по ширине (ratio ≈1.9 вместо 2.35): акторы и шлюз вынесены
    # из ряда каскада в верхнюю полосу, поэтому пять слоёв L0→L4 занимают почти
    # всю ширину и крупно читаются на слайде.
    fig, ax = plt.subplots(figsize=(15.0, 7.6))
    ax.set_xlim(0, 22.6)
    ax.set_ylim(-0.15, 10.5)
    ax.set_aspect("equal")
    ax.axis("off")

    flow_cy = 3.9  # центральная линия каскада

    # ─── Каскад слоёв — полноширинный ряд (центральный объект схемы) ─────────
    # Отдельная рамка не нужна: пять блоков «Слой N» сами образуют каскад
    # внутри контейнера ML-сервиса.
    cs_x, cs_w = 1.0, 17.4

    layer_labels = [
        "Слой 0\nМаршрутизация\nпо категории",
        "Слой 1\nИзвлечение\nправилами",
        "Слой 2\nКлассификация\n(SBERT+XGBoost)",
        "Слой 3\nВалидация\nбайес-сетью",
        "Слой 4\nЗапасной слой\n(языковая\nмодель)",
    ]
    n = len(layer_labels)
    bw, bh = 2.9, 2.1
    inner = cs_w - 0.5
    gap = (inner - n * bw) / (n - 1)  # ≈0.6 — заметные стрелки между слоями
    layer_rects = []
    for i, lbl in enumerate(layer_labels):
        x = cs_x + 0.25 + i * (bw + gap)
        y = flow_cy - bh / 2
        _block(ax, x, y, bw, bh, lbl, font_size=11.5)
        layer_rects.append((x, y, bw, bh))

    l0c = layer_rects[0][0] + layer_rects[0][2] / 2
    l2c = layer_rects[2][0] + layer_rects[2][2] / 2
    l3c = layer_rects[3][0] + layer_rects[3][2] / 2
    l4c = layer_rects[4][0] + layer_rects[4][2] / 2

    # ─── Контейнеры: сначала граница системы (без заливки), затем ML-сервис
    #     с лёгкой заливкой и отступом — чтобы его рамка была хорошо видна. ────
    sys_x, sys_y, sys_w, sys_h = 0.55, 0.2, 18.3, 7.1
    _container(ax, sys_x, sys_y, sys_w, sys_h, "")
    ax.text((sys_x + sys_w / 2), sys_y + sys_h - 0.45,
            "Функции системы обогащения товарных данных",
            fontsize=12, fontstyle="italic", color=STROKE, ha="center", va="top")

    ml_x, ml_y, ml_w, ml_h = 0.9, 0.35, 17.6, 5.2
    _container(ax, ml_x, ml_y, ml_w, ml_h, "", facecolor="#eef2fb")
    # Подпись ML-сервиса — по центру верхней кромки, чтобы не пересекаться
    # со стрелкой Шлюз → Слой 0.
    ax.text(ml_x + ml_w / 2, ml_y + ml_h - 0.32, "ML-сервис (Python / FastAPI)",
            fontsize=11.5, fontstyle="italic", color=STROKE, ha="center", va="top")

    # Хранилища данных под каскадом (опущены — длинные стрелки от слоёв)
    cyl_w, cyl_h, cyl_y = 2.35, 1.45, 0.45
    cyl_xgb = _cylinder(ax, l2c - cyl_w / 2, cyl_y, cyl_w, cyl_h,
                        "Модели\nXGBoost", font_size=11)
    cyl_bn = _cylinder(ax, l3c - cyl_w / 2, cyl_y, cyl_w, cyl_h,
                       "Байесовские\nсети", font_size=11)

    # ─── Шлюз API (вход в систему) ──────────────────────────────────────────
    gw_w, gw_h = 3.0, 1.5
    gw_x, gw_y = l0c - gw_w / 2, 5.7
    _block(ax, gw_x, gw_y, gw_w, gw_h,
           "Шлюз API (Go)\nвалидация,\nмаршрутизация", font_size=10.5)

    # ─── Внешние акторы и большая языковая модель (верхняя полоса) ──────────
    partner = _actor(ax, l0c, 9.6, 2.9, 1.55, "Партнёр\n(продавец)", font_size=12)
    llm_w = 3.8
    llm = _predef(ax, l4c - llm_w / 2, 8.6, llm_w, 1.55,
                  "Большая\nязыковая модель", font_size=11)
    pim = _actor(ax, 21.2, flow_cy, 2.4, 1.55, "PIM-\nсистема", font_size=12)
    cm  = _actor(ax, 21.2, 9.3, 2.4, 1.5, "Контент-\nменеджер", font_size=11)

    # ─── Arrows ──────────────────────────────────────────────────────────────
    # Партнёр → Шлюз (вертикально); подпись «поля партнёра» — над границей системы
    _arrow(ax, (l0c, partner[1] - partner[3] / 2), (l0c, gw_y + gw_h),
           route="straight")
    def _flow_label(fx, fy, ftext):
        ax.text(fx, fy, ftext, fontsize=11.5, color=STROKE, ha="center", va="center",
                fontstyle="italic",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5), zorder=5)
    _flow_label(l0c - 1.4, gw_y + gw_h + 0.55, "данные\nтоваров")
    # Шлюз → Слой 0 (вертикально вниз в каскад)
    l0 = layer_rects[0]
    _arrow(ax, (l0c, gw_y), (l0c, l0[1] + l0[3]), route="straight")

    # Цепочка L0 → L1 → … → L4
    for i in range(n - 1):
        r1, r2 = layer_rects[i], layer_rects[i + 1]
        _arrow(ax, (r1[0] + r1[2], r1[1] + r1[3] / 2),
               (r2[0], r2[1] + r2[3] / 2))

    # Слой 2 ↔ XGBoost
    l2 = layer_rects[2]
    _arrow(ax, (l2c - 0.32, l2[1]), (l2c - 0.32, cyl_y + cyl_h))
    _arrow(ax, (l2c + 0.32, cyl_y + cyl_h), (l2c + 0.32, l2[1]))
    _flow_label(l2c + 1.5, (l2[1] + cyl_y + cyl_h) / 2 + 0.05, "запрос /\nмодель")
    # Слой 3 ↔ Байесовские сети
    l3 = layer_rects[3]
    _arrow(ax, (l3c - 0.32, l3[1]), (l3c - 0.32, cyl_y + cyl_h))
    _arrow(ax, (l3c + 0.32, cyl_y + cyl_h), (l3c + 0.32, l3[1]))
    _flow_label(l3c + 1.6, (l3[1] + cyl_y + cyl_h) / 2 + 0.05, "контекст /\nплотность")

    # Слой 4 ↔ большая языковая модель (вертикально, одна подпись по центру)
    l4 = layer_rects[4]
    llm_bottom = llm[1]
    _arrow(ax, (l4c - 0.32, l4[1] + l4[3]), (l4c - 0.32, llm_bottom))
    _arrow(ax, (l4c + 0.32, llm_bottom), (l4c + 0.32, l4[1] + l4[3]))
    _flow_label(l4c - 1.5, 6.55, "промпт-\nзапрос")
    _flow_label(l4c + 1.45, 6.55, "ответ\nмодели")

    # Слой 4 → PIM-система (обогащённая карточка покидает систему)
    _arrow(ax, (l4[0] + l4[2], flow_cy),
           (_ellipse_edge(pim, flow_cy, "left"), flow_cy), route="straight")
    _flow_label((l4[0] + l4[2] + pim[0] - pim[2] / 2) / 2, flow_cy + 0.6,
                "обогащённые\nатрибуты")

    # PIM ↔ Контент-менеджер (выборочная верификация 5–10 % ячеек)
    pim_top = pim[1] + pim[3] / 2
    cm_bot = cm[1] - cm[3] / 2
    _arrow(ax, (pim[0] - 0.32, pim_top), (cm[0] - 0.32, cm_bot))
    _arrow(ax, (cm[0] + 0.32, cm_bot), (pim[0] + 0.32, pim_top))
    _flow_label(pim[0] - 1.45, (pim_top + cm_bot) / 2, "выборочная\nверификация")

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / out_name
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.2, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    render()
