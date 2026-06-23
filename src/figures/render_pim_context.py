"""Render context view — Рисунок «Место системы обогащения в инфраструктуре».

Двухдорожечное контекстное представление (ГОСТ Р 57100-2016), повторяет
схему слайда «Место и роль программного средства»:
  • верхняя дорожка — промышленная эксплуатация магазина
    (Покупатели → Витрина → PIM-система → Контент-менеджер);
  • нижняя дорожка — обогащение карточки товара
    (Партнёр → гибридный каскад → Open Food Facts).
Подписи на русском (без англицизмов): «Слои 0→4», «обогащённая карточка» и т.п.
Заголовок не рисуется — его несёт \caption в TeX.

Usage:
    OMP_NUM_THREADS=1 python -m src.figures.render_pim_context
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "images"

rcParams["font.family"] = "Times New Roman"
rcParams["font.size"] = 11

STROKE = "#5b6472"
TEXT = "#1f2430"
MUTE = "#6b7280"
LANE_FILL = "#fcfcf0"
LANE_EDGE = "#c9c9b0"
ACTOR_FILL = "#eef0f3"
ACTOR_EDGE = "#7a828f"
EXT_FILL = "#e6eefb"
EXT_EDGE = "#3a6ea5"
DEV_FILL = "#dff0d6"
DEV_EDGE = "#4a8a3f"
DATA_FILL = "#fce6cf"
DATA_EDGE = "#c8853a"
LW = 1.2


def _lane(ax, x, y, w, h, label):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.18",
        linewidth=1.0, edgecolor=LANE_EDGE, facecolor=LANE_FILL, zorder=1))
    ax.text(x + 0.25, y + h - 0.28, label, fontsize=10, fontstyle="italic",
            color=MUTE, ha="left", va="top", zorder=2)


def _box(ax, cx, cy, w, h, label, *, fill, edge, fs=10.5, bold=False, lw=LW):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.1",
        linewidth=lw, edgecolor=edge, facecolor=fill, zorder=3))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fs,
            color=TEXT, fontweight="bold" if bold else "normal", zorder=4)
    return (cx, cy, w, h)


def _cylinder(ax, cx, cy, w, h, label, *, fs=10.5):
    x, y = cx - w / 2, cy - h / 2
    cap = min(0.34, h * 0.26)
    ax.add_patch(Rectangle((x, y + cap / 2), w, h - cap, linewidth=0,
                           facecolor=DATA_FILL, zorder=3))
    ax.plot([x, x], [y + cap / 2, y + h - cap / 2], color=DATA_EDGE, lw=LW, zorder=3)
    ax.plot([x + w, x + w], [y + cap / 2, y + h - cap / 2], color=DATA_EDGE, lw=LW, zorder=3)
    ax.add_patch(Ellipse((cx, y + cap / 2), w, cap, linewidth=LW,
                         edgecolor=DATA_EDGE, facecolor=DATA_FILL, zorder=3))
    ax.add_patch(Ellipse((cx, y + h - cap / 2), w, cap, linewidth=LW,
                         edgecolor=DATA_EDGE, facecolor=DATA_FILL, zorder=4))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fs, color=TEXT, zorder=5)
    return (cx, cy, w, h)


def _arrow(ax, p_from, p_to, label="", *, dashed=False, rad=0.0, loff=(0.0, 0.0), fs=9):
    ax.add_patch(FancyArrowPatch(
        p_from, p_to, arrowstyle="-|>,head_length=6,head_width=4",
        linewidth=1.0, color=STROKE, linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2, zorder=2))
    if label:
        mx = (p_from[0] + p_to[0]) / 2 + loff[0]
        my = (p_from[1] + p_to[1]) / 2 + loff[1]
        ax.text(mx, my, label, fontsize=fs, color=MUTE, ha="center", va="center",
                fontstyle="italic",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.4), zorder=5)


def render(out_name: str = "pim_role_context.png") -> Path:
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.4)
    ax.set_aspect("equal")
    ax.axis("off")

    y_top, y_bot = 6.2, 2.3
    aw, ah = 2.5, 1.15

    # ─── дорожки ─────────────────────────────────────────────────────────────
    _lane(ax, 0.4, 4.95, 15.2, 2.55, "Промышленная эксплуатация (магазин)")
    _lane(ax, 0.4, 0.95, 13.3, 2.55, "Обогащение карточки товара")

    # ─── верхняя дорожка ─────────────────────────────────────────────────────
    cust = _box(ax, 1.9, y_top, aw, ah, "Покупатели\nмагазина", fill=ACTOR_FILL, edge=ACTOR_EDGE)
    store = _box(ax, 5.4, y_top, 2.7, ah, "Витрина\nмагазина", fill=EXT_FILL, edge=EXT_EDGE)
    pim = _box(ax, 9.9, y_top, 2.7, ah, "PIM-система", fill=EXT_FILL, edge=EXT_EDGE)
    cm = _box(ax, 13.9, y_top, 2.6, ah, "Контент-\nменеджер", fill=ACTOR_FILL, edge=ACTOR_EDGE)

    # ─── нижняя дорожка ──────────────────────────────────────────────────────
    partner = _box(ax, 1.9, y_bot, aw, ah, "Партнёр /\nпоставщик", fill=ACTOR_FILL, edge=ACTOR_EDGE)
    cascade = _box(ax, 6.6, y_bot, 4.6, 1.5,
                   "ГИБРИДНЫЙ КАСКАД\n(разрабатываемое ПО)\nСлои 0 → 1 → 2 → 3 → 4",
                   fill=DEV_FILL, edge=DEV_EDGE, fs=10.5, bold=True, lw=1.5)
    off = _cylinder(ax, 11.9, y_bot, 2.4, 1.5, "Open Food\nFacts")

    # ─── связи: верхняя дорожка ──────────────────────────────────────────────
    _arrow(ax, (cust[0] + aw / 2, y_top), (store[0] - store[2] / 2, y_top),
           "поиск,\nфильтры", loff=(0, 0.42))
    _arrow(ax, (pim[0] - pim[2] / 2, y_top), (store[0] + store[2] / 2, y_top),
           "выкладка", loff=(0, 0.32))
    _arrow(ax, (cm[0] - cm[2] / 2, y_top), (pim[0] + pim[2] / 2, y_top),
           "выборочная\nверификация 5–10 %", loff=(0, 0.5))

    # ─── связи: нижняя дорожка ───────────────────────────────────────────────
    _arrow(ax, (partner[0] + aw / 2, y_bot), (cascade[0] - cascade[2] / 2, y_bot),
           "JSON / REST", loff=(0, 0.34))
    _arrow(ax, (off[0] - off[2] / 2, y_bot), (cascade[0] + cascade[2] / 2, y_bot),
           "обучение,\nкалибровка", loff=(0, 0.42))

    # ─── межуровневые связи ──────────────────────────────────────────────────
    _arrow(ax, (cascade[0] + 0.6, y_bot + 0.75), (pim[0] - 0.6, y_top - ah / 2),
           "обогащённая\nкарточка", rad=0.12, loff=(0.7, 0.0))
    _arrow(ax, (partner[0] + aw / 2 - 0.2, y_bot + ah / 2),
           (cm[0] - 0.4, y_top - ah / 2),
           "ручное заполнение (сегодня)", dashed=True, rad=-0.18, loff=(0.5, 0.55))

    # ─── условные обозначения ────────────────────────────────────────────────
    ly = 0.35
    items = [("актор", ACTOR_FILL, ACTOR_EDGE), ("внешняя система", EXT_FILL, EXT_EDGE),
             ("разрабатываемое ПО", DEV_FILL, DEV_EDGE), ("хранилище данных", DATA_FILL, DATA_EDGE)]
    lx = 0.5
    ax.text(lx, ly, "Условные обозначения:", fontsize=9, fontstyle="italic",
            color=MUTE, ha="left", va="center")
    lx = 3.4
    for text, fill, edge in items:
        ax.add_patch(Rectangle((lx, ly - 0.18), 0.55, 0.36, linewidth=1.0,
                               edgecolor=edge, facecolor=fill, zorder=2))
        ax.text(lx + 0.72, ly, text, fontsize=9, color=TEXT, ha="left", va="center")
        lx += 1.15 + 0.16 * len(text)

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / out_name
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.2, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    render()
