"""Render architecture CONTEXT view — Рисунок «Контекстный вид архитектуры».

Соответствует ГОСТ Р 57100-2016 (ISO/IEC/IEEE 42010): система как целое в
окружении заинтересованных сторон и внешних систем/хранилищ, с условными
обозначениями (легендой). Линии связи идут от каждой сущности
непосредственно к центральному блоку системы.

Usage:
    OMP_NUM_THREADS=1 python -m src.figures.render_context_view
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

STROKE = "#1f1f4d"
FILL_SYS = "white"
FILL_STAKE = "#dbe7f3"   # заинтересованная сторона — голубой
FILL_EXT = "#fce6cf"     # внешняя система — оранжевый
FILL_STORE = "#e6e6e6"   # внешнее хранилище — серый
LW = 1.2


# ─── primitives (center-anchored) ────────────────────────────────────────────

def _rrect(ax, cx, cy, w, h, label, fill, *, fs=10.5, z=2):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.18",
        linewidth=LW, edgecolor=STROKE, facecolor=fill, zorder=z))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fs,
            color=STROKE, zorder=z + 1)
    return (cx, cy, w, h)


def _cylinder(ax, cx, cy, w, h, label, fill, *, fs=10):
    x, y = cx - w / 2, cy - h / 2
    cap = min(0.34, h * 0.26)
    ax.add_patch(Rectangle((x, y + cap / 2), w, h - cap,
                           linewidth=0, facecolor=fill, zorder=2))
    ax.plot([x, x], [y + cap / 2, y + h - cap / 2], color=STROKE, lw=LW, zorder=2)
    ax.plot([x + w, x + w], [y + cap / 2, y + h - cap / 2], color=STROKE, lw=LW, zorder=2)
    ax.add_patch(Ellipse((cx, y + cap / 2), w, cap, linewidth=LW,
                         edgecolor=STROKE, facecolor=fill, zorder=2))
    ax.add_patch(Ellipse((cx, y + h - cap / 2), w, cap, linewidth=LW,
                         edgecolor=STROKE, facecolor=fill, zorder=3))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fs,
            color=STROKE, zorder=4)
    return (cx, cy, w, h)


def _system(ax, cx, cy, w, h):
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, linewidth=1.7,
                           edgecolor=STROKE, facecolor=FILL_SYS, zorder=3))
    ax.text(cx, cy + 0.85,
            "Гибридная каскадная система\nобогащения товарных данных",
            ha="center", va="center", fontsize=12.5, fontweight="bold",
            color=STROKE, zorder=4)
    ax.text(cx, cy - 0.95,
            "Слой 0 $\\to$ Слой 1 $\\to$ Слой 2 $\\to$ Слой 3 $\\to$ Слой 4",
            ha="center", va="center", fontsize=10.5, color=STROKE, zorder=4)
    return (cx, cy, w, h)


def _link(ax, p_from, p_to, label="", *, double=False, fs=9.5, loff=(0.0, 0.45)):
    """Соединительная линия от точки к точке (от грани блока к грани блока)."""
    style = "<|-|>" if double else "-|>"
    ax.add_patch(FancyArrowPatch(
        p_from, p_to,
        arrowstyle=f"{style},head_length=7,head_width=4.5",
        linewidth=0.9, color=STROKE, shrinkA=0, shrinkB=0, zorder=2))
    if label:
        mx = (p_from[0] + p_to[0]) / 2 + loff[0]
        my = (p_from[1] + p_to[1]) / 2 + loff[1]
        ax.text(mx, my, label, fontsize=fs, color=STROKE, ha="center",
                va="center", fontstyle="italic",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6), zorder=5)


def render(out_name: str = "fig_arch_context.png") -> Path:
    fig, ax = plt.subplots(figsize=(13.5, 7.4))
    ax.set_xlim(0, 27)
    ax.set_ylim(0, 14)
    ax.set_aspect("equal")
    ax.axis("off")

    # центр — исследуемая система
    sys_cx, sys_cy, sys_w, sys_h = 13.5, 7.7, 9.4, 4.4
    _system(ax, sys_cx, sys_cy, sys_w, sys_h)
    L = sys_cx - sys_w / 2          # левая грань системы
    R = sys_cx + sys_w / 2          # правая грань системы
    y_up, y_mid, y_lo = sys_cy + 1.4, sys_cy, sys_cy - 1.4   # точки подключения

    # ряды сущностей
    top, mid, bot = 11.4, 7.7, 4.0
    st_w, st_h, st_cx = 4.8, 1.55, 2.9
    ex_w, ex_h, ex_cx = 4.8, 1.7, 24.1

    # ─── заинтересованные стороны (слева) ───────────────────────────────────
    partner = _rrect(ax, st_cx, top, st_w, st_h, "Партнёр\n(поставщик данных)", FILL_STAKE, fs=9.5)
    operator = _rrect(ax, st_cx, mid, st_w, st_h, "Оператор каталога\n(контент-менеджер)", FILL_STAKE, fs=9.5)
    admin = _rrect(ax, st_cx, bot, st_w, st_h, "Администратор\nсистемы", FILL_STAKE, fs=9.5)
    st_r = st_cx + st_w / 2         # правая грань левых блоков

    # ─── внешние системы и хранилища (справа) ───────────────────────────────
    catalog = _cylinder(ax, ex_cx, top, ex_w, ex_h, "Каталог\nинтернет-магазина", FILL_STORE, fs=9.5)
    off = _cylinder(ax, ex_cx, mid, ex_w, ex_h, "Open Food Facts\n(эталон, нутриенты)", FILL_STORE, fs=9.5)
    llm = _rrect(ax, ex_cx, bot, ex_w, ex_h, "Большая\nязыковая модель", FILL_EXT, fs=9.5)
    ex_l = ex_cx - ex_w / 2         # левая грань правых блоков

    # ─── связи: каждая линия соединяет грань блока с гранью системы ──────────
    _link(ax, (st_r, top), (L, y_up), "неполные карточки", loff=(0.2, 0.5))
    _link(ax, (st_r, mid), (L, y_mid), "правки, верификация", double=True, loff=(0.0, 0.5))
    _link(ax, (st_r, bot), (L, y_lo), "настройка порогов", loff=(0.2, -0.55))

    _link(ax, (R, y_up), (ex_l, top), "обогащённые карточки", loff=(-0.2, 0.5))
    _link(ax, (R, y_mid), (ex_l, mid), "теги, нутриенты\n(эталон разметки)", double=True, loff=(0.0, 0.75))
    _link(ax, (R, y_lo), (ex_l, bot), "остаточные ячейки /\nответы модели", double=True, loff=(0.0, 0.85))

    # ─── условные обозначения (легенда) ─────────────────────────────────────
    ly = 1.1
    ax.text(0.4, ly + 0.7, "Условные обозначения:", fontsize=10,
            fontstyle="italic", color=STROKE, ha="left", va="center")
    legend = [
        ("Исследуемая система", FILL_SYS),
        ("Заинтересованная сторона", FILL_STAKE),
        ("Внешняя система", FILL_EXT),
        ("Внешнее хранилище", FILL_STORE),
    ]
    lx = 0.6
    for text, fill in legend:
        ax.add_patch(Rectangle((lx, ly - 0.28), 0.75, 0.56, linewidth=LW,
                               edgecolor=STROKE, facecolor=fill, zorder=2))
        ax.text(lx + 1.0, ly, text, fontsize=9.5, color=STROKE,
                ha="left", va="center")
        lx += 1.9 + 0.165 * len(text)

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / out_name
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.2, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    render()
