"""Render UML Sequence Diagram (Рисунок 3.1) — request flow.

Style (UML 2.5 conventions, GOST-compatible visual):
- Actor/object headers at top (rectangles with name in italic)
- Vertical dashed lifelines
- Solid arrows for synchronous calls, dashed for returns
- Activation bars (thin filled rectangles on lifelines during operation)
- Self-call arrows loop back to the same lifeline
- Times New Roman, black-on-white, 300 dpi

Usage:
    OMP_NUM_THREADS=1 python -m src.figures.render_sequence_diagram
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "thesis" / "figures"

rcParams["font.family"] = "Times New Roman"
rcParams["font.size"] = 10

STROKE = "#1f1f4d"
LW_THICK = 1.1
LW_THIN = 0.7


# ─── primitives ──────────────────────────────────────────────────────────────

def _lifeline_head(ax, cx, top_y, w, h, label, font_size=10):
    """Object/actor box at top of a lifeline."""
    box = FancyBboxPatch(
        (cx - w / 2, top_y - h), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.10",
        linewidth=LW_THICK, edgecolor=STROKE, facecolor="white", zorder=3,
    )
    ax.add_patch(box)
    ax.text(cx, top_y - h / 2, label, ha="center", va="center",
            fontsize=font_size, color=STROKE, zorder=4)


def _lifeline(ax, cx, top_y, bottom_y):
    """Vertical dashed lifeline."""
    ax.plot([cx, cx], [top_y, bottom_y],
            color=STROKE, linewidth=LW_THIN, linestyle=(0, (4, 3)), zorder=1)


def _activation(ax, cx, y_top, y_bot, width=0.18):
    """Thin filled activation bar on a lifeline."""
    rect = Rectangle((cx - width / 2, y_bot), width, y_top - y_bot,
                     linewidth=LW_THIN, edgecolor=STROKE,
                     facecolor="white", zorder=2)
    ax.add_patch(rect)


def _message(ax, p_from, p_to, label, *, style="solid"):
    """Horizontal message arrow with label above."""
    if style == "solid":
        ls = "solid"
        arrowstyle = "-|>,head_length=6,head_width=4"
    elif style == "dashed":
        ls = (0, (4, 3))
        arrowstyle = "-|>,head_length=6,head_width=4"
    else:
        raise ValueError(style)
    arrow = FancyArrowPatch(
        p_from, p_to,
        arrowstyle=arrowstyle, linewidth=LW_THIN, color=STROKE,
        linestyle=ls, connectionstyle="arc3,rad=0", zorder=3,
    )
    ax.add_patch(arrow)
    mx = (p_from[0] + p_to[0]) / 2
    my = max(p_from[1], p_to[1]) + 0.18
    ax.text(mx, my, label, ha="center", va="bottom",
            fontsize=9, fontstyle="italic", color=STROKE, zorder=4)


def _self_message(ax, cx, y, label):
    """Self-call: small loop back to the same lifeline."""
    loop_w = 0.7
    loop_h = 0.45
    # Outgoing horizontal segment
    ax.plot([cx, cx + loop_w], [y, y],
            color=STROKE, linewidth=LW_THIN, zorder=3)
    # Vertical right segment
    ax.plot([cx + loop_w, cx + loop_w], [y, y - loop_h],
            color=STROKE, linewidth=LW_THIN, zorder=3)
    # Incoming horizontal segment back (with arrowhead)
    arrow = FancyArrowPatch(
        (cx + loop_w, y - loop_h), (cx + 0.12, y - loop_h),
        arrowstyle="-|>,head_length=6,head_width=4",
        linewidth=LW_THIN, color=STROKE, zorder=3,
    )
    ax.add_patch(arrow)
    # Label to the right of the loop
    ax.text(cx + loop_w + 0.25, y - loop_h / 2, label,
            ha="left", va="center", fontsize=9,
            fontstyle="italic", color=STROKE, zorder=4)


# ─── scene ───────────────────────────────────────────────────────────────────

def render(out_name: str = "fig_3_1_sequence_diagram.png") -> Path:
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 14.7)  # aspect ≈ 3:2
    ax.set_aspect("equal")
    ax.axis("off")

    # Lifelines layout
    actors = [
        ("Партнёр\n(продавец)", 2.5),
        ("Шлюз API\n(Go)",      8.0),
        ("ML-сервис\n(Python / FastAPI)", 14.0),
        ("API внешней\nбольшой языковой модели", 20.0),
    ]
    head_w, head_h = 3.5, 1.4
    head_top_y = 14.0
    head_bot_y = head_top_y - head_h
    lifeline_bot_y = 1.0

    for label, cx in actors:
        _lifeline_head(ax, cx, head_top_y, head_w, head_h, label)
        _lifeline(ax, cx, head_bot_y, lifeline_bot_y)

    xs = {name.split("\n")[0]: cx for name, cx in actors}
    P = xs["Партнёр"]
    G = xs["Шлюз API"]
    M = xs["ML-сервис"]
    L = xs["API внешней"]

    # Activation bars and messages, top to bottom
    y = head_bot_y - 0.4

    # 1. Партнёр → Шлюз: POST /enrich
    _message(ax, (P + 0.15, y), (G - 0.15, y), "POST /enrich (поля партнёра)")
    g_act_top = y - 0.05
    y -= 0.95

    # 2. Шлюз self: валидация JSON-схемы
    _self_message(ax, G, y, "валидация JSON-схемы")
    y -= 1.1

    # 3. Шлюз → ML-сервис: enrich()
    _message(ax, (G + 0.15, y), (M - 0.15, y), "enrich(товар, категория)")
    m_act_top = y - 0.05
    y -= 0.95

    # 4–7. ML self-calls: Слой 0..3
    for label in [
        "Слой 0: классификация категории",
        "Слой 1: regex-извлечение",
        "Слой 2: SBERT + XGBoost",
        "Слой 3: байесовская валидация",
    ]:
        _self_message(ax, M, y, label)
        y -= 1.1

    # 8. ML → API LLM: запасной слой
    _message(ax, (M + 0.15, y), (L - 0.15, y),
             "prompt() — для неуверенных ячеек")
    l_act_top = y - 0.05
    y -= 0.95

    # 9. API LLM → ML: ответ (dashed)
    _message(ax, (L - 0.15, y), (M + 0.15, y),
             "ответ большой языковой модели", style="dashed")
    l_act_bot = y + 0.05
    y -= 1.1

    # 10. ML → Шлюз: атрибуты + уверенности (dashed)
    _message(ax, (M - 0.15, y), (G + 0.15, y),
             "атрибуты + уверенности", style="dashed")
    m_act_bot = y + 0.05
    y -= 1.1

    # 11. Шлюз → Партнёр: HTTP 200 OK (dashed)
    _message(ax, (G - 0.15, y), (P + 0.15, y),
             "HTTP 200 OK + JSON-ответ", style="dashed")
    g_act_bot = y + 0.05

    # Draw activation bars (after computing top/bottom y's)
    _activation(ax, G, g_act_top, g_act_bot)
    _activation(ax, M, m_act_top, m_act_bot)
    _activation(ax, L, l_act_top, l_act_bot)

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / out_name
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.2, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    render()
