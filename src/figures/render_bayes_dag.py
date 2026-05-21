"""Render Bayes DAG figure for thesis (Рисунок 3.5) via graphviz.

Pipeline: dot → SVG (preserves TNR font-family) → rsvg-convert → PNG (300 dpi).
The macOS graphviz PNG renderer ignores the requested font, so we go through SVG.

Style:
- shape=box (rectangles), thin black borders, white fill
- Times New Roman 12
- splines=ortho (right-angle edges), filled arrowheads
- Real DAG structure from the trained chocolate Bayesian network

Usage:
    OMP_NUM_THREADS=1 python -m src.figures.render_bayes_dag
"""
from __future__ import annotations

import pickle
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
OUT = ROOT / "docs" / "thesis" / "figures"

NODE_LABELS_RU = {
    "chocolate_type":            "Тип шоколада",
    "cocoa_percentage":          "Содержание\nкакао",
    "is_organic":                "Органический\nпродукт",
    "brand":                     "Бренд",
    "brand_has_organic_marker":  "Маркер «organic»\nв бренде",
    "nutri_score_grade":         "Nutri-Score",
    "protein_class":             "Класс белка",
    "contains_nuts":             "Содержит орехи",
    "chocolate_extra":           "Дополнительные\nатрибуты",
}


def render(category: str = "chocolate_stratified",
           out_name: str = "fig_3_5_bayes_dag.png") -> Path:
    with open(MODELS / f"{category}_bayesian.pkl", "rb") as f:
        model = pickle.load(f)
    edges = list(model.edges())
    nodes = sorted({n for e in edges for n in e})

    lines = [
        "digraph G {",
        '  graph [rankdir=TB, splines=ortho, nodesep=0.45, ranksep=0.6, bgcolor="white"];',
        '  node  [shape=box, style=filled, fillcolor="white",'
        '         fontname="Times New Roman", fontsize=12,'
        '         color="black", penwidth=1.0, margin="0.18,0.10"];',
        '  edge  [color="black", arrowsize=0.7, penwidth=0.8, arrowhead=normal];',
        "",
    ]
    for n in nodes:
        label = NODE_LABELS_RU.get(n, n.replace("_", " "))
        lines.append(f'  "{n}" [label="{label}"];')
    lines.append("")
    for src, dst in edges:
        lines.append(f'  "{src}" -> "{dst}";')
    lines.append("}")

    dot_src = "\n".join(lines)
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / out_name

    # dot → SVG → rsvg-convert → PNG (only this path renders TNR correctly on macOS)
    dot_proc = subprocess.run(
        ["dot", "-Tsvg"],
        input=dot_src.encode("utf-8"),
        capture_output=True, check=False,
    )
    if dot_proc.returncode != 0:
        raise RuntimeError(f"dot failed: {dot_proc.stderr.decode()}")

    rsvg_proc = subprocess.run(
        ["rsvg-convert", "-d", "300", "-p", "300", "-o", str(out_path)],
        input=dot_proc.stdout, capture_output=True, check=False,
    )
    if rsvg_proc.returncode != 0:
        raise RuntimeError(f"rsvg-convert failed: {rsvg_proc.stderr.decode()}")

    print(f"Saved: {out_path}")
    print(f"Nodes: {len(nodes)}, edges: {len(edges)}")
    return out_path


if __name__ == "__main__":
    render()
