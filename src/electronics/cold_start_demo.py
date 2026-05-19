"""
Cold-start Bayesian demo для phones — главный артефакт Phase 11.

Из одного evidence (brand) выводятся остальные 6 атрибутов через Bayesian inference.
Это лучшая демонстрация архитектуры на защите дипломной — на food такого не получится,
потому что food атрибуты слабо связаны (kefir может быть от любого бренда), а phones —
сильно (Apple → iOS, premium specs / Samsung → Android, bar/foldable).

Usage:
    python -m src.electronics.cold_start_demo
    python -m src.electronics.cold_start_demo --brand Apple --json
"""

import argparse
import json
import logging
import os
import pickle
import sys

from src.common import MODELS_DIR, setup_logging

logger = logging.getLogger(__name__)

DEMO_BRANDS = ["Apple", "Samsung", "Xiaomi", "Huawei", "OnePlus", "Sony", "Other"]
TARGETS = ["os", "form_factor", "screen_size_class",
           "ram_class", "storage_class", "release_year_class"]


def cold_start(brand: str, model, inference) -> dict:
    """Возвращает {target: [(value, prob), ...]} по убыванию prob."""
    cpd = model.get_cpds("brand")
    if brand not in cpd.state_names["brand"]:
        return {"error": f"brand '{brand}' not in train data"}
    out = {}
    for target in TARGETS:
        if target not in model.nodes():
            continue
        try:
            res = inference.query([target], evidence={"brand": brand}, show_progress=False)
            probs = sorted(
                [(str(s), float(res.values[i])) for i, s in enumerate(res.state_names[target])],
                key=lambda x: -x[1],
            )
            out[target] = probs
        except Exception as e:
            out[target] = {"error": str(e)}
    return out


def fmt_human(brand: str, results: dict) -> list[str]:
    lines = [f"  {brand}:"]
    for target, probs in results.items():
        if isinstance(probs, dict) and "error" in probs:
            lines.append(f"    {target:<22} ERROR: {probs['error']}")
            continue
        top1 = probs[0]
        top2 = probs[1] if len(probs) > 1 else None
        line = f"    {target:<22} {top1[0]:>10}={top1[1]:.2f}"
        if top2 and top2[1] > 0.15:
            line += f"   {top2[0]}={top2[1]:.2f}"
        lines.append(line)
    return lines


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--brand", default=None,
                   help="Один brand. Default: прогон по DEMO_BRANDS")
    p.add_argument("--json", action="store_true",
                   help="JSON output для notebook'а")
    p.add_argument("--model", default=os.path.join(MODELS_DIR, "electronics_bayesian.pkl"))
    args = p.parse_args()

    from pgmpy.inference import VariableElimination
    with open(args.model, "rb") as f:
        model = pickle.load(f)
    inf = VariableElimination(model)

    brands = [args.brand] if args.brand else DEMO_BRANDS

    if args.json:
        data = {b: cold_start(b, model, inf) for b in brands}
        sys.stdout.write(json.dumps(data, indent=2, default=str) + "\n")
        return

    logger.info("=" * 70)
    logger.info("COLD-START Bayesian inference (electronics, brand-only evidence)")
    logger.info("=" * 70)
    logger.info("Edges in graph: %d", len(list(model.edges())))
    logger.info("")
    for b in brands:
        results = cold_start(b, model, inf)
        for line in fmt_human(b, results):
            logger.info(line)
        logger.info("")


if __name__ == "__main__":
    main()
