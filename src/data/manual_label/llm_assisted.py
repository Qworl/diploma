"""
LLM-assisted gold labeling: сильная LLM (Sonnet 4.5 по умолчанию) ставит
черновую разметку в colu `manual_*` существующего CSV, а человек верифицирует.

Это академически нормальный воркфлоу для weak-supervision проектов: первый
проход делает сильная модель, второй — человек, который проверяет и правит
несогласия. Результат — НЕ "чистая ручная разметка", а "верифицированная
LLM-разметка"; именно так это и нужно называть в §6.12 / §7.1.

Ключевые оговорки методологии:
- Для разметки используется модель *сильнее* той, что стоит в production-каскаде
  (Sonnet 4.5 vs Haiku 4.5 в каскаде/baseline) — иначе была бы циркулярность
  с direct LLM baseline.
- Модель работает по тому же `enrich_product`, что и Layer 4, но с другой моделью.
- Человеку рекомендуется отдельно проверить все продукты, где Sonnet и Haiku
  расходятся (см. вспомогательный отчёт `disagreement_*.csv`).

Usage:
    python scripts/llm_assisted_label.py --all
    python scripts/llm_assisted_label.py --category pasta --model anthropic/claude-sonnet-4.5
"""
from __future__ import annotations

import argparse
import logging
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd

from src.common import setup_logging
from src.pipeline.schemas import (
    BEVERAGE_SCHEMA,
    CHEESES_SCHEMA,
    CEREALS_SCHEMA,
    CHOCOLATE_SCHEMA,
    COSMETICS_SCHEMA,
    PASTA_SCHEMA,
)
from src.pipeline.llm_fallback import enrich_product

logger = logging.getLogger(__name__)

SCHEMAS = {
    "pasta": PASTA_SCHEMA,
    "chocolate": CHOCOLATE_SCHEMA,
    "beverages": BEVERAGE_SCHEMA,
    "cosmetics": COSMETICS_SCHEMA,
    "cheeses": CHEESES_SCHEMA,
    "cereals": CEREALS_SCHEMA,
}
LABEL_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets", "manual_label")
INPUT_FIELDS = ["product_name", "brands", "ingredients_text", "quantity"]


def label_category(category: str, model: str, force: bool) -> None:
    in_path = os.path.join(LABEL_DIR, f"{category}_to_label.csv")
    out_path = os.path.join(LABEL_DIR, f"{category}_labeled.csv")

    if not os.path.exists(in_path):
        logger.warning("[%s] %s не найден — пропуск", category, in_path)
        return

    df = pd.read_csv(in_path)
    # удалить строку легенды (первая строка с '##' в code)
    df = df[~df["code"].astype(str).str.startswith("##")].copy().reset_index(drop=True)
    schema = SCHEMAS[category]

    # сохранить уже заполненные руками значения, если они есть и пользователь не --force
    out_existing = pd.read_csv(out_path) if (os.path.exists(out_path) and not force) else None
    if out_existing is not None:
        out_existing = out_existing[
            ~out_existing["code"].astype(str).str.startswith("##")
        ].copy()
        out_existing["code"] = out_existing["code"].astype(str)

    df["code"] = df["code"].astype(str)
    logger.info("[%s] %d продуктов, model=%s", category, len(df), model)

    t0 = time.time()
    for i, row in df.iterrows():
        # пропустить продукт, у которого все manual_* уже заполнены (резюм)
        manual_cols = [f"manual_{a}" for a in schema]
        if out_existing is not None and row["code"] in out_existing["code"].values:
            existing_row = out_existing[out_existing["code"] == row["code"]].iloc[0]
            existing_filled = sum(
                1 for c in manual_cols
                if c in existing_row.index
                and pd.notna(existing_row[c])
                and str(existing_row[c]).strip() != ""
            )
            if existing_filled == len(manual_cols):
                # переносим без вызова LLM
                for c in manual_cols:
                    if c in existing_row.index:
                        df.at[i, c] = existing_row[c]
                continue

        product = {f: row.get(f) for f in INPUT_FIELDS if f in df.columns}
        product["code"] = row["code"]
        try:
            pred = enrich_product(
                product, schema, backend="openrouter",
                model=model, enforce_json=True,
            )
        except Exception as exc:
            logger.warning("  [%d/%d] %s — LLM error: %s",
                            i + 1, len(df), row["code"], exc)
            continue

        for attr in schema:
            mcol = f"manual_{attr}"
            if mcol not in df.columns:
                continue
            v = pred.get(attr)
            df.at[i, mcol] = "" if v is None else str(v)

        if (i + 1) % 10 == 0 or (i + 1) == len(df):
            df.to_csv(out_path, index=False, encoding="utf-8")
            logger.info("  [%s] %d/%d, %.1fs (%.1fs/product) -> %s",
                         category, i + 1, len(df),
                         time.time() - t0, (time.time() - t0) / (i + 1), out_path)

    df.to_csv(out_path, index=False, encoding="utf-8")
    logger.info("[%s] DONE -> %s", category, out_path)


def disagreement_report(category: str) -> None:
    """Сравнить manual_* с silver_* и сохранить случаи расхождений для ручной верификации."""
    out_path = os.path.join(LABEL_DIR, f"{category}_labeled.csv")
    if not os.path.exists(out_path):
        return
    df = pd.read_csv(out_path)
    df = df[~df["code"].astype(str).str.startswith("##")].copy()
    schema = SCHEMAS[category]
    rows = []
    for _, r in df.iterrows():
        for attr in schema:
            silver_col = f"silver_{attr}"
            manual_col = f"manual_{attr}"
            if silver_col not in df.columns or manual_col not in df.columns:
                continue
            s = "" if pd.isna(r[silver_col]) else str(r[silver_col]).strip()
            m = "" if pd.isna(r[manual_col]) else str(r[manual_col]).strip()
            if s == m:
                continue
            rows.append({
                "code": r["code"],
                "product_name": r.get("product_name", ""),
                "attr": attr,
                "silver": s,
                "llm_assisted": m,
            })
    out_disagreement = os.path.join(LABEL_DIR, f"{category}_disagreement.csv")
    pd.DataFrame(rows).to_csv(out_disagreement, index=False, encoding="utf-8")
    logger.info("[%s] %d расхождений -> %s", category, len(rows), out_disagreement)


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--category", choices=list(SCHEMAS.keys()))
    p.add_argument("--all", action="store_true")
    p.add_argument("--model", default="anthropic/claude-sonnet-4.5",
                   help="OpenRouter model id (default: claude-sonnet-4.5)")
    p.add_argument("--force", action="store_true",
                   help="Игнорировать частично заполненный _labeled.csv и переписать")
    args = p.parse_args()

    cats = list(SCHEMAS.keys()) if args.all else ([args.category] if args.category else [])
    if not cats:
        p.error("укажите --category или --all")

    for cat in cats:
        label_category(cat, model=args.model, force=args.force)
        disagreement_report(cat)


if __name__ == "__main__":
    main()
