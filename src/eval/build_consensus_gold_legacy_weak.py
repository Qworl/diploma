"""Build consensus gold standard from WEAK-SEED LLM eval artefacts (legacy 2026-05-12).

Сохранён для §6.14.5 scaling-law reference. Текущий главный consensus —
build_consensus_gold.py (strong-seed: Sonnet 4.5 + GPT-4o + Gemini 2.5 Flash).

Build consensus gold standard from existing direct LLM eval artefacts.

Use case
--------
В §6.12 настоящая ручная gold-разметка отсутствует (есть только Sonnet 4.5
first-pass на 50 продуктах × 6 категорий без верификации). Этот скрипт
строит **emulated gold** комбинируя:

1. **Silver-эталон** на парах (category, attr) с tier ∈ {gold, close_to_gold}
   по таксономии §6.12.1 (`src/eval/validation_sources.py`). Для нутри-классов,
   INCI-regex, OFF labels_tags и NOVA-derived — silver = детерминированно
   вычислено из реальных полей OFF (см. label_silver.py).

2. **3-LLM majority consensus** на парах с tier=silver_strong (категориальные
   с текстовой семантикой). Берутся GPT-4o-mini + gpt-oss-120b + llama-3.2-3b
   из существующих `direct_llm_eval_*_stratified[_suffix].parquet`. **Haiku 4.5
   исключён** — это Layer 4 каскада, его участие в consensus создаёт
   циркулярность с router/cascade оценкой.

3. **Arbitrage stub** для пар без 2-of-3 majority — сохраняются в отдельный
   CSV для последующей ручной проверки (но они уже не блокируют пересчёт
   метрик: учитываются как "no_consensus" пропуск).

Output
------
- `datasets/processed/consensus_gold_weak.parquet` — long format (legacy weak-seed output)
    columns = [category, code, attr, source, tier, gt_consensus, n_votes,
               agreement_strength, silver_value]
- `datasets/processed/consensus_arbitrage_candidates.csv` — only no-consensus pairs
    columns = [category, code, attr, product_name, silver,
               gpt4omini, gptoss, llama3b, your_arbitrage]

Usage
-----
    python -m src.eval.build_consensus_gold
    # параметры по умолчанию: 3 LLM (без Haiku), threshold = 2-of-3
"""
from __future__ import annotations

import logging
import os
import sys

import pandas as pd

from src.common import PROCESSED_DIR, setup_logging
from src.eval.validation_sources import (
    VALIDATION_SOURCE,
    SOURCE_TIER,
    Source,
    SourceTier,
    get_source,
    get_tier,
)

logger = logging.getLogger(__name__)

# 3 LLM для consensus. Haiku 4.5 исключён (=Layer 4 каскада).
# Suffix → метка для CSV.
CONSENSUS_LLMS = [
    ("_gpt4omini", "gpt4omini"),
    ("_gptoss",    "gptoss"),
    ("_llama3b",   "llama3b"),
]

FOOD_CATS = ["pasta", "chocolate", "beverages", "cheeses", "cereals", "cosmetics"]

# Path к arbitrage CSVs (created by build_arbitrage_csv.py, filled in arbitrage_app.py)
ARBITRAGE_DIR = "datasets/manual_label"


def _load_arbitrage(cat: str) -> dict[tuple[str, str], str]:
    """Загрузить ручной арбитраж для категории.

    Ищет datasets/manual_label/arbitrage_{cat}_*.csv. Если файл есть —
    возвращает {(code, attr): your_arbitrage} для строк с непустой колонкой.
    """
    out: dict[tuple[str, str], str] = {}
    import glob
    paths = glob.glob(f"{ARBITRAGE_DIR}/arbitrage_{cat}_*.csv")
    for p in paths:
        try:
            df = pd.read_csv(p, dtype={"code": str})
        except Exception as e:
            logger.warning("Cannot read arbitrage CSV %s: %s", p, e)
            continue
        if "your_arbitrage" not in df.columns:
            continue
        # attr из имени файла: arbitrage_{cat}_{attr}.csv
        fname = os.path.basename(p).replace(".csv", "")
        attr = fname.replace(f"arbitrage_{cat}_", "", 1)
        for _, row in df.iterrows():
            arb = str(row.get("your_arbitrage") or "").strip()
            if not arb:
                continue
            code = str(row.get("code") or "").strip()
            if code:
                out[(code, attr)] = arb.lower()
        logger.info("[%s] loaded %d manual arbitrage labels from %s",
                    cat, sum(1 for k in out if k[1] == attr), p)
    return out


def _normalize(v) -> str | None:
    """Нормализовать значение для сравнения (case-insensitive str)."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().lower()
    if s in ("", "none", "nan", "null"):
        return None
    return s


def _majority_vote(votes: list[str | None]) -> tuple[str | None, int, str]:
    """Majority из 3 голосов.

    Returns
    -------
    (consensus, n_non_null_votes, strength)
        consensus    — мажоритарный ответ (str или None)
        n_non_null   — сколько LLM ответили не-None
        strength     — "unanimous" / "majority_2of3" / "no_majority"
    """
    non_null = [v for v in votes if v is not None]
    if len(non_null) == 0:
        return None, 0, "no_majority"
    if len(set(non_null)) == 1:
        return non_null[0], len(non_null), "unanimous" if len(non_null) == 3 else "majority_2of3"
    # бывает 2 vs 1
    from collections import Counter
    cnt = Counter(non_null)
    top, top_count = cnt.most_common(1)[0]
    if top_count >= 2:
        return top, len(non_null), "majority_2of3"
    return None, len(non_null), "no_majority"


def load_llm_votes(cat: str) -> pd.DataFrame:
    """Загрузить per-product predictions от 3 LLM для категории.

    Returns long DataFrame: (code, attr, gpt4omini, gptoss, llama3b).
    """
    base = f"{PROCESSED_DIR}/direct_llm_eval_{cat}_stratified"
    rows = []
    for suffix, label in CONSENSUS_LLMS:
        path = f"{base}{suffix}.parquet"
        if not os.path.exists(path):
            logger.warning("Skip LLM %s for %s (no %s)", label, cat, path)
            continue
        df = pd.read_parquet(path)
        df = df[["code", "attr", "pred"]].rename(columns={"pred": label})
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["code", "attr", "gpt4omini", "gptoss", "llama3b"])
    # outer merge на (code, attr)
    out = rows[0]
    for r in rows[1:]:
        out = out.merge(r, on=["code", "attr"], how="outer")
    for label in ("gpt4omini", "gptoss", "llama3b"):
        if label not in out.columns:
            out[label] = None
    return out


def build_consensus(cat: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Построить consensus gold для всех атрибутов категории.

    Returns
    -------
    (consensus_df, arbitrage_df)
        consensus_df  — final long-format gold (manual arbitrage > silver > consensus per pair)
        arbitrage_df  — пары без 2-of-3 majority (silver_strong only)
    """
    silver_path = f"{PROCESSED_DIR}/{cat}_stratified_silver_standard.parquet"
    silver = pd.read_parquet(silver_path)

    llm_votes = load_llm_votes(cat)
    manual_arb = _load_arbitrage(cat)  # {(code, attr): str}

    rows = []
    arbitrage_rows = []

    for attr in [a for (c, a) in VALIDATION_SOURCE.keys() if c == cat]:
        src = get_source(cat, attr)
        tier = get_tier(cat, attr)
        # silver values for this attribute
        if attr not in silver.columns:
            logger.warning("[%s/%s] attr not in silver", cat, attr)
            continue

        for _, row in silver.iterrows():
            code = str(row["code"])
            silver_val = _normalize(row[attr])

            # Manual arbitrage перебивает всё (если задан)
            manual = manual_arb.get((code, attr))

            if manual is not None:
                rows.append({
                    "category": cat,
                    "code": code,
                    "attr": attr,
                    "source": src.value,
                    "tier": tier.value,
                    "gt_consensus": manual,
                    "n_votes": 1,
                    "agreement_strength": "manual_arbitrage",
                    "silver_value": silver_val,
                })
                continue

            if tier in (SourceTier.GOLD, SourceTier.CLOSE_TO_GOLD):
                # Berem silver as-is. На непокрытом срезе (silver=None)
                # — пропускаем (для метрик такие пары не считаются).
                rows.append({
                    "category": cat,
                    "code": code,
                    "attr": attr,
                    "source": src.value,
                    "tier": tier.value,
                    "gt_consensus": silver_val,
                    "n_votes": 1,
                    "agreement_strength": "silver_authoritative",
                    "silver_value": silver_val,
                })
            else:
                # silver_strong: majority of 3 LLM
                v = llm_votes[(llm_votes.code == code) & (llm_votes.attr == attr)]
                if v.empty:
                    consensus, n, strength = None, 0, "no_llm_data"
                    votes_per_llm = (None, None, None)
                else:
                    r = v.iloc[0]
                    votes = (
                        _normalize(r.get("gpt4omini")),
                        _normalize(r.get("gptoss")),
                        _normalize(r.get("llama3b")),
                    )
                    consensus, n, strength = _majority_vote(list(votes))
                    votes_per_llm = votes

                rows.append({
                    "category": cat,
                    "code": code,
                    "attr": attr,
                    "source": src.value,
                    "tier": tier.value,
                    "gt_consensus": consensus,
                    "n_votes": n,
                    "agreement_strength": strength,
                    "silver_value": silver_val,
                })

                if strength == "no_majority" and n >= 2:
                    # пары с 2 расхождением — кандидаты на ручной арбитраж
                    arbitrage_rows.append({
                        "category": cat,
                        "code": code,
                        "attr": attr,
                        "product_name": row.get("product_name", ""),
                        "brands": row.get("brands", ""),
                        "ingredients_text": str(row.get("ingredients_text", "") or "")[:300],
                        "silver": silver_val,
                        "gpt4omini": votes_per_llm[0],
                        "gptoss": votes_per_llm[1],
                        "llama3b": votes_per_llm[2],
                        "your_arbitrage": "",
                    })

    consensus_df = pd.DataFrame(rows)
    arbitrage_df = pd.DataFrame(arbitrage_rows)
    return consensus_df, arbitrage_df


def main():
    setup_logging()
    all_consensus = []
    all_arbitrage = []
    for cat in FOOD_CATS:
        cdf, adf = build_consensus(cat)
        logger.info("[%s] consensus=%d  arbitrage=%d", cat, len(cdf), len(adf))
        if len(cdf):
            all_consensus.append(cdf)
        if len(adf):
            all_arbitrage.append(adf)

    out_c = f"{PROCESSED_DIR}/consensus_gold_weak.parquet"
    out_a = f"{PROCESSED_DIR}/consensus_arbitrage_candidates_weak.csv"
    cons = pd.concat(all_consensus, ignore_index=True)
    cons.to_parquet(out_c, index=False)
    logger.info("Saved %d rows -> %s", len(cons), out_c)

    if all_arbitrage:
        arb = pd.concat(all_arbitrage, ignore_index=True)
        arb.to_csv(out_a, index=False, encoding="utf-8")
        logger.info("Saved %d arbitrage candidates -> %s", len(arb), out_a)
    else:
        logger.info("No arbitrage candidates (every silver_strong pair has 2-of-3 majority).")

    # Сводка
    print()
    print("=== Consensus gold summary ===")
    print(f"Total pairs: {len(cons)}")
    print()
    print("По tier:")
    print(cons.groupby("tier")["gt_consensus"].agg(
        n="count", n_with_gt=lambda s: s.notna().sum()
    ).to_string())
    print()
    print("Agreement strength (silver_strong only):")
    ss = cons[cons.tier == "silver_strong"]
    if len(ss):
        print(ss["agreement_strength"].value_counts().to_string())
    else:
        print("  (нет silver_strong пар)")


if __name__ == "__main__":
    main()
