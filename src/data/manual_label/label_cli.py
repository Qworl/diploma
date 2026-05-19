"""
Ручная разметка hold-out выборки для honest accuracy-якоря.

Зачем: silver standard размечен Haiku 4.5. Бенчмарк cheap-LLM (notebook 05 § 5)
против такой разметки меряет agreement-with-Haiku, а не accuracy. Чтобы ноутбук
выдавал защитимые цифры, нужен hold-out с человеческой разметкой.

Что делает скрипт:
- Берёт тот же test split, что run_experiments.py (TEST_SIZE=0.2, RANDOM_STATE=42)
  из <category>_stratified_silver_standard.parquet — сбалансированного по языкам
  (Phase 13). Из test_idx стабильно отбирает первые --per-category товаров.
- Для каждого товара переводит product_name + ingredients_text на русский через
  OpenRouter (cheap model, кэш в JSON — переводим каждый товар один раз).
- Спрашивает значение каждого атрибута по схеме. Silver-метку НЕ показывает, чтобы
  не было якорения. Поддерживает skip ('s'), back ('b'), quit ('q').
- Пишет ответы в JSONL после каждого ответа (resumable) и в parquet снапшот.

Подкоманды:
    label   — собственно разметка (default)
    compare — agreement vs silver standard по атрибутам
    status  — сколько уже размечено

Usage:
    python scripts/manual_label.py --category pasta --per-category 50
    python scripts/manual_label.py --category chocolate compare
    python scripts/manual_label.py --category beverages status
    # ограничить набор атрибутов (быстрее):
    python scripts/manual_label.py --category pasta --attrs is_organic,grain_type
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from src.common import PROCESSED_DIR, RANDOM_STATE, TEST_SIZE, setup_logging
from src.data.manual_label.hints import Hint, compute_hints
from src.pipeline.schemas import BEVERAGE_SCHEMA, CHOCOLATE_SCHEMA, PASTA_SCHEMA
from src.llm import call_openrouter

logger = logging.getLogger(__name__)

CATEGORY_CONFIG = {
    "pasta": {"silver": "pasta_stratified_silver_standard.parquet", "schema": PASTA_SCHEMA},
    "chocolate": {"silver": "chocolate_stratified_silver_standard.parquet", "schema": CHOCOLATE_SCHEMA},
    "beverages": {"silver": "beverages_stratified_silver_standard.parquet", "schema": BEVERAGE_SCHEMA},
}

OUTPUT_DIR = Path(PROCESSED_DIR) / "human_labels"
TRANSLATION_MODEL_DEFAULT = "google/gemini-2.5-flash-lite"
TRANSLATION_FIELDS = ("product_name", "ingredients_text", "generic_name", "categories_tags")

# Минимальный ANSI без зависимостей
class C:
    R = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"


# --- I/O ---

def out_paths(category: str) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "jsonl": OUTPUT_DIR / f"human_labels_{category}.jsonl",
        "parquet": OUTPUT_DIR / f"human_labels_{category}.parquet",
        "translations": OUTPUT_DIR / "translations.json",
    }


def load_translations(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def save_translations(path: Path, cache: dict) -> None:
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


def load_done_codes(jsonl_path: Path) -> set:
    if not jsonl_path.exists():
        return set()
    codes = set()
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                codes.add(str(rec["code"]))
            except Exception:
                continue
    return codes


def append_record(jsonl_path: Path, record: dict) -> None:
    with jsonl_path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def rebuild_parquet(jsonl_path: Path, parquet_path: Path) -> None:
    if not jsonl_path.exists():
        return
    rows = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if rows:
        pd.DataFrame(rows).to_parquet(parquet_path, index=False)


# --- Sampling ---

def select_pool(category: str, per_category: int) -> pd.DataFrame:
    cfg = CATEGORY_CONFIG[category]
    df = pd.read_parquet(Path(PROCESSED_DIR) / cfg["silver"]).reset_index(drop=True)
    _, test_idx = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    test_idx = np.array(test_idx)
    take = min(per_category, len(test_idx))
    pool = df.iloc[test_idx[:take]].reset_index(drop=True)
    pool["code"] = pool["code"].astype(str)
    return pool


# --- Translation ---

def translate_product(row: pd.Series, cache: dict, model: str, api_key: str) -> dict:
    """Перевод 4 полей (name/desc/ingredients/categories) → русский. Один LLM вызов на товар."""
    code = str(row["code"])
    if code in cache:
        return cache[code]

    payload = {f: ("" if pd.isna(row.get(f)) else str(row.get(f, "")))[:600] for f in TRANSLATION_FIELDS}
    if not any(payload.values()):
        cache[code] = payload
        return payload

    prompt = (
        "Translate the following food product fields to Russian for a human annotator "
        "who does not read other languages. Translate descriptive words (e.g. 'vasito de "
        "arroz redondo' → 'стаканчик круглого риса', 'pâtes complètes' → 'цельнозерновые "
        "макароны', 'sin azúcar añadido' → 'без добавленного сахара'). "
        "Keep registered brand names and proper nouns as-is (Barilla, Lindt). "
        "Keep ingredient lists as a comma-separated Russian list with percentages preserved. "
        "If a field is empty, return empty string. "
        "Return ONLY a JSON object with the same keys.\n\n"
        f"Input JSON:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    try:
        _resp = call_openrouter(
            [{"role": "user", "content": prompt}],
            model=model, api_key=api_key, enforce_json=True,
        )
        raw = _resp["raw"] if isinstance(_resp, dict) else _resp
        translated = json.loads(raw)
        if not isinstance(translated, dict):
            raise ValueError("not a dict")
        out = {f: str(translated.get(f, payload[f])) for f in TRANSLATION_FIELDS}
    except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning("translation failed for %s: %s — using original", code, e)
        out = payload

    cache[code] = out
    return out


# --- UI ---

def fmt_nutri(row: pd.Series) -> str:
    parts = []
    for col, label in [
        ("fat_100g", "fat"),
        ("sugars_100g", "sugar"),
        ("proteins_100g", "protein"),
        ("carbohydrates_100g", "carbs"),
    ]:
        v = row.get(col)
        if pd.notna(v):
            try:
                parts.append(f"{label}={float(v):.1f}g/100g")
            except (TypeError, ValueError):
                pass
    return ", ".join(parts) if parts else "—"


def render_product(idx: int, total: int, row: pd.Series, translations: dict) -> None:
    print()
    print(f"{C.BOLD}{C.CYAN}[{idx+1}/{total}]{C.R} {C.BOLD}{row.get('product_name', '—')}{C.R}")
    print(f"  {C.DIM}code={row['code']}  brand={row.get('brands', '—')}  qty={row.get('quantity', '—')}{C.R}")
    print(f"  {C.DIM}OFF: https://world.openfoodfacts.org/product/{row['code']}{C.R}")
    print(f"  {C.DIM}nutri: {fmt_nutri(row)}{C.R}")

    print(f"\n  {C.YELLOW}Оригинал:{C.R}")
    for f in ("product_name", "generic_name", "ingredients_text"):
        v = row.get(f)
        if pd.notna(v) and str(v).strip():
            print(f"    {f}: {str(v)[:240]}")
    cats = row.get("categories_tags")
    if pd.notna(cats):
        print(f"    categories_tags: {str(cats)[:200]}")

    print(f"\n  {C.GREEN}Перевод (RU):{C.R}")
    for f in TRANSLATION_FIELDS:
        v = translations.get(f, "")
        if v:
            print(f"    {f}: {v[:240]}")


def _fmt_hint_value(v) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)


def ask_attr(attr: str, spec: dict, hint: Hint | None = None) -> tuple:
    """Returns (status, value, hint_taken: bool).
    status: 'answer' | 'skip' | 'back' | 'quit'.
    """
    desc = spec.get("description", "")
    nullable = spec.get("nullable", False)
    typ = spec["type"]

    print(f"\n  {C.MAGENTA}{C.BOLD}{attr}{C.R} {C.DIM}({desc}){C.R}")
    if hint is not None:
        print(f"    {C.YELLOW}подсказка: {C.BOLD}{_fmt_hint_value(hint.value)}{C.R}"
              f"{C.YELLOW}  [{hint.source}: {hint.detail}]  →  '=' принять{C.R}")

    if typ == "bool":
        prompt = f"    [y]es / [n]o"
        if hint is not None:
            prompt += " / [=]hint"
        prompt += f" / [s]kip{' / [u]nknown→null' if nullable else ''} / [b]ack / [q]uit > "
        while True:
            ans = input(prompt).strip().lower()
            if ans == "=" and hint is not None and isinstance(hint.value, bool):
                return ("answer", hint.value, True)
            if ans in ("y", "yes", "д", "да"):
                return ("answer", True, False)
            if ans in ("n", "no", "н", "нет"):
                return ("answer", False, False)
            if ans == "s":
                return ("skip", None, False)
            if ans == "b":
                return ("back", None, False)
            if ans == "q":
                return ("quit", None, False)
            if nullable and ans == "u":
                return ("answer", None, False)
            print(f"    {C.RED}invalid — попробуй снова{C.R}")

    # enum / int with values
    values = list(spec.get("values", []))
    print("    варианты:")
    for i, v in enumerate(values, 1):
        marker = ""
        if hint is not None and str(hint.value) == str(v):
            marker = f"  {C.YELLOW}← подсказка{C.R}"
        print(f"      {i}) {v}{marker}")
    hint_in_values = hint is not None and any(str(v) == str(hint.value) for v in values)
    extras = ""
    if hint_in_values:
        extras += " / [=]hint"
    extras += " / [s]kip / [b]ack / [q]uit"
    if nullable:
        extras = " / [u]nknown→null" + extras
    while True:
        ans = input(f"    [1-{len(values)}]{extras} > ").strip().lower()
        if ans == "=" and hint_in_values:
            return ("answer", hint.value, True)
        if ans == "s":
            return ("skip", None, False)
        if ans == "b":
            return ("back", None, False)
        if ans == "q":
            return ("quit", None, False)
        if nullable and ans == "u":
            return ("answer", None, False)
        if ans.isdigit():
            i = int(ans)
            if 1 <= i <= len(values):
                v = values[i - 1]
                taken_from_hint = hint is not None and str(hint.value) == str(v)
                return ("answer", v, taken_from_hint)
        print(f"    {C.RED}invalid — попробуй снова{C.R}")


# --- Main label loop ---

@dataclass
class LoopState:
    quit_requested: bool = False


def label_one(row: pd.Series, schema: dict, attrs: list[str], category: str,
              translations: dict, idx: int, total: int, state: LoopState) -> dict | None:
    """Returns record dict (with answers) or None если quit."""
    render_product(idx, total, row, translations)
    hints = compute_hints(row, category, attrs)
    if hints:
        print(f"\n  {C.DIM}доступны подсказки для: {', '.join(hints.keys())}{C.R}")

    answers: dict = {}
    skipped: list[str] = []
    hint_metadata: dict = {}  # attr → {value, source, detail, taken: bool}
    i = 0
    while i < len(attrs):
        attr = attrs[i]
        hint = hints.get(attr)
        status, val, taken_from_hint = ask_attr(attr, schema[attr], hint=hint)
        if status == "quit":
            state.quit_requested = True
            return None
        if status == "back":
            if i == 0:
                print(f"    {C.YELLOW}первый атрибут — некуда возвращаться{C.R}")
                continue
            i -= 1
            prev = attrs[i]
            answers.pop(prev, None)
            hint_metadata.pop(prev, None)
            if prev in skipped:
                skipped.remove(prev)
            print(f"    {C.YELLOW}↶ откатили '{prev}'{C.R}")
            continue
        if status == "skip":
            skipped.append(attr)
        else:
            answers[attr] = val
        if hint is not None:
            hint_metadata[attr] = {
                "value": hint.value,
                "source": hint.source,
                "detail": hint.detail,
                "taken": taken_from_hint,
                "matches_answer": (status == "answer" and str(hint.value) == str(val)),
            }
        i += 1

    return {
        "code": str(row["code"]),
        "product_name": str(row.get("product_name") or ""),
        "brands": str(row.get("brands") or ""),
        "quantity": str(row.get("quantity") or ""),
        "answers": answers,
        "skipped": skipped,
        "hints": hint_metadata,
    }


def cmd_label(args):
    if args.category not in CATEGORY_CONFIG:
        sys.exit(f"unknown category: {args.category}")

    schema = CATEGORY_CONFIG[args.category]["schema"]
    if args.attrs:
        requested = [a.strip() for a in args.attrs.split(",") if a.strip()]
        unknown = [a for a in requested if a not in schema]
        if unknown:
            sys.exit(f"unknown attrs for {args.category}: {unknown}")
        attrs = requested
    else:
        attrs = list(schema.keys())

    paths = out_paths(args.category)
    pool = select_pool(args.category, args.per_category)
    done = load_done_codes(paths["jsonl"])
    todo = pool[~pool["code"].isin(done)].reset_index(drop=True)

    print(f"{C.BOLD}Категория:{C.R} {args.category}")
    print(f"{C.BOLD}Атрибуты:{C.R} {', '.join(attrs)}")
    print(f"{C.BOLD}Hold-out из stratified silver standard:{C.R} {len(pool)} (test split, deterministic)")
    print(f"{C.BOLD}Уже размечено:{C.R} {len(done)} → осталось {len(todo)}")
    if len(todo) == 0:
        print(f"{C.GREEN}Всё размечено для этой категории.{C.R} compare/status — отдельные команды.")
        return

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY не установлен (нужен для перевода)")

    translations_cache = load_translations(paths["translations"])

    state = LoopState()
    print(f"\n{C.DIM}Клавиши: '=' — принять подсказку, 's' — пропустить (если непонятно),{C.R}")
    print(f"{C.DIM}         'b' — назад, 'q' — сохранить и выйти.{C.R}")
    print(f"{C.DIM}Подсказка — детерминированная (нутриенты/labels_tags/regex), не от LLM.{C.R}")
    print(f"{C.DIM}В сомнительных случаях лучше skip — ложноположительная разметка хуже пропуска.{C.R}\n")

    for i, (_, row) in enumerate(todo.iterrows()):
        # Перевод (cached)
        try:
            translated = translate_product(row, translations_cache, args.translate_model, api_key)
            save_translations(paths["translations"], translations_cache)
        except Exception as e:
            logger.error("translation hard-fail for %s: %s", row["code"], e)
            translated = {}

        record = label_one(row, schema, attrs, args.category, translated, i, len(todo), state)
        if record is None:
            break
        # Add language hint from cache (if we ran langdetect later — placeholder for now)
        record["translation"] = translated
        append_record(paths["jsonl"], record)
        rebuild_parquet(paths["jsonl"], paths["parquet"])
        print(f"  {C.GREEN}✓ записано ({len(load_done_codes(paths['jsonl']))} всего){C.R}")

    if state.quit_requested:
        print(f"\n{C.YELLOW}Прервано пользователем — прогресс сохранён.{C.R}")
    else:
        print(f"\n{C.GREEN}Готово.{C.R}")
    print(f"  jsonl: {paths['jsonl']}")
    print(f"  parquet: {paths['parquet']}")


# --- Compare ---

def cmd_compare(args):
    paths = out_paths(args.category)
    if not paths["jsonl"].exists():
        sys.exit("нет размеченных данных — сначала запусти label")
    schema = CATEGORY_CONFIG[args.category]["schema"]
    attrs = list(schema.keys())

    human_rows = []
    with paths["jsonl"].open() as f:
        for line in f:
            line = line.strip()
            if line:
                human_rows.append(json.loads(line))
    if not human_rows:
        sys.exit("jsonl пустой")

    silver = pd.read_parquet(Path(PROCESSED_DIR) / CATEGORY_CONFIG[args.category]["silver"])
    silver["code"] = silver["code"].astype(str)
    silver_idx = silver.set_index("code")

    print(f"\n{C.BOLD}Agreement: human vs Haiku silver standard ({args.category}){C.R}")
    print(f"{C.DIM}n_labeled={len(human_rows)}{C.R}\n")
    print(f"  {'attr':<22} {'n':>4} {'agree':>6} {'silver_null':>11}  disagreements (human → silver)")
    print(f"  {'-'*22} {'-'*4} {'-'*6} {'-'*11}  {'-'*40}")

    summary_rows = []
    for attr in attrs:
        n_compared = 0
        n_agree = 0
        n_silver_null = 0
        disagreements = []
        for rec in human_rows:
            if attr not in rec.get("answers", {}):
                continue  # человек skipped
            human_val = rec["answers"][attr]
            if rec["code"] not in silver_idx.index:
                continue
            silver_val = silver_idx.loc[rec["code"], attr]
            if pd.isna(silver_val):
                # silver не разметил — human разметил → отдельный bucket
                if human_val is not None:
                    n_silver_null += 1
                continue
            n_compared += 1
            human_str = "None" if human_val is None else str(human_val)
            silver_str = str(silver_val)
            if human_str == silver_str:
                n_agree += 1
            else:
                disagreements.append((rec["code"], human_str, silver_str,
                                       rec.get("product_name", "")[:40]))

        agree_rate = (n_agree / n_compared * 100) if n_compared else float("nan")
        summary_rows.append({
            "attr": attr, "n": n_compared, "agree": n_agree,
            "agree_rate": agree_rate, "silver_null_filled": n_silver_null,
        })
        rate_str = f"{agree_rate:5.1f}%" if n_compared else "  —"
        color = C.GREEN if agree_rate >= 90 else C.YELLOW if agree_rate >= 75 else C.RED
        first_dis = ""
        if disagreements:
            d = disagreements[0]
            first_dis = f"e.g. {d[0]}: {d[1]!r} → {d[2]!r}"
        print(f"  {attr:<22} {n_compared:>4} {color}{rate_str}{C.R} {n_silver_null:>11}  {first_dis}")

    df = pd.DataFrame(summary_rows)
    out_path = paths["parquet"].parent / f"agreement_{args.category}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n  saved: {out_path}")

    if args.show_disagreements:
        print(f"\n{C.BOLD}Все расхождения:{C.R}")
        for attr in attrs:
            local = []
            for rec in human_rows:
                if attr not in rec.get("answers", {}):
                    continue
                if rec["code"] not in silver_idx.index:
                    continue
                silver_val = silver_idx.loc[rec["code"], attr]
                if pd.isna(silver_val):
                    continue
                hv = "None" if rec["answers"][attr] is None else str(rec["answers"][attr])
                sv = str(silver_val)
                if hv != sv:
                    local.append((rec["code"], hv, sv, rec.get("product_name", "")[:60]))
            if local:
                print(f"\n  {C.MAGENTA}{attr}:{C.R}")
                for code, hv, sv, name in local:
                    print(f"    {code} | human={hv!r} silver={sv!r} | {name}")


# --- Status ---

def cmd_status(args):
    paths = out_paths(args.category)
    pool = select_pool(args.category, args.per_category)
    done = load_done_codes(paths["jsonl"])
    print(f"{args.category}: {len(done)}/{len(pool)} размечено")
    if paths["parquet"].exists():
        df = pd.read_parquet(paths["parquet"])
        print(f"  parquet: {paths['parquet']} ({len(df)} строк)")
    schema = CATEGORY_CONFIG[args.category]["schema"]
    if not done:
        return
    rows = []
    with paths["jsonl"].open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"\n  Покрытие по атрибутам:")
    print(f"    {'attr':<22} {'answered':>8} {'skipped':>8} {'hint_avail':>10} {'hint_taken':>10}")
    for attr in schema:
        n_answered = sum(1 for r in rows if attr in r.get("answers", {}))
        n_skipped = sum(1 for r in rows if attr in r.get("skipped", []))
        n_hint_avail = sum(1 for r in rows if attr in r.get("hints", {}))
        n_hint_taken = sum(1 for r in rows
                           if r.get("hints", {}).get(attr, {}).get("taken"))
        print(f"    {attr:<22} {n_answered:>8} {n_skipped:>8} {n_hint_avail:>10} {n_hint_taken:>10}")


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True, choices=list(CATEGORY_CONFIG.keys()))
    p.add_argument("--per-category", type=int, default=50,
                   help="Сколько товаров отбирать из test split (deterministic)")
    p.add_argument("--attrs", default=None, help="Comma-separated subset атрибутов")
    p.add_argument("--translate-model", default=TRANSLATION_MODEL_DEFAULT)
    p.add_argument("--show-disagreements", action="store_true",
                   help="(compare) распечатать все расхождения с silver, не только агрегаты")
    p.add_argument("subcommand", nargs="?", default="label",
                   choices=["label", "compare", "status"])
    args = p.parse_args()

    if args.subcommand == "label":
        cmd_label(args)
    elif args.subcommand == "compare":
        cmd_compare(args)
    elif args.subcommand == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
