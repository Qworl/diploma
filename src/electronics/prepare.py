"""
Подготовка smartphone dataset → silver standard для ELECTRONICS_SCHEMA.

Source: HuggingFace `Nadirova/Phone_SpecsDataset_25K` (зеркало phonedb.net, 25008 phones).
Sink:   datasets/processed/electronics_silver_standard.parquet — структурно совместимо
        с food silver standards (есть `code`, `product_name`, `brands`, attribute columns).

Phones — особый случай: spec values уже структурированы (не natural language), поэтому
silver standard собирается **детерминированно** из dataset, а не через LLM-разметку.
Это и есть «true ground truth» для phones — мы НЕ выдумываем новые поля, только маппим
существующие spec keys в ELECTRONICS_SCHEMA.

ELECTRONICS_SCHEMA имеет 8 атрибутов; в этом dataset покрыты 7 из 8:
  brand, os, form_factor, screen_size_class, ram_class, storage_class, release_year_class.
`price_tier` отсутствует в источнике → останется NaN (в audit будет drop).
Это не «новое поле» — это просто missing data для одного из existing schema attrs.

Usage:
    python -m src.electronics.prepare
    python -m src.electronics.prepare --max 1000  # subsample for dev
"""

import argparse
import logging
import os
import re

import pandas as pd

from src.common import PROCESSED_DIR, RAW_DIR, setup_logging

logger = logging.getLogger(__name__)

RAW_PATH = os.path.join(RAW_DIR, "electronics_phonedb_raw.parquet")
OUT_PATH = os.path.join(PROCESSED_DIR, "electronics_silver_standard.parquet")


# --- Schema brands (mirrors ELECTRONICS_SCHEMA, but used here as labelling vocabulary) ---
SCHEMA_BRANDS = {
    "apple": "Apple", "samsung": "Samsung", "xiaomi": "Xiaomi", "huawei": "Huawei",
    "oppo": "Oppo", "vivo": "Vivo", "oneplus": "OnePlus", "google": "Google", "sony": "Sony",
}

# BBK family — parent of Vivo/Oppo/OnePlus/Realme. Split via title substring.
BBK_DISAMBIG = {
    "oneplus": "OnePlus", "realme": "Other", "vivo": "Vivo", "oppo": "Oppo", "iqoo": "Vivo",
}


def _spec_value(spec: dict, key: str):
    """Извлекает spec[key].value (phonedb wraps everything в {description,key,value} dicts).
    Возвращает None если ключ отсутствует / value пустое."""
    if not isinstance(spec, dict):
        return None
    v = spec.get(key)
    if v is None:
        return None
    if isinstance(v, dict):
        v = v.get("value")
    if v is None:
        return None
    s = str(v).strip()
    if not s or s == "\xa0":
        return None
    return s


def map_brand(spec: dict, title: str) -> str | None:
    raw = _spec_value(spec, "Brand")
    if not raw:
        return None
    raw_low = raw.lower().strip()
    if raw_low in SCHEMA_BRANDS:
        return SCHEMA_BRANDS[raw_low]
    if raw_low == "bbk":
        title_low = (title or "").lower()
        for kw, brand in BBK_DISAMBIG.items():
            if kw in title_low:
                return brand
        return "Other"
    return "Other"


def map_os(spec: dict) -> str | None:
    """OS из 'Operating System' или 'Platform'. Bucket в schema values."""
    raw = _spec_value(spec, "Operating System") or _spec_value(spec, "Platform")
    if not raw:
        return None
    rl = raw.lower()
    if "ios" in rl or "iphone os" in rl or "iPadOS" in raw:
        return "iOS"
    if "harmony" in rl:
        return "HarmonyOS"
    if "android" in rl:
        return "Android"
    return "Other"


_FOLD_RE = re.compile(r"\b(fold|fold[35-9]|fold\s*[0-9]+)\b", re.IGNORECASE)
_FLIP_RE = re.compile(r"\b(flip[345]|flip\s*[0-9]+|razr|w20\s*5g)\b", re.IGNORECASE)
_RUGGED_RE = re.compile(r"\b(rugged|xcover|toughbook|cat\s*s\d+|land\s*rover|active\s*pro|armor)\b",
                         re.IGNORECASE)


def map_form_factor(spec: dict, title: str) -> str | None:
    """Form factor: bar / foldable / flip / rugged / other. Используем title + Device Category.
    'Smartphone' не различает foldable от bar → проверяем по подстрокам."""
    title = title or ""
    if _FLIP_RE.search(title):
        return "flip"
    if _FOLD_RE.search(title):
        return "foldable"
    if _RUGGED_RE.search(title):
        return "rugged"
    # default for smartphones
    cat = _spec_value(spec, "Device Category")
    if cat and cat.lower() == "smartphone":
        return "bar"
    return None


_DISPLAY_RE = re.compile(r"([\d.]+)\s*mm")


def map_screen_size_class(spec: dict) -> str | None:
    """Display Diagonal: '170.9 mm' → bucket. 1 inch = 25.4 mm.
    small <5", medium 5-6", large 6-6.7", phablet >6.7"."""
    raw = _spec_value(spec, "Display Diagonal")
    if not raw:
        return None
    m = _DISPLAY_RE.search(raw)
    if not m:
        return None
    try:
        mm = float(m.group(1))
    except ValueError:
        return None
    inches = mm / 25.4
    if inches < 5.0:
        return "small"
    if inches < 6.0:
        return "medium"
    if inches < 6.7:
        return "large"
    return "phablet"


_RAM_RE = re.compile(r"([\d.]+)\s*(GiB|GB|MiB|MB)\s*RAM", re.IGNORECASE)


def map_ram_class(spec: dict) -> str | None:
    """'16 GiB RAM' → bucket. Schema: 2GB/4GB/6GB/8GB/12GB+."""
    raw = _spec_value(spec, "RAM Capacity (converted)") or _spec_value(spec, "RAM Capacity")
    if not raw:
        return None
    m = _RAM_RE.search(raw)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    unit = m.group(2).lower()
    if unit.startswith("m"):  # MiB / MB
        v = v / 1024.0
    # GiB ≈ GB для bucket purposes
    if v < 3:
        return "2GB"
    if v < 5:
        return "4GB"
    if v < 7:
        return "6GB"
    if v < 12:
        return "8GB"
    return "12GB+"


_STORAGE_RE = re.compile(r"([\d.]+)\s*(TB|GB|MB)", re.IGNORECASE)


def map_storage_class(spec: dict) -> str | None:
    """'1000 GB ROM' → bucket. Schema: 32GB/64GB/128GB/256GB/512GB+."""
    raw = (_spec_value(spec, "Non-volatile Memory Capacity (converted)")
           or _spec_value(spec, "Non-volatile Memory Capacity"))
    if not raw:
        return None
    m = _STORAGE_RE.search(raw)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    unit = m.group(2).upper()
    if unit == "TB":
        v *= 1000
    elif unit == "MB":
        v /= 1024
    if v < 48:
        return "32GB"
    if v < 96:
        return "64GB"
    if v < 192:
        return "128GB"
    if v < 384:
        return "256GB"
    return "512GB+"


_YEAR_RE = re.compile(r"(\d{4})")


def map_release_year_class(spec: dict) -> str | None:
    """'2025 May 22' → bucket. Schema: pre-2020/2020-2022/2023-2024/2025+."""
    raw = _spec_value(spec, "Released") or _spec_value(spec, "Announced")
    if not raw:
        return None
    m = _YEAR_RE.search(raw)
    if not m:
        return None
    try:
        y = int(m.group(1))
    except ValueError:
        return None
    if y < 2020:
        return "pre-2020"
    if y < 2023:
        return "2020-2022"
    if y < 2025:
        return "2023-2024"
    return "2025+"


# --- Pipeline ---

def is_smartphone(spec: dict) -> bool:
    cat = _spec_value(spec, "Device Category")
    return bool(cat) and cat.lower() == "smartphone"


def synth_text(title: str, spec: dict) -> dict:
    """
    Синтезирует partner-input-style поля для совместимости с food pipeline:
      - product_name: title (бренд+модель)
      - brands: Brand spec value (для embeddings)
      - ingredients_text: пусто (для phones inapplicable)
      - quantity: пусто
    Все остальные cols — атрибуты (silver standard) или metadata.
    """
    return {
        "product_name": title or "",
        "brands": _spec_value(spec, "Brand") or "",
        "ingredients_text": "",
        "quantity": "",
        "categories_tags": "",
        "labels_tags": "",
    }


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--max", type=int, default=None,
                   help="Subsample N rows (для dev / smoke test). Default: all.")
    p.add_argument("--in-path", default=RAW_PATH)
    p.add_argument("--out-path", default=OUT_PATH)
    p.add_argument("--modern-only", action="store_true",
                   help="Keep only release_year >= 2020. Убирает Windows Mobile/Symbian "
                        "из os=Other и делает form_factor более realistic (foldable/flip "
                        "видны в 3-4%% доле, не 0.1%% как pre-2020).")
    args = p.parse_args()

    logger.info("Loading raw: %s", args.in_path)
    df = pd.read_parquet(args.in_path)
    logger.info("Total rows: %d", len(df))

    # Filter to Smartphone only
    smart_mask = df["specs"].apply(is_smartphone)
    df = df[smart_mask].copy()
    logger.info("After Device Category=Smartphone filter: %d", len(df))

    if args.max:
        df = df.head(args.max).copy()
        logger.info("Subsampled to: %d", len(df))

    # Apply mappings
    rows = []
    for _, r in df.iterrows():
        spec = r["specs"]
        title = r.get("title", "")
        text_fields = synth_text(title, spec)
        attrs = {
            "brand": map_brand(spec, title),
            "os": map_os(spec),
            "form_factor": map_form_factor(spec, title),
            "screen_size_class": map_screen_size_class(spec),
            "ram_class": map_ram_class(spec),
            "storage_class": map_storage_class(spec),
            # price_tier — отсутствует в источнике → NaN (НЕ придумываем поле)
            "price_tier": None,
            "release_year_class": map_release_year_class(spec),
        }
        # Stable code: hash url since phonedb urls unique per device
        url = r.get("url", "") or ""
        code = url.split("id=")[-1].split("&")[0] if "id=" in url else str(hash(url) & 0xFFFFFFFF)
        rows.append({"code": str(code), "url": url, **text_fields, **attrs})

    out = pd.DataFrame(rows)

    if args.modern_only:
        before = len(out)
        out = out[out["release_year_class"].isin(["2020-2022", "2023-2024", "2025+"])].copy()
        logger.info("Modern-only filter (>=2020): %d -> %d", before, len(out))

    logger.info("Schema fill rates:")
    for col in ["brand", "os", "form_factor", "screen_size_class", "ram_class",
                "storage_class", "price_tier", "release_year_class"]:
        n = out[col].notna().sum()
        logger.info("  %-22s %5d / %5d  (%5.1f%%)",
                    col, n, len(out), n / len(out) * 100)

    out.to_parquet(args.out_path, index=False)
    logger.info("Saved -> %s (%d rows)", args.out_path, len(out))


if __name__ == "__main__":
    main()
