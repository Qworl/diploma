"""
Filter downloaded OFF/OBF/OPFF dump into category-specific parquet files.

OFF dumps: tab-separated, flat columns (nutriments as separate columns),
categories_tags as comma-separated string. Source is auto-detected by category
(pasta/chocolate/beverages → OFF, cosmetics → OBF, pet_food → OPFF).
"""

import argparse
import logging
import os

import pandas as pd

from src.common import PROCESSED_DIR, RAW_DIR, setup_logging

logger = logging.getLogger(__name__)

# Категория → источник (off/obf/opff). Источник определяет, какой dump читать.
CATEGORY_TO_SOURCE = {
    "pasta": "off",
    "chocolate": "off",
    "beverages": "off",
    "baby": "off",
    "cheeses": "off",
    "cereals": "off",
    "cosmetics": "obf",
}

SOURCE_FILES = {
    "off":  ("en.openfoodfacts.org.products.parquet",
             ["en.openfoodfacts.org.products.csv.gz", "en.openfoodfacts.org.products.csv"]),
    "obf":  ("en.openbeautyfacts.org.products.parquet",
             ["en.openbeautyfacts.org.products.csv.gz"]),
    "opff": ("en.openpetfoodfacts.org.products.parquet",
             ["en.openpetfoodfacts.org.products.csv.gz"]),
}

# Per-category cleanup config (include_substrings, exclude_tags).
# include_substrings: substring matched against categories_tags (legacy semantics).
# exclude_tags: full OFF tags ("en:..."); rows where categories_tags contains any
#               of these are dropped. Used to remove non-target products
#               (e.g. chips/potato/lentil from pasta).
CATEGORY_CLEANUP = {
    "pasta": {
        "output": "pasta_raw.parquet",
        # Strict pasta tags only — старая версия include_substrings=["pastas",
        # "cereals","grains","rice","noodles"] давала 75% non-pasta (хлеб,
        # рис, мука, овсяные напитки). Теперь только en:pastas family.
        "include_tags": [
            "en:pastas", "en:fresh-pastas", "en:dried-pastas",
            "en:noodles", "en:cereal-pastas", "en:filled-pastas",
            "en:wholemeal-pastas", "en:whole-grain-pastas",
            "en:gluten-free-pastas", "en:rice-pastas", "en:corn-pastas",
            "en:lasagna", "en:lasagna-sheets",
        ],
        "exclude_tags": [
            "en:potato-products", "en:chips", "en:crisps",
            "en:legumes-and-their-products", "en:lentils",
            "en:potatoes", "en:fries",
            # Прочие cross-domain
            "en:breads", "en:breakfast-cereals", "en:rices",
            "en:flours", "en:beverages",
        ],
    },
    "chocolate": {
        "output": "chocolate_raw.parquet",
        "include_substrings": [
            "chocolates", "chocolate-bars", "chocolate-confectioneries",
            "dark-chocolates", "milk-chocolates", "white-chocolates",
            "filled-chocolates", "chocolate-tablets",
        ],
        # Note: NOT excluding "en:cocoa-and-its-products" — it's a generic
        # parent tag in OFF taxonomy that ~99% of real chocolate products
        # carry. Excluding it drops nearly everything we want.
        "exclude_tags": [
            "en:cocoa-powders",
            "en:hot-chocolates",
            "en:chocolate-spreads",
            "en:chocolate-cakes",
            "en:chocolate-biscuits",
            "en:ice-creams-and-sorbets",
            "en:chocolate-cereals",
            "en:chocolate-cookies",
            "en:chocolate-mousses",
            "en:chocolate-puddings",
            "en:chocolate-syrups",
        ],
    },
    "beverages": {
        "output": "beverages_raw.parquet",
        # Use exact tag matching (include_tags) instead of substrings.
        # Substring matching catches `en:plant-based-foods-and-beverages`
        # which is a generic parent for foods AND beverages — unusable.
        "include_tags": [
            "en:beverages", "en:waters", "en:mineral-waters",
            "en:spring-waters", "en:fruit-juices", "en:juices",
            "en:soft-drinks", "en:sodas", "en:colas", "en:teas",
            "en:iced-teas", "en:coffees", "en:coffee-drinks",
            "en:carbonated-drinks", "en:dairy-drinks", "en:milk-drinks",
            "en:sport-drinks", "en:energy-drinks",
        ],
        "exclude_tags": [
            "en:plant-based-milk-alternatives",
            "en:beverages-and-beverages-preparations",
            "en:syrups",
            "en:powdered-beverages",
            "en:hot-chocolates",
            "en:smoothies",
        ],
    },
    "baby": {
        "output": "baby_raw.parquet",
        # ВАЖНО: сужено до молочных смесей. Без сужения domain включает
        # фруктовые пюре, мясные блюда и мюсли — на них milk_type/minimal_age
        # семантически N/A, и coverage atributов 14-39%. На milk-only — 80%+,
        # причинные связи (milk_type → is_lactose_free → minimal_age) валидны.
        # Альтернативное расширение в baby_meals (фруктовые/мясные блюда) —
        # отдельный sub-domain со своей схемой (flavour/is_organic/...).
        "include_tags": [
            "en:baby-milks", "en:infant-formulas", "en:growth-milks",
            "en:baby-formula", "en:baby-milks-in-powder",
            "en:baby-follow-on-milk-from-5-months",
            "en:dairy-dessert-for-baby",
        ],
        "exclude_tags": [
            "en:baby-care", "en:baby-products",  # non-food consumables
        ],
    },
    "cheeses": {
        "output": "cheeses_raw.parquet",
        # OFF cheeses: en:cheeses + sub-categories (cow/sheep/goat/blue/etc).
        "include_tags": [
            "en:cheeses", "en:cow-cheeses", "en:goat-cheeses",
            "en:sheep-s-milk-cheeses", "en:hard-cheeses", "en:soft-cheeses",
            "en:fresh-cheeses", "en:blue-cheeses", "en:cream-cheeses",
            "en:processed-cheeses", "en:french-cheeses", "en:italian-cheeses",
            "en:spanish-cheeses", "en:swiss-cheeses",
        ],
        "exclude_tags": [
            "en:cheese-spreads",  # отделим от main cheese-block если нужно
            "en:cheese-flavored",  # это snacks, не сыр
        ],
    },
    "cereals": {
        "output": "cereals_raw.parquet",
        # Только breakfast cereals (en:breakfast-cereals и явные подкатегории),
        # БЕЗ pasta/bread/chips, чтобы не пересекаться с pasta domain.
        "include_tags": [
            "en:breakfast-cereals", "en:mueslis", "en:granolas",
            "en:corn-flakes", "en:porridges", "en:rolled-oats",
            "en:chocolate-cereals", "en:multigrain-cereals", "en:rice-puffs",
            "en:puffed-rice", "en:wheat-flakes",
        ],
        "exclude_tags": [
            "en:pastas", "en:breads", "en:chips", "en:crisps",
            "en:potato-products",
        ],
    },
    # OBF (Open Beauty Facts). Substring filter — cover cosmetic-related parents.
    # OBF почти весь cosmetic, но 7% продуктов имеют en:non-food-products
    # (помечены как мусор сообществом).
    "cosmetics": {
        "output": "cosmetics_raw.parquet",
        "include_substrings": [
            "cosmetic", "hygiene", "hair", "face", "body", "makeup",
            "skin", "suncare", "sunscreen", "shampoo", "soap", "deodor",
            "shower", "toothpaste", "lipstick", "perfume", "fragrance",
            "lip-balm", "lip-care", "shower-gel", "lotion", "cream",
        ],
        "exclude_tags": [
            "en:non-cosmetic-products",
            # Не отрезаем en:non-food-products: в OBF ~7% продуктов помечены так,
            # но это не значит "не cosmetics" — community просто отделяет от food.
        ],
    },
}

KEEP_COLUMNS = [
    "code", "product_name", "generic_name", "brands",
    "categories_tags", "labels_tags",
    "ingredients_text", "ingredients_tags", "ingredients_analysis_tags",
    "allergens_tags", "traces_tags",
    "quantity", "serving_size",
    "completeness", "data_quality_tags",
    "countries_tags", "image_url",
    "fat_100g", "proteins_100g", "carbohydrates_100g",
    "energy-kcal_100g", "sugars_100g", "fiber_100g",
    "salt_100g", "sodium_100g", "alcohol_100g",
    "nutriscore_grade", "nova_group",
]


def find_source_path(source: str) -> tuple[str, str]:
    """Return (path, kind) for given source ('off'/'obf'/'opff'). Prefer parquet."""
    parquet_name, csv_names = SOURCE_FILES[source]
    parquet_path = os.path.join(RAW_DIR, parquet_name)
    if os.path.exists(parquet_path):
        return parquet_path, "parquet"
    for name in csv_names:
        path = os.path.join(RAW_DIR, name)
        if os.path.exists(path):
            return path, "csv"
    raise FileNotFoundError(
        f"No source file for {source} in {RAW_DIR}. "
        f"Expected: {parquet_name} or one of {csv_names}. "
        f"Run: python scripts/download_open_facts.py --source {source}"
    )


def load_source(path: str, kind: str, category: str | None = None) -> pd.DataFrame:
    if kind == "parquet":
        logger.info("Loading parquet %s ...", path)
        df = pd.read_parquet(path)
        normalized_keep = [c.replace("-", "_") for c in KEEP_COLUMNS]
        keep = [c for c in normalized_keep if c in df.columns]
        df = df[keep]
        logger.info("Loaded %d products (cols: %d/%d)", len(df), len(keep), len(KEEP_COLUMNS))
        return df

    # CSV — chunked load to avoid OOM on 12 GB raw file.
    available_cols = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    usecols = [c for c in KEEP_COLUMNS if c in available_cols]
    logger.info("Loading CSV %s in chunks (cols: %d/%d, category=%s)",
                path, len(usecols), len(KEEP_COLUMNS), category)

    # Pre-compute include substrings for early per-chunk filter (saves memory).
    # Категории с include_tags (например, beverages, pet_food) не фильтруются
    # в чанках — full include_tags применяется в apply_cleanup_filter.
    include_subs: list[str] = []
    if category and category in CATEGORY_CLEANUP:
        include_subs = [s.lower()
                        for s in CATEGORY_CLEANUP[category].get("include_substrings", [])]

    chunks: list[pd.DataFrame] = []
    total_rows = 0
    chunksize = 200_000
    # Force string dtype to avoid expensive type inference per chunk.
    reader = pd.read_csv(
        path, sep="\t", usecols=usecols,
        dtype=str, engine="c",
        low_memory=False, on_bad_lines="skip", chunksize=chunksize,
    )
    for ci, chunk in enumerate(reader):
        n_scanned = len(chunk)
        total_rows += n_scanned
        if include_subs and "categories_tags" in chunk.columns:
            cats = chunk["categories_tags"].fillna("").str.lower()
            mask = pd.Series(False, index=chunk.index)
            for sub in include_subs:
                mask |= cats.str.contains(sub, regex=False)
            chunk = chunk[mask].copy()
        n_kept = len(chunk)
        chunks.append(chunk)
        logger.info("  chunk %d: scanned=%d (cumulative=%d), kept=%d",
                    ci, n_scanned, total_rows, n_kept)

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)
    logger.info("Loaded %d products (post pre-include) from %d total rows", len(df), total_rows)
    return df


def apply_cleanup_filter(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Apply per-category include filter + exclude_tags cleanup.

    Supports two modes (one or the other, not both):
    - include_substrings: substring match against lowercased categories_tags
      (looser, may catch parent tags like en:plant-based-foods-and-beverages).
    - include_tags: exact OFF tag match in the comma-separated categories_tags list
      (stricter, recommended for narrow categories like beverages).

    exclude_tags: exact OFF tag match (rows containing any of these are dropped).
    """
    cfg = CATEGORY_CLEANUP[category]
    cats_str = df["categories_tags"].fillna("").str.lower()

    n_before_include = len(df)
    if cfg.get("include_tags"):
        # Exact tag match
        include_set = {t.lower() for t in cfg["include_tags"]}

        def has_any_include(cats: str) -> bool:
            if not cats:
                return False
            for t in cats.split(","):
                if t.strip() in include_set:
                    return True
            return False

        include_mask = cats_str.apply(has_any_include)
    else:
        # Substring fallback
        include_mask = pd.Series(False, index=df.index)
        for sub in cfg["include_substrings"]:
            include_mask = include_mask | cats_str.str.contains(sub, regex=False)

    df = df[include_mask].copy()
    logger.info("Include filter: %d -> %d rows", n_before_include, len(df))

    if not cfg.get("exclude_tags"):
        return df

    exclude_set = {t.lower() for t in cfg["exclude_tags"]}

    def has_any_exclude(cats):
        if not cats:
            return False
        for t in cats.split(","):
            if t.strip() in exclude_set:
                return True
        return False

    cats_lc = df["categories_tags"].fillna("").str.lower()
    excl_mask = cats_lc.apply(has_any_exclude)
    n_excl = int(excl_mask.sum())
    df = df[~excl_mask].copy()
    logger.info("Exclude filter: dropped %d rows (-%.1f%%), %d remaining",
                n_excl, (n_excl / max(n_before_include, 1)) * 100, len(df))
    return df


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Filter OFF dump into category-specific parquet files")
    parser.add_argument(
        "--category",
        choices=list(CATEGORY_CLEANUP.keys()),
        required=True,
        help="Category to filter (uses include_substrings or include_tags + exclude_tags)",
    )
    args = parser.parse_args()

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    source = CATEGORY_TO_SOURCE[args.category]
    src_path, src_kind = find_source_path(source)
    logger.info("Source: %s (%s)", source, src_path)
    df = load_source(src_path, src_kind, category=args.category)

    df.columns = [c.replace("-", "_") for c in df.columns]

    if "code" in df.columns:
        df["code"] = df["code"].astype(str)

    cfg = CATEGORY_CLEANUP[args.category]
    cleaned = apply_cleanup_filter(df, args.category)
    out_path = os.path.join(PROCESSED_DIR, cfg["output"])
    cleaned.to_parquet(out_path, index=False)
    logger.info("  %s: %d products -> %s", args.category, len(cleaned), out_path)


if __name__ == "__main__":
    main()
