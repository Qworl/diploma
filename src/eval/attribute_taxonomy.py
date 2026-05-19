"""Attribute signal taxonomy: classify each (category, attr) by primary
silver-derivation source.

See spec §3.4 for definitions:
- tag_derived:  silver value comes from labels_tags / categories_tags
- text_derived: silver value comes from regex over ingredients_text / product_name
- nutri_derived: silver value comes from numeric nutriments thresholds

Multi-source attrs (e.g. is_organic, derivable from BOTH labels_tags and
ingredients_text regex) are tagged with primary + secondary path.

Used in §6.12.0 of the notebook to disclose what each accuracy number
actually measures.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

from src.pipeline.schemas.beverages import BEVERAGE_SCHEMA
from src.pipeline.schemas.cereals import CEREALS_SCHEMA
from src.pipeline.schemas.cheeses import CHEESES_SCHEMA
from src.pipeline.schemas.chocolate import CHOCOLATE_SCHEMA
from src.pipeline.schemas.cosmetics import COSMETICS_SCHEMA
from src.pipeline.schemas.pasta import PASTA_SCHEMA


@dataclass(frozen=True)
class _SignalType:
    TAG: str = "tag_derived"
    TEXT: str = "text_derived"
    NUTRI: str = "nutri_derived"


SIGNAL_TYPE: Final = _SignalType()


# Explicit mapping for attributes that appear in label_silver.py.
# Keys: (category, attr). Values: (signal_type, primary_path, secondary_path or None).
# primary_path is the OFF field that label_silver uses by default.
# secondary_path is set when Layer 1 regex extracts the same attr from a different field.
_TAXONOMY: dict[tuple[str, str], tuple[str, str, str | None]] = {
    # ---- pasta ----
    ("pasta", "pasta_shape"):       ("text_derived",  "product_name",       None),
    ("pasta", "grain_type"):        ("text_derived",  "ingredients_text",   "product_name"),
    ("pasta", "is_filled"):         ("text_derived",  "product_name",       None),
    ("pasta", "is_organic"):        ("tag_derived",   "labels_tags",        "ingredients_text"),
    ("pasta", "is_gluten_free"):    ("tag_derived",   "labels_tags",        None),
    ("pasta", "is_vegan"):          ("tag_derived",   "labels_tags",        "ingredients_text"),
    ("pasta", "nutri_score_grade"): ("nutri_derived", "nutriments",         None),
    ("pasta", "protein_class"):     ("nutri_derived", "nutriments",         None),

    # ---- chocolate ----
    ("chocolate", "chocolate_type"):    ("text_derived",  "product_name",     "ingredients_text"),
    ("chocolate", "cocoa_percentage"):  ("text_derived",  "ingredients_text", "product_name"),
    ("chocolate", "contains_nuts"):     ("text_derived",  "ingredients_text", "allergens_tags"),
    ("chocolate", "chocolate_extra"):   ("text_derived",  "ingredients_text", None),
    ("chocolate", "is_organic"):        ("tag_derived",   "labels_tags",      "ingredients_text"),
    ("chocolate", "nutri_score_grade"): ("nutri_derived", "nutriments",       None),
    ("chocolate", "protein_class"):     ("nutri_derived", "nutriments",       None),

    # ---- cheeses ----
    ("cheeses", "milk_source"):        ("text_derived",  "product_name",     "ingredients_text"),
    ("cheeses", "texture"):            ("text_derived",  "product_name",     "categories_tags"),
    ("cheeses", "country_of_origin"):  ("tag_derived",   "categories_tags",  "manufacturing_places"),
    ("cheeses", "fat_class"):          ("nutri_derived", "nutriments",       None),
    ("cheeses", "is_pdo"):             ("tag_derived",   "labels_tags",      None),
    ("cheeses", "is_organic"):         ("tag_derived",   "labels_tags",      "ingredients_text"),
    ("cheeses", "is_ultra_processed"): ("tag_derived",   "categories_tags",  "ingredients_text"),

    # ---- beverages ----
    ("beverages", "beverage_type"):    ("text_derived",  "product_name",     "categories_tags"),
    ("beverages", "is_carbonated"):    ("text_derived",  "ingredients_text", "categories_tags"),
    ("beverages", "is_organic"):       ("tag_derived",   "labels_tags",      "ingredients_text"),
    ("beverages", "is_vegan"):         ("tag_derived",   "labels_tags",      "ingredients_text"),
    ("beverages", "nova_group"):       ("nutri_derived", "nutriments",       None),
    ("beverages", "nutri_score_grade"):("nutri_derived", "nutriments",       None),
    ("beverages", "protein_class"):    ("nutri_derived", "nutriments",       None),
    ("beverages", "sugar_class"):      ("nutri_derived", "nutriments",       None),

    # ---- cereals ----
    ("cereals", "cereal_type"):        ("text_derived",  "product_name",     "ingredients_text"),
    ("cereals", "grain_type"):         ("text_derived",  "ingredients_text", "product_name"),
    ("cereals", "is_organic"):         ("tag_derived",   "labels_tags",      "ingredients_text"),
    ("cereals", "is_low_sugar"):       ("nutri_derived", "nutriments",       None),
    ("cereals", "is_high_fibre"):      ("nutri_derived", "nutriments",       None),
    ("cereals", "nova_class"):         ("nutri_derived", "nutriments",       None),
    ("cereals", "is_vegan"):           ("tag_derived",   "labels_tags",      "ingredients_text"),
    ("cereals", "is_whole_grain"):     ("text_derived",  "ingredients_text", "product_name"),

    # ---- cosmetics ----
    ("cosmetics", "body_area"):        ("text_derived",  "product_name",     "categories_tags"),
    ("cosmetics", "product_type"):     ("text_derived",  "product_name",     "categories_tags"),
    ("cosmetics", "form_factor"):      ("text_derived",  "product_name",     "categories_tags"),
    ("cosmetics", "has_sulfates"):     ("text_derived",  "ingredients_text", None),
    ("cosmetics", "has_silicones"):    ("text_derived",  "ingredients_text", None),
    ("cosmetics", "is_organic"):       ("tag_derived",   "labels_tags",      "ingredients_text"),
}


_DOMAIN_SCHEMAS = {
    "pasta": PASTA_SCHEMA,
    "chocolate": CHOCOLATE_SCHEMA,
    "cheeses": CHEESES_SCHEMA,
    "beverages": BEVERAGE_SCHEMA,
    "cereals": CEREALS_SCHEMA,
    "cosmetics": COSMETICS_SCHEMA,
}


def classify_attribute(category: str, attr: str) -> str:
    """Return signal_type for (category, attr). Defaults to text_derived."""
    if (category, attr) in _TAXONOMY:
        return _TAXONOMY[(category, attr)][0]
    return SIGNAL_TYPE.TEXT


def build_taxonomy_dataframe() -> pd.DataFrame:
    """Build a long-format taxonomy DataFrame covering all attrs in all
    six food schemas.
    """
    rows = []
    for category, schema in _DOMAIN_SCHEMAS.items():
        for attr in schema.keys():
            entry = _TAXONOMY.get((category, attr))
            if entry is not None:
                signal, primary, secondary = entry
            else:
                signal, primary, secondary = SIGNAL_TYPE.TEXT, "product_name", None
            rows.append({
                "category": category,
                "attr": attr,
                "signal_type": signal,
                "primary_path": primary,
                "secondary_path": secondary,
                "multi_source": secondary is not None,
            })
    return pd.DataFrame(rows)


def main(out_path: str = "datasets/processed/attribute_signal_taxonomy.parquet") -> None:
    """CLI: build and save the taxonomy parquet."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = build_taxonomy_dataframe()
    from pathlib import Path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logging.info("Wrote %s rows to %s", len(df), out_path)
    logging.info("Signal type counts:\n%s", df["signal_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
