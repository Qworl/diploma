"""Schemas for the 7 supported product domains.

Re-exports SCHEMA + EXAMPLES per domain, plus INPUT_FIELDS and EXAMPLES_BY_SCHEMA.
"""

from src.pipeline.schemas.pasta import PASTA_SCHEMA, PASTA_EXAMPLES
from src.pipeline.schemas.chocolate import CHOCOLATE_SCHEMA, CHOCOLATE_EXAMPLES
from src.pipeline.schemas.beverages import BEVERAGE_SCHEMA, BEVERAGE_EXAMPLES
from src.pipeline.schemas.electronics import ELECTRONICS_SCHEMA, ELECTRONICS_EXAMPLES
from src.pipeline.schemas.cosmetics import COSMETICS_SCHEMA, COSMETICS_EXAMPLES
from src.pipeline.schemas.cheeses import CHEESES_SCHEMA, CHEESES_EXAMPLES
from src.pipeline.schemas.cereals import CEREALS_SCHEMA, CEREALS_EXAMPLES

INPUT_FIELDS = ["product_name", "brands", "categories_tags", "ingredients_text", "quantity"]

EXAMPLES_BY_SCHEMA = {
    id(PASTA_SCHEMA): PASTA_EXAMPLES,
    id(CHOCOLATE_SCHEMA): CHOCOLATE_EXAMPLES,
    id(BEVERAGE_SCHEMA): BEVERAGE_EXAMPLES,
    id(ELECTRONICS_SCHEMA): ELECTRONICS_EXAMPLES,
    id(COSMETICS_SCHEMA): COSMETICS_EXAMPLES,
    id(CHEESES_SCHEMA): CHEESES_EXAMPLES,
    id(CEREALS_SCHEMA): CEREALS_EXAMPLES,
}

__all__ = [
    "PASTA_SCHEMA", "PASTA_EXAMPLES",
    "CHOCOLATE_SCHEMA", "CHOCOLATE_EXAMPLES",
    "BEVERAGE_SCHEMA", "BEVERAGE_EXAMPLES",
    "ELECTRONICS_SCHEMA", "ELECTRONICS_EXAMPLES",
    "COSMETICS_SCHEMA", "COSMETICS_EXAMPLES",
    "CHEESES_SCHEMA", "CHEESES_EXAMPLES",
    "CEREALS_SCHEMA", "CEREALS_EXAMPLES",
    "INPUT_FIELDS",
    "EXAMPLES_BY_SCHEMA",
]
