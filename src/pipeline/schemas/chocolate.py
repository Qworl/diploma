"""Chocolate domain schema and few-shot examples for LLM prompts."""

CHOCOLATE_SCHEMA = {
    "chocolate_type": {
        "type": "enum",
        "values": ["dark", "milk", "white", "filled", "other"],
        "description": "Type of chocolate",
    },
    "cocoa_percentage": {
        "type": "enum",
        "values": ["<30", "30-50", "50-70", "70-85", "85+"],
        "nullable": True,
        "description": "Cocoa percentage bucket",
    },
    "contains_nuts": {
        "type": "bool",
        "description": "Whether the product contains nuts",
    },
    # palm_oil_status удалён (degenerate, 95% palm-oil-free), заменён на chocolate_extra
    "chocolate_extra": {
        "type": "enum",
        "values": ["plain", "with_nuts", "with_fruit", "with_caramel",
                   "with_cookie", "filled", "with_alcohol", "with_coffee", "other"],
        "nullable": True,
        "description": "Add-ins / inclusions in the chocolate (plain/nuts/fruit/etc.)",
    },
    "is_organic": {
        "type": "bool",
        "description": "Whether the product is organic",
    },
    "nutri_score_grade": {
        "type": "enum",
        "values": ["A", "B", "C", "D", "E"],
        "nullable": True,
        "description": "Nutri-Score grade",
    },
    "protein_class": {
        "type": "enum",
        "values": ["0", "low", "med", "high"],
        "nullable": True,
        "description": "Protein content class (0/low/med/high) bucketed from proteins_100g",
    },
}

CHOCOLATE_EXAMPLES = [
    (
        {
            "product_name": "Lindt Excellence 70% Cocoa Dark",
            "brands": "Lindt",
            "ingredients_text": "Cocoa mass, sugar, cocoa butter, vanilla.",
            "quantity": "100g",
        },
        {
            "chocolate_type": "dark",
            "cocoa_percentage": "50-70",
            "contains_nuts": False,
            "palm_oil_status": "palm-oil-free",
            "is_organic": False,
            "nutri_score_grade": "E",
            "protein_class": "med",
        },
    ),
    (
        {
            "product_name": "Milka Noisettes",
            "brands": "Milka",
            "ingredients_text": "Sugar, cocoa butter, skim milk powder, hazelnuts (13%), cocoa mass.",
            "quantity": "100 g",
        },
        {
            "chocolate_type": "milk",
            "cocoa_percentage": "<30",
            "contains_nuts": True,
            "palm_oil_status": "palm-oil-free",
            "is_organic": False,
            "nutri_score_grade": "E",
            "protein_class": "med",
        },
    ),
    (
        {
            "product_name": "Chocolat blanc bio fourré framboise",
            "brands": "Alter Eco",
            "ingredients_text": "Sucre de canne*, beurre de cacao*, framboise*. *bio.",
            "quantity": "80g",
        },
        {
            "chocolate_type": "filled",
            "cocoa_percentage": "<30",
            "contains_nuts": False,
            "palm_oil_status": "palm-oil-free",
            "is_organic": True,
            "nutri_score_grade": "E",
            "protein_class": "low",
        },
    ),
]
