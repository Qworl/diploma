"""Beverages domain schema and few-shot examples for LLM prompts."""

BEVERAGE_SCHEMA = {
    "beverage_type": {
        "type": "enum",
        "values": ["water", "juice", "soda", "tea", "coffee",
                   "dairy", "sport", "other"],
        "description": "Type of beverage",
    },
    "sugar_class": {
        "type": "enum",
        "values": ["0", "low", "med", "high"],
        "nullable": True,
        "description": "Sugar content class (0/low/med/high) bucketed from sugars_100g",
    },
    "is_organic": {
        "type": "bool",
        "description": "Whether the beverage is organic / bio",
    },
    # is_no_added_sugar удалён (degenerate, 94% False), заменён на is_carbonated
    "is_carbonated": {
        "type": "bool",
        "description": "Whether the beverage is carbonated (soda/sparkling water)",
    },
    "nutri_score_grade": {
        "type": "enum",
        "values": ["A", "B", "C", "D", "E"],
        "nullable": True,
        "description": "Nutri-Score grade A-E",
    },
    "nova_group": {
        "type": "int",
        "values": [1, 2, 3, 4],
        "nullable": True,
        "description": "NOVA ultra-processed food classification (1-4)",
    },
    "protein_class": {
        "type": "enum",
        "values": ["0", "low", "med", "high"],
        "nullable": True,
        "description": "Protein content class (0/low/med/high) bucketed from proteins_100g",
    },
    "is_vegan": {
        "type": "bool",
        "description": "Веганский напиток (без молока/мёда; en:vegan или ingredients_analysis=en:vegan)",
    },
}

BEVERAGE_EXAMPLES = [
    (
        {
            "product_name": "Evian Natural Mineral Water",
            "brands": "Evian",
            "quantity": "1.5L",
        },
        {
            "beverage_type": "water",
            "sugar_class": "0",
            "is_organic": False,
            "is_no_added_sugar": True,
            "nutri_score_grade": "A",
            "nova_group": 1,
            "protein_class": "0",
        },
    ),
    (
        {
            "product_name": "Coca-Cola Classic",
            "brands": "Coca-Cola",
            "ingredients_text": "Carbonated water, sugar, caramel E150d, phosphoric acid, caffeine.",
            "quantity": "330ml",
        },
        {
            "beverage_type": "soda",
            "sugar_class": "high",
            "is_organic": False,
            "is_no_added_sugar": False,
            "nutri_score_grade": "E",
            "nova_group": 4,
            "protein_class": "0",
        },
    ),
    (
        {
            "product_name": "Jus d'orange 100% pur jus bio",
            "brands": "Vitamont",
            "ingredients_text": "Jus d'orange* (100%). *Issu de l'agriculture biologique.",
            "quantity": "1L",
        },
        {
            "beverage_type": "juice",
            "sugar_class": "med",
            "is_organic": True,
            "is_no_added_sugar": True,
            "nutri_score_grade": "C",
            "nova_group": 1,
            "protein_class": "low",
        },
    ),
]
