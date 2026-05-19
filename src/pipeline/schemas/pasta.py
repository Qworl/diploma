"""Pasta domain schema and few-shot examples for LLM prompts."""

PASTA_SCHEMA = {
    "grain_type": {
        "type": "enum",
        "values": ["wheat", "spelt", "rice", "corn", "buckwheat", "oat", "mixed", "other"],
        "description": "Primary grain type",
    },
    # is_whole_grain удалён (degenerate, 99.5% False), заменён на is_filled
    "is_filled": {
        "type": "bool",
        "description": "Whether the pasta is filled (ravioli/tortellini/cappelletti)",
    },
    "is_organic": {
        "type": "bool",
        "description": "Whether the product is organic / bio",
    },
    "is_gluten_free": {
        "type": "bool",
        "description": "Whether the product is gluten-free",
    },
    "pasta_shape": {
        "type": "enum",
        "values": ["spaghetti", "penne", "fusilli", "macaroni", "farfalle",
                    "tagliatelle", "lasagna", "noodles", "rigatoni", "vermicelli",
                    "linguine", "other"],
        "nullable": True,
        "description": "Shape of pasta (only for pasta products)",
    },
    "is_vegan": {
        "type": "bool",
        "description": "Whether the product is vegan",
    },
    "nutri_score_grade": {
        "type": "enum",
        "values": ["A", "B", "C", "D", "E"],
        "nullable": True,
        "description": "Nutri-Score grade A-E",
    },
    "protein_class": {
        "type": "enum",
        "values": ["0", "low", "med", "high"],
        "nullable": True,
        "description": "Protein content class (0/low/med/high) bucketed from proteins_100g",
    },
}

PASTA_EXAMPLES = [
    (
        {
            "product_name": "Barilla Spaghetti n.5",
            "brands": "Barilla",
            "ingredients_text": "Durum wheat semolina, water",
            "quantity": "500g",
        },
        {
            "grain_type": "wheat",
            "pasta_shape": "spaghetti",
            "is_whole_grain": False,
            "is_organic": False,
            "is_gluten_free": False,
            "is_vegan": True,
            "nutri_score_grade": "A",
            "protein_class": "med",
        },
    ),
    (
        {
            "product_name": "Pâtes de riz complètes sans gluten BIO",
            "brands": "Markal",
            "ingredients_text": "Farine de riz complet*. *Issu de l'agriculture biologique.",
            "quantity": "250 g",
        },
        {
            "grain_type": "rice",
            "pasta_shape": None,
            "is_whole_grain": True,
            "is_organic": True,
            "is_gluten_free": True,
            "is_vegan": True,
            "nutri_score_grade": "A",
            "protein_class": "low",
        },
    ),
    (
        {
            "product_name": "Buchweizen Fusilli",
            "brands": "Alb-Gold",
            "ingredients_text": "Buchweizenmehl, Wasser. Glutenfrei.",
            "quantity": "400 g",
        },
        {
            "grain_type": "buckwheat",
            "pasta_shape": "fusilli",
            "is_whole_grain": False,
            "is_organic": False,
            "is_gluten_free": True,
            "is_vegan": True,
            "nutri_score_grade": "A",
            "protein_class": "med",
        },
    ),
]
