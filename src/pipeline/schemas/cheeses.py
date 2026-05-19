"""Cheeses domain schema and few-shot examples for LLM prompts."""

CHEESES_SCHEMA = {
    "milk_source": {
        "type": "enum",
        "values": ["cow", "goat", "sheep", "buffalo", "mixed", "other"],
        "nullable": True,
        "description": "Source of milk used for the cheese",
    },
    "texture": {
        "type": "enum",
        "values": ["hard", "soft", "fresh", "cream", "blue", "processed", "other"],
        "nullable": True,
        "description": "Cheese texture/type (hard / soft / fresh / cream / blue / processed)",
    },
    "country_of_origin": {
        "type": "enum",
        "values": ["france", "italy", "spain", "germany", "uk", "us",
                   "switzerland", "netherlands", "other"],
        "nullable": True,
        "description": "Geographic origin of the cheese variety",
    },
    "fat_class": {
        "type": "enum",
        "values": ["low", "medium", "high", "very_high"],
        "nullable": True,
        "description": "Fat content class bucketed from fat_100g (low<15, med 15-25, high 25-32, very_high>32)",
    },
    "is_pdo": {
        "type": "bool",
        "description": "Protected Designation of Origin (PDO/AOP/DOP) — geographic protection",
    },
    "is_organic": {
        "type": "bool",
        "description": "Whether the cheese is organic / bio",
    },
    "is_ultra_processed": {
        "type": "bool",
        "description": "Ультра-обработанный сыр (NOVA group 4 — плавленый, спреды, slices)",
    },
}

CHEESES_EXAMPLES = [
    (
        {
            "product_name": "Comté 15 mois",
            "brands": "Jura flore",
            "categories_tags": "en:dairies,en:fermented-foods,en:fermented-milk-products,en:cheeses,en:cow-cheeses,en:fresh-foods,en:hard-cheeses,en:french-cheeses,en:unpasteurised-cheeses,en:comte,en:aoc-cheeses",
            "ingredients_text": "_Lait_ cru de vache, ferments lactiques (_lait_), sel, présure,",
            "quantity": "200 g",
        },
        {
            "milk_source": "cow",
            "texture": "hard",
            "country_of_origin": "france",
            "fat_class": "very_high",
            "is_pdo": True,
            "is_organic": False,
            "is_ultra_processed": False,
        },
    ),
    (
        {
            "product_name": "Queso de Murcia fresco",
            "brands": "La Purisima",
            "categories_tags": "en:dairies,en:fermented-foods,en:fermented-milk-products,en:cheeses,en:goat-cheeses,en:fresh-cheeses,en:spanish-cheeses,es:queso-de-murcia",
            "ingredients_text": "Leche pasteurizada de cabra murciana-granadina, sal, estabilizante: cloruro cálcico, cuajo, fermentos lácticos y conservador E-202.",
            "quantity": "",
        },
        {
            "milk_source": "goat",
            "texture": "fresh",
            "country_of_origin": "spain",
            "fat_class": "high",
            "is_pdo": True,
            "is_organic": False,
            "is_ultra_processed": False,
        },
    ),
    (
        {
            "product_name": "Brique de brebis",
            "brands": "Marque Repère, Les Croisés",
            "categories_tags": "en:dairies,en:fermented-foods,en:fermented-milk-products,en:cheeses,en:soft-cheeses,en:fresh-foods,en:soft-cheeses-with-bloomy-rind,en:french-cheeses,en:pasteurized-cheeses,en:sheep-s-milk-cheeses",
            "ingredients_text": "Lait de brebis pasteurisé, ferments (dont lait), sel, présure, Lait origine France [facultatif - étiqueté]",
            "quantity": "150 g",
        },
        {
            "milk_source": "sheep",
            "texture": "soft",
            "country_of_origin": "france",
            "fat_class": "high",
            "is_pdo": False,
            "is_organic": False,
            "is_ultra_processed": False,
        },
    ),
]
