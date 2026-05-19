"""Cereals domain schema and few-shot examples for LLM prompts."""

CEREALS_SCHEMA = {
    "cereal_type": {
        "type": "enum",
        "values": ["muesli", "granola", "corn_flakes", "oat_cereal",
                   "chocolate_cereal", "puffed_rice", "mixed", "other"],
        "nullable": True,
        "description": "Type of breakfast cereal",
    },
    "grain_type": {
        "type": "enum",
        "values": ["oat", "wheat", "corn", "rice", "multigrain", "other"],
        "nullable": True,
        "description": "Primary grain in the cereal",
    },
    "is_low_sugar": {
        "type": "bool",
        "description": "True if tagged no-added-sugar/sugar-free OR sugars_100g ≤ 5g",
    },
    "is_high_fibre": {
        "type": "bool",
        "description": "True if tagged high-fibre/source-of-fibre OR fiber_100g ≥ 6g",
    },
    "nova_class": {
        "type": "enum",
        "values": ["natural", "processed", "ultra_processed"],
        "nullable": True,
        "description": "Степень обработки (NOVA): 1-2=natural, 3=processed, 4=ultra_processed",
    },
    "is_vegan": {
        "type": "bool",
        "description": "Веганский продукт (без молока/яиц/мёда; en:vegan или ingredients_analysis=en:vegan)",
    },
    "is_whole_grain": {
        "type": "bool",
        "description": "Whether the cereal is whole-grain",
    },
    "is_organic": {
        "type": "bool",
        "description": "Whether the cereal is organic / bio",
    },
}

CEREALS_EXAMPLES = [
    (
        {
            "product_name": "Muesli Bio Croustillant Amande Vanille",
            "brands": "Charles Vignon",
            "categories_tags": "en:plant-based-foods-and-beverages,en:plant-based-foods,en:breakfasts,en:cereals-and-potatoes,en:cereals-and-their-products,en:breakfast-cereals,en:mueslis",
            "ingredients_text": "Céréales* complètes 65,2% (flocons d'avoine* 49,1%, riz*), sucre de canne complet non raffiné, huile de tournesol*, amandes* entières 7%, miel*, extrait de vanille*. *issu de l'agriculture biologique",
            "quantity": "375 g",
        },
        {
            "cereal_type": "muesli",
            "grain_type": "multigrain",
            "is_low_sugar": False,
            "is_high_fibre": False,
            "nova_class": "processed",
            "is_vegan": False,
            "is_whole_grain": False,
            "is_organic": True,
        },
    ),
    (
        {
            "product_name": "Corn flakes sans sucres ajoutés",
            "brands": "Terres et Céréales bio",
            "categories_tags": "en:plant-based-foods-and-beverages,en:plant-based-foods,en:breakfasts,en:cereals-and-potatoes,en:cereals-and-their-products,en:breakfast-cereals,en:flaked-cereals,en:corn-flakes",
            "ingredients_text": "Maïs* - * Produit issu de l'Agriculture Biologique",
            "quantity": "450 g",
        },
        {
            "cereal_type": "corn_flakes",
            "grain_type": "corn",
            "is_low_sugar": True,
            "is_high_fibre": False,
            "nova_class": "natural",
            "is_vegan": True,
            "is_whole_grain": False,
            "is_organic": True,
        },
    ),
    (
        {
            "product_name": "Extra Chocolat au lait",
            "brands": "Kellog's",
            "categories_tags": "en:plant-based-foods-and-beverages,en:plant-based-foods,en:breakfasts,en:cereals-and-potatoes,en:cereals-and-their-products,en:breakfast-cereals,en:chocolate-cereals",
            "ingredients_text": "_Avoine_ complète (48%), sucre, morceaux de chocolat au _lait_ (14%) (sucre, beurre de cacao, _lait_ entier en poudre, pâte de cacao, graisses végétales)",
            "quantity": "500 g",
        },
        {
            "cereal_type": "chocolate_cereal",
            "grain_type": "multigrain",
            "is_low_sugar": False,
            "is_high_fibre": False,
            "nova_class": "ultra_processed",
            "is_vegan": False,
            "is_whole_grain": False,
            "is_organic": False,
        },
    ),
]
