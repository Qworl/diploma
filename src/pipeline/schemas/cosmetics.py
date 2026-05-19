"""Cosmetics domain schema and few-shot examples for LLM prompts."""

COSMETICS_SCHEMA = {
    "product_type": {
        "type": "enum",
        "values": ["shampoo", "deodorant", "soap", "shower_gel", "toothpaste",
                   "makeup", "sunscreen", "face_cream", "body_cream", "hand_cream",
                   "hair_care", "lip_balm", "other"],
        "description": "Product category (cosmetics)",
    },
    "form_factor": {
        "type": "enum",
        "values": ["liquid", "cream_gel", "solid_bar", "powder", "spray",
                   "stick", "wipe", "other"],
        "nullable": True,
        "description": "Physical form of the product",
    },
    "body_area": {
        "type": "enum",
        "values": ["body", "face", "hair", "oral", "deo", "sun",
                   "makeup", "fragrance", "intimate", "other"],
        "nullable": True,
        "description": "Часть тела / назначение (выводимо из categories_tags: shower-gel→body, shampoo→hair, toothpaste→oral)",
    },
    # fragrance_status removed 2026-05-11: 97.5% fragranced (мажор), Bayes Δ=0,
    # ML тривиально predict мажор-класс (acc 0.982 = pos rate). Тот же критерий
    # как is_no_added_sugar/palm_oil_status (95%+ мажор → degenerate).
    # is_vegan removed 2026-05-11: 91% NaN в OBF ingredients_analysis_tags.
    "has_sulfates": {
        "type": "bool",
        "description": "SLS/SLES в составе (sodium lauryl/laureth sulfate) — характерно для shampoo/shower_gel",
    },
    "has_silicones": {
        "type": "bool",
        "description": "Silicones в составе (dimethicone/cyclomethicone/siloxan) — характерно для hair_care/makeup",
    },
    "is_organic": {
        "type": "bool",
        "description": "Whether the product carries an organic certification (cosmebio/ecocert/cosmos)",
    },
}

COSMETICS_EXAMPLES = [
    (
        {
            "product_name": "Shampooing nourrissant à la mangue et au beurre de noix",
            "brands": "Les Cosmétiques Design Paris, Nectar of Beauty",
            "categories_tags": "en:hair,en:shampoos,en:shampoo-for-dry-hair,en:open-beauty-facts",
            "ingredients_text": "AQUA, SODIUM LAURETH SULFATE, COCAMIDOPROPYL BETAINE, SODIUM CHLORIDE, PARFUM, GLYCERIN, SORBITAN CAPRYLATE, PPG-1-PEG-9 LAURYL GLYCOL ETHER, STYRENE/ACRYLATES COPOLYMER, CITRIC ACID, PROPANEDIOL, SODIUM BICARBONATE, BENZOIC ACID, PEG-40 HYDROGENATED CASTOR OIL, POLYQUATERNIUM-10, MANGIFERA INDICA FRUIT EXTRACT, BUTYROSPERMUM PARKII BUTTER",
            "quantity": "250 ml",
        },
        {
            "product_type": "shampoo",
            "form_factor": "cream_gel",
            "body_area": "hair",
            "has_sulfates": True,
            "has_silicones": False,
            "is_organic": False,
        },
    ),
    (
        {
            "product_name": "Dentifrice à la menthe à l'extrait de menthe bio",
            "brands": "Carrefour",
            "categories_tags": "en:hygiene,en:toothpastes",
            "ingredients_text": "AQUA, GLYCERIN, MENTHA PIPERITA LEAF WATER, SILICA, XANTHAN GUM, TITANIUM DIOXIDE, POTASSIUM SORBATE, AROMA, CALCIUM CARBONATE, SODIUM HYDROXIDE, MENTHA VIRIDIS LEAF OIL",
            "quantity": "75 ml",
        },
        {
            "product_type": "toothpaste",
            "form_factor": "cream_gel",
            "body_area": "oral",
            "has_sulfates": False,
            "has_silicones": False,
            "is_organic": True,
        },
    ),
    (
        {
            "product_name": "Jardins du monde Fleur de coton d'Inde",
            "brands": "Yves Rocher",
            "categories_tags": "en:hygiene,en:deodorants,en:anti-perspirants,en:roll-on-deodorants",
            "ingredients_text": "AQUA, ALUMINUM CHLOROHYDRATE, PPG-15 STEARYL ETHER, HAMAMELIS VIRGINIANA WATER, STEARETH-2, GOSSYPIUM HERBACEUM EXTRACT, STEARETH-20, SODIUM BENZOATE, PARFUM, PROPYLENE GLYCOL, LIMONENE",
            "quantity": "50 mL",
        },
        {
            "product_type": "deodorant",
            "form_factor": "stick",
            "body_area": "deo",
            "has_sulfates": False,
            "has_silicones": False,
            "is_organic": False,
        },
    ),
]
