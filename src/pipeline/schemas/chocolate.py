"""Chocolate domain schema and few-shot examples for LLM prompts."""

CHOCOLATE_SCHEMA = {
    "chocolate_type": {
        "type": "enum",
        "values": ["dark", "milk", "white", "filled", "other"],
        "description": (
            "Type of chocolate by base cocoa formulation. "
            "'dark' = chocolat noir, semi-sweet, bittersweet, ≥50% cocoa typical "
            "(includes 'extra dark', semi-sweet chips, baking dark); "
            "'milk' = chocolat au lait, milk chocolate, hot chocolate, ≤45% cocoa "
            "(includes Mexican hot chocolate tablets, milk chocolate truffle bars); "
            "'white' = chocolat blanc, white chocolate (no cocoa solids); "
            "'filled' = ONLY when product is structurally a SHELL+FILLING (truffles, pralines, "
            "bonbons, liqueur-filled chocolates, brandy beans, lava cake) and the base type "
            "cannot be determined. A milk chocolate bar with caramel/cookie/nut inclusions is "
            "'milk', NOT 'filled' — use chocolate_extra for the inclusion. "
            "Use 'other' for non-chocolate items mis-categorized (cocoa powder, drink mixes, "
            "chocolate-flavored cookies)."
        ),
    },
    # cocoa_percentage исключён из LLM scope — он deterministically computable из
    # product_name через regex + bucketize (src/pipeline/off_labels/rules.py:TYPE_C_RULES).
    # Silver regex точнее LLM на 7pp; numeric bucketize — задача для rules, не для LLM.
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
        "description": (
            "Add-ins / inclusions. **PRIORITY:** if multiple add-ins, pick the dominant by name. "
            "'plain' = чистый chocolate без inclusions (chocolat noir/au lait/blanc plain bar, "
            "plain chips, plain truffles без inclusion mention); "
            "'with_nuts' = hazelnuts/almonds/peanuts/pistachios/walnuts/cashews visible chunks "
            "OR marzipan/almond paste/almond butter (almond derivatives count as nuts); "
            "'with_fruit' = berries/orange/cherry/raisins/zarzamora/orange flavoured pieces; "
            "'with_caramel' = caramel/toffee/butterscotch/dulce de leche (with or without salt); "
            "'with_cookie' = biscuit/wafer/oreo/cookie dough chunks (Hello Panda, Tim Tam); "
            "'filled' = praline/ganache/cream/lava filling без named add-ins (Lindor classic, "
            "Ferrero Rocher без named nut в основном слое); "
            "'with_alcohol' = rum/cognac/whisky/liqueur/brandy infused (brandy beans); "
            "'with_coffee' = espresso/mocha/coffee beans; "
            "'other' для shaped novelty WITHOUT inclusions (advent calendar, plain chocolate letters)."
        ),
    },
    "is_organic": {
        "type": "bool",
        "description": "Whether the product is organic",
    },
    # nutri_score_grade и protein_class исключены — TYPE_C deterministic из nutriments.
    "flavor_profile": {
        "type": "enum",
        "values": ["sweet_creamy", "intense_bitter", "fruity", "spiced",
                    "salty_caramel", "nutty", "floral", "other"],
        "nullable": True,
        "description": (
            "Dominant flavor character. **STRICT PRIORITY ORDER when multiple flavors present** "
            "(pick the FIRST matching specific category, fall back to base type): "
            "1. 'fruity' (fruit dominates) = orange, raspberry, cherry, strawberry, berries, "
            "zarzamora, dried fruit; "
            "2. 'spiced' = chili, cinnamon, ginger, cardamom, pepper, masala, Mexican hot chocolate; "
            "3. 'salty_caramel' = caramel / toffee / butterscotch / dulce de leche / creme brulee "
            "**(with OR without salt — plain 'caramel' / 'toffee' product → salty_caramel)**, "
            "sea salt-only dark fits here too; "
            "4. 'nutty' = hazelnut / almond / peanut / peanut butter / pistachio / walnut / pecan / "
            "marzipan as significant flavor (peanut butter blossoms → nutty); "
            "5. 'floral' = lavender, rose, jasmine, elderflower, geranium. "
            "**Tie-breaker for caramel+nuts** (e.g. 'toffee bark with pecans'): salty_caramel "
            "wins (caramel is more distinctive). "
            "**Base type fallback** (when no specific flavor or product name uninformative): "
            "plain milk chocolate / white chocolate / kinder / milka classic / Hershey's milk / "
            "milk chocolate с plain filling (LINDOR classic, lava cake) / cookie dough variants → "
            "'sweet_creamy'; "
            "plain dark chocolate (chocolat noir, semi-sweet, 50-95% cocoa, dark chocolate без "
            "specific flavor) → 'intense_bitter'. "
            "'other' ONLY для shaped novelty без flavor info (advent calendar, chocolate letters, "
            "Easter shapes) — но обычно даже их можно классифицировать по chocolate_type."
        ),
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
            "contains_nuts": False,
            "chocolate_extra": "plain",
            "is_organic": False,
            "flavor_profile": "intense_bitter",
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
            "contains_nuts": True,
            "chocolate_extra": "with_nuts",
            "is_organic": False,
            "flavor_profile": "nutty",
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
            "contains_nuts": False,
            "chocolate_extra": "with_fruit",
            "is_organic": True,
            "flavor_profile": "fruity",
        },
    ),
    (
        {
            "product_name": "Milka Tendre Lait",
            "brands": "Milka",
            "ingredients_text": "Sucre, beurre de cacao, lait écrémé en poudre, pâte de cacao, lactosérum en poudre",
            "quantity": "100 g",
        },
        {
            "chocolate_type": "milk",
            "contains_nuts": False,
            "chocolate_extra": "plain",
            "is_organic": False,
            "flavor_profile": "sweet_creamy",
        },
    ),
]
