"""OFF/OBF tag taxonomies and regex constants used by the off_labels layer.

This module only holds data — no logic. Imported by `rules.py` and `apply.py`.
"""

import re

# === Organic ===
ORGANIC_TAGS = {
    "en:organic", "en:eu-organic", "fr:ab-agriculture-biologique",
    "de:eg-öko-verordnung", "en:bio", "en:bio-suisse",
    "en:demeter", "en:naturland",
    # Cosmetics-specific (OBF)
    "en:cosmos-organic", "en:cosmos-natural",
    "en:ecocert", "fr:cosmetique-bio-charte-cosmebio",
    "en:cosmebio", "en:nature-progres",
}
ORGANIC_PATTERNS = ("-bio-", ":bio-", "-organic-", "-cosmos-")

# === Gluten-free ===
GLUTEN_FREE_TAGS = {
    "en:gluten-free", "en:no-gluten", "en:sans-gluten",
    "en:senza-glutine", "en:sin-gluten", "en:glutenfrei",
    "en:dzg-gluten-free", "en:crossed-grain-trademark", "en:afdiag",
}

# === Vegan ===
VEGAN_TAGS = {"en:vegan", "fr:vegan", "en:vegan-society"}
VEGAN_INGREDIENTS_ANALYSIS_POSITIVE = {"en:vegan"}
VEGAN_INGREDIENTS_ANALYSIS_NEGATIVE = {"en:non-vegan"}

# === Fair trade ===
FAIR_TRADE_TAGS = {
    "en:fair-trade", "en:fairtrade", "en:fairtrade-international",
    "en:max-havelaar", "fr:max-havelaar",
    "en:fairtrade-cocoa", "en:fairtrade-cotton", "en:fairtrade-sugar",
    "en:fair-trade-usa", "en:rainforest-alliance",
    "fr:commerce-equitable", "es:comercio-justo",
    "de:fair-trade", "it:commercio-equo-e-solidale",
}

# === Palm oil ===
PALM_OIL_INGREDIENTS_ANALYSIS_FREE = {"en:palm-oil-free"}
PALM_OIL_INGREDIENTS_ANALYSIS_CONTAINS = {"en:palm-oil"}
PALM_OIL_TAGS = {
    "en:palm-oil-free": "palm-oil-free",
    "en:contains-palm-oil": "contains",
    "en:may-contain-palm-oil": "may-contain",
}

# === Sugar / fibre ===
NO_ADDED_SUGAR_TAGS = {"en:no-added-sugar", "en:no-sugar-added", "en:sugar-free"}

LOW_SUGAR_TAGS = {
    "en:low-sugar", "en:low-sugars", "en:reduced-sugar",
    "en:no-added-sugar", "en:no-sugar-added", "en:sugar-free",
    "fr:sans-sucre-ajoute", "fr:faible-en-sucres", "fr:peu-sucre",
    "de:zuckerarm", "de:ohne-zuckerzusatz",
    "es:bajo-en-azucar", "es:sin-azucares-anadidos",
    "it:senza-zuccheri-aggiunti", "it:poco-zucchero",
}

HIGH_FIBRE_TAGS = {
    "en:high-fibre", "en:high-fibres", "en:high-fiber", "en:high-fibers",
    "en:source-of-fibre", "en:source-of-fibres", "en:source-of-fiber",
    "fr:riche-en-fibres", "fr:source-de-fibres",
    "de:ballaststoffreich", "de:quelle-von-ballaststoffen",
    "es:rico-en-fibra", "es:fuente-de-fibra",
    "it:ricco-di-fibre", "it:fonte-di-fibre",
}

# === Cheeses: PDO/AOP/DOP — Protected Designation of Origin ===
PDO_TAGS = {
    "en:pdo", "en:aop", "en:dop",
    "en:protected-designation-of-origin",
    "fr:appellation-d-origine-protegee",
    "fr:aop", "it:dop", "es:dop",
}

# === Lactose-free ===
LACTOSE_FREE_TAGS = {
    "en:no-lactose", "en:lactose-free", "en:sans-lactose",
    "en:senza-lattosio", "en:sin-lactosa", "en:laktosefrei",
}

# === Grain-free (pet food) ===
GRAIN_FREE_TAGS = {
    "en:grain-free", "en:no-grain", "en:no-grains",
    "en:gluten-free", "en:no-gluten",
    "fr:sans-cereales", "fr:sans-cereale", "fr:sans-gluten",
    "de:ohne-getreide", "es:sin-cereales",
}

# === Whole grain ===
WHOLE_GRAIN_TAGS = {"en:whole-grain", "en:whole-grains"}
WHOLE_GRAIN_CATEGORY_TAGS = {"en:whole-grain-pastas", "en:whole-grain-flours"}

# === Pasta categories ===
FILLED_PASTA_CATEGORY_TAGS = {
    "en:stuffed-pastas", "en:filled-pastas", "en:ravioli", "en:tortellini",
    "en:tortelloni", "en:cappelletti", "en:cannelloni",
}

# === Beverage categories ===
CARBONATED_CATEGORY_TAGS = {"en:carbonated-drinks", "en:carbonated-beverages"}
CAFFEINE_CATEGORY_TAGS = {"en:beverages-with-caffeine", "en:caffeinated-drinks"}

# === Allergens ===
NUTS_ALLERGEN_TAGS = {
    "en:nuts", "en:tree-nuts", "en:hazelnuts", "en:almonds",
    "en:walnuts", "en:cashews", "en:pistachios", "en:peanuts",
}

# === Regex patterns ===
WHOLE_GRAIN_REGEX = re.compile(
    r"\b(wholemeal|whole[- ]?(grain|wheat|oat|rye|barley|spelt|rice|corn)|"
    # FR: AVOINE/BLÉ/RIZ + complet[s,e,es]
    r"(avoine|bl[ée]|riz|seigle|orge|épeautre|sarrasin)\s+complet[es]?|"
    # IT/ES/DE: integrale, integro/integral, vollkorn
    r"integral(e|es|i)?|integro|vollkorn|"
    # Russian "цельнозерновой"
    r"цельно[- ]?зерн)\w*",
    re.IGNORECASE,
)

NUTS_REGEX = re.compile(
    r"\b(nuts?|hazelnut|almond|walnut|cashew|pistachio|peanut|"
    r"noisette|amande|noix|cacahu[èe]te|nuez|nocciol|mandel|haselnuss)\w*",
    re.IGNORECASE,
)

SULFATES_REGEX = re.compile(
    r"\b(sodium\s+laureth\s+sulfate|sodium\s+lauryl\s+sulfate|"
    r"ammonium\s+laureth\s+sulfate|ammonium\s+lauryl\s+sulfate|"
    r"sls|sles|laureth[-\s]*\d*\s*sulfate|lauryl\s+sulfate)\b",
    re.IGNORECASE,
)

SILICONES_REGEX = re.compile(
    r"\b(dimethicone|cyclomethicone|cyclopentasiloxane|cyclohexasiloxane|"
    r"phenyl\s+trimethicone|amodimethicone|cetearyl\s+methicone|"
    r"trimethylsiloxysilicate|siloxan\w*|silicone\w*)\b",
    re.IGNORECASE,
)
