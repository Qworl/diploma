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
        "description": (
            "Cheese texture/type. **PRIORITY ORDER** — pick the FIRST matching: "
            "'processed' = melted/spread processed cheese с emulsifiers (kraft singles, laughing cow, "
            "vache qui rit, kiri, processed slices, american singles, velveeta, cheese sauces "
            "— NOVA-4 industrial). **Pre-shredded/grated/sliced regular cheese is NOT processed** "
            "even с anti-caking (cellulose, potato starch). "
            "'blue' = blue-veined с penicillium roqueforti (roquefort, gorgonzola, stilton, "
            "bleu d'auvergne, danish blue, fourme d'ambert, cabrales). "
            "'cream' = spreadable creamy fresh (cream cheese, philadelphia, fromage frais à tartiner, "
            "mascarpone, ricotta when spreadable). "
            "'fresh' = unripened solid in original form (whole mozzarella ball, low-moisture "
            "mozzarella block, fresh feta, paneer, queso fresco, queso blanco, fromage blanc, "
            "cottage, burrata, halloumi-style fresh, fresh marinated mozzarella). "
            "**EXCEPTION:** breaded/fried snack formats like 'mozzarella sticks' or 'cheese "
            "sticks' (the deep-fried appetizer) → 'processed'. Plain string cheese sticks "
            "(uncooked, single-serve packaging) → 'fresh'. "
            "'hard' = aged firm cheeses sold as bars/wedges/blocks/shredded (parmesan, cheddar "
            "ALL varieties — mild/medium/sharp/extra/white, comte, gouda, manchego, edam, gruyère, "
            "pecorino, raclette, BABYBEL, queso curado/semicurado, aged provolone, monterey jack, "
            "colby, colby jack, pepper jack, BellaVitano, Tillamook, Sargento, "
            "all American-style blocks/shreds/slices, 'Mexican blend' shredded). "
            "Prefer 'hard' for industry-standard aged wheels/shreds even if relatively semi-firm. "
            "'soft' = bloomy-rind or washed-rind soft cheeses + traditional semi-soft "
            "(brie, camembert, taleggio, munster, reblochon, livarot, brique, scamorza, "
            "**morbier, havarti, fontina, port salut, esrom, danbo**). "
            "**Do NOT use soft for babybel, edam, gouda, cheddar, monterey jack, colby, "
            "halloumi — those are 'hard' or 'fresh'.** "
            "'other' ONLY for truly rare exotic (do NOT use for any cheddar/mozzarella/"
            "queso/standard cheese). "
            "**IMPORTANT:** if product is NOT actually a cheese (crackers like Goldfish, "
            "biscuits, snack bars, egg dishes, cheese-flavored chips, cereal, dip mixes) → "
            "return null."
        ),
    },
    "country_of_origin": {
        "type": "enum",
        "values": ["france", "italy", "spain", "germany", "uk", "us",
                   "switzerland", "netherlands", "greece", "denmark",
                   "cyprus", "india", "mexico", "belgium", "bulgaria",
                   "ireland", "norway", "russia", "other"],
        "nullable": True,
        "description": (
            "Country where the cheese **variety/recipe** originated (traditional origin, NOT "
            "where this particular product was manufactured). "
            "**KEY RULE:** Generic name of an Italian/French/Spanish variety → that country, "
            "even if made in USA. Examples: 'Parmesan grated cheese' (Kraft USA) → 'italy' "
            "(parmesan recipe is Italian); 'Camembert' generic → 'france'; 'Feta' → 'greece' "
            "always; 'Mozzarella sticks' → 'italy' (variety); 'Cheddar' generic → 'uk' "
            "(unless brand is clearly American). "
            "**EXCEPTION — American-only varieties + US snack formats always 'us':** "
            "BellaVitano, Tillamook, Cabot, Land O'Lakes, Sargento, Kraft Singles, Velveeta, "
            "American Singles, Cheez Whiz, Cracker Barrel cheese, Monterey Jack, Colby, "
            "Colby Jack, Pepper Jack, cream cheese (US-style), 'Mexican blend' shredded, "
            "**US snack formats (string cheese, mozzarella sticks, cheese sticks 'n' crackers, "
            "shredded cheese for tacos, sliced 'American'-style cheese) → 'us'** even when "
            "based on Italian variety. Rule: if the FORMAT is a US convenience product → 'us'; "
            "only the cheese variety in original form keeps Italian/French/etc. origin. "
            "Other mappings: "
            "feta → 'greece'; halloumi → 'cyprus'; paneer → 'india'; havarti, danish blue → 'denmark'; "
            "queso fresco, queso cotija, queso blanco, queso oaxaca, queso del pais → 'mexico'; "
            "chimay, herve, brunch (Bel brand) → 'belgium' (но brunch by Bel = french); "
            "кашкавал → 'bulgaria'; cashel blue, dubliner → 'ireland'; jarlsberg, brunost → 'norway'; "
            "емменталь, gruyère, raclette, appenzeller → 'switzerland'; "
            "comté, brie, camembert, roquefort, brunch (Bel) → 'france'; "
            "parmesan, mozzarella, gorgonzola, pecorino, scamorza, taleggio, asiago → 'italy'; "
            "manchego, cabrales, mahón → 'spain'; "
            "cheddar (если British origin, Wensleydale, Stilton) → 'uk'; "
            "emmental (Allemand), tilsiter → 'germany'; "
            "cream cheese (US-style) → 'us'; "
            "gouda, edam, leyden, beemster → 'netherlands'. "
            "'other' только если страна не в списке (Кавказ, Балканы кроме Bulgaria и т.д.); "
            "null если совсем неясно."
        ),
    },
    # fat_class исключён из LLM scope — TYPE_C deterministic из fat_100g
    # (см. src/pipeline/off_labels/rules.py:TYPE_C_RULES, recalibrated 12/22/32).
    "aging": {
        "type": "enum",
        "values": ["fresh", "young", "aged"],
        "nullable": True,
        "description": (
            "Maturation stage — 3-class semantic classification. **Coarse buckets** to avoid "
            "subjective young/medium boundary noise. "
            "'fresh' = ONLY unripened cheeses sold within days, no maturation "
            "(mozzarella, ricotta, feta, paneer, fromage frais, cottage, burrata, queso fresco, "
            "queso blanco, cream cheese, string cheese, halloumi). "
            "'young' = up to ~6 months maturation, mild-to-balanced flavor — includes ALL "
            "bloomy-rind soft cheeses (brie, camembert, munster, reblochon, taleggio), "
            "mild/medium cheddars (Cornish Cove, Double Gloucester, mid-cheddars), "
            "basic gouda/edam/colby/monterey jack/havarti/BellaVitano basic, "
            "semi-curado, raclette, regular blue cheeses. "
            "'aged' = 12+ months long maturation, intense flavor — sharp/extra sharp/vintage "
            "cheddar, vieux comté/gouda, manchego viejo/añejo, parmigiano reggiano (any age), "
            "grana padano riserva, stravecchio, aged stilton, BellaVitano espresso/merlot. "
            "Use name keywords: 'mild'/'jeune'/'doux'/'medium'→young; "
            "'sharp'/'extra sharp'/'aged'/'vieux'/'añejo'/'viejo'/'vintage'/'stravecchio'/"
            "'riserva'/'reggiano'→aged. "
            "Default for processed/spreadable cheeses → 'fresh'. "
            "If aging unclear AND cheese type is generic (e.g. 'Cheddar' with no qualifier) → null."
        ),
    },
    "is_pdo": {
        "type": "bool",
        "description": (
            "Protected Designation of Origin (PDO/AOP/DOP/AOC) — geographic protection. "
            "**TRUE ONLY when:** (a) product name explicitly contains 'AOP', 'PDO', 'DOP', "
            "'AOC', 'POO', 'IGP-PDO'; OR (b) the cheese is a specifically named PDO-protected "
            "variety (Parmigiano Reggiano, Grana Padano, Pecorino Romano, Camembert de "
            "Normandie, Roquefort, Comté, Reblochon, Manchego, Mahón, Gorgonzola, "
            "Mozzarella di Bufala Campana, Feta, Stilton, West Country Farmhouse Cheddar). "
            "**FALSE for generic names:** generic 'Parmesan' (NOT Parmigiano Reggiano), "
            "'Camembert d'Isigny' (NOT Camembert de Normandie AOP), 'Mozzarella' (NOT di "
            "Bufala), 'Cheddar' (NOT West Country Farmhouse), 'Brie' (NOT Brie de Meaux AOP). "
            "If unsure → false."
        ),
    },
    "is_organic": {
        "type": "bool",
        "description": "Whether the cheese is organic / bio",
    },
    "is_ultra_processed": {
        "type": "bool",
        "description": (
            "True ONLY for NOVA group 4 industrial cheese products: melted/processed cheese "
            "(kraft singles, laughing cow, vache qui rit, kiri, velveeta, american singles), "
            "cheese spreads с emulsifiers (E331/E339/E452), processed cheese slices, "
            "cheese sauces с stabilizers, cheez whiz, easy cheese spray cans. "
            "**FALSE for** regular traditional cheeses (parmesan, cheddar all varieties, comte, "
            "brie, mozzarella, feta, gouda, etc.) **even when** pasteurized, pre-shredded, "
            "pre-sliced, pre-grated, or sold с anti-caking agents (cellulose, potato starch, "
            "natamycin). Shredded/grated/sliced cheese is NOT ultra-processed."
        ),
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
            "is_pdo": True,
            "is_organic": False,
            "is_ultra_processed": False,
            "aging": "young",
        },
    ),
    (
        {
            "product_name": "Mozzarella di bufala",
            "brands": "Galbani",
            "categories_tags": "en:dairies,en:cheeses,en:italian-cheeses,en:buffalo-cheeses,en:fresh-cheeses,en:mozzarella",
            "ingredients_text": "Lait de bufflonne pasteurisé, sel, présure",
            "quantity": "125 g",
        },
        {
            "milk_source": "buffalo",
            "texture": "fresh",
            "country_of_origin": "italy",
            "is_pdo": True,
            "is_organic": False,
            "is_ultra_processed": False,
            "aging": "fresh",
        },
    ),
    (
        {
            "product_name": "Roquefort AOP",
            "brands": "Société",
            "categories_tags": "en:dairies,en:cheeses,en:french-cheeses,en:sheep-s-milk-cheeses,en:blue-cheeses,en:roquefort,en:aoc-cheeses,en:pdo-cheeses",
            "ingredients_text": "Lait de brebis pasteurisé, sel, ferments (penicillium roqueforti), présure",
            "quantity": "100 g",
        },
        {
            "milk_source": "sheep",
            "texture": "blue",
            "country_of_origin": "france",
            "is_pdo": True,
            "is_organic": False,
            "is_ultra_processed": False,
            "aging": "aged",
        },
    ),
    (
        {
            "product_name": "La Vache qui rit, 16 portions",
            "brands": "La Vache qui rit, Bel",
            "categories_tags": "en:dairies,en:cheeses,en:processed-cheeses,en:spreadable-cheeses,en:french-cheeses",
            "ingredients_text": "Fromages (60%), eau, beurre, lait écrémé en poudre, protéines de lait, sels émulsifiants (E331, E452), sel, ferments",
            "quantity": "267 g",
        },
        {
            "milk_source": "cow",
            "texture": "processed",
            "country_of_origin": "france",
            "is_pdo": False,
            "is_organic": False,
            "is_ultra_processed": True,
            "aging": "fresh",
        },
    ),
    (
        {
            "product_name": "Morbier AOP au lait cru",
            "brands": "Monts & Terroirs",
            "categories_tags": "en:dairies,en:cheeses,en:french-cheeses,en:cow-cheeses,en:morbier,en:aop-cheeses,en:semi-soft-cheeses",
            "ingredients_text": "Lait cru de vache, sel, ferments, présure, charbon végétal (couche centrale)",
            "quantity": "200 g",
        },
        {
            "milk_source": "cow",
            "texture": "soft",
            "country_of_origin": "france",
            "is_pdo": True,
            "is_organic": False,
            "is_ultra_processed": False,
            "aging": "young",
        },
    ),
    (
        {
            "product_name": "Halloumi traditional",
            "brands": "Pittas",
            "categories_tags": "en:dairies,en:cheeses,en:cypriot-cheeses,en:halloumi,en:sheep-cheeses,en:semi-soft-cheeses",
            "ingredients_text": "Lait pasteurisé de chèvre, brebis et vache, sel, présure, menthe",
            "quantity": "225 g",
        },
        {
            "milk_source": "mixed",
            "texture": "soft",
            "country_of_origin": "cyprus",
            "is_pdo": True,
            "is_organic": False,
            "is_ultra_processed": False,
            "aging": "fresh",
        },
    ),
    (
        {
            "product_name": "Feta AOP",
            "brands": "Mevgal",
            "categories_tags": "en:dairies,en:cheeses,en:greek-cheeses,en:feta,en:sheep-cheeses,en:fresh-cheeses,en:aop-cheeses",
            "ingredients_text": "Lait pasteurisé de brebis (70%) et de chèvre (30%), sel, ferments lactiques, présure",
            "quantity": "200 g",
        },
        {
            "milk_source": "mixed",
            "texture": "fresh",
            "country_of_origin": "greece",
            "is_pdo": True,
            "is_organic": False,
            "is_ultra_processed": False,
            "aging": "fresh",
        },
    ),
]
