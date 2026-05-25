"""Pasta domain schema and few-shot examples for LLM prompts."""

PASTA_SCHEMA = {
    "grain_type": {
        "type": "enum",
        "values": ["wheat", "spelt", "rice", "corn", "buckwheat", "oat",
                    "potato", "legume", "mixed", "other"],
        "description": (
            "Primary starch base. Use: "
            "'wheat' (semoule de blé dur, durum, semolina, standard pasta), "
            "'spelt' (épeautre, dinkel), 'rice' (riz), 'corn' (maïs), "
            "'buckwheat' (blé noir, Buchweizen, sarrasin), 'oat' (avoine), "
            "'potato' (gnocchi, Spätzle где potato base), "
            "'legume' (red lentil pasta, chickpea pasta, black bean pasta, edamame pasta), "
            "'mixed' (multiple grains as primary base), 'other' rare."
        ),
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
                    "linguine", "shells", "gnocchi", "orzo", "other"],
        "nullable": True,
        "description": (
            "Pasta shape with grouping rules: "
            "'spaghetti' = standard long thin (capellini, angel hair, spaghettoni); "
            "'linguine' = long flat; "
            "'tagliatelle' = ribbon-flat (fettuccine, pappardelle, mafaldine); "
            "'penne' = short tube cut diagonally (penne lisce, penne rigate); "
            "'fusilli' = spiral/twist (rotini, riccioli, fusillotti, cavatappi, torsades, "
            "serpentini, eliche, trottole) — **all curly/spiral shapes → fusilli, not 'other'**; "
            "'rigatoni' = wide ridged tube; "
            "'macaroni' = short tube straight (elbows, ditalini, ziti, mostaccioli, pipette, "
            "tubetti rigati); "
            "'farfalle' = bow-tie; 'lasagna' = sheets (lasagne, dumpling sheets); "
            "'vermicelli' = very thin long; "
            "'shells' = shell-shape (coquillettes, conchiglie, conchigliette); "
            "'gnocchi' = potato dumplings (also Schupfnudeln); "
            "'orzo' = rice-shaped tiny pasta — includes 'semi di X' (semi di orzo, semi di melone), "
            "kritharaki, risoni, puntalette, anelletti; "
            "'noodles' = Asian noodles (ramen, udon, soba, rice noodles, mie, soup noodles, "
            "Chinese dumpling sheets, lo mein); "
            "'other' = ONLY rare specialty (cannelloni, orecchiette, gemelli, radiatori); "
            "do NOT use 'other' for any shape with a clear Italian name. "
            "null OK для filled pastas (ravioli, tortellini, cappelletti — set is_filled=true вместо shape)."
        ),
    },
    "is_vegan": {
        "type": "bool",
        "description": (
            "Whether the pasta is vegan. DEFAULT TRUE for plain pasta made only from "
            "wheat/grain + water (spaghetti, penne, fusilli, lasagna sheets, тагliatelle dry — "
            "no eggs/dairy). FALSE only if ingredients contain eggs ('uova', '_oeufs_', 'eggs', "
            "'_eier_'), milk/cheese ('lait', 'fromage', 'cheese'), or non-vegan filling (meat, "
            "shrimp). Fresh egg pasta (tagliolini all'uovo, fresh tagliatelle 'aux oeufs') = false. "
            "Filled pasta with meat/cheese/fish = false. Dry plain pasta from any grain = true."
        ),
    },
    # TYPE_C attrs (nutri_score_grade, protein_class) исключены из LLM scope —
    # они deterministically computeable из raw nutriments через src/pipeline/off_labels/rules.py.
    # См. отдельный pipeline silver_type_c_fresh для их заполнения.
    "cuisine_origin": {
        "type": "enum",
        "values": ["italian", "asian", "german_alpine", "other_regional", "other"],
        "nullable": True,
        "description": (
            "Cultural cuisine tradition the pasta belongs to. Semantic classification "
            "based on shape/name/ingredients context: "
            "'italian' = spaghetti, penne, fusilli, lasagna, tagliatelle, farfalle, "
            "rigatoni, vermicelli (Italian wheat pasta); "
            "'asian' = noodles, ramen, udon, soba, rice noodles, glass noodles, instant noodles; "
            "'german_alpine' = spätzle, knöpfle, käsespätzle; "
            "'other_regional' = couscous/freekeh/harissa pasta (north_african), pierogi/halušky/"
            "kluski/лапша (eastern_european), middle_eastern wheat varieties; "
            "'other' = не подпадает под выше. Use product_name and ingredients_text context."
        ),
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
            "is_filled": False,
            "is_organic": False,
            "is_gluten_free": False,
            "is_vegan": True,
            "cuisine_origin": "italian",
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
            "is_filled": False,
            "is_organic": True,
            "is_gluten_free": True,
            "is_vegan": True,
            "cuisine_origin": "asian",
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
            "is_filled": False,
            "is_organic": False,
            "is_gluten_free": True,
            "is_vegan": True,
            "cuisine_origin": "german_alpine",
        },
    ),
    (
        {
            "product_name": "Gnocchi di patate al pesto",
            "brands": "Rana",
            "categories_tags": "en:plant-based-foods-and-beverages,en:plant-based-foods,en:cereals-and-potatoes,en:meals,en:filled-pastas,en:gnocchi,en:potato-gnocchi",
            "ingredients_text": "Patate (60%), farina di grano tenero, uova, sale, pesto",
            "quantity": "350 g",
        },
        {
            "grain_type": "potato",
            "pasta_shape": "gnocchi",
            "is_filled": False,
            "is_organic": False,
            "is_gluten_free": False,
            "is_vegan": False,
            "cuisine_origin": "italian",
        },
    ),
    (
        {
            "product_name": "Ravioli ricotta e spinaci",
            "brands": "Giovanni Rana",
            "categories_tags": "en:plant-based-foods,en:cereals-and-potatoes,en:pastas,en:stuffed-pastas,en:fresh-stuffed-pasta,en:ravioli",
            "ingredients_text": "Semoule de blé dur, eau, oeufs, ricotta (15%), épinards (8%), parmesan, sel",
            "quantity": "250 g",
        },
        {
            "grain_type": "wheat",
            "pasta_shape": None,
            "is_filled": True,
            "is_organic": False,
            "is_gluten_free": False,
            "is_vegan": False,
            "cuisine_origin": "italian",
        },
    ),
    (
        {
            "product_name": "Organic Red Lentil Penne",
            "brands": "Tolerant",
            "categories_tags": "en:plant-based-foods,en:cereals-and-potatoes,en:pastas,en:gluten-free-pastas,en:legume-pastas",
            "ingredients_text": "Organic red lentil flour",
            "quantity": "227 g",
        },
        {
            "grain_type": "legume",
            "pasta_shape": "penne",
            "is_filled": False,
            "is_organic": True,
            "is_gluten_free": True,
            "is_vegan": True,
            "cuisine_origin": "other",
        },
    ),
]
