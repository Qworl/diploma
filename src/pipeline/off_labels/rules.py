"""Rule-based labelers (TYPE_A..TYPE_F) and private _type_* helpers.

Each TYPE_X_RULES table maps attribute name → spec for that rule family.
Used internally by apply.py.
"""

import re

import pandas as pd

from src.pipeline.off_labels.tags import (
    ORGANIC_TAGS, ORGANIC_PATTERNS, GLUTEN_FREE_TAGS,
    VEGAN_TAGS, VEGAN_INGREDIENTS_ANALYSIS_POSITIVE, VEGAN_INGREDIENTS_ANALYSIS_NEGATIVE,
    FAIR_TRADE_TAGS,
    PALM_OIL_INGREDIENTS_ANALYSIS_FREE, PALM_OIL_INGREDIENTS_ANALYSIS_CONTAINS, PALM_OIL_TAGS,
    NO_ADDED_SUGAR_TAGS, LOW_SUGAR_TAGS, HIGH_FIBRE_TAGS,
    PDO_TAGS, LACTOSE_FREE_TAGS, GRAIN_FREE_TAGS,
    WHOLE_GRAIN_TAGS, WHOLE_GRAIN_CATEGORY_TAGS,
    FILLED_PASTA_CATEGORY_TAGS,
    CARBONATED_CATEGORY_TAGS, CAFFEINE_CATEGORY_TAGS,
    NUTS_ALLERGEN_TAGS,
    WHOLE_GRAIN_REGEX, NUTS_REGEX, SULFATES_REGEX, SILICONES_REGEX,
)


def _split_off_tags(value) -> list[str]:
    """labels_tags/categories_tags/allergens_tags from OFF — comma string OR list. Lowercase normalised."""
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, str):
        return [t.strip().lower() for t in value.split(",") if t.strip()]
    if hasattr(value, "__iter__"):
        return [str(t).strip().lower() for t in value if str(t).strip()]
    return []


def _has_any(tags: list[str], wanted: set[str], patterns: tuple[str, ...] = ()) -> bool:
    if any(t in wanted for t in tags):
        return True
    if patterns and any(any(p in t for p in patterns) for t in tags):
        return True
    return False


# Маппинг атрибутов на (источник, набор тегов, паттерны).
# Значение — либо одиночный кортеж (source_field, wanted, patterns),
# либо список таких кортежей, если атрибут может выводиться из нескольких полей.
TYPE_A_RULES = {
    "is_organic": ("labels_tags", ORGANIC_TAGS, ORGANIC_PATTERNS),
    "is_gluten_free": ("labels_tags", GLUTEN_FREE_TAGS, ()),
    # is_vegan обрабатывается _type_vegan_specific (3-state: positive/negative/unknown).
    # Pasta-pipeline по-прежнему получит True/False через positive labels или
    # negative ingredients_analysis_tag.
    # is_whole_grain moved to TYPE_AC (tag + regex hybrid). Оставлено пустым в TYPE_A
    # чтобы pasta-pipeline по-прежнему получал {True/False/None} от TYPE_AC.
    # Pasta: filled через categories_tags
    "is_filled": ("categories_tags", FILLED_PASTA_CATEGORY_TAGS, ()),
    "is_no_added_sugar": ("labels_tags", NO_ADDED_SUGAR_TAGS, ()),
    "contains_nuts": ("allergens_tags", NUTS_ALLERGEN_TAGS, ()),
    # Baby food
    "is_lactose_free": ("labels_tags", LACTOSE_FREE_TAGS, ()),
    # Beverages: carbonated через categories_tags
    "is_carbonated": [
        ("categories_tags", CARBONATED_CATEGORY_TAGS, ()),
        ("categories_tags", {"en:sodas", "en:colas", "en:sparkling-waters",
                              "en:tonic-waters", "en:carbonated-waters",
                              "en:beer", "en:beers"}, ()),
    ],
    # Cheeses: PDO/AOP
    "is_pdo": ("labels_tags", PDO_TAGS, ()),
}


def _type_a_bool(row: dict, attr: str):
    """Type A: bool атрибут из labels_tags / categories_tags / allergens_tags.

    Поддерживает несколько источников для одного атрибута (list of rules).

    Returns:
        True — если хотя бы в одном источнике есть positive tag
        False — если хотя бы одно из source-полей присутствует, но positive tag отсутствует
        None — если ни одно из source-полей не присутствует в row (unknown)
    """
    if attr not in TYPE_A_RULES:
        return None
    rule = TYPE_A_RULES[attr]
    rules_list = rule if isinstance(rule, list) else [rule]

    any_field_present = False
    for source_field, wanted, patterns in rules_list:
        if source_field not in row or row[source_field] is None:
            continue
        raw = row[source_field]
        # NaN guard for float NaN values from pandas
        if isinstance(raw, float) and pd.isna(raw):
            continue
        any_field_present = True
        tags = _split_off_tags(raw)
        if _has_any(tags, wanted, patterns):
            return True
    return False if any_field_present else None


# Type AC: hybrid bool — tag matching OR numeric threshold OR regex match.
# Атрибут считается True если хотя бы один источник дал True; False если есть
# данные (теги/число/текст), но ни один источник не выдал True; None если нет
# никаких данных.

TYPE_AC_RULES = {
    "is_low_sugar": {
        "tag_source": "labels_tags",
        "tag_set": LOW_SUGAR_TAGS,
        "numeric_field": "sugars_100g",
        "numeric_op": "le",
        "numeric_threshold": 5.0,
    },
    "is_high_fibre": {
        "tag_source": "labels_tags",
        "tag_set": HIGH_FIBRE_TAGS,
        "numeric_field": "fiber_100g",
        "numeric_op": "ge",
        "numeric_threshold": 6.0,
    },
    # is_whole_grain: tag (en:whole-grain) OR categories_tags OR regex по составу
    "is_whole_grain": {
        "tag_sources": [
            ("labels_tags", None),         # set заполняется ниже
            ("categories_tags", None),
        ],
        "regex_fields": ("ingredients_text", "product_name"),
        "regex_pattern": WHOLE_GRAIN_REGEX,
    },
}


def _type_vegan_specific(row: dict, attr: str):
    """3-state vegan detection.

    Returns:
      True: labels_tags содержит en:vegan/fr:vegan/etc., либо
            ingredients_analysis_tags содержит en:vegan.
      False: ingredients_analysis_tags содержит en:non-vegan.
      None: ingredients_analysis_tags = en:maybe-vegan / en:vegan-status-unknown
            или ни одного из источников нет (хвост для LLM/Bayes).
    """
    if attr != "is_vegan":
        return None

    labels = _split_off_tags(row.get("labels_tags"))
    if _has_any(labels, VEGAN_TAGS, ()):
        return True

    ana = _split_off_tags(row.get("ingredients_analysis_tags"))
    if _has_any(ana, VEGAN_INGREDIENTS_ANALYSIS_POSITIVE, ()):
        return True
    if _has_any(ana, VEGAN_INGREDIENTS_ANALYSIS_NEGATIVE, ()):
        return False

    return None


def _type_palm_oil_free_specific(row: dict, attr: str):
    """3-state palm-oil-free detection (для baby).

    Returns:
      True: ingredients_analysis_tags содержит en:palm-oil-free.
      False: содержит en:palm-oil (но НЕ en:may-contain-palm-oil).
      None: en:may-contain-palm-oil или ничего (хвост для LLM/Bayes).
    """
    if attr != "is_palm_oil_free":
        return None

    ana = _split_off_tags(row.get("ingredients_analysis_tags"))
    if _has_any(ana, PALM_OIL_INGREDIENTS_ANALYSIS_FREE, ()):
        return True
    if _has_any(ana, PALM_OIL_INGREDIENTS_ANALYSIS_CONTAINS, ()):
        return False
    return None


def _type_ultra_processed_specific(row: dict, attr: str):
    """is_ultra_processed = (nova_group == 4). Для cheeses."""
    if attr != "is_ultra_processed":
        return None
    v = row.get("nova_group")
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return int(float(v)) == 4
    except (TypeError, ValueError):
        return None


def _type_ac_hybrid(row: dict, attr: str):
    if attr not in TYPE_AC_RULES:
        return None
    rule = TYPE_AC_RULES[attr]

    has_data = False
    # 1) Tag match — multi-source if rule has "tag_sources" list
    if "tag_sources" in rule:
        for source, tag_set in rule["tag_sources"]:
            val = row.get(source)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            has_data = True
            tags = _split_off_tags(val)
            if tag_set is None:
                # Special: is_whole_grain uses globals WHOLE_GRAIN_TAGS/_CATEGORY_TAGS
                if source == "labels_tags" and _has_any(tags, WHOLE_GRAIN_TAGS, ()):
                    return True
                if source == "categories_tags" and _has_any(tags, WHOLE_GRAIN_CATEGORY_TAGS, ()):
                    return True
            elif _has_any(tags, tag_set, ()):
                return True
    elif "tag_source" in rule:
        val = row.get(rule["tag_source"])
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            has_data = True
            tags = _split_off_tags(val)
            if _has_any(tags, rule["tag_set"], ()):
                return True

    # 2) Numeric threshold
    if "numeric_field" in rule:
        num_val = row.get(rule["numeric_field"])
        if num_val is not None and not (isinstance(num_val, float) and pd.isna(num_val)):
            try:
                value = float(num_val)
                has_data = True
                op = rule["numeric_op"]
                threshold = rule["numeric_threshold"]
                if op == "le" and value <= threshold:
                    return True
                if op == "ge" and value >= threshold:
                    return True
            except (TypeError, ValueError):
                pass

    # 3) Regex match on free-text fields
    if "regex_fields" in rule:
        for field in rule["regex_fields"]:
            text = row.get(field)
            if isinstance(text, str) and text:
                has_data = True
                if rule["regex_pattern"].search(text):
                    return True

    return False if has_data else None


# Type B: categories_tags → multiclass mapping
# Формат: {attr_name: {category_tag: enum_value, ...}}
TYPE_B_MAPPINGS = {
    "pasta_shape": {
        # ORDER MATTERS: specific tags must come BEFORE generic "en:noodles"
        # (see Trek D audit: silver mis-tagged Italian/Spanish ribbon pasta
        # as noodles because en:noodles appeared first in product tag lists).
        # Iteration is mapping-order, NOT tag-order — see _type_b_multiclass.
        "en:egg-tagliatelle": "tagliatelle",
        "en:fresh-tagliatelle": "tagliatelle",
        "en:tagliatelle": "tagliatelle",
        "en:rice-vermicelli": "vermicelli",
        "en:soy-vermicelli": "vermicelli",
        "en:cellophane-noodles": "vermicelli",
        "en:durum-wheat-macaroni": "macaroni",
        "en:durum-wheat-fusilli": "fusilli",
        "en:durum-wheat-penne": "penne",
        "en:durum-wheat-spirals": "fusilli",
        "en:durum-wheat-spaghetti": "spaghetti",
        "en:durum-wheat-linguine": "linguine",
        "en:egg-pappardelle": "tagliatelle",
        "en:pappardelle": "tagliatelle",
        "en:fettuccine": "tagliatelle",
        "en:spaghetti": "spaghetti",
        "en:penne": "penne",
        "en:fusilli": "fusilli",
        "en:macaroni": "macaroni",
        "en:farfalle": "farfalle",
        "en:lasagna": "lasagna",
        "en:lasagna-sheets": "lasagna",
        "en:rigatoni": "rigatoni",
        "en:vermicelli": "vermicelli",
        "en:linguine": "linguine",
        # Asian-style noodle subtypes (more specific than generic "noodles")
        "en:instant-noodles": "noodles",
        "en:rice-noodles": "noodles",
        "en:ramen": "noodles",
        "en:udon": "noodles",
        "en:soba": "noodles",
        # Generic "noodles" tag — LAST so specific shapes win
        "en:noodles": "noodles",
        # Filled & specialty named shapes — schema bucket "other".
        # These come AFTER generic noodles because if a product has both
        # en:noodles and e.g. en:ravioli, "other" should still win (ravioli is
        # filled pasta, not noodles). But order vs en:noodles doesn't matter
        # in practice since these tags rarely co-occur.
        "en:ravioli": "other",
        "en:tortellini": "other",
        "en:tortelloni": "other",
        "en:cappelletti": "other",
        "en:agnolotti": "other",
        "en:gnocchi": "other",
        "en:potato-gnocchi": "other",
        "en:stuffed-pastas": "other",
        "en:fresh-stuffed-pasta": "other",
        "en:pasta-stuffed-with-meat": "other",
        "en:pasta-stuffed-with-cheese": "other",
        "en:pasta-stuffed-with-vegetables": "other",
        "en:orecchiette": "other",
        "en:orzo": "other",
        "en:ditalini": "other",
        "en:conchiglie": "other",
        "en:conchigliette": "other",
        "en:spaetzle": "other",
        "en:pastina": "other",
        "en:trofie": "other",
    },
    "grain_type": {
        # ORDER MATTERS: specific grain tags must come BEFORE generic fallbacks
        # like en:fresh-pasta / en:cereal-pastas / en:stuffed-pastas that would
        # otherwise default to wheat. See _type_b_multiclass — iteration is
        # mapping-order.
        # --- Specific non-wheat grains first ---
        "en:rice-pastas": "rice",
        "en:rices": "rice",
        "en:long-grain-rices": "rice",
        "en:white-rices": "rice",
        "en:aromatic-rices": "rice",
        "en:indica-rices": "rice",
        "en:rice-based-drinks": "rice",
        "en:corn-pastas": "corn",
        "en:corn": "corn",
        "en:sweet-corn": "corn",
        "en:oat-pastas": "oat",
        "en:rolled-oats": "oat",
        "en:oat-based-drinks": "oat",
        "en:buckwheat-pastas": "buckwheat",
        "en:buckwheats": "buckwheat",
        # --- Legume / non-cereal pastas — schema "other" (no legume bucket) ---
        # OFF taxonomy is inconsistent about plural/singular; cover both.
        "en:legume-pastas": "other",
        "en:legume-pasta": "other",
        "en:lentil-pastas": "other",
        "en:lentil-pasta": "other",
        "en:dry-lentil-pasta": "other",
        "en:chickpea-pastas": "other",
        "en:chickpea-pasta": "other",
        "en:pea-pastas": "other",
        "en:pea-pasta": "other",
        "en:pulse-pastas": "other",
        "en:pulse-pasta": "other",
        "en:konjac": "other",
        "en:konjac-pasta": "other",
        "it:konjac": "other",
        "en:shirataki": "other",
        # --- Wheat specific ---
        "en:durum-wheat-pasta": "wheat",
        "en:dry-durum-wheat-pasta": "wheat",
        "en:whole-durum-wheat-pasta": "wheat",
        "en:wheat-pastas": "wheat",
        "en:wheat-flours": "wheat",
        "en:wheat-breads": "wheat",
        # --- Wheat generic fallback (only fire if no specific grain tag) ---
        "en:fresh-pasta": "wheat",
        "en:egg-pastas": "wheat",
        "en:cereal-pastas": "wheat",  # default
        "en:stuffed-pastas": "wheat",
        # Mixed
        "en:multigrain-pastas": "mixed",
        "en:gluten-free-pasta": "mixed",  # как правило mix rice+corn
        # Cereals (breakfast) — расширение для cereals-domain. Атрибут "grain_type"
        # shared with pasta_stratified, mappings не конфликтуют (разные категории).
        # NB: для cereals в Schema используются другие values (multigrain вместо mixed),
        # но Type B возвращает строку — TYPE_B вернёт "mixed", silver сохранит "mixed",
        # ML обучится. Если нужна строгая «multigrain» — переопределить override.
        "en:corn-flakes": "corn",
        "en:cornflakes": "corn",
        "en:flakes-cereals": "corn",
        "en:rice-puffs": "rice",
        "en:puffed-rice": "rice",
        "en:rice-cereals": "rice",
        "en:wheat-cereals": "wheat",
        "en:wheat-flakes": "wheat",
        "en:mueslis": "multigrain",
        "en:granolas": "multigrain",
        "en:multigrain-cereals": "multigrain",
        "en:porridges": "oat",
        "en:oat-meals": "oat",
    },
    "chocolate_type": {
        "en:dark-chocolates": "dark",
        "en:milk-chocolates": "milk",
        "en:white-chocolates": "white",
        "en:filled-chocolates": "filled",
        "en:chocolate-bars": None,  # generic — leaves to LLM
    },
    "beverage_type": {
        "en:waters": "water",
        "en:mineral-waters": "water",
        "en:spring-waters": "water",
        "en:fruit-juices": "juice",
        "en:juices": "juice",
        "en:soft-drinks": "soda",
        "en:sodas": "soda",
        "en:colas": "soda",
        "en:teas": "tea",
        "en:iced-teas": "tea",
        "en:coffees": "coffee",
        "en:coffee-drinks": "coffee",
        "en:dairy-drinks": "dairy",
        "en:milk-drinks": "dairy",
        "en:sport-drinks": "sport",
        "en:energy-drinks": "sport",
    },
    # Cosmetics (OBF)
    "product_type": {
        "en:shampoos": "shampoo",
        "en:hair-shampoos": "shampoo",
        "en:dry-shampoos": "shampoo",
        "en:deodorants": "deodorant",
        "en:antiperspirants": "deodorant",
        "en:soaps": "soap",
        "en:hand-soaps": "soap",
        "en:bar-soaps": "soap",
        "en:liquid-soaps": "soap",
        "en:shower-gels": "shower_gel",
        "en:showers-and-baths": "shower_gel",
        "en:bath-products": "shower_gel",
        "en:toothpastes": "toothpaste",
        "en:dental-care": "toothpaste",
        "en:makeup": "makeup",
        "en:lipsticks": "makeup",
        "en:foundations": "makeup",
        "en:mascaras": "makeup",
        "en:eyeshadows": "makeup",
        "en:nail-polishes": "makeup",
        "en:sunscreen": "sunscreen",
        "en:in-sun-protections": "sunscreen",
        "en:suncare": "sunscreen",
        "en:sun-protection": "sunscreen",
        "en:facial-creams": "face_cream",
        "en:face-creams": "face_cream",
        "en:moisturizing-creams": "face_cream",
        "en:anti-aging-creams": "face_cream",
        "en:body-creams": "body_cream",
        "en:body-lotions": "body_cream",
        "en:body-milks": "body_cream",
        "en:hand-creams": "hand_cream",
        "en:hand-care": "hand_cream",
        "en:hair-care": "hair_care",
        "en:hair-conditioners": "hair_care",
        "en:hair-masks": "hair_care",
        "en:hair-oils": "hair_care",
        "en:lip-balms": "lip_balm",
        "en:lip-care": "lip_balm",
    },
    # Cosmetics: body_area — куда наносится продукт.
    # Перекрывает TYPE_F target_audience (вырожденный, 96% unisex).
    "body_area": {
        # Hair
        "en:hair-care": "hair", "en:hair": "hair",
        "en:shampoos": "hair", "en:hair-shampoos": "hair",
        "en:dry-shampoos": "hair", "en:conditioners": "hair",
        "en:hair-conditioners": "hair", "en:hair-coloring": "hair",
        "en:hair-dyes": "hair", "en:hair-styling": "hair",
        "en:hair-sprays": "hair", "en:hair-oils": "hair",
        "en:hair-masks": "hair", "en:hair-removal": "hair",
        "fr:coiffants": "hair", "fr:produits-depilatoires": "hair",
        # Face / lips / eye
        "en:face": "face", "en:facial-creams": "face",
        "en:face-creams": "face", "en:facial-cleansers": "face",
        "en:cleansers": "face", "en:cleansing-milks": "face",
        "en:face-masks": "face", "en:facial-care": "face",
        "en:face-lotions": "face", "en:facial-toners": "face",
        "en:moisturizing-creams": "face", "en:moisturizers": "face",
        "en:anti-aging-creams": "face", "en:lip-balms": "face",
        "en:lip-care": "face", "en:eye-care": "face",
        "en:eye-makeup-remover": "face",
        "fr:lotion-de-beaute": "face", "fr:eau-de-rose": "face",
        # Body / hands / feet
        "en:body": "body", "en:body-care": "body",
        "en:body-creams": "body", "en:body-oils": "body",
        "en:body-lotions": "body", "en:body-milks": "body",
        "en:body-scrubs": "body", "en:hand-creams": "body",
        "en:hand-care": "body", "en:foot-care": "body",
        "en:soaps": "body", "en:hand-soaps": "body",
        "en:bar-soaps": "body", "en:liquid-soaps": "body",
        "en:shower-gels": "body", "en:showers-and-baths": "body",
        "en:bath-products": "body", "en:body-washes": "body",
        "en:marseille-soaps": "body",
        "fr:soin-corporel-et-capillaire": "body", "fr:lotion": "body",
        # Oral
        "en:toothpastes": "oral", "en:toothpaste": "oral",
        "en:mouthwashes": "oral", "en:dental-care": "oral",
        # Deo
        "en:deodorants": "deo", "en:antiperspirants": "deo",
        "fr:deodorant": "deo", "fr:deodorant-vegetal-24h": "deo",
        # Sun
        "en:suncare": "sun", "en:sunscreen": "sun",
        "en:in-sun-protections": "sun", "en:sun-protections": "sun",
        "en:after-sun": "sun",
        # Makeup / fragrance / intimate
        "en:makeup": "makeup", "en:lipsticks": "makeup",
        "en:foundations": "makeup", "en:mascaras": "makeup",
        "en:eyeshadows": "makeup", "en:nail-polishes": "makeup",
        "en:eye-makeup": "makeup",
        "en:perfumes": "fragrance", "en:fragrances": "fragrance",
        "en:intimate-hygiene": "intimate",
        "en:feminine-hygiene": "intimate",
    },
    # Pet food (OPFF)
    "pet_type": {
        "en:cat-food": "cat",
        "en:wet-cat-food": "cat",
        "en:dry-cat-food": "cat",
        "en:adult-cat-food": "cat",
        "en:cat-biscuits": "cat",
        "en:dog-food": "dog",
        "en:wet-dog-food": "dog",
        "en:dry-dog-food": "dog",
        "en:dog-biscuit": "dog",
        "en:dog-and-cat-food": "dog_and_cat",
        "en:bird-food": "bird",
        "en:fish-food": "fish",
        "en:rodent-food": "small_animal",
        "en:rabbit-food": "small_animal",
    },
    "food_form": {
        "en:dry-pet-food": "dry",
        "en:dry-cat-food": "dry",
        "en:dry-dog-food": "dry",
        "en:dry-food": "dry",
        "fr:croquettes": "dry",
        "en:wet-pet-food": "wet",
        "en:wet-cat-food": "wet",
        "en:wet-dog-food": "wet",
        "en:wet-food": "wet",
        "fr:nourriture-humide-pour-chat": "wet",
        "en:dog-biscuit": "treats",
        "en:cat-biscuits": "treats",
        "en:snacks": "treats",
        "en:treats": "treats",
        "en:supplements": "supplement",
        "en:dietary-supplements": "supplement",
    },
    "life_stage": {
        "en:puppy-food": "puppy_kitten",
        "en:kitten-food": "puppy_kitten",
        "en:junior-food": "puppy_kitten",
        "en:adult-cat-food": "adult",
        "en:adult-dog-food": "adult",
        "en:adult-food": "adult",
        "en:senior-cat-food": "senior",
        "en:senior-dog-food": "senior",
        "en:senior-food": "senior",
        "en:all-life-stages": "all_stages",
        "en:all-stages": "all_stages",
    },
    # Cheeses (OFF) — три атрибута через categories_tags
    "milk_source": {
        "en:cow-cheeses": "cow",
        "en:cow-milk-cheeses": "cow",
        "en:goat-cheeses": "goat",
        "en:goat-milk-cheeses": "goat",
        "en:sheep-s-milk-cheeses": "sheep",
        "en:sheep-cheeses": "sheep",
        "en:buffalo-mozzarella": "buffalo",
        "en:buffalo-cheeses": "buffalo",
        "en:mixed-milk-cheeses": "mixed",
    },
    "texture": {
        "en:hard-cheeses": "hard",
        "en:uncooked-pressed-cheeses": "hard",
        "en:cooked-pressed-cheeses": "hard",
        "en:semi-hard-cheeses": "hard",
        "en:soft-cheeses": "soft",
        "en:soft-cheeses-with-bloomy-rind": "soft",
        "en:soft-cheeses-with-washed-rind": "soft",
        "en:fresh-cheeses": "fresh",
        "en:stretched-curd-cheeses": "fresh",
        "en:mozzarella": "fresh",
        "en:ricotta": "fresh",
        "en:feta": "fresh",
        "en:cream-cheeses": "cream",
        "en:cheese-spreads": "cream",
        "en:blue-cheeses": "blue",
        "en:blue-veined-cheeses": "blue",
        "en:roquefort": "blue",
        "en:gorgonzola": "blue",
        "en:processed-cheeses": "processed",
        "en:processed-cheese-products": "processed",
    },
    "country_of_origin": {
        "en:french-cheeses": "france",
        "en:cheeses-from-france": "france",
        "en:italian-cheeses": "italy",
        "en:cheeses-from-italy": "italy",
        "en:spanish-cheeses": "spain",
        "en:cheeses-from-spain": "spain",
        "en:german-cheeses": "germany",
        "en:cheeses-from-germany": "germany",
        "en:cheeses-from-the-united-kingdom": "uk",
        "en:cheeses-from-england": "uk",
        "en:british-cheeses": "uk",
        "en:cheeses-from-the-united-states": "us",
        "en:american-cheeses": "us",
        "en:swiss-cheeses": "switzerland",
        "en:cheeses-from-switzerland": "switzerland",
        "en:dutch-cheeses": "netherlands",
        "en:cheeses-of-the-netherlands": "netherlands",
    },
    # Cereals (OFF breakfast cereals) — два атрибута
    "cereal_type": {
        "en:mueslis": "muesli",
        "en:muesli": "muesli",
        "en:granolas": "granola",
        "en:granola": "granola",
        "en:corn-flakes": "corn_flakes",
        "en:cornflakes": "corn_flakes",
        "en:flakes-cereals": "corn_flakes",
        "en:porridges": "oat_cereal",
        "en:porridge": "oat_cereal",
        "en:oat-meals": "oat_cereal",
        "en:rolled-oats": "oat_cereal",
        "en:chocolate-cereals": "chocolate_cereal",
        "en:cocoa-cereals": "chocolate_cereal",
        "en:puffed-rice": "puffed_rice",
        "en:rice-puffs": "puffed_rice",
        "en:multigrain-cereals": "mixed",
    },
    # grain_type для cereals (отдельно от pasta — разные категории)
    # Используется тот же атрибут "grain_type" в cereals schema, наследует
    # mapping из pasta TYPE_B (en:wheat-pastas → wheat и т.д.). Расширяем под cereals:
}


def _type_b_multiclass(row: dict, attr: str):
    """Type B: multiclass из categories_tags.

    Iterates over MAPPING keys in their defined order (not over the product's
    tag list), so specific tags configured earlier in TYPE_B_MAPPINGS beat
    generic fallbacks. Critical for shapes like en:tagliatelle winning over
    en:noodles when both are present on a product (silver-noise audit, Trek D
    pivot — silver mis-tagged ribbon pasta as noodles because en:noodles
    happened to come first in raw tag lists).

    Returns specific enum value if there's a clear category match. None otherwise (LLM gap).
    """
    if attr not in TYPE_B_MAPPINGS:
        return None
    if "categories_tags" not in row or row["categories_tags"] is None:
        return None
    raw = row["categories_tags"]
    if isinstance(raw, float) and pd.isna(raw):
        return None
    tags = set(_split_off_tags(raw))
    mapping = TYPE_B_MAPPINGS[attr]
    for tag, value in mapping.items():
        if tag in tags:
            return value
    return None


# Type C: numeric thresholding rules
TYPE_C_RULES = {
    "sugar_class": {
        "field": "sugars_100g",
        "buckets": [(0.5, "0"), (5.0, "low"), (10.0, "med"), (float("inf"), "high")],
    },
    "alcohol_class": {
        "field": "alcohol_100g",  # OFF column name in our parquet (was alcohol_value in spec)
        "buckets": [(0.5, "0"), (5.0, "low"), (15.0, "med"), (float("inf"), "high")],
    },
    "cocoa_percentage": {
        # Special: regex from product_name
        "regex_field": "product_name",
        "regex_pattern": r"(\d{2,3})\s*%",
        # Buckets follow industry convention "X-Y" = [X, Y): 70% cocoa lands
        # in the "70-85" bucket, matching how chocolate labels are read in
        # practice ("70%+ cocoa" = dark chocolate). The original silver
        # convention (X, Y] put 70% into "50-70" — Trek E Opus audit found
        # 14/15 cocoa_percentage overrides correcting this off-by-one.
        "buckets": [(30, "<30"), (50, "30-50"), (70, "50-70"), (85, "70-85"), (float("inf"), "85+")],
    },
    "protein_class": {
        "field": "proteins_100g",
        "buckets": [(1.0, "0"), (5.0, "low"), (15.0, "med"), (float("inf"), "high")],
        # Per-schema override включён ниже: PET_FOOD_SCHEMA использует
        # {low/medium/high} с границами 10/25 — bucket override приходит
        # из schema[attr]["buckets"] если он задан.
    },
    # Cheeses: fat content. Original thresholds (15/25/32) clustered most
    # cheeses in "medium" or "high"; Trek E Opus audit showed the silver
    # consistently undercalled hard cheeses (53× "medium → high",
    # 33× "high → very_high"). Adjusted thresholds (15/20/28) align with
    # Opus's reading and with the global cheese fat distribution
    # (median 25 g/100g, p75 30 g/100g). See audit_findings_cheeses.md.
    "fat_class": {
        "field": "fat_100g",
        "buckets": [(15.0, "low"), (20.0, "medium"),
                    (28.0, "high"), (float("inf"), "very_high")],
    },
}


def _type_c_numeric(row: dict, attr: str, schema: dict | None = None):
    """Type C: numeric thresholding.

    Если schema задан и в schema[attr] есть ключ "buckets" — он переопределяет
    TYPE_C_RULES[attr]["buckets"] (per-schema override, см. PET_FOOD_SCHEMA).
    """
    if attr not in TYPE_C_RULES:
        return None
    rule = TYPE_C_RULES[attr]

    # Schema-level bucket override (например, PET_FOOD_SCHEMA для protein_class)
    buckets = rule["buckets"]
    if schema is not None and attr in schema:
        attr_def = schema[attr]
        if isinstance(attr_def, dict) and "buckets" in attr_def:
            buckets = attr_def["buckets"]

    if "regex_field" in rule:
        # Special case for cocoa_percentage — extract from text
        field_val = row.get(rule["regex_field"])
        if not field_val or not isinstance(field_val, str):
            return None
        match = re.search(rule["regex_pattern"], field_val)
        if not match:
            return None
        try:
            value = float(match.group(1))
        except ValueError:
            return None
    else:
        field_val = row.get(rule["field"])
        if field_val is None:
            return None
        if isinstance(field_val, float) and pd.isna(field_val):
            return None
        try:
            value = float(field_val)
        except (TypeError, ValueError):
            return None

    for threshold, bucket in buckets:
        if value < threshold:
            return bucket
    return buckets[-1][1]


# Type E: regex по тексту (ingredients_text, product_name) + traces_tags fallback.
# Используется когда в OFF нет dedicated allergens-колонки, но есть signal в составе.

TYPE_E_RULES = {
    "contains_nuts": {
        "trace_tags_field": "traces_tags",
        "trace_tags": NUTS_ALLERGEN_TAGS,
        "regex_fields": ("ingredients_text", "product_name"),
        "regex": NUTS_REGEX,
    },
    "has_sulfates": {
        "regex_fields": ("ingredients_text",),
        "regex": SULFATES_REGEX,
    },
    "has_silicones": {
        "regex_fields": ("ingredients_text",),
        "regex": SILICONES_REGEX,
    },
}


# Type F: regex multiclass — first-match-wins по списку паттернов с метками.
# Используется для атрибутов где OFF не имеет dedicated тегов, но в тексте
# (ingredients_text/product_name/categories_tags) есть устойчивый сигнал.
# Cosmetics: fragrance_status, form_factor, target_audience.
# Pet food: primary_protein_source.

TYPE_F_RULES = {
    # Cosmetics — product_type (TYPE_F fallback после TYPE_B mapping категорий).
    # OBF имеет короткие названия и редкие category tags; regex по product_name
    # вытаскивает product_type для FR/EN/ES/DE/IT когда категория отсутствует.
    "product_type": {
        "fields": ("product_name", "ingredients_text"),
        "patterns": [
            # Order: специфичные продукты первыми, чтобы избежать ловушек
            # (e.g. "shampoo для тела" → shampoo, не body_cream).
            (re.compile(
                r"\b(shampoo|shampooing|champ[uú]|szampon|шампунь)\b", re.I), "shampoo"),
            (re.compile(
                r"\b(d[ée]odorant|deo|deodorant|desodorante|antiperspirant|"
                r"antitranspirant|antitranspirante|deodorante)\b", re.I), "deodorant"),
            (re.compile(
                r"\b(toothpaste|dentifrice|dentifricio|dent[ií]frico|zahnpasta|"
                r"зубная\s+паста)\b", re.I), "toothpaste"),
            (re.compile(
                r"\b(gel\s+douche|shower\s+gel|gel\s+de\s+ducha|duschgel|"
                r"bagnoschiuma|shower\s+wash)\b", re.I), "shower_gel"),
            (re.compile(
                r"\b(soap|savon|jab[oó]n|seife|sapone|мыло|saponetta)\b", re.I), "soap"),
            (re.compile(
                r"\b(cr[èe]me\s+visage|crema\s+facial|face\s+cream|"
                r"face\s+moistur|gesichtscreme|crema\s+viso)\b", re.I), "face_cream"),
            (re.compile(
                r"\b(cr[èe]me\s+(corps|mains?|pieds?)|hand\s+cream|body\s+cream|"
                r"body\s+lotion|body\s+milk|hand\s+lotion|body\s+butter|"
                r"hand\s+creme|crema\s+corporal|crema\s+mani|k[öo]rpercreme|"
                r"handcreme|handcr[èe]me)\b", re.I), "body_cream"),
            (re.compile(
                r"\b(cr[èe]me\s+solaire|sunscreen|suncream|protector\s+solar|"
                r"sonnencreme|crema\s+solare|spf\s*\d+)\b", re.I), "sunscreen"),
            (re.compile(
                r"\b(lipstick|rouge\s+[aà]\s+l[èe]vres|labial|lippenstift|"
                r"rossetto|mascara|m[aá]scara|wimperntusche|"
                r"foundation|fond\s+de\s+teint|base|grundierung|"
                r"eyeshadow|fard\s+[aà]\s+paupi[èe]res|sombra|lidschatten|"
                r"nail\s+polish|vernis\s+[aà]\s+ongles|esmalte|nagellack|"
                r"smalto|blush|fard|rouge|colorete|"
                r"eyeliner|kohl|maquillaje|makeup|make[-\s]up)\b", re.I), "makeup"),
            (re.compile(
                r"\b(baume\s+l[èe]vres?|lip\s+balm|b[aá]lsamo\s+labial|"
                r"lippenbalsam|balsamo\s+labbra|stick\s+l[èe]vres?)\b", re.I), "lip_balm"),
            (re.compile(
                r"\b(apr[èe]s[-\s]shampooing|conditioner|acondicionador|"
                r"haarsp[üu]lung|balsamo\s+capelli|hair\s+mask|masque\s+capillaire|"
                r"hair\s+oil|huile\s+capillaire|hair\s+conditioner|coloration|"
                r"hair\s+dye|teinture|tinte\s+capilar|haarf[äa]rbung)\b", re.I), "hair_care"),
        ],
        # default None — LLM gap
    },
    # Cosmetics
    "fragrance_status": {
        "fields": ("ingredients_text", "product_name", "labels_tags"),
        # Order matters — first match wins. fragrance_free check first
        # (иначе слово "fragrance" в "fragrance-free" поймает fragranced).
        "patterns": [
            (re.compile(
                r"(fragrance[-_ ]?free|fragrance\s*-\s*free|perfume[-_ ]?free|"
                r"unscented|scent[-_ ]?free|"
                r"sans[-_ ]parfum|ohne[-_ ]duftstoff|sin[-_ ]aroma|"
                r"без[-_ ]аромат|no[-_ ]fragrance|no[-_ ]perfume|no[-_ ]parfum|"
                r"hypoallergenic|en:no-fragrance|en:fragrance-free|"
                r"en:no-perfume|en:hypoallergenic)",
                re.IGNORECASE), "fragrance_free"),
            (re.compile(
                r"\b(parfum|fragrance|aroma|fragranza|duftstoff|perfume|"
                r"essence)\b|en:perfumes|en:fragrances",
                re.IGNORECASE), "fragranced"),
        ],
        # default: None (LLM gap)
    },
    "form_factor": {
        "fields": ("product_name", "categories_tags", "ingredients_text"),
        "patterns": [
            # Spray: aerosol formats first (specific)
            (re.compile(r"\b(spray|aerosol|spritz|brume|atomizer|mist)\b|\ben:spray", re.I), "spray"),
            # Stick: roll-on/deostick first
            (re.compile(r"\b(roll[-_ ]?on|deostick|deo[-_ ]?stick|lipstick|"
                        r"l[aá]piz|barrita)\b", re.I), "stick"),
            (re.compile(r"\bstick\b", re.I), "stick"),
            # Solid bar: shampoo bars, soap bars
            (re.compile(
                r"\b(bar[-_ ]soap|soap[-_ ]bar|savon[-_ ]solide|jab[oó]n[-_ ]barra|"
                r"seifenst[üu]ck|sapone[-_ ]solido|"
                r"solid[-_ ]bar|shampoo[-_ ]bar|conditioner[-_ ]bar|"
                r"savon[-_ ]de[-_ ]marseille)\b|en:bar-soaps|en:solid-soaps|"
                r"en:solid-shampoos", re.I), "solid_bar"),
            # Powder
            (re.compile(r"\b(powder|poudre|polvo|pulver|polvere|pudr|"
                        r"talc|talcum)\b|en:powders", re.I), "powder"),
            # Wipe
            (re.compile(r"\b(wipes?|lingettes?|toallitas?|servietten?|"
                        r"salviette)\b|en:wipes|en:facial-wipes",
                        re.I), "wipe"),
            # Cream/gel/balm/lotion (consolidated viscous formats)
            (re.compile(
                r"\b(cream|cr[èe]me|crema|creme|gel|balm|baume|b[aá]lsamo|"
                r"loci[oó]n|lotion|serum|s[eé]rum|mousse|foam|"
                r"oil|huile|aceite|oleo|[oóö]l|"
                r"butter|beurre|manteca)\b|en:creams|en:lotions|en:gels|"
                r"en:serums|en:balms|en:oils|en:butters", re.I), "cream_gel"),
            # Liquid as default for shampoos/shower-gels/body-wash
            (re.compile(
                r"\b(liquid|liquide|l[ií]quido|fl[üu]ssig|shampoo|shower[-_ ]gel|"
                r"body[-_ ]wash|tonic|toner|tonique)\b|en:shampoos|en:liquid-soaps|"
                r"en:shower-gels|en:toners|en:body-washes", re.I), "liquid"),
        ],
        # default: None — let LLM/ML decide
    },
    # target_audience removed (96% degenerate "unisex"). Replaced by body_area
    # (TYPE_B mapping from categories_tags — см. TYPE_B_MAPPINGS["body_area"]).
    # Baby food
    "milk_type": {
        # Применяется ТОЛЬКО если продукт — молочная смесь / йогурт для baby /
        # напиток на базе молока. Без gate'а regex ловит "riz" в любом блюде
        # («Sole et Riz»).
        "requires": {
            "categories_tags": re.compile(
                r"en:baby-milks|en:infant-formulas|en:baby-formula|"
                r"en:growth-milks|en:baby-milks-in-powder|"
                r"en:baby-follow-on-milk-from-5-months|"
                r"en:dairy-dessert-for-baby|"
                r"en:milks|en:plant-based-milk-alternatives|"
                r"en:fermented-milk-products|en:yogurts",
                re.I)
        },
        "fields": ("product_name", "categories_tags", "ingredients_text"),
        "patterns": [
            # Specific milk sources first (более специфичные паттерны)
            (re.compile(
                r"\b(soy|soja|soya|soia)\b|en:soy-based|en:soy-formulas",
                re.I), "soy"),
            (re.compile(
                r"\b(goat|ch[èe]vre|cabra|ziege|capra)\b|en:goat-milk|"
                r"en:goat-milks", re.I), "goat"),
            (re.compile(
                r"\b(rice|riso|riz|arroz|reis)\b|en:rice-based|en:rice-formulas",
                re.I), "rice"),
            (re.compile(
                r"\b(almond|amande|almendra|mandel|mandorla)\b|en:almond-based",
                re.I), "almond"),
            (re.compile(
                r"\b(hypoallergenic|hypoallerg[eé]nique|hipoalerg[eé]nico|"
                r"HA[-_ ]?formula|extensively[-_ ]?hydrolyzed)\b|"
                r"en:hypoallergenic-formula|en:hypoallergenic-formulas",
                re.I), "hypoallergenic"),
            # follow_on / growing-up — это НЕ source, это stage. Мы не возвращаем
            # отсюда milk_type, эти случаи попадают в minimal_age (см. ниже).
            # Если соответствующий продукт — coo молочная смесь, default cow ниже
            # сработает по «lait/leche/cow».
            # Cow as default — общие индикаторы молока (французский lait, испанский
            # leche и т.п.) ловят молочные йогурты для baby типа Nestlé Petit Brassé.
            (re.compile(
                r"\b(cow|vache|vaca|kuh|mucca|"
                r"infant[-_ ]?formula|baby[-_ ]?milk|first[-_ ]?milk|"
                r"stage[-_ ]?[123]|[123][èe][me]?[-_ ]age|"
                r"follow[-_ ]?on|growing[-_ ]?up|growth[-_ ]?milk|toddler|"
                r"lait[a-z]*|leche|latte|milch|laitage|laiti[èe]re?|"
                r"folgemilch|s[äa]uglingsmilch)\b|"
                r"en:baby-milks|en:infant-formulas|en:baby-formula|"
                r"en:growth-milks|en:baby-follow-on-milk-from-5-months|"
                r"en:dairy-dessert-for-baby|en:fermented-milk-products|"
                r"en:yogurts",
                re.I), "cow"),
        ],
    },
    "minimal_age": {
        "fields": ("product_name", "categories_tags"),
        "patterns": [
            # 0-3m: newborn / first months / 1er age
            (re.compile(
                r"(\b0[-_ ]?3m|first[-_ ]?milk|newborn|naissance|"
                r"recien[-_ ]?nacido|stage[-_ ]?1|1er[-_ ]?age|1[ée]re?[-_ ]?age|"
                r"0[-_ ]?6\s*mois|0[-_ ]?6m\b|0[-_ ]?\+\s*m)|"
                r"en:from-birth", re.I), "0-3m"),
            # 3-6m
            (re.compile(
                r"(\b3[-_ ]?6m|3m\s*[+]|4m\s*[+]|from[-_ ]?3[-_ ]?months|"
                r"d[èe]s[-_ ]?3[-_ ]?mois)", re.I), "3-6m"),
            # 6-12m — включая follow-on / 2ème age / Stage 2 (это переходные смеси
            # для возраста 6+ месяцев, не отдельный milk source).
            (re.compile(
                r"(\b6[-_ ]?12m|6m\s*[+]|9m\s*[+]|10m\s*[+]|"
                r"from[-_ ]?6[-_ ]?months|"
                r"d[èe]s[-_ ]?6[-_ ]?mois|d[èe]s[-_ ]?9[-_ ]?mois|"
                r"baby[-_ ]?food[-_ ]?stage[-_ ]?2|"
                r"\bstage[-_ ]?2|\b2[èe]me[-_ ]?age\b|\b2nd[-_ ]?age\b|"
                r"follow[-_ ]?on|formule[-_ ]?2)|"
                r"en:baby-follow-on-milk-from-5-months", re.I), "6-12m"),
            # 12m+ — toddlers / growing-up / 3ème age / growth-milks
            (re.compile(
                r"(\b12m\s*[+]|1y\s*[+]|toddler|growing[-_ ]?up|"
                r"growth[-_ ]?milk|croissance|"
                r"\b3[èe]me[-_ ]?age\b|\b3rd[-_ ]?age\b|stage[-_ ]?3|"
                r"formule[-_ ]?3)|en:growth-milks", re.I), "12m+"),
        ],
    },
    # Baby food formula — функциональное назначение (HA / AR / Comfort / etc.)
    "feeding_purpose": {
        "fields": ("product_name", "categories_tags"),
        "patterns": [
            # Hypoallergenic — HA / extensively hydrolyzed / amino acid
            (re.compile(
                r"(\bHA\b|\bH\.A\.|hypoallerg[ée]niqu[e]?|hypoallergenic|"
                r"hipoalerg[eé]nico|hipoallergeen|"
                r"\bHA[-_ ]?formula|extensively[-_ ]?hydrolyzed|"
                r"amino[-_ ]?acid[-_ ]?formula|allergic[-_ ]?care)|"
                r"en:hypoallergenic-formula|en:hypoallergenic-formulas",
                re.I), "hypoallergenic"),
            # Anti-reflux — AR / spit-up / thickened
            (re.compile(
                r"(\bAR\b|\bA\.R\.|anti[-_ ]?reflux|anti[-_ ]?r[ée]gurgitation|"
                r"\banti[-_ ]?spit|\bthickened|nutrilon[-_ ]?ar)|"
                r"en:anti-reflux-formula", re.I), "anti_reflux"),
            # Anti-colic / Comfort — для пищеварения
            (re.compile(
                r"\b(comfort|anti[-_ ]?colic|anti[-_ ]?coliques|"
                r"sensitive|gentle|easy[-_ ]?digest|verdauung|"
                r"sin[-_ ]?c[oó]licos|colick)\w*\b",
                re.I), "anti_colic"),
            # Lactose-free formula
            (re.compile(
                r"(lactose[-_ ]?free|sans[-_ ]?lactose|sin[-_ ]?lactosa|"
                r"laktosefrei|senza[-_ ]?lattosio|no[-_ ]?lactose)|"
                r"en:no-lactose|en:lactose-free", re.I), "lactose_free"),
            # Pre-term / для недоношенных
            (re.compile(
                r"(\bpre[-_ ]?nan\b|\bpre[-_ ]?term\b|pr[ée]matur[ée]?|"
                r"premature|low[-_ ]?birth[-_ ]?weight|fr[üu]hgeborene|"
                r"prematuro)|en:premature-formula", re.I), "pre_term"),
            # Default — regular (если есть текстовый сигнал, но не специфичный)
        ],
        "default_if_text": "regular",
    },
    "format": {
        "fields": ("product_name", "categories_tags", "quantity"),
        "patterns": [
            # Ready-to-feed — готовая жидкость
            (re.compile(
                r"(ready[-_ ]?to[-_ ]?feed|\bRTF\b|"
                r"pr[eê]t[-_ ]?[aà][-_ ]?l['']emploi|"
                r"listo[-_ ]?para[-_ ]?usar|trinkfertig)|"
                r"en:ready-to-feed", re.I), "ready_to_feed"),
            # Liquid concentrate — нужно разбавить
            (re.compile(
                r"(liquid[-_ ]?concentrate|concentr[eé]?[-_ ]?liquide|"
                r"concentrate|fl[üu]ssig[-_ ]?konzentrat)|"
                r"en:liquid-concentrate", re.I), "liquid_concentrate"),
            # Powder — основной формат
            (re.compile(
                r"(\bpowder\b|en[-_ ]?poudre|en[-_ ]?polvo|in[-_ ]?polvere|"
                r"\bpulver\b|formule[-_ ]?en[-_ ]?poudre)|"
                r"en:baby-milks-in-powder|en:powdered-formulas",
                re.I), "powder"),
        ],
        # default: powder (~95% baby formula в OFF — порошок)
        "default_if_text": "powder",
    },
    # Pet food
    "life_stage": {
        "fields": ("product_name", "categories_tags"),
        "patterns": [
            # Puppy/kitten — самые распространённые маркеры на товарах
            (re.compile(
                r"\b(puppy|puppies|kitten|kittens|junior|growth|baby[-_ ]?dog|"
                r"baby[-_ ]?cat|chiot|chaton|cachorro|gatito|welpe|k[äa]tzchen|"
                r"cucciolo|gattino|szczeniak|kotek)\b|en:puppy-food|"
                r"en:kitten-food|en:junior-food",
                re.I), "puppy_kitten"),
            (re.compile(
                r"\b(senior|mature|aging|geriatric|7[+]?[-_ ]?year|"
                r"a[gd]e[d]?|s[eé]nior|alter|anziano|s[eé]nior)\b|en:senior-food|"
                r"en:senior-cat-food|en:senior-dog-food",
                re.I), "senior"),
            (re.compile(
                r"\b(adult|adulte|adulto|ausgewachsen|adulta)\b|en:adult-food|"
                r"en:adult-cat-food|en:adult-dog-food",
                re.I), "adult"),
            (re.compile(
                r"\b(all[-_ ]life[-_ ]stages?|all[-_ ]ages|tous[-_ ]ages|"
                r"todas[-_ ]las[-_ ]edades)\b|en:all-life-stages|en:all-stages",
                re.I), "all_stages"),
        ],
        # default: None (LLM gap)
    },
    "primary_protein_source": {
        "fields": ("ingredients_text", "product_name"),
        "patterns": [
            # Order: more specific / culturally dominant first.
            (re.compile(
                r"\b(salmon|tuna|cod|trout|sardine|fish|seafood|shrimp|"
                r"saumon|thon|poisson|pesce|fisch|losos)\w*", re.I), "fish_seafood"),
            (re.compile(
                r"\b(chicken|poultry|turkey|hen|duck[-_ ]with[-_ ]chicken|"
                r"poulet|dinde|pollo|pavo|hähnchen|huhn|truthahn)\w*",
                re.I), "chicken_poultry"),
            (re.compile(
                r"\b(beef|veal|lamb|mutton|boeuf|veau|agneau|"
                r"carne|manzo|cordero|rind|kalb|lamm)\w*", re.I), "beef_lamb"),
            (re.compile(
                r"\b(duck|goose|rabbit|venison|wild[-_ ]game|"
                r"canard|oie|lapin|conejo|ente|hase)\w*", re.I), "duck_other_meat"),
            (re.compile(
                r"\b(vegetarian|vegan|plant[-_ ]based|grain|cereal|wheat|corn|rice|"
                r"legume|v[eé]g[eé]tarien|cereal[ie]s?|trigo|c[eé]r[eé]ales)\w*",
                re.I), "vegetable_grain"),
        ],
        # default: None
    },
    # Pet food: функциональное назначение корма
    "food_purpose": {
        "fields": ("product_name", "categories_tags"),
        "patterns": [
            (re.compile(
                r"\b(weight[-_ ]?control|weight[-_ ]?management|"
                r"light|low[-_ ]calorie|obesity|"
                r"poids|gewicht|peso[-_ ]?control|adelgazante)\b",
                re.I), "weight_control"),
            (re.compile(
                r"\b(urinary|urinaire|struvite|crystal[-_ ]?prevention|"
                r"\bk\/d\b|\bs\/d\b|renal|kidney|niere)\b",
                re.I), "urinary_health"),
            (re.compile(
                r"\b(dental|teeth|tartar|oral[-_ ]?care|"
                r"dent|zahn|dentes)\b",
                re.I), "dental_care"),
            (re.compile(
                r"\b(sensitive|gentle|easy[-_ ]?digest|digest[ie]|"
                r"hairball|hair[-_ ]?ball|"
                r"sensible|sensibilidad|empfindlich)\b",
                re.I), "sensitive_digestion"),
            (re.compile(
                r"\b(joint|mobility|articulation|gelenk|articula[cç][ãa]o|"
                r"hip|chondro)\b",
                re.I), "joint_mobility"),
            (re.compile(
                r"\b(skin|coat|derma|fur|fell|piel|pelo)[-_ ]?(?:care|health)?\b",
                re.I), "skin_coat"),
            (re.compile(
                r"\b(indoor|appartement|inside)\b",
                re.I), "indoor"),
        ],
        "default_if_text": "regular",
    },
    # Chocolate: добавки / включения в плитке
    "chocolate_extra": {
        "fields": ("product_name", "categories_tags", "ingredients_text"),
        "patterns": [
            # Filled — пралине, трюфели, начинка (specific first)
            (re.compile(
                r"\b(truffle|truffles|praline|pralines|ganache|filled|"
                r"with[-_ ]?cream|gianduja|relleno)\w*\b|"
                r"en:filled-chocolates|en:pralines|en:truffles",
                re.I), "filled"),
            # With cookies/biscuit/wafer
            (re.compile(
                r"\b(cookie|cookies|biscuit|biscuits|wafer|gaufrette|"
                r"galleta|keks|crunch|crunchy)\w*\b",
                re.I), "with_cookie"),
            # With caramel / nougat / fudge
            (re.compile(
                r"\b(caramel|caramelo|nougat|fudge|toffee)\w*\b",
                re.I), "with_caramel"),
            # With nuts (hazelnut, almond, pistachio, etc.)
            (re.compile(
                r"\b(hazelnut|almond|pistachio|peanut|cashew|walnut|"
                r"noisette|amande|pistache|cacahu[èe]te|noix|"
                r"avellan|alm[eé]ndra|nocciol|mandorl|pistacchi|"
                r"haselnuss|mandel|nut|nuts)\w*\b",
                re.I), "with_nuts"),
            # With fruit
            (re.compile(
                r"\b(orange|raspberry|strawberry|cherry|blueberry|cranberry|"
                r"banana|apple|fruit|berries|"
                r"framboise|fraise|cerise|myrtille|"
                r"naranja|fresa|cereza|"
                r"lampone|fragola|ciliegia)\w*\b",
                re.I), "with_fruit"),
            # With coffee / espresso / tea
            (re.compile(
                r"\b(coffee|espresso|cappuccino|caf[eé]|kaffee|tea|matcha)\b",
                re.I), "with_coffee"),
            # With alcohol (rum, whisky, etc.)
            (re.compile(
                r"\b(rum|whisky|whiskey|cognac|brandy|liqueur|liqu[eé]r|"
                r"champagne|wine|vino|rhum|alcohol)\w*\b",
                re.I), "with_alcohol"),
        ],
        # default_if_text: plain (никаких add-ins)
        "default_if_text": "plain",
    },
}


def _type_f_regex_multiclass(row: dict, attr: str):
    """Type F: regex multiclass — first-match-wins по списку (pattern, label).

    Если в правиле задан `requires` (dict field→regex) — атрибут применяется только
    если все указанные поля содержат match. Используется чтобы запретить ложные
    срабатывания: например, milk_type='rice' применяется лишь к молочным продуктам,
    а не ко всем продуктам со словом «riz» в названии.
    """
    if attr not in TYPE_F_RULES:
        return None
    rule = TYPE_F_RULES[attr]

    # Gating: если задан requires — все поля должны иметь match.
    requires = rule.get("requires")
    if requires:
        for req_field, req_pat in requires.items():
            val = row.get(req_field)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return None
            if not isinstance(val, str) or not val:
                return None
            if not req_pat.search(val):
                return None

    text_signal_present = False
    for field in rule["fields"]:
        val = row.get(field)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        if not isinstance(val, str) or not val:
            continue
        text_signal_present = True
        for pattern, label in rule["patterns"]:
            if pattern.search(val):
                return label

    # default_if_text fallback (например, unisex для target_audience если есть текст)
    if text_signal_present and rule.get("default_if_text") is not None:
        return rule["default_if_text"]

    return None


def _type_e_regex(row: dict, attr: str):
    """Type E: contains_nuts через ingredients_text/product_name regex.

    Семантика "is ingredient" (не "may contain"):
    - regex match в ingredients_text или product_name → True (актуальный ингредиент)
    - traces_tags содержит nuts allergen, но regex не сработал → False
      (allergen warning есть, но в составе nuts не указаны = "может содержать
       следы" → формально не contains)
    - всё пусто → None (unknown)
    """
    if attr not in TYPE_E_RULES:
        return None
    rule = TYPE_E_RULES[attr]

    # 1) regex on text fields — high-confidence True
    text_signal_present = False
    for field in rule["regex_fields"]:
        val = row.get(field)
        if not isinstance(val, str) or not val:
            continue
        text_signal_present = True
        if rule["regex"].search(val):
            return True

    # 2) traces_tags as fallback (optional) — disambiguates False vs unknown
    traces_field = rule.get("trace_tags_field")
    if traces_field and traces_field in row and row[traces_field] is not None:
        raw = row[traces_field]
        if not (isinstance(raw, float) and pd.isna(raw)):
            tags = _split_off_tags(raw)
            if any(t in rule["trace_tags"] for t in tags):
                # Allergen warning есть, но в составе nuts не упомянуты — False
                return False
            # traces_tags есть и nuts там нет: это negative signal
            return False

    # Если ingredients_text был, но nut signal отсутствует — False
    if text_signal_present:
        return False

    return None


def _type_d_direct(row: dict, attr: str):
    """Type D: прямые OFF fields."""
    if attr == "nutri_score_grade":
        v = row.get("nutriscore_grade")
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        s = str(v).upper()
        if s in {"UNKNOWN", "NOT-APPLICABLE", "NA", ""}:
            return None
        if s not in {"A", "B", "C", "D", "E"}:
            return None
        return s

    if attr == "nova_group":
        v = row.get("nova_group")
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None

    if attr == "nova_class":
        v = row.get("nova_group")
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        try:
            n = int(float(v))
        except (TypeError, ValueError):
            return None
        if n in (1, 2):
            return "natural"
        if n == 3:
            return "processed"
        if n == 4:
            return "ultra_processed"
        return None

    if attr == "palm_oil_status":
        tags = _split_off_tags(row.get("ingredients_analysis_tags"))
        for tag in tags:
            if tag in PALM_OIL_TAGS:
                return PALM_OIL_TAGS[tag]
        return None

    return None
