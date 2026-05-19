"""
Coverage extension via deterministic rules for silver_strong attributes.

Bottleneck §6.12.2: silver coverage 41-67% на некоторых silver_strong
атрибутах (pasta_shape, milk_source, texture, beverage_type, ...).
На long-tail (silver=None) cascade обучен без сигнала. Этот модуль
заполняет long-tail через regex/lookup правила по generic_name,
categories_tags, product_name, ingredients_text — те же поля, которые
silver labeler уже частично использует, но с расширенными словарями.

Каждое правило прозрачно (regex + lookup table), reason записывается
в отдельную колонку. Принцип: EXTEND (заполнять только silver=None),
не OVERRIDE.

Output: silver_extended_{cat}_standard.parquet с дополнительными
колонками `{attr}_source` ∈ {silver, ext_rule, still_none}.

Usage:
    python -m src.eval.coverage_extension [--category cat] [--write]
"""
from __future__ import annotations

import argparse
import logging
import os
import re
from typing import Callable

import pandas as pd

from src.common import PROCESSED_DIR, setup_logging

logger = logging.getLogger(__name__)


def _t(s) -> str:
    return str(s or "").lower() if s is not None and not (isinstance(s, float) and pd.isna(s)) else ""


CLASSIFIERS: dict[tuple[str, str], Callable] = {}


def _register(category: str, attr: str):
    def deco(fn):
        CLASSIFIERS[(category, attr)] = fn
        return fn
    return deco


# ============================================================================
# pasta/pasta_shape — silver coverage ~41%
# ============================================================================
@_register("pasta", "pasta_shape")
def _pasta_shape(row: dict) -> tuple[str | None, str]:
    pn = _t(row.get("product_name"))
    gn = _t(row.get("generic_name"))
    ct = _t(row.get("categories_tags"))
    full = f"{pn} | {gn}"

    # === Layer 0: non-pasta products (картофельные клёцки, gnocchi) → other ===
    if "en:gnocchi" in ct or "en:potato-gnocchi" in ct or "en:gnocchis" in ct:
        return "other", "cat:gnocchi"
    if re.search(r"\b(schupfnudeln|gnocchi|gnocchetti|gnocchis|"
                 r"kartoffel[ \-]?(klö(ß|ss)e|kn(ö|oe)del|teig|nudel))\b", full):
        return "other", "kw:potato-dumpling"
    if re.search(r"\b(spaetzle|sp[äa]tzle|sp[äa]ezle|sp[äa]tzli|knöpfle)\b", full):
        return "other", "kw:spaetzle"

    # === Layer 1: filled pasta (нет в schema enum, → other) ===
    # Tortelloni/tortellini/tortelli — явный список вместо broken regex
    # tortell?on?[ie] (баг: не матчил tortellini).
    if re.search(r"\b(tortelloni|tortellini|tortelli|tortellone|"
                 r"cappelletti|cappellett[oe]|cappellacc[ie]|cappellaccio|"
                 r"agnolotti|agnolott[oe]|"
                 r"mezzelune|mezzaluna|"
                 r"ravioli|ravioles|raviol[ie]|raviolon[ie]|"
                 r"girasol[ie]|girasoles?|"
                 r"panzerotti|panzerott[oe]|"
                 r"fagottini|fagottin[oe]|"
                 r"sacchetti|sacchett[oe]|"
                 r"caramelle|rotolo|"
                 r"cannellon[ie]|cannelloni|"
                 r"maultaschen|pierogi|piroshki)", full):
        return "other", "kw:filled-pasta-no-class"

    # === Layer 2: specific named shapes без enum match → other ===
    if re.search(r"\b(trulli|conchigli?[ie]|orecchiett?[ie]|paccher?[ie]|lumache|"
                 r"cresta|strozzapret[ie]|trof?[ie]e?|gemelli|pici|busiat[ie]|"
                 r"cavatelli|gigli|radiator[ie]|ditali|ditalin?[ie]|"
                 r"orzo|rosmarino|stelline|risoni|anelli|tubetti|casarecce|"
                 r"casereccia)", full):
        return "other", "kw:specific-shape-no-class"

    # === Layer 3: главные shapes (учитывают plurals/regional names) ===
    rules = [
        # Plural-аware: убран trailing \b чтобы tagliatelles/spaghettis матчились.
        (r"\b(spaghett[oi]|spaghettin[ie]|esparguete|espagueti)", "spaghetti"),
        (r"\b(penne|pennette|pennoni|pennetta|mezzi[ \-]penne|"
         r"mostachol|mostaccioli)", "penne"),
        (r"\b(fusilli|spirelli|spirale|spiral|eliche|tortiglion|"
         r"rotini|torsades|trompetti)", "fusilli"),
        (r"\b(macaroni|maccheroni|macarrones|macarrón|coquillettes|hörnchen|"
         r"coditos)", "macaroni"),
        (r"\b(farfalle|bowtie|bow[ \-]tie|schmetterling|moñitos|lazi)", "farfalle"),
        (r"\b(tagliatelle|tagliatelles|tagliarini|fettucce|fettuccine|"
         r"nudeln[ \-]band|bandnudeln|tallarín|tallarines)", "tagliatelle"),
        (r"\b(lasagne?|lasagn[ae]s|lasaña|lasaganas?)", "lasagna"),
        # noodles — Asian-style + generic noodles в product_name (egg/chicken/
        # instant noodles обычно Asian-inspired, OFF tag это подтверждает).
        (r"\b(asia[ \-]?nudeln|mie|udon|ramen|soba|chow[ \-]mein|"
         r"glasnudeln|reisnudeln|chinese[ \-]noodles|"
         r"egg[ \-]?noodles|chicken[ \-]?noodles?|instant[ \-]?noodles?|"
         r"oriental[ \-]?(noodles?|nouilles)|nouilles[ \-]?(asiatiques?|japonaises?))",
         "noodles"),
        (r"\b(rigatoni|tortiglioni|elicoidali)", "rigatoni"),
        (r"\b(vermicelli|vermicelles|capelli|cappellini|"
         r"angel[ \-]hair|fideos|fideo)", "vermicelli"),
        (r"\b(linguine|linguini|trenette|bavette|tagliolini|tagliolin)", "linguine"),
    ]
    for pat, label in rules:
        if re.search(pat, full):
            return label, f"kw:{label}"
    return None, "unmatched"


# ============================================================================
# pasta/grain_type — silver coverage ~62%
# ============================================================================
@_register("pasta", "grain_type")
def _pasta_grain_type(row: dict) -> tuple[str | None, str]:
    ct = _t(row.get("categories_tags"))
    pn = _t(row.get("product_name"))
    gn = _t(row.get("generic_name"))
    ing = _t(row.get("ingredients_text"))
    full = f"{pn} | {gn}"
    ing_top = ing[:200]  # первые 1-3 ингредиента — главный grain source

    # === Layer 0: legume/alternative pasta → other (нет grain в enum) ===
    # silver часто default к wheat для legume pasta — это noise.
    if re.search(r"\b(pois[ \-]?(?:cass[ée]s?|chiches?)|garbanzos?|chickpeas?|"
                 r"lentejas|lentill?es?|lenticchie|lentils?|linsen|"
                 r"edamames?|sojabohnen|soybeans?|"
                 r"erbsenmehl|farine[ \-]de[ \-]pois|harina[ \-]de[ \-]"
                 r"(?:lentejas?|garbanzos?)|legumi[ \-]?secchi|"
                 r"konjac|konnyaku|shirataki|seitan)\b", full + ing_top):
        return "other", "kw:legume-pasta"
    # Legume cat tags
    if any(x in ct for x in ["en:legume-pasta", "en:legume-pastas",
                              "en:chickpea-pasta", "en:lentil-pasta",
                              "en:pea-pasta"]):
        return "other", "cat:legume-pasta"

    # === Layer 1: OFF categories tags ===
    if "en:rice-pastas" in ct or "en:rice-noodles" in ct or "en:rice-pasta" in ct:
        return "rice", "cat:rice"
    if "en:corn-pasta" in ct or "en:maize-pasta" in ct or "en:corn-pastas" in ct:
        return "corn", "cat:corn"
    if "en:buckwheat-pasta" in ct or "en:buckwheat-pastas" in ct:
        return "buckwheat", "cat:buckwheat"
    if "en:multigrain-pastas" in ct or "en:multi-cereal-pastas" in ct:
        return "mixed", "cat:multigrain"

    # === Layer 2: explicit multigrain в названии ===
    if re.search(r"multi[ \-]?(grain|cereal|cereali|cereales)|"
                 r"m[eé]lange[ \-]?c[ée]r[ée]ales|mehrkorn|multicereali",
                 full):
        return "mixed", "kw:multigrain"

    # === Layer 3: ingredient-first detection (главный grain в первых ~200 chars) ===
    # Если первый/доминантный ингредиент — конкретный grain.
    ing_first_m = re.match(
        r"^([0-9]+(?:[.,][0-9]+)?\s*%?\s*)?"
        r"(?:semoule\s+de\s+|farine\s+de\s+|harina\s+de\s+|"
        r"farina\s+(?:di\s+|integrale\s+(?:di\s+)?)|"
        r"semola\s+(?:di\s+)?|"
        r"hartweizen(?:grieß|gries|mehl)?|"
        r"whole\s+grain\s+|integral\s+)?"
        # buckwheat и grano-saraceno проверяются ПЕРВЫМИ (длиннее, не fall to plain "grano"→wheat)
        r"(buckwheat|sarrasin|grano\s+saraceno|buchweizen|"
        r"rice|riz|riso|reis|arroz|"
        r"corn|ma[iï]s|maize|ma[ií]z|granmais|"
        r"oat|avoine|avena|hafer|"
        r"durum|wheat|bl[ée]|frumento|trigo|weizen|spelt|épeautre|"
        r"grano(?:\s+(?:duro|tenero))?)",
        ing_top.strip())
    if ing_first_m:
        grain = ing_first_m.group(2)
        if grain in ("rice", "riz", "riso", "reis", "arroz"):
            return "rice", f"ing:{grain}-first"
        if grain in ("corn", "maïs", "mais", "maize", "maíz", "granmais"):
            return "corn", f"ing:{grain}-first"
        if grain in ("oat", "avoine", "avena", "hafer"):
            return "oat", f"ing:{grain}-first"
        if grain in ("buckwheat", "sarrasin", "grano saraceno", "buchweizen"):
            return "buckwheat", f"ing:{grain}-first"
        # durum/wheat/blé/frumento/trigo/weizen/spelt/grano/grano duro/grano tenero
        return "wheat", f"ing:{grain}-first"

    # === Layer 4: keyword match в product_name/generic_name ===
    rules = [
        (r"\b(rice|riz|riso|reis|arroz)\b", "rice"),
        (r"\b(corn|ma[iï]s|maize|ma[ií]z)\b", "corn"),
        (r"\b(oat|avoine|avena|hafer)\b", "oat"),
        (r"\b(buckwheat|sarrasin|grano[ \-]?saraceno|buchweizen|trigo[ \-]?sarraceno)\b", "buckwheat"),
        (r"\b(durum|wheat|bl[ée]|frumento|trigo|weizen|spelt|épeautre|spelta|dinkel|"
         r"semoule[ \-]de[ \-]bl[ée]|semolina|hartweizen|integrale)\b", "wheat"),
    ]
    matches = []
    for pat, label in rules:
        if re.search(pat, full):
            matches.append(label)
    if len(matches) == 1:
        return matches[0], f"kw:{matches[0]}"
    if len(matches) >= 2:
        return "mixed", "kw:multi-grain-detected"

    # === Layer 5: default wheat для standard Italian pasta shape без grain signal ===
    # Если в имени есть pasta shape (spaghetti/penne/fusilli/...) И нет других
    # grain markers — это **wheat по умолчанию** (durum wheat = standard pasta).
    pasta_shape_re = (r"\b(spaghetti|penne|fusilli|macaroni|maccheroni|farfalle|"
                       r"tagliatelle|tagliatelles|lasagne?|lasaña|rigatoni|"
                       r"vermicelli|vermicelles|linguine|cappellini|"
                       r"orecchiette|paccheri|trofie|gemelli|fettuccine|fettucce|"
                       r"tortellini|tortelloni|tortelli|ravioli|raviol[ie]?s?|"
                       r"capp?ellett?i|cap?elletti|cannelloni|cannellon[ie]|"
                       r"agnolotti|panzerotti|mezzelune|sacchetti|fagottini|"
                       r"pasta|pâtes|coquillettes|nudeln|mostachol|tallarines?|"
                       r"casarecce|fideos|torsades|rotini)\b")
    if re.search(pasta_shape_re, full):
        return "wheat", "default:pasta-shape-implies-wheat"

    return None, "unmatched"


# ============================================================================
# chocolate/chocolate_type — silver coverage ~84%
# ============================================================================
@_register("chocolate", "chocolate_type")
def _chocolate_type(row: dict) -> tuple[str | None, str]:
    pn = _t(row.get("product_name"))
    gn = _t(row.get("generic_name"))
    ct = _t(row.get("categories_tags"))
    ing = _t(row.get("ingredients_text"))
    full = f"{pn} | {gn} | {ct}"

    # === Layer 1: OFF category tags (base type priority) ===
    # ВАЖНО: продукты часто имеют ОДНОВРЕМЕННО en:milk-chocolates И
    # en:filled-chocolates (milk chocolate с начинкой). silver выбирает
    # base type как primary, filling — отдельный atribut chocolate_extra.
    # Поэтому specific base type проверяется ПЕРЕД filled.
    if "en:white-chocolates" in ct:
        return "white", "cat:white"
    if "en:milk-chocolates" in ct:
        return "milk", "cat:milk"
    if "en:dark-chocolates" in ct or "en:plain-dark-chocolates" in ct:
        return "dark", "cat:dark"
    # Filled только если base type не detected (true filled assortments
    # типа pralines/bonbons box без specified milk/dark base).
    if "en:filled-chocolates" in ct or "en:pralines" in ct or "en:truffles" in ct:
        return "filled", "cat:filled"

    # === Layer 2: name keywords (high confidence — клиент прямо сказал) ===
    if re.search(r"\b(noir|dark|fondente|negro|amaro|zartbitter|extra[ \-]?bitter|"
                 r"chocolat[ \-]?noir|cioccolato[ \-]?fondente)\b", full):
        return "dark", "kw:dark"
    if re.search(r"\b(white|blanc|bianco|blanco|wei[sß]e?[ \-]?schokolade|"
                 r"chocolat[ \-]?blanc|cioccolato[ \-]?bianco)\b", full):
        return "white", "kw:white"
    if re.search(r"\b(vollmilch|milchschokolade|chocolat[ \-]?au[ \-]?lait|"
                 r"cioccolato[ \-]?al[ \-]?latte|chocolate[ \-]?con[ \-]?leche)\b", full):
        return "milk", "kw:milk-explicit"
    # Filled-specific keywords (only когда это явно описывает форму продукта)
    if re.search(r"\b(pralines?|truffles?|trufas?|trüff?eln?|bonbons?|"
                 r"chocolate[ \-]?box|boîte[ \-]?de[ \-]?chocolats?|"
                 r"pralinen[ \-]?(mischung|sortiment)|assortiment[ \-]de[ \-]chocolats)\b", full):
        return "filled", "kw:filled-named"

    # === Layer 3: ingredient-based detection (если name не дал signal) ===
    # Top ingredients first 250 chars. Boundary \b убран чтобы match'ить
    # немецкие compound words (Magermilchpulver, Milchschokolade, Vollmilchpulver).
    ing_top = ing[:250]
    has_milk = bool(re.search(r"(milch|latte|lait|milk|leche|"
                                 r"magermilch|vollmilch|"
                                 r"sólidos[ \-]?de[ \-]?la[ \-]?leche|"
                                 r"poudre[ \-]?de[ \-]?lait|"
                                 r"leche[ \-]?en[ \-]?polvo|"
                                 r"latte[ \-]?in[ \-]?polvere)", ing_top))
    has_cocoa = bool(re.search(r"(cocoa|cacao|kakao|cacau|kakaomasse|"
                                  r"pasta[ \-]di[ \-]cacao|p[âa]te[ \-]de[ \-]cacao|"
                                  r"chocolate|chocolat|cioccolato|schokolade|schoko)", ing_top))
    if has_cocoa and not has_milk:
        return "dark", "ing:cocoa-no-milk"
    if has_cocoa and has_milk:
        return "milk", "ing:cocoa-with-milk"

    return None, "unmatched"


# ============================================================================
# beverages/beverage_type — silver coverage ~56%
# ============================================================================
@_register("beverages", "beverage_type")
def _beverage_type(row: dict) -> tuple[str | None, str]:
    ct = _t(row.get("categories_tags"))
    pn = _t(row.get("product_name"))
    gn = _t(row.get("generic_name"))
    full = f"{pn} | {gn}"

    # === Layer 0: alcoholic beverages → other (нет в schema enum) ===
    if any(x in ct for x in ["en:alcoholic-beverages", "en:beers", "en:wines",
                              "en:liqueurs", "en:spirits", "en:champagnes",
                              "en:ciders", "en:sparkling-wines", "en:rosé-wines",
                              "en:white-wines", "en:red-wines", "en:vermouths",
                              "en:gin", "en:vodka", "en:whisky", "en:whiskeys",
                              "en:rum", "en:cocktails"]):
        return "other", "cat:alcoholic"
    if re.search(r"\b(beer|bière|cerveza|birra|bier|"
                 r"wine|vin|vino|wein|"
                 r"liqueur|liquore|likör|"
                 r"vodka|whisky|whiskey|gin|rum|tequila|cognac|brandy|"
                 r"champagne|champán|prosecco|cava|"
                 r"cidre|cider|sidra|"
                 r"amaretto|sambuca|grappa|absinth|aperol|campari|"
                 r"vermouth|porto|sherry)\b", full):
        return "other", "kw:alcoholic"

    # === Layer 1: OFF categories tags ===
    if any(x in ct for x in ["en:waters", "en:mineral-waters", "en:still-waters",
                              "en:sparkling-waters", "en:flavored-waters"]):
        return "water", "cat:water"
    if any(x in ct for x in ["en:fruit-juices", "en:juices", "en:nectars",
                              "en:fruit-nectars", "en:fruit-beverages",
                              "en:vegetable-juices"]):
        return "juice", "cat:juice"
    if any(x in ct for x in ["en:sodas", "en:carbonated-drinks", "en:colas",
                              "en:sweetened-beverages", "en:carbonated-soft-drinks"]):
        return "soda", "cat:soda"
    # Tea (вкл. herbal teas, infusions, tisanes — все типы tea-like)
    if any(x in ct for x in ["en:teas", "en:tea-based-beverages", "en:herbal-teas",
                              "en:black-teas", "en:green-teas", "en:white-teas",
                              "en:infusions", "en:tisanes", "en:fruit-teas",
                              "en:rooibos", "en:matcha"]):
        return "tea", "cat:tea"
    if any(x in ct for x in ["en:coffees", "en:coffee-based-beverages",
                              "en:instant-coffees", "en:ground-coffees",
                              "en:espresso", "en:cappuccinos"]):
        return "coffee", "cat:coffee"
    if any(x in ct for x in ["en:dairy-drinks", "en:plant-based-milks", "en:milks",
                              "en:milk-substitutes", "en:milk-beverages",
                              "en:yogurts", "en:drinking-yogurts", "en:kefirs",
                              "en:fermented-milks"]):
        return "dairy", "cat:dairy"
    if any(x in ct for x in ["en:sports-drinks", "en:energy-drinks",
                              "en:isotonic-drinks"]):
        return "sport", "cat:sport"

    # === Layer 2: generic_name / product_name fallback ===
    rules = [
        (r"\b(water|eau|agua|acqua|wasser)\b", "water"),
        (r"\b(juice|jus|jugo|succo|saft|nectar)\b", "juice"),
        (r"\b(soda|cola|sprite|fanta|pepsi|tonic|ginger[ \-]?ale|7up|"
         r"limonade|lemonade|limonata|gaseosa)\b", "soda"),
        # Tea — добавлены herbal/infusion variants (camomille, manzanilla, tisane)
        (r"\b(tea|th[ée]|t[èé]|tee|chai|infusion|tisane|manzanilla|"
         r"camomille|camomilla|kamille|rooibos|matcha)\b", "tea"),
        (r"\b(coffee|caf[ée]|caff[èe]|kaffee|espresso|latte|cappuccino|mocca|moka)\b", "coffee"),
        (r"\b(milk|lait|latte|leche|milch|yogurt|yaourt|kefir|smoothie|"
         r"buttermilk|buttermilch)\b", "dairy"),
        (r"\b(gatorade|powerade|red[ \-]?bull|monster|energy[ \-]?drink|"
         r"sport[ \-]?drink|isotonic)\b", "sport"),
    ]
    for pat, label in rules:
        if re.search(pat, full):
            return label, f"kw:{label}"
    return None, "unmatched"


# ============================================================================
# cheeses/milk_source — silver coverage ~49%
# ============================================================================
@_register("cheeses", "milk_source")
def _milk_source(row: dict) -> tuple[str | None, str]:
    pn = _t(row.get("product_name"))
    gn = _t(row.get("generic_name"))
    ing = _t(row.get("ingredients_text"))
    ct = _t(row.get("categories_tags"))
    full = f"{pn} | {gn} | {ing}"

    # specific named cheeses (high confidence)
    named = {
        "feta":         "sheep",
        "mozzarella di bufala": "buffalo",
        "mozzarella":   "cow",
        "halloumi":     "sheep",
        "manchego":     "sheep",
        "pecorino":     "sheep",
        "roquefort":    "sheep",
        "chèvre":       "goat",
        "cabra":        "goat",
        "ziegen":       "goat",
        "comté":        "cow",
        "camembert":    "cow",
        "brie":         "cow",
        "cheddar":      "cow",
        "gouda":        "cow",
        "parmesan":     "cow",
        "parmigiano":   "cow",
        "ricotta":      "cow",
        "edam":         "cow",
        "emmental":     "cow",
        "gruyère":      "cow",
        "burrata":      "cow",
        "mascarpone":   "cow",
    }
    for name, src in named.items():
        if name in full:
            return src, f"named:{name}"

    # generic detection
    if re.search(r"\b(buffalo|bufala|büffel|b[uú]falo)\b", full):
        return "buffalo", "kw:buffalo"
    if re.search(r"\b(goat|ch[èe]vre|capra|cabra|ziege|ziegen|geiß)\b", full):
        return "goat", "kw:goat"
    if re.search(r"\b(sheep|brebis|pecora|oveja|schaf|ovino)\b", full):
        return "sheep", "kw:sheep"
    if re.search(r"\bmixed\b|gemischt|mehr[ \-]?(milch|tier)", full):
        return "mixed", "kw:mixed"
    if re.search(r"\b(cow|vache|mucca|vaca|kuh|kuhmilch)\b", full):
        return "cow", "kw:cow"
    # generic "milk" without specific animal → assume cow (most common)
    if re.search(r"\b(milk|lait|latte|leche|milch)\b", full):
        return "cow", "kw:milk-default-cow"
    return None, "unmatched"


# ============================================================================
# cheeses/texture — silver coverage ~49%
# ============================================================================
@_register("cheeses", "texture")
def _cheese_texture(row: dict) -> tuple[str | None, str]:
    pn = _t(row.get("product_name"))
    gn = _t(row.get("generic_name"))
    ct = _t(row.get("categories_tags"))
    full = f"{pn} | {gn} | {ct}"

    # named-cheese lookups (texture однозначная)
    named = {
        "feta":          "fresh",
        "mozzarella":    "fresh",
        "ricotta":       "fresh",
        "cottage":       "fresh",
        "mascarpone":    "cream",
        "boursin":       "cream",
        "philadelphia":  "cream",
        "frischk":       "cream",
        "cheddar":       "hard",
        "comté":         "hard",
        "parmesan":      "hard",
        "parmigiano":    "hard",
        "manchego":      "hard",
        "gouda":         "hard",
        "edam":          "hard",
        "emmental":      "hard",
        "gruyère":       "hard",
        "pecorino":      "hard",
        "camembert":     "soft",
        "brie":          "soft",
        "munster":       "soft",
        "reblochon":     "soft",
        "burrata":       "fresh",
        "halloumi":      "soft",
        "roquefort":     "blue",
        "gorgonzola":    "blue",
        "bleu":          "blue",
        "blue":          "blue",
        "blauschimmel":  "blue",
        "stilton":       "blue",
        "danish blue":   "blue",
        "processed":     "processed",
        "fondue":        "processed",
        "schmelzkäse":   "processed",
        "spreadable":    "processed",
        "à tartiner":    "processed",
    }
    for name, tex in named.items():
        if name in full:
            return tex, f"named:{name}"

    # generic patterns
    if "en:blue-cheeses" in ct or "en:blue-mold-cheeses" in ct or "en:blue-mould-cheeses" in ct:
        return "blue", "cat:blue"
    if "en:hard-cheeses" in ct or "en:semi-hard-cheeses" in ct or "en:extra-hard-cheeses" in ct:
        return "hard", "cat:hard"
    if "en:soft-cheeses" in ct or "en:soft-ripened-cheeses" in ct or "en:soft-cheeses-with-bloomy-rind" in ct:
        return "soft", "cat:soft"
    # Fresh cheeses — расширены fromages-blancs / petits-suisses / skyr /
    # quark / curd cheeses / fermented dairy desserts (фактически spreadable
    # yogurt-style свежие сыры).
    if any(x in ct for x in ["en:fresh-cheeses", "en:cottage-cheeses",
                              "en:fromages-blancs-petit-suisses-and-skyr",
                              "en:fromages-blancs", "en:petits-suisses",
                              "en:skyr", "en:quark", "en:curd-cheeses",
                              "en:fermented-dairy-desserts",
                              "en:plain-fermented-dairy-desserts",
                              "en:plain-petit-suisse"]):
        return "fresh", "cat:fresh"
    if "en:cream-cheeses" in ct or "en:spreadable-cheeses" in ct or "en:cheese-spreads" in ct:
        return "cream", "cat:cream"
    if "en:processed-cheeses" in ct or "en:melted-cheeses" in ct:
        return "processed", "cat:processed"

    # Fromage blanc / petit suisse / quark / skyr / cottage — fresh keyword
    if re.search(r"\b(fromage[ \-]?blanc|petit[ \-]?suisse|skyr|quark|"
                 r"requesón|cottage[ \-]?cheese|topfen|kvarg)\b", full):
        return "fresh", "kw:fromage-blanc-style"
    if re.search(r"\b(fresh|frais|frische|fresca|fresco|"
                 r"chèvre[ \-]?frais|ricotta|mozzarella|burrata)\b", full):
        return "fresh", "kw:fresh"
    if re.search(r"\b(cream|cr[eé]me|crema|crème|frischk[äa]se|spread|tartiner|"
                 r"cream[ \-]?cheese)\b", full):
        return "cream", "kw:cream"
    if re.search(r"\b(hard|sec|sechi|harter|firm|aged|affin[ée]|mature|stagionato|"
                 r"curado|reifer|alt|vieux)\b", full):
        return "hard", "kw:hard"
    if re.search(r"\b(soft|tendre|morbido|weich|suave|"
                 r"goat[' ]?s?[ \-]?cheese|chèvre|brie|camembert)\b", full):
        return "soft", "kw:soft"
    return None, "unmatched"


# ============================================================================
# cereals/cereal_type — silver coverage ~67%
# Уже реализовано в src/eval/auto_arbitrage_cereal_type.py
# ============================================================================
@_register("cereals", "cereal_type")
def _cereal_type(row: dict) -> tuple[str | None, str]:
    from src.eval.auto_arbitrage_cereal_type import classify_cereal_type
    return classify_cereal_type(row)


# ============================================================================
# cosmetics/body_area — silver coverage ~79%
# ============================================================================
@_register("cosmetics", "body_area")
def _body_area(row: dict) -> tuple[str | None, str]:
    pn = _t(row.get("product_name"))
    gn = _t(row.get("generic_name"))
    ct = _t(row.get("categories_tags"))
    full = f"{pn} | {gn} | {ct}"

    # === Layer 0: non-cosmetic products (food supplements, vitamins) → other ===
    # OBF иногда содержит supplements (Berberine, Omega-3, Vitamin D), которые
    # не косметика. Detect через product_name keywords.
    if re.search(r"\b(berberine|berberina|vitamin[ae]?\s*[a-z0-9]*|"
                 r"omega[ \-]?[369]|magnesium|calcium|zinc|biotin|collagen|"
                 r"compl[ée]ment|supplement|capsules?|tablets?|pills?|"
                 r"comprim[ée]s?|gelules?|mg|iu|extract|extrakt)\b", pn):
        # extra check: но не если это маска для лица с витамином C, etc.
        if not re.search(r"\b(cream|crema|cr[ée]me|lotion|serum|mask|maska|maschera|"
                          r"creme|sérum|emulsion|gel)\b", full):
            return "other", "kw:supplement-not-cosmetic"

    # === Layer 1: OFF category tags (high confidence) ===
    # Hair (включая oils, masks, sprays)
    if any(x in ct for x in ["en:hair-products", "en:shampoos", "en:conditioners",
                              "en:hair-oils", "en:hair-care", "en:hair-masks",
                              "en:hair-sprays", "en:hair-colors"]):
        return "hair", "cat:hair"
    # Oral
    if any(x in ct for x in ["en:dental-care", "en:toothpastes", "en:mouthwashes",
                              "en:mouth-wash", "en:dental-hygiene"]):
        return "oral", "cat:oral"
    # Sun
    if any(x in ct for x in ["en:sun-care", "en:sunscreens", "en:sun-protection",
                              "en:after-sun"]):
        return "sun", "cat:sun"
    # Deo
    if any(x in ct for x in ["en:deodorants", "en:antiperspirants"]):
        return "deo", "cat:deo"
    # Makeup
    if any(x in ct for x in ["en:make-up", "en:makeup", "en:lipsticks", "en:mascaras",
                              "en:nail-polishes", "en:eye-shadows", "en:foundations"]):
        return "makeup", "cat:makeup"
    # Fragrance
    if any(x in ct for x in ["en:fragrances", "en:perfumes", "en:eaux-de-toilette",
                              "en:eaux-de-parfum"]):
        return "fragrance", "cat:fragrance"
    # Face
    if any(x in ct for x in ["en:face-care", "en:facial-care", "en:face-creams",
                              "en:face-masks", "en:facial-cleansers", "en:face-serums"]):
        return "face", "cat:face"
    # Intimate
    if any(x in ct for x in ["en:intimate-hygiene", "en:feminine-care"]):
        return "intimate", "cat:intimate"
    # Body (last — generic, чтобы face/hair/sun/etc. забрали приоритет)
    if any(x in ct for x in ["en:body-care", "en:shower-gels", "en:soaps",
                              "en:body-creams", "en:body-oils", "en:body-lotions",
                              "en:body-scrubs", "en:bath-products"]):
        return "body", "cat:body"
    # Very generic en:body / en:hair как fallback
    if "en:hair" in ct.split(",") if isinstance(ct, str) else False:
        return "hair", "cat:hair-generic"
    if "en:body" in ct.split(",") if isinstance(ct, str) else False:
        return "body", "cat:body-generic"

    # === Layer 2: keyword fallback ===
    rules = [
        (r"\b(shampoo|shampoing|shampooing|champ[uú]|conditioner|"
         r"haarspülung|hair[ \-]?(care|mask|oil|spray|colour|color))\b", "hair"),
        (r"\b(face[ \-]?(cream|wash|mask|serum|cleanser|toner)|"
         r"cr[ée]me[ \-]?visage|gesichts(creme|maske|wasser)|"
         r"cleanser|toner|exfoliant)\b", "face"),
        (r"\b(body[ \-]?(cream|wash|lotion|oil|scrub|butter)|"
         r"soap|savon|jab[oó]n|seife|shower[ \-]?gel|gel[ \-]?douche|"
         r"douche|duschgel|bath[ \-]?(salt|bomb|oil|foam))\b", "body"),
        # oral — расширен испанским "enjuague/enguague bucal" и итальянским "collutorio"
        (r"\b(toothpaste|dentifrice|dentif[rí]cio|zahnpasta|"
         r"mouth[ \-]?wash|mundwasser|dental|"
         r"en[gj]uague[ \-]?(bucal|dental)?|colutorio|collutorio|colluttorio|"
         r"bain[ \-]?de[ \-]?bouche)\b", "oral"),
        (r"\b(sun[ \-]?(screen|block|protection|cream|spray|lotion)|"
         r"spf\s?\d|protector[ \-]solar|sonnencreme|after[ \-]?sun)\b", "sun"),
        (r"\b(deodorant|d[eé]odorant|deo|desodorante|antiperspirant)\b", "deo"),
        (r"\b(lipstick|mascara|eyeliner|foundation|blush|nail[ \-]?polish|"
         r"rouge[ \-]à[ \-]l[ée]vres|maquillaje|makeup|"
         r"eye[ \-]?shadow|lip[ \-]?gloss)\b", "makeup"),
        (r"\b(perfume|parfum|cologne|eau[ \-]de[ \-]toilette|eau[ \-]de[ \-]parfum|"
         r"fragrance|parfümwasser)\b", "fragrance"),
        (r"\b(intimate|intim|f[ée]minine[ \-]hygiene)\b", "intimate"),
        # Coconut/argan/jojoba oil без specific context → body (массажное масло)
        (r"\b(huile\s+(?:v[ée]g[ée]tale|vierge)\s+(?:de\s+)?(?:noix\s+de\s+)?coco|"
         r"argan[ \-]?oil|jojoba[ \-]?oil|cocos[ \-]?nucifera[ \-]?oil)\b", "body"),
    ]
    for pat, label in rules:
        if re.search(pat, full):
            return label, f"kw:{label}"
    return None, "unmatched"


# ============================================================================
# Driver
# ============================================================================
ATTRS_WITH_RULES = sorted(set(CLASSIFIERS.keys()))


def extend(silver_df: pd.DataFrame, category: str) -> tuple[pd.DataFrame, dict]:
    """Apply extension rules. Returns (extended_df, stats_per_attr)."""
    out = silver_df.copy()
    stats = {}
    for (c, attr), cls_fn in CLASSIFIERS.items():
        if c != category or attr not in out.columns:
            continue
        before = out[attr].notna().sum()
        mask_empty = out[attr].isna()
        labels, reasons = [], []
        for _, row in out[mask_empty].iterrows():
            try:
                lab, reason = cls_fn(row.to_dict())
            except Exception as e:
                lab, reason = None, f"err:{e}"
            labels.append(lab)
            reasons.append(reason)
        new_labels_n = sum(1 for l in labels if l is not None)
        out.loc[mask_empty, attr] = labels
        # source provenance
        src_col = f"{attr}_source"
        if src_col not in out.columns:
            out[src_col] = ""
        out.loc[~mask_empty, src_col] = "silver"
        out.loc[mask_empty, src_col] = ["ext_rule" if l else "still_none" for l in labels]
        after = out[attr].notna().sum()
        stats[attr] = {
            "before": before,
            "after": after,
            "extended": new_labels_n,
            "total": len(out),
        }
    return out, stats


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--category", default=None,
                   help="Один category (e.g. pasta). Default: все 6.")
    p.add_argument("--write", action="store_true",
                   help="Сохранить silver_extended_{cat}_standard.parquet.")
    args = p.parse_args()

    cats = [args.category] if args.category else \
           ["pasta", "chocolate", "beverages", "cheeses", "cereals", "cosmetics"]

    print(f"{'category':<12} {'attr':<22} {'silver_cov':>11} {'+extension':>11} {'final_cov':>11}")
    print("-" * 75)
    for cat in cats:
        path = f"{PROCESSED_DIR}/{cat}_stratified_silver_standard.parquet"
        if not os.path.exists(path):
            continue
        sv = pd.read_parquet(path)
        ext, stats = extend(sv, cat)
        for attr, s in stats.items():
            cov_before = s["before"] / s["total"] * 100
            cov_after = s["after"] / s["total"] * 100
            ext_pct = s["extended"] / s["total"] * 100
            print(f"{cat:<12} {attr:<22} {cov_before:>10.0f}% +{ext_pct:>9.0f}% {cov_after:>10.0f}%")

        if args.write:
            out_path = f"{PROCESSED_DIR}/{cat}_stratified_silver_extended.parquet"
            ext.to_parquet(out_path, index=False)
            logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
