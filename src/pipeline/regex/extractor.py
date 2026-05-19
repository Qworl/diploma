"""
Regex-based attribute extraction (Layer 1 of hybrid system).

Extracts structured attributes from product text fields:
- fat_content: "3.2%", "жирность 2,5%"
- minimal_age: "6+", "от 6 мес", "from 12 months"
- weight/volume: "200г", "1л", "500ml"
- cooking_time: "варить 10 мин", "cook 8 min"
- КБЖУ validation: cross-check with nutriment fields
"""

import re
from dataclasses import dataclass


@dataclass
class ExtractionResult:
    value: str | float | None
    confidence: float  # 1.0 for regex match, 0.0 for no match
    source: str  # which field it was extracted from


class RegexExtractor:

    # --- Fat content ---
    FAT_PATTERNS = [
        # "жирн. 2.5%", "жирность 3,2%"
        re.compile(r"(?:жирн(?:ость)?\.?\s*)(\d+[.,]\d+)\s*%"),
        # "fat 3.2%", "fat: 2.5%"
        re.compile(r"fat\s*:?\s*(\d+[.,]\d+)\s*%", re.I),
        # "M.G. 3,5%" (matière grasse, French)
        re.compile(r"M\.?G\.?\s*(\d+[.,]\d+)\s*%", re.I),
        # standalone "2.5%" — but NOT followed by "free", "off", "discount", etc.
        re.compile(r"\b(\d+[.,]\d+)\s*%(?!\s*(?:free|off|discount|de réduction|rabatt|от нормы))", re.I),
    ]

    # --- Minimal age ---
    AGE_PATTERNS = [
        (re.compile(r"(\d+)\s*\+"), "name"),
        (re.compile(r"от\s*(\d+)\s*мес", re.I), "name"),
        (re.compile(r"from\s*(\d+)\s*months?", re.I), "name"),
        (re.compile(r"(\d+)\s*months?\s*(?:and\s*(?:up|over|older)|\+)", re.I), "name"),
        (re.compile(r"stage\s*(\d+)", re.I), "name"),
        (re.compile(r"step\s*(\d+)", re.I), "name"),
        # "1er âge", "2ème âge" (French, common in OFF)
        (re.compile(r"(\d+)(?:er|ème|e)\s*[aâ]ge", re.I), "name"),
    ]

    # Age stage mapping (stage 1 = 0+, stage 2 = 6+, stage 3 = 12+)
    STAGE_TO_MONTHS = {"1": "0", "2": "6", "3": "12"}

    # --- Weight / Volume ---
    MEASURE_PATTERNS = [
        # "200г", "200 г", "200g", "1.5кг", "1,5 kg"
        re.compile(r"(\d+[.,]?\d*)\s*(кг|г|kg|g)\b", re.I),
        # "1л", "500мл", "1.5l", "500ml"
        re.compile(r"(\d+[.,]?\d*)\s*(л|мл|l|ml)\b", re.I),
        # "200 oz", "16 fl oz"
        re.compile(r"(\d+[.,]?\d*)\s*(?:fl\.?\s*)?(oz)\b", re.I),
    ]

    # --- Cooking time ---
    COOKING_TIME_PATTERNS = [
        re.compile(r"(?:варить|варка|готовить)\s*(\d+)\s*мин", re.I),
        re.compile(r"cook(?:ing)?\s*(?:time)?\s*:?\s*(\d+)\s*min", re.I),
        re.compile(r"(\d+)\s*min(?:utes?)?\s*cook", re.I),
        # French: "cuisson 12 min", "temps de cuisson: 10 min"
        re.compile(r"cuisson\s*:?\s*(\d+)\s*min", re.I),
        # German: "Kochzeit 7 Minuten", "7 Min. kochen"
        re.compile(r"Kochzeit\s*:?\s*(\d+)\s*Min", re.I),
        re.compile(r"(\d+)\s*Min(?:uten?)?\s*kochen", re.I),
    ]

    def extract_fat_content(self, text: str) -> ExtractionResult:
        for pattern in self.FAT_PATTERNS:
            match = pattern.search(text)
            if match:
                value = float(match.group(1).replace(",", "."))
                if 0 < value < 50:  # real fat content is < 50%
                    return ExtractionResult(value=value, confidence=1.0, source="regex")
        return ExtractionResult(value=None, confidence=0.0, source="regex")

    def extract_minimal_age(self, text: str) -> ExtractionResult:
        for pattern, _ in self.AGE_PATTERNS:
            match = pattern.search(text)
            if match:
                raw = match.group(1)
                # Convert stage to months if applicable
                if "stage" in pattern.pattern.lower() or "step" in pattern.pattern.lower() or "âge" in pattern.pattern or "[aâ]ge" in pattern.pattern:
                    months = self.STAGE_TO_MONTHS.get(raw, raw)
                else:
                    months = raw
                return ExtractionResult(value=f"{months}+", confidence=1.0, source="regex")
        return ExtractionResult(value=None, confidence=0.0, source="regex")

    def extract_measure(self, text: str) -> ExtractionResult:
        for pattern in self.MEASURE_PATTERNS:
            match = pattern.search(text)
            if match:
                value = match.group(1).replace(",", ".")
                unit = match.group(2).lower()
                return ExtractionResult(value=f"{value} {unit}", confidence=1.0, source="regex")
        return ExtractionResult(value=None, confidence=0.0, source="regex")

    def extract_cooking_time(self, text: str) -> ExtractionResult:
        for pattern in self.COOKING_TIME_PATTERNS:
            match = pattern.search(text)
            if match:
                minutes = int(match.group(1))
                if 1 <= minutes <= 120:  # sanity check
                    return ExtractionResult(value=minutes, confidence=1.0, source="regex")
        return ExtractionResult(value=None, confidence=0.0, source="regex")

    def validate_kbju(
        self, calories: float | None, proteins: float | None,
        fats: float | None, carbs: float | None
    ) -> dict[str, bool]:
        """
        Cross-validate КБЖУ values.
        Expected: calories ≈ proteins*4 + fats*9 + carbs*4 (±20%)
        """
        result = {"all_present": False, "consistent": False}
        if all(v is not None for v in [calories, proteins, fats, carbs]):
            result["all_present"] = True
            expected = proteins * 4 + fats * 9 + carbs * 4
            if expected > 0:
                ratio = calories / expected
                result["consistent"] = 0.8 <= ratio <= 1.2
        return result

    # --- Grain type ---
    # ORDER MATTERS: legume / non-cereal patterns first so they beat the generic
    # "wheat" match when a product is e.g. "100% farine de pois cassés".
    # Trek D audit found 8+ products where silver collapsed legume/konjac/
    # sweet-potato pasta to wheat by tag inheritance; this regex catches them
    # from ingredients_text/product_name.
    GRAIN_PATTERNS = [
        # Legume / pulse — schema "other"
        (re.compile(
            r"\b(?:pois\s+(?:cass[ée]s|chiches)|chickpea|lentil|lenticchie|lentejas?|"
            r"farine\s+de\s+lentilles|harina\s+de\s+lentejas|"
            r"green\s+pea\s+flour|gr[üu]ne[nrs]?\s+erbsen|"
            r"split[\s-]?pea|garbanzo)\b",
            re.I), "other"),
        # Konjac / shirataki / sweet potato starch — schema "other"
        (re.compile(
            r"\b(?:konjac|konnyaku|shirataki|amorphophallus|"
            r"sweet\s+potato\s+starch|patate?\s+dolc[ei]\s+amido|"
            r"farina\s+di\s+konjac)\b",
            re.I), "other"),
        # Cereals (existing)
        (re.compile(r"\b(?:durum\s+)?wheat|blé|weizen|grano\s+duro|trigo\b", re.I), "wheat"),
        (re.compile(r"\brice|riz|reis|riso|arroz\b", re.I), "rice"),
        (re.compile(r"\bcorn|maïs|mais|maíz\b", re.I), "corn"),
        (re.compile(r"\bbuckwheat|sarrasin|buchweizen|grano\s+saraceno\b", re.I), "buckwheat"),
        (re.compile(r"\boat|avoine|hafer|avena\b", re.I), "oat"),
        # Spelt/dinkel: schema includes "spelt" (Trek D pivot) but ML model is
        # trained on legacy data where these collapse to wheat. Until retrain,
        # keep mapping to wheat to match training labels.
        (re.compile(r"\bspelt|épeautre|dinkel|farro\b", re.I), "wheat"),
    ]

    # --- Pasta shape ---
    # ORDER MATTERS: specific named shapes first, then Asian-style noodles,
    # then "other"-bucket specialty shapes. Trek D audit revealed silver
    # over-applied en:noodles to Italian/Spanish/German ribbon pasta;
    # multilingual synonyms below close that gap on ingredients_text/product_name.
    PASTA_SHAPE_PATTERNS = [
        (re.compile(r"\bspaghett[io]\b", re.I), "spaghetti"),
        (re.compile(r"\bpenne\b", re.I), "penne"),
        # Fusilli + German "Spirelli", Spanish "tirabuzón", "Drelli",
        # "Schlemmerlinge" (twirl), "Spiralen" (spirals)
        (re.compile(
            r"\b(?:fusill[ei]|spirell[ei]|tirabuz[óo]n|drelli|schlemmerling|spirale[ns]?|spirelli)\b",
            re.I), "fusilli"),
        # Macaroni + Spanish "macarrón(es)", "codo(s)/codito(s)", German "Hörnchen",
        # English "elbow"
        (re.compile(
            r"\bmacaron[i]?|maccheroni|macarr[óo]n(?:es)?|cod(?:o|os|ito|itos)|h[öo]rnchen|elbow\b",
            re.I), "macaroni"),
        (re.compile(r"\bfarfall?e\b", re.I), "farfalle"),
        # Tagliatelle + Spanish "tallarín(es)", German "Bandnudeln",
        # Italian "pappardelle"/"fettuccine"
        (re.compile(
            r"\btagliatell?e|tallar[íi]n(?:es)?|bandnudel[n]?|pappardell?e|fettuccine\b",
            re.I), "tagliatelle"),
        (re.compile(r"\blasagn[ea]\b", re.I), "lasagna"),
        (re.compile(r"\brigatoni\b", re.I), "rigatoni"),
        (re.compile(r"\blinguine\b", re.I), "linguine"),
        # Vermicelli + Spanish "fideos de arroz / chinos", cellophane noodles
        (re.compile(
            r"\b(?:vermicell[ei]|fideos?\s+(?:de\s+arroz|chinos?)|cellophane)\b",
            re.I), "vermicelli"),
        # Asian-style noodles (specific subtypes — beat generic "noodles" tag)
        (re.compile(
            r"\b(?:ramen|udon|soba|mie\b|yakisoba|nouilles\s+(?:chinois|asiatiq)|"
            r"chinese[\s-]?style\s+noodles?|asian[\s-]?style\s+noodles?)\b",
            re.I), "noodles"),
        # Filled & specialty shapes — schema bucket "other"
        # (tortelloni, gnocchi, spätzle, fleckerl, trofie, orecchiette, orzo,
        # ditalini, pastina, trulli, conchiglie/conchigliette, pipette rigate,
        # schupfnudeln). Silver previously mis-mapped most of these to noodles.
        (re.compile(
            r"\b(?:tortelloni|tortelli|ravioli|cappelletti|agnolotti|"
            r"sp(?:ä|ae)tzle|sp(?:ä|ae)tzli|fleckerl|trofie|orecchiette|"
            r"orzo|ditalini|pastina|trulli|conchigli(?:ne|ette|e)|"
            r"pipette\s+rigate|schupfnudel[n]?)\b",
            re.I), "other"),
    ]

    def extract_grain_type(self, text: str) -> ExtractionResult:
        for pattern, grain in self.GRAIN_PATTERNS:
            if pattern.search(text):
                return ExtractionResult(value=grain, confidence=1.0, source="regex")
        return ExtractionResult(value=None, confidence=0.0, source="regex")

    def extract_pasta_shape(self, text: str) -> ExtractionResult:
        for pattern, shape in self.PASTA_SHAPE_PATTERNS:
            if pattern.search(text):
                return ExtractionResult(value=shape, confidence=1.0, source="regex")
        return ExtractionResult(value=None, confidence=0.0, source="regex")

    # --- Chocolate ---
    COCOA_PCT_RE = re.compile(r"(\d{2,3})\s*%")

    # Chocolate type keywords (multilingual). Order matters — "white" before "milk"
    # so "Lindt White" is not classified as milk via fallback.
    CHOCOLATE_TYPE_PATTERNS = [
        # filled / praline / truffle first — usually overrides milk/dark
        (re.compile(r"\b(?:filled|fourr[ée]e?|praline|praliné|truffle|truffe|gianduja)\b", re.I), "filled"),
        (re.compile(r"\b(?:white|blanc(?:he)?|wei[sß]e?|bianco|blanco|branco)\b", re.I), "white"),
        (re.compile(r"\b(?:dark|noir|fondente|extra-?fin|zartbitter|schwarz|amargo|negro)\b", re.I), "dark"),
        (re.compile(r"\b(?:milk|lait|milch|latte|leche|leite)\b", re.I), "milk"),
    ]

    def extract_cocoa_percentage(self, text: str) -> ExtractionResult:
        """Extract cocoa percentage bucket from product text.

        Buckets: <30, 30-50, 50-70, 70-85, 85+ (matches CHOCOLATE_SCHEMA).
        """
        match = self.COCOA_PCT_RE.search(text)
        if not match:
            return ExtractionResult(value=None, confidence=0.0, source="regex")
        try:
            pct = int(match.group(1))
        except ValueError:
            return ExtractionResult(value=None, confidence=0.0, source="regex")
        if not (0 < pct <= 100):
            return ExtractionResult(value=None, confidence=0.0, source="regex")
        # Industry convention "X-Y" = [X, Y): 70% → "70-85" (matches the
        # silver TYPE_C_RULES post-Trek-E fix). Trek E Opus audit found
        # silver/cascade systematically labeling 70%-cocoa products as
        # "50-70"; the bug was an off-by-one boundary inherited from the
        # original `(X, Y]` reading.
        if pct < 30:
            bucket = "<30"
        elif pct < 50:
            bucket = "30-50"
        elif pct < 70:
            bucket = "50-70"
        elif pct < 85:
            bucket = "70-85"
        else:
            bucket = "85+"
        return ExtractionResult(value=bucket, confidence=1.0, source="regex")

    def extract_chocolate_type(self, text: str) -> ExtractionResult:
        """Определить тип шоколада (dark/milk/white/filled).

        ВАЖНО: вход должен быть `product_name + brands + quantity` БЕЗ
        `ingredients_text`. Триггер-слова milk/lait/Milch/latte/leche массово
        встречаются в составе (молочный порошок, эмульгатор лецитин и т.д.) у
        тёмного и filled-шоколада и приводят к 19 п.п. потери точности.
        Диагностика: 85 из 126 regex-ошибок chocolate_type на gold возникают
        из-за вхождения «milk-маркеров» в ингредиентах при gold=dark/filled.

        Дополнительно: при одновременном матче нескольких типов в названии
        (например, «Lait Noir Praliné») extractor отказывается решать и
        передаёт ячейку на Layer 2 — двусмысленные названия гарантированно
        ловят ML.
        """
        matched = []
        for pattern, ctype in self.CHOCOLATE_TYPE_PATTERNS:
            if pattern.search(text) and ctype not in matched:
                matched.append(ctype)
        if len(matched) != 1:
            return ExtractionResult(value=None, confidence=0.0, source="regex")
        return ExtractionResult(value=matched[0], confidence=1.0, source="regex")

    # Nut markers across languages (EN/FR/DE/IT/ES/PT).
    # Используется и для contains_nuts, и для chocolate_extra=with_nuts.
    NUT_RE = re.compile(
        r"\b(?:nuts?|hazelnuts?|noisettes?|haselnuss|nocciol[ae]|avellanas?|"
        r"almonds?|amandes?|mandel[n]?|mandorl[ae]|almendras?|"
        r"walnuts?|noix|walnuss|noci|nueces|"
        r"pistachios?|pistaches?|pistazie[n]?|pistacchi[oi]|pistachos?|"
        r"cashew[s]?|noix\s+de\s+cajou|anacardi[oi]|"
        r"pecans?|noix\s+de\s+pécan|"
        r"macadamias?|"
        r"praline[s]?|pralin[ée]e?s?|pralinen|"
        r"gianduja|gianduiotto|nougat|turr[óo]n|krokant)\b",
        re.I,
    )

    # chocolate_extra (subset) — additional inclusions in chocolate.
    # ORDER MATTERS: more specific patterns first; "with_nuts" via NUT_RE handled
    # separately below. "plain" не извлекается regex — это значение по умолчанию,
    # требующее отрицания всех маркеров.
    CHOCOLATE_EXTRA_PATTERNS = [
        # alcohol — rum, whisky, liqueurs
        (re.compile(
            r"\b(?:rum|rhum|whisk[ey]y|cognac|bourbon|amaretto|kirsch|"
            r"liqueur|liquor|alcool|alkohol|brandy)\b",
            re.I), "with_alcohol"),
        # coffee / mocha / espresso
        (re.compile(
            r"\b(?:coffee|caf[ée]|kaffee|caffè|mocha|moka|espresso|cappuccino)\b",
            re.I), "with_coffee"),
        # caramel / toffee / dulce de leche
        (re.compile(
            r"\b(?:caramel|karamell|caramello|toffee|fudge|dulce\s+de\s+leche)\b",
            re.I), "with_caramel"),
        # cookie / biscuit / wafer / crisp
        (re.compile(
            r"\b(?:cookie|biscuit|biscot|wafer|waffeln?|crisp(?:ies)?|"
            r"galleta|crunch(?:y)?|cereal[ies]?|riso\s+soffiato|"
            r"puffed\s+rice|reis(?:knusper)?)\b",
            re.I), "with_cookie"),
        # fruit / berries / orange / raisin
        (re.compile(
            r"\b(?:fruit[s]?|frutt[ai]|frut[as]?|fr[üu]chte?|"
            r"raisin[s]?|rosinen|uvetta|pasas|"
            r"orange|arancia|naranja|"
            r"berry|berries|baies|beeren|bacche|bayas|"
            r"strawberry|fraise|erdbeer|fragol[ae]|fresa|"
            r"cherry|cerise|kirsche|ciliegi[ae]|cereza|"
            r"cranberr(?:y|ies)|"
            r"abricot|aprikose|albicocch?[ae]|albaricoque|"
            r"banana|banane|"
            r"lemon|citron|zitrone|limone|lim[óo]n|"
            r"mint|menthe|minze|menta|hierbabuena)\b",
            re.I), "with_fruit"),
    ]

    def extract_contains_nuts(self, text: str) -> ExtractionResult:
        if self.NUT_RE.search(text):
            return ExtractionResult(value=True, confidence=0.95, source="regex_keyword")
        return ExtractionResult(value=None, confidence=0.0, source="regex_keyword")

    def extract_chocolate_extra(self, text: str) -> ExtractionResult:
        """Определить тип добавки в шоколаде (with_nuts/with_fruit/...).

        ВАЖНО: как и `chocolate_type`, требует **сужённого** текста
        (product_name + brands + quantity, без ingredients_text), иначе
        примеси-маркеры из состава (изюм/орехи/печенье как ингредиенты
        несвязанных вкусов) дают 37 % точности regex. На однозначном
        product_name точность поднимается до приемлемого уровня.

        Дополнительно: при одновременном матче нескольких категорий
        (например, и nuts, и caramel в названии) — abstain.
        """
        matched: list[str] = []
        if self.NUT_RE.search(text):
            matched.append("with_nuts")
        for pattern, extra in self.CHOCOLATE_EXTRA_PATTERNS:
            if pattern.search(text) and extra not in matched:
                matched.append(extra)
        if len(matched) != 1:
            return ExtractionResult(value=None, confidence=0.0, source="regex")
        return ExtractionResult(value=matched[0], confidence=0.85, source="regex")

    def extract_beverage_type(self, text: str) -> ExtractionResult:
        text_lower = text.lower()
        keywords = {
            "water": ["water", "eau", "agua", "wasser", "acqua", "mineral"],
            "juice": ["juice", "jus", "zumo", "succo", "saft"],
            "soda": ["soda", "cola", "lemonade", "fizz"],
            "tea": ["tea", "thé", " the ", "tee", " tè ", " té "],
            "coffee": ["coffee", "café", " cafe ", "kaffee", "caffè"],
            "sport": ["energy drink", "sport drink", "isotonic"],
        }
        for btype, kws in keywords.items():
            for kw in kws:
                if kw in text_lower:
                    return ExtractionResult(value=btype, confidence=0.85, source="regex_keyword")
        return ExtractionResult(value=None, confidence=0.0, source="regex_keyword")

    def extract_is_carbonated(self, text: str) -> ExtractionResult:
        kws = ["sparkling", "carbonated", "fizzy", "gazeuse", "gasificada", "kohlensäure", "frizzante"]
        text_lower = text.lower()
        if any(kw in text_lower for kw in kws):
            return ExtractionResult(value=True, confidence=0.9, source="regex_keyword")
        return ExtractionResult(value=None, confidence=0.0, source="regex_keyword")

    def extract_caffeine_present(self, text: str) -> ExtractionResult:
        # caffeine indicators in name/ingredients
        kws = ["caffeine", "caffeinated", "cola", "coffee", "café", "espresso", "energy"]
        text_lower = text.lower()
        if any(kw in text_lower for kw in kws):
            return ExtractionResult(value=True, confidence=0.85, source="regex_keyword")
        return ExtractionResult(value=None, confidence=0.0, source="regex_keyword")

    def extract_all(self, product_name: str, description: str = "",
                    quantity: str = "", category: str = "baby",
                    brands: str = "", ingredients_text: str = "") -> dict:
        """Extract all regex-based attributes from product text.

        Для chocolate type-attribute обрабатывается на сужённом тексте
        (product_name + brands + quantity), без ingredients_text — иначе
        молочные ингредиенты у тёмного шоколада маркируют его как milk
        (см. extract_chocolate_type).
        """
        full_text = f"{product_name} {description} {brands} {quantity} {ingredients_text}"
        name_text = f"{product_name} {brands} {quantity}"
        results = {
            "fat_content": self.extract_fat_content(full_text),
            "minimal_age": self.extract_minimal_age(full_text),
            "measure": self.extract_measure(quantity or full_text),
            "cooking_time": self.extract_cooking_time(full_text),
        }
        if category == "pasta":
            results["grain_type"] = self.extract_grain_type(full_text)
            results["pasta_shape"] = self.extract_pasta_shape(full_text)
        elif category == "chocolate":
            results["cocoa_percentage"] = self.extract_cocoa_percentage(full_text)
            # chocolate_type — только product_name+brands+quantity (без ingredients):
            # молочные ингредиенты у dark/filled тянут точность regex до 81 %.
            results["chocolate_type"] = self.extract_chocolate_type(name_text)
            results["contains_nuts"] = self.extract_contains_nuts(full_text)
            # chocolate_extra намеренно НЕ извлекается через Layer 1: даже на
            # сужённом тексте regex успевает «снять» простые ячейки, которые
            # ML обрабатывал с 99 % точностью, оставив Layer 2 более трудный
            # остаток (acc падает с 99 % до 75 %). ML-слой превосходит regex
            # на этом атрибуте; метод extract_chocolate_extra доступен, но
            # не вызывается из extract_all.
        elif category == "beverages":
            results["beverage_type"] = self.extract_beverage_type(full_text)
            results["is_carbonated"] = self.extract_is_carbonated(full_text)
            results["caffeine_present"] = self.extract_caffeine_present(full_text)
        return results
