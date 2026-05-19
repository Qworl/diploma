"""
Rule-based classifier for cereals/cereal_type using full OFF context.

В отличие от silver labeler (`apply_off_labels`), использует:
- generic_name (часто содержит structured описание: "sweetened puffed wheat cereal")
- categories_tags (en:granolas, en:mueslis, en:corn-flakes, en:puffed-rice)
- product_name (последний fallback)
- ingredients_text (для tie-break по dominant grain)

Это **детерминированный код** (regex + lookup), не LLM. Воспроизводимо,
прозрачно, объяснимо на защите. Если accuracy ≥ 85% против ручного
арбитража — cereal_type переводится из CONSENSUS_NEEDED → TEXT_REGEX
(gold tier по таксономии §6.12.1).

Output: добавляет колонку `auto_arbitrage` в arbitrage_cereals_cereal_type.csv.
"""
from __future__ import annotations

import argparse
import logging
import os
import re

import pandas as pd

from src.common import setup_logging

logger = logging.getLogger(__name__)


def _t(s) -> str:
    return str(s or "").lower()


_RE_CHOCO_WORD = r"(?:choco|cocoa|chocolat|kakao|schoko|schokolade|nesquik|nougat|cacao)"
_RE_WHEAT     = r"(?:wheat|bl[ée]|frumento|trigo|weizen|froment|spelt|épeautre|spelta|dinkel)"
_RE_OAT       = r"(?:oat|avoine|avena|hafer|haferflocken|haferfleks|haferpops)"
_RE_RICE      = r"(?:rice|riz|riso|reis|arroz)"
_RE_CORN      = r"(?:corn|ma[iï]s|maíz|maíz|granmais)"
_RE_OTHER_GRAIN = r"(?:quinoa|amaranth|amaranto|buchweizen|sarrazin|millet|miglio|hirse|seigle|rye|orzo|barley|cebada|cebada)"
_RE_FRUIT     = r"(?:fruit|fruits|berry|berries|frutto|frutta|baies|fr(?:ü|u)cht|red[ -]?fruit)"

# Branded chocolate cereals — детская сладкая каша с какао/шоколадом
_RE_BRANDED_CHOCO = re.compile(
    r"(?:chocapic|nesquik|cini[ \-]?minis?|kakao[ \-]?(?:düsis|pops|cushi)"
    r"|kriskao|chocoleo|cocoa[ \-]?(?:krispies|puffs|pebbles)|cocoa[ \-]?(?:pebbles)"
    r"|coco[ \-]?pops|cookie[ \-]?crisp|krave|reese.s[ \-]?puffs)",
    re.IGNORECASE,
)


def classify_cereal_type(row: dict) -> tuple[str | None, str]:
    """
    Returns (label, reason). label is None when we're not confident enough.
    """
    pn = _t(row.get("product_name"))
    gn = _t(row.get("generic_name"))
    ct = _t(row.get("categories_tags"))
    ing = _t(row.get("ingredients_text"))
    full = f"{pn} | {gn}"
    # Topful содержит первые ~120 символов ingredients (главные/доминирующие
    # ингредиенты), используется для chocolate-cereal detection через состав.
    ing_top = ing[:150]

    # === Layer 0: non-cereal products в датасете (OFF misclassification) ===
    # Mука/farinas/Mehl, oils, и т.д. — это не breakfast cereal, тег
    # en:breakfast-cereals в OFF иногда ошибочный.
    if any(x in ct for x in ["en:flours", "en:cereal-flours", "en:cornmeal",
                              "en:wheat-flours", "en:rice-flours", "en:vegetable-oils"]):
        return "other", "cat:non-cereal-product"

    # === Layer 1: OFF categories_tags ===
    if "en:chocolate-cereals" in ct or "en:cocoa-cereals" in ct or "en:chocolate-breakfast-cereals" in ct:
        return "chocolate_cereal", "cat:chocolate-cereals"

    # === Layer 1.5: chocolate keyword anywhere → chocolate_cereal (приоритет
    # перед muesli/granola — кейсы вида "Schoko Müsli", "Granola Chocolate"
    # классифицируются как chocolate_cereal, потому что шоколад — главный
    # отличительный признак продукта). ===
    if re.search(_RE_CHOCO_WORD, full):
        return "chocolate_cereal", "kw:choco-priority"
    # Chocolate-rich продукт без слова "choco" в названии (e.g. Chejoy: brand
    # name only, но 35% chocolate cream в ingredients). Если шоколад/какао
    # в первых ~150 символах ingredients — это chocolate cereal по сути.
    #
    # НО: если в product_name явно указан specific cereal type (Corn Flakes,
    # Müsli, Granola, Oat Cereal, Cheerios), то chocolate в ingredients —
    # это add-in (chocolate drops), не base. Trust the marketing name.
    has_specific_type_in_name = re.search(
        r"\b(corn[ \-]?flakes?|cornflakes|m(?:ue|ü|u)sli|granola|"
        r"oat[ \-]?(?:cereal|flakes|rings|crisp)|cheerios|"
        r"puffed[ \-]?(?:rice|wheat)|cornflocken)\b", full)
    if not has_specific_type_in_name and \
            re.search(r"(?:cioccolat|chocolat|cacao|kakao|schoko)", ing_top):
        return "chocolate_cereal", "ing:choco-in-top-ingredients"

    # Branded chocolate cereals без word "choco" в названии (NESTLE CRUNCH,
    # NESTLE LION, KIT KAT cereal, Coco Pops, и т.д.)
    if _RE_BRANDED_CHOCO.search(full):
        return "chocolate_cereal", "branded:choco"
    if re.search(r"\b(?:nestle|nestl[eé])\s*(?:crunch|lion|kit[ -]?kat)", full):
        return "chocolate_cereal", "branded:nestle-choco"
    if re.search(r"\b(?:trix|froot[ -]?loops|fruity[ -]?pebbles)\b", full):
        # Эти бренды — fruit-flavored, но в schema нет fruit_cereal → other
        # (или mixed если базовая каша multigrain)
        return "other", "branded:fruit-flavored"

    # === Layer 2: остальные категории tags ===
    if "en:granolas" in ct or "en:granola" in ct:
        return "granola", "cat:granolas"
    if "en:mueslis" in ct or "en:muesli" in ct:
        return "muesli", "cat:mueslis"
    if "en:corn-flakes" in ct or "en:cornflakes" in ct:
        return "corn_flakes", "cat:corn-flakes"
    if "en:puffed-rice" in ct or "en:rice-cereals" in ct:
        return "puffed_rice", "cat:puffed-rice"
    if "en:oat-flakes" in ct or "en:porridges" in ct or "en:oat-cereals" in ct:
        return "oat_cereal", "cat:oat-cereal"

    # === Layer 4: granola ===
    # "crunchy" в cereals домене — granola signature (baked sweetened oats with oil/sugar).
    # Berücksichtigt: "Bio-Hafer Crunchy", "Country Crisp", "Crunchy Müsli with chocolate".
    # Если "crunchy" + "muesli" → muesli (см. Layer 5). Иначе crunchy = granola.
    if "granola" in full or "country crisp" in full:
        return "granola", "kw:granola"
    if re.search(r"\bcrunchy\b|\bknusprig", full) and not re.search(r"m(?:ue|ü|u)sli", full):
        return "granola", "kw:crunchy"

    # === Layer 5: muesli (всякое müsli/musli/musli — самый частый кейс) ===
    if re.search(r"m(?:ue|ü|u)sli", full):
        return "muesli", "kw:muesli"

    # === Layer 6: corn flakes ===
    if re.search(r"corn[ \-]?flakes?", full) or "cornflakes" in full:
        return "corn_flakes", "kw:corn-flakes"

    # === Layer 7: puffed rice / rice krispies / arroz/riso popcorn ===
    if re.search(rf"puffed[ \-]?{_RE_RICE}|{_RE_RICE}[ \-]?(?:pop|puff|cris|crisp|krisp)|"
                 r"rice[ \-]?krispies", full):
        return "puffed_rice", "kw:puffed-rice"
    if re.search(rf"{_RE_RICE}\s+soffiat|arroz\s+inflado|reis[ \-]?pops|reisflocken", full):
        return "puffed_rice", "kw:rice-puffed-translated"

    # === Layer 8: oat (cheerios, oat flakes, haferflocken, fiocchi avena, copos avena) ===
    if re.search(r"cheerios|haferfleks|haferpops|haferflocken|fiocchi[ \-]?d[ie][ \-]?avena|"
                 r"flocons[ \-]?d[ie][ \-]?avoine|copos[ \-]?de[ \-]?avena|"
                 r"crusca[ \-]?d[ie][ \-]?avena|oatmeal|porridge|knusprige[ \-]?bio[ \-]?hafer",
                 full):
        return "oat_cereal", "kw:oat-cereal"
    # Layer 8 НЕ включает "crunchy/crunch" в format-list — это granola signature,
    # уже отлавливается выше в Layer 4.
    if re.search(rf"{_RE_OAT}[ \-]?(?:cereal|flakes|rings|crisp|pops|biscuits)", full):
        return "oat_cereal", "kw:oat+format"

    # === Layer 9: multigrain / multi-cereal / multi-grain mix ===
    if re.search(r"multi[ \-]?(?:grain|cereal|gr[au]nos|gr[au]ni)|multicereal|cereal[ \-]?mix|"
                 r"misture[ \-]?cereal|m[eé]lange[ \-]?c[ée]r[ée]ales|mehrkorn", full):
        return "mixed", "kw:multigrain"
    # "rice and wheat", "wheat and oat", "frumento e avena" etc — 2+ зерна в названии
    grains_in_text = set()
    if re.search(_RE_OAT, full):   grains_in_text.add("oat")
    if re.search(_RE_WHEAT, full): grains_in_text.add("wheat")
    if re.search(_RE_RICE, full):  grains_in_text.add("rice")
    if re.search(_RE_CORN, full):  grains_in_text.add("corn")
    if len(grains_in_text) >= 2:
        return "mixed", f"kw:multi-grain-{'_'.join(sorted(grains_in_text))}"

    # === Layer 10: wheat / barley / rye / spelt / other-grain alone → other ===
    # (schema не имеет wheat_cereal класса)
    if re.search(rf"\b(puffed[ \-]?{_RE_WHEAT}|{_RE_WHEAT}[ \-]?(?:flakes|puffs|biscuits|crisp|"
                 r"bran|sticks|fl[oa]cons|fiocchi|flakes|copos|hojuelas|formelle|bisc?uits))",
                 full):
        return "other", "kw:wheat-no-class"
    if re.search(rf"{_RE_OTHER_GRAIN}[ \-]?(?:flakes|puffs|soffiat|pops|crisp|"
                 r"flocons|fiocchi|copos)", full):
        return "other", "kw:other-grain-no-class"
    # Pseudo-grains (amaranth, quinoa, buckwheat) в форме popped/poppies/
    # soffiato → other (нет в schema enum). Учитывает: amaranth-poppies,
    # quinoa soffiata, amaranthvollkorn gepoppt (в ingredients).
    pseudo_grain_re = r"\b(amaranth|amaranto|quinoa|buchweizen|sarrasin|grano\s+saraceno)"
    puffed_form_re = r"(?:popp|poppy|poppies|pops|puffed|soffiat|crisp|crunch|gepoppt|inflad|soffi)"
    if re.search(pseudo_grain_re, full + " " + ing_top):
        if re.search(puffed_form_re, full + " " + ing_top):
            return "other", "kw:pseudo-grain-puffed"

    # === Layer 11: special K, Fitness — wheat-based diet cereals → other ===
    if re.search(r"special[ \-]?k|fitness", pn) and re.search(_RE_FRUIT, full):
        return "other", "branded:wheat-diet"

    # === Layer 12: tie-break по dominant grain в ingredients_text ===
    if ing:
        m = re.match(r"^([0-9]+(?:[.,][0-9]+)?%?\s*)?(?:whole\s*grain\s+|integrale\s+|"
                     r"complet[ea]?\s+)?"
                     r"(oat|avena|hafer|avoine|wheat|bl[éaée]|frumento|trigo|weizen|"
                     r"rice|riz|riso|reis|arroz|"
                     r"corn|ma[iï]s|maize|maíz|"
                     r"barley|orzo|cebada|rye|seigle|spelt|épeautre|dinkel)",
                     ing.strip())
        if m:
            grain = m.group(2)
            if grain in ("oat", "avena", "hafer", "avoine"):
                return "oat_cereal", f"ing:{grain}-first"
            if grain in ("corn", "maïs", "mais", "maize", "maíz"):
                return "corn_flakes", f"ing:{grain}-first"
            if grain in ("rice", "riz", "riso", "reis", "arroz"):
                return "puffed_rice", f"ing:{grain}-first"
            # wheat-only → other (schema gap)
            return "other", f"ing:{grain}-first"

    return None, "unmatched"


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--csv",
                   default="datasets/manual_label/arbitrage_cereals_cereal_type.csv")
    p.add_argument("--overwrite-arbitrage", action="store_true",
                   help="Overwrite your_arbitrage even if non-empty (default: only fill empty)")
    args = p.parse_args()

    df = pd.read_csv(args.csv, dtype=str)
    for col in ("your_arbitrage", "note"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    labels: list[str | None] = []
    reasons: list[str] = []
    for _, row in df.iterrows():
        lab, reason = classify_cereal_type(row.to_dict())
        labels.append(lab)
        reasons.append(reason)

    df["auto_arbitrage"] = [l if l else "" for l in labels]
    df["auto_reason"] = reasons

    # Fill your_arbitrage из auto если пусто
    filled_before = (df["your_arbitrage"].str.strip() != "").sum()
    if args.overwrite_arbitrage:
        df.loc[df["auto_arbitrage"] != "", "your_arbitrage"] = df.loc[
            df["auto_arbitrage"] != "", "auto_arbitrage"
        ]
    else:
        empty = df["your_arbitrage"].str.strip() == ""
        df.loc[empty & (df["auto_arbitrage"] != ""), "your_arbitrage"] = df.loc[
            empty & (df["auto_arbitrage"] != ""), "auto_arbitrage"
        ]

    df.to_csv(args.csv, index=False, encoding="utf-8")
    filled_after = (df["your_arbitrage"].str.strip() != "").sum()

    print(f"Saved -> {args.csv}")
    print()
    print(f"Auto-classified: {(df['auto_arbitrage'] != '').sum()} / {len(df)} "
          f"({(df['auto_arbitrage'] != '').mean()*100:.0f}%)")
    print(f"your_arbitrage filled: {filled_before} → {filled_after} (+{filled_after-filled_before})")
    print()
    print("Distribution of auto labels:")
    print(df[df["auto_arbitrage"] != ""]["auto_arbitrage"].value_counts().to_string())
    print()
    print("Coverage by status (priority subsets):")
    for st in ["no_majority", "silver_diff", "silver_missing", "agree", "no_llm_data"]:
        sub = df[df["status"] == st]
        if len(sub) == 0:
            continue
        cov = (sub["auto_arbitrage"] != "").sum()
        print(f"  {st:<16} {cov:>4} / {len(sub):>4} auto-filled ({cov/len(sub)*100:.0f}%)")
    print()
    print("Agreement vs silver (where both present, no_llm_data excluded):")
    non_data = df[df["status"] != "no_llm_data"]
    has_silver = non_data[non_data["silver"].fillna("").str.strip() != ""]
    has_auto = has_silver[has_silver["auto_arbitrage"] != ""]
    if len(has_auto):
        agree = (has_auto["auto_arbitrage"] == has_auto["silver"].str.lower()).sum()
        print(f"  n={len(has_auto)}, agreement={agree/len(has_auto)*100:.1f}%")


if __name__ == "__main__":
    main()
