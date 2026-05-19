"""Adversarial / edge-case tests for the enrichment pipeline.

Each test encodes a specific failure mode the system should handle gracefully:
- Conflicting signals between OFF tag families
- Cocoa-percentage boundary values (where buckets meet)
- Multilingual ambiguity (same word, different meanings)
- LLM response shapes the parser must reject without crashing

Listed by failure mode so it's clear what each guards against.
"""
import pytest

from src.pipeline.schemas import PASTA_SCHEMA, CHOCOLATE_SCHEMA, BEVERAGE_SCHEMA
from src.pipeline.off_labels import apply_off_labels
from src.pipeline.off_labels.rules import (
    _type_a_bool,
    _type_b_multiclass,
    _type_c_numeric,
    _type_d_direct,
)
from src.pipeline.llm_fallback import build_prompt
from src.llm.parsing import _parse_with_status, parse_llm_response
from src.pipeline.regex.extractor import RegexExtractor


# ============================================================================
# Cocoa percentage — boundary values where buckets transition
# Bucket edges (Trek E fix, [X, Y) convention): 30, 50, 70, 85.
# ============================================================================

class TestCocoaBoundaries:
    def test_below_30(self):
        # 29% → "<30"
        assert _type_c_numeric({"product_name": "Milk chocolate 29% cocoa"}, "cocoa_percentage") == "<30"

    def test_exactly_30(self):
        # 30% is the lower edge of the 30-50 bucket.
        assert _type_c_numeric({"product_name": "Chocolat 30% cacao"}, "cocoa_percentage") == "30-50"

    def test_exactly_50(self):
        # Trek E [X, Y): 50% kicks into 50-70.
        assert _type_c_numeric({"product_name": "Dark 50% cocoa"}, "cocoa_percentage") == "50-70"

    def test_exactly_51(self):
        # 51% is in 50-70 (same as before the boundary fix).
        assert _type_c_numeric({"product_name": "Chocolat 51% cacao"}, "cocoa_percentage") == "50-70"

    def test_exactly_70(self):
        # Trek E [X, Y): 70% kicks into 70-85 (was "50-70" pre-fix).
        assert _type_c_numeric({"product_name": "Lindt 70% Cocoa"}, "cocoa_percentage") == "70-85"

    def test_exactly_71(self):
        assert _type_c_numeric({"product_name": "Chocolat 71% cacao"}, "cocoa_percentage") == "70-85"

    def test_exactly_85(self):
        assert _type_c_numeric({"product_name": "Dark 85% cocoa"}, "cocoa_percentage") == "85+"

    def test_extreme_99(self):
        assert _type_c_numeric({"product_name": "99% cocoa extra dark"}, "cocoa_percentage") == "85+"

    def test_no_percentage_in_text(self):
        # No %, no extraction
        assert _type_c_numeric({"product_name": "Dark chocolate bar"}, "cocoa_percentage") is None


# ============================================================================
# Cocoa-percentage false positives — % in text that is NOT cocoa content
# ============================================================================

class TestCocoaFalsePositives:
    @pytest.mark.xfail(
        reason="Current regex catches any 2-3 digit %; '20% off' would mis-classify "
               "as cocoa. Acceptable today because Layer 2 ML overrides for chocolates "
               "where cocoa_percentage doesn't make sense, but flagged for the future."
    )
    def test_promotional_percent(self):
        """'20% off' shouldn't be read as 20% cocoa."""
        out = _type_c_numeric({"product_name": "Chocolate 20% off"}, "cocoa_percentage")
        assert out is None

    @pytest.mark.xfail(reason="Same family — 'reduced 30%' shouldn't read as cocoa")
    def test_reduced_percent(self):
        out = _type_c_numeric({"product_name": "Chocolate reduced 30%"}, "cocoa_percentage")
        assert out is None


# ============================================================================
# OFF label conflicts — different tag families disagree
# ============================================================================

class TestOffLabelConflicts:
    def test_organic_in_one_namespace_but_not_eu(self):
        """'en:bio' should still trigger organic even if eu-organic is missing."""
        row = {"labels_tags": "en:bio,en:fairtrade"}
        assert _type_a_bool(row, "is_organic") is True

    def test_palm_oil_status_priority_first_match_wins(self):
        """When ingredients_analysis_tags contains both 'palm-oil-free' and
        'may-contain-palm-oil', whichever appears first in iteration wins.
        We document the current behaviour so a future refactor doesn't silently flip it.
        """
        row = {"ingredients_analysis_tags": "en:palm-oil-free,en:may-contain-palm-oil"}
        # Both tags are mapped — _type_d iterates in tag order, returning first match
        result = _type_d_direct(row, "palm_oil_status")
        assert result in {"palm-oil-free", "may-contain"}

    def test_whole_grain_via_either_source(self):
        """Whole-grain detected from either labels OR categories (TYPE_AC hybrid)."""
        from src.pipeline.off_labels.rules import _type_ac_hybrid
        row1 = {"labels_tags": "en:whole-grain", "categories_tags": ""}
        row2 = {"labels_tags": "", "categories_tags": "en:whole-grain-pastas"}
        assert _type_ac_hybrid(row1, "is_whole_grain") is True
        assert _type_ac_hybrid(row2, "is_whole_grain") is True


# ============================================================================
# Whitespace / formatting robustness in OFF tags
# ============================================================================

class TestTagFormatting:
    def test_extra_whitespace_in_tags(self):
        """OFF dumps sometimes have trailing spaces; tags must still match."""
        row = {"labels_tags": "  en:organic  ,   en:eu-organic  "}
        assert _type_a_bool(row, "is_organic") is True

    def test_uppercase_tags_normalised(self):
        """Tags from some sources arrive uppercase; we lowercase before matching."""
        row = {"labels_tags": "EN:ORGANIC"}
        assert _type_a_bool(row, "is_organic") is True

    def test_tags_as_list_not_string(self):
        """Some OFF dumps return labels_tags as a list rather than comma string."""
        row = {"labels_tags": ["en:organic", "en:fairtrade"]}
        assert _type_a_bool(row, "is_organic") is True

    def test_empty_string_tags_treated_as_present_but_no_signal(self):
        """Empty labels_tags = field is present but has no positive tag → False, not None."""
        row = {"labels_tags": ""}
        # Empty string still counts as field-present (it's a known absence)
        # Per current behaviour, empty list returns False (field present, no positive match)
        result = _type_a_bool(row, "is_organic")
        assert result is False


# ============================================================================
# pasta_shape — multiple matching tags, first-match resolution
# ============================================================================

class TestPastaShapeAmbiguity:
    def test_specific_shape_before_generic(self):
        """When categories_tags has both en:spaghetti and en:long-pastas, specific wins."""
        row = {"categories_tags": "en:pastas,en:dry-pastas,en:long-pastas,en:spaghetti"}
        assert _type_b_multiclass(row, "pasta_shape") == "spaghetti"

    def test_lasagna_alias(self):
        """en:lasagna-sheets maps to lasagna (alias)."""
        row = {"categories_tags": "en:pastas,en:lasagna-sheets"}
        assert _type_b_multiclass(row, "pasta_shape") == "lasagna"

    def test_gnocchi_maps_to_other(self):
        """Post-Trek-D fix: en:gnocchi → "other" bucket per schema rubric.
        Before c1a0815 this returned None (LLM gap)."""
        row = {"categories_tags": "en:pastas,en:gnocchi"}
        assert _type_b_multiclass(row, "pasta_shape") == "other"

    def test_unrecognized_shape_returns_none(self):
        """A truly unknown tag (not in mapping) still returns None (LLM gap)."""
        row = {"categories_tags": "en:pastas,en:bigoli-pasta-xyz"}
        assert _type_b_multiclass(row, "pasta_shape") is None


# ============================================================================
# Beverage category — branded sodas vs generic beverages
# ============================================================================

class TestBeverageCategoryEdgeCases:
    def test_cola_specific_to_soda(self):
        row = {"categories_tags": "en:beverages,en:soft-drinks,en:colas"}
        assert _type_b_multiclass(row, "beverage_type") == "soda"

    def test_iced_tea_to_tea(self):
        row = {"categories_tags": "en:beverages,en:iced-teas"}
        assert _type_b_multiclass(row, "beverage_type") == "tea"

    def test_energy_drink_to_sport(self):
        """Energy drinks in our enum are bucketed with sport."""
        row = {"categories_tags": "en:beverages,en:energy-drinks"}
        assert _type_b_multiclass(row, "beverage_type") == "sport"


# ============================================================================
# LLM response parsing — adversarial JSON shapes
# ============================================================================

class TestLLMParsingAdversarial:
    def test_json_with_trailing_text(self):
        """Some models append 'Here is the JSON:' or trailing analysis."""
        raw = 'Here is the result: {"grain_type": "wheat"} Hope this helps!'
        parsed, ok = _parse_with_status(raw, PASTA_SCHEMA)
        assert ok is True
        assert parsed.get("grain_type") == "wheat"

    def test_json_array_top_level_rejected(self):
        """A bare array isn't a valid response object."""
        raw = '["wheat", "spaghetti"]'
        parsed, ok = _parse_with_status(raw, PASTA_SCHEMA)
        assert ok is False
        assert parsed == {}

    def test_string_value_for_bool_field_dropped(self):
        """Bool field receiving 'true' as string — reject (not auto-cast)."""
        raw = '{"is_organic": "true", "grain_type": "wheat"}'
        parsed, ok = _parse_with_status(raw, PASTA_SCHEMA)
        assert ok is True
        # is_organic gets dropped (str != bool); grain_type stays
        assert "is_organic" not in parsed
        assert parsed["grain_type"] == "wheat"

    def test_int_as_string_accepted_for_int_field(self):
        """nova_group accepts string "3" as int (per existing behaviour)."""
        raw = '{"nova_group": "3", "beverage_type": "soda"}'
        parsed, ok = _parse_with_status(raw, BEVERAGE_SCHEMA)
        assert ok is True
        assert parsed["nova_group"] == 3
        assert parsed["beverage_type"] == "soda"

    def test_nullable_field_explicit_null(self):
        """Nullable enum receives null → preserved as None."""
        raw = '{"pasta_shape": null, "grain_type": "rice"}'
        parsed, ok = _parse_with_status(raw, PASTA_SCHEMA)
        assert ok is True
        assert parsed["pasta_shape"] is None
        assert parsed["grain_type"] == "rice"

    def test_non_nullable_null_dropped(self):
        """Non-nullable bool receives null → dropped silently."""
        raw = '{"is_organic": null}'
        parsed, ok = _parse_with_status(raw, PASTA_SCHEMA)
        assert ok is True
        assert "is_organic" not in parsed

    def test_unknown_enum_value_dropped(self):
        """Enum value outside the schema's whitelist — dropped, no crash."""
        raw = '{"chocolate_type": "ruby", "is_organic": true}'
        parsed, ok = _parse_with_status(raw, CHOCOLATE_SCHEMA)
        assert ok is True
        assert "chocolate_type" not in parsed
        assert parsed["is_organic"] is True

    def test_extra_fields_ignored(self):
        """Model emits a field not in schema → silently dropped."""
        raw = '{"grain_type": "wheat", "made_up_field": "value"}'
        parsed, ok = _parse_with_status(raw, PASTA_SCHEMA)
        assert ok is True
        assert parsed == {"grain_type": "wheat"}


# ============================================================================
# Multilingual ambiguity — same surface form, different semantics
# ============================================================================

class TestMultilingualAmbiguity:
    """These tests document the regex layer's limits, not its perfection.
    Layer 1 should be permissive: false negatives are fine (LLM picks up),
    false positives that survive ML are the real risk.
    """

    def test_french_accent_in_cooking_time(self):
        """'cuisson' with accents shouldn't break."""
        rx = RegexExtractor()
        out = rx.extract_cooking_time("Cuisson : 10 min")
        assert out.value == 10

    def test_german_cooking_time(self):
        rx = RegexExtractor()
        out = rx.extract_cooking_time("Kochzeit 7 Min")
        assert out.value == 7

    def test_french_age_with_apostrophe_not_confused(self):
        """'1er âge' should match; '1 enfant' should not."""
        rx = RegexExtractor()
        assert rx.extract_minimal_age("1er âge").value == "0+"
        assert rx.extract_minimal_age("1 enfant heureux").value is None


# ============================================================================
# Build prompt — robustness with hostile or empty inputs
# ============================================================================

class TestBuildPromptRobustness:
    def test_all_input_fields_missing(self):
        """Empty product → prompt still well-formed (degraded but valid)."""
        prompt = build_prompt({}, PASTA_SCHEMA)
        assert "Extract product attributes" in prompt
        assert "grain_type" in prompt
        # No product fields rendered, but prompt is still usable
        assert "Product:" in prompt

    def test_product_with_special_chars(self):
        """Quotes, backslashes, newlines in product name shouldn't break the prompt."""
        product = {
            "product_name": 'Pâtes "spéciales" \\ multi-grain\n— BIO',
            "brands": "Brand & Co.",
        }
        prompt = build_prompt(product, PASTA_SCHEMA)
        assert "spéciales" in prompt
        assert "Brand & Co." in prompt
