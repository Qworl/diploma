"""Tests for regex extractor."""
import pytest
from src.pipeline.regex.extractor import RegexExtractor

rx = RegexExtractor()


class TestFatContent:
    def test_explicit_fat(self):
        assert rx.extract_fat_content("Milk fat 3.2%").value == 3.2

    def test_russian_fat(self):
        assert rx.extract_fat_content("жирность 2,5%").value == 2.5

    def test_standalone_percent(self):
        assert rx.extract_fat_content("Lait 1.5%").value == 1.5

    def test_false_positive_fat_free(self):
        assert rx.extract_fat_content("99.9% fat free").value is None

    def test_false_positive_off(self):
        assert rx.extract_fat_content("50% off sale").value is None

    def test_false_positive_free(self):
        assert rx.extract_fat_content("100% lactose free").value is None

    def test_false_positive_discount(self):
        assert rx.extract_fat_content("Buy 2 get 30% discount").value is None


class TestMinimalAge:
    def test_plus_notation(self):
        assert rx.extract_minimal_age("Baby food 6+").value == "6+"

    def test_from_months(self):
        assert rx.extract_minimal_age("from 12 months").value == "12+"

    def test_russian(self):
        assert rx.extract_minimal_age("от 6 мес").value == "6+"

    def test_french_age(self):
        assert rx.extract_minimal_age("2ème âge").value == "6+"

    def test_stage(self):
        assert rx.extract_minimal_age("Stage 1 formula").value == "0+"

    def test_no_match(self):
        assert rx.extract_minimal_age("Regular milk").value is None


class TestMeasure:
    def test_grams(self):
        assert rx.extract_measure("200g").value == "200 g"

    def test_ml(self):
        assert rx.extract_measure("500 ml").value == "500 ml"

    def test_kg_decimal(self):
        assert rx.extract_measure("1,5 kg").value == "1.5 kg"


class TestCookingTime:
    def test_english(self):
        assert rx.extract_cooking_time("cooking time: 8 min").value == 8

    def test_russian(self):
        assert rx.extract_cooking_time("варить 10 мин").value == 10

    def test_french(self):
        assert rx.extract_cooking_time("cuisson 12 min").value == 12

    def test_german(self):
        assert rx.extract_cooking_time("Kochzeit 7 Minuten").value == 7

    def test_minutes_cook(self):
        assert rx.extract_cooking_time("10 minutes cooking").value == 10

    def test_no_match(self):
        assert rx.extract_cooking_time("Baby milk formula").value is None


class TestPastaExtractors:
    def test_grain_type_wheat(self):
        assert rx.extract_grain_type("Durum wheat spaghetti").value == "wheat"

    def test_grain_type_rice(self):
        assert rx.extract_grain_type("Rice noodles").value == "rice"

    def test_grain_type_buckwheat(self):
        assert rx.extract_grain_type("Soba buckwheat noodles").value == "buckwheat"

    def test_grain_type_french(self):
        assert rx.extract_grain_type("Pâtes au blé complet").value == "wheat"

    def test_grain_type_no_match(self):
        assert rx.extract_grain_type("Organic pasta").value is None

    def test_pasta_shape_spaghetti(self):
        assert rx.extract_pasta_shape("Barilla Spaghetti No.5").value == "spaghetti"

    def test_pasta_shape_penne(self):
        assert rx.extract_pasta_shape("Penne rigate bio").value == "penne"

    def test_pasta_shape_fusilli(self):
        assert rx.extract_pasta_shape("Fusilli tricolore").value == "fusilli"

    # Trek D audit findings — multilingual pasta_shape synonyms
    def test_pasta_shape_spanish_codo_maps_to_macaroni(self):
        assert rx.extract_pasta_shape("Codo No. 2 Barilla").value == "macaroni"

    def test_pasta_shape_spanish_coditos_maps_to_macaroni(self):
        assert rx.extract_pasta_shape("Coditos integrales").value == "macaroni"

    def test_pasta_shape_german_hoernchen_maps_to_macaroni(self):
        assert rx.extract_pasta_shape("Bio Hörnchen Vollkorn").value == "macaroni"

    def test_pasta_shape_spanish_tallarin_maps_to_tagliatelle(self):
        assert rx.extract_pasta_shape("Tallarín N°5").value == "tagliatelle"

    def test_pasta_shape_german_bandnudeln_maps_to_tagliatelle(self):
        assert rx.extract_pasta_shape("Bandnudeln 8mm").value == "tagliatelle"

    def test_pasta_shape_pappardelle_maps_to_tagliatelle(self):
        assert rx.extract_pasta_shape("Pappardelle aux oeufs").value == "tagliatelle"

    def test_pasta_shape_spanish_tirabuzon_maps_to_fusilli(self):
        assert rx.extract_pasta_shape("Tirabuzón Integral").value == "fusilli"

    def test_pasta_shape_german_spirelli_maps_to_fusilli(self):
        assert rx.extract_pasta_shape("Dinkel Spirelli Vollkorn").value == "fusilli"

    def test_pasta_shape_fideos_arroz_maps_to_vermicelli(self):
        assert rx.extract_pasta_shape("Fideos de arroz vermicelli").value == "vermicelli"

    def test_pasta_shape_cellophane_maps_to_vermicelli(self):
        assert rx.extract_pasta_shape("Cellophane noodles").value == "vermicelli"

    def test_pasta_shape_ramen_maps_to_noodles(self):
        assert rx.extract_pasta_shape("Kimchi Ramen instant").value == "noodles"

    def test_pasta_shape_udon_maps_to_noodles(self):
        assert rx.extract_pasta_shape("Udon thick wheat noodles").value == "noodles"

    def test_pasta_shape_tortelloni_maps_to_other(self):
        assert rx.extract_pasta_shape("Tortelloni Vier Käse").value == "other"

    def test_pasta_shape_spaetzle_maps_to_other(self):
        assert rx.extract_pasta_shape("Bio Spätzle Eier").value == "other"

    def test_pasta_shape_fleckerl_maps_to_other(self):
        assert rx.extract_pasta_shape("Bio Dinkel Fleckerl").value == "other"

    def test_pasta_shape_ditalini_maps_to_other(self):
        assert rx.extract_pasta_shape("Ditalini Rigati Barilla").value == "other"

    # Trek D audit findings — grain_type legume / non-cereal patterns
    def test_grain_type_chickpea_maps_to_other(self):
        assert rx.extract_grain_type("Torsades pois chiches").value == "other"

    def test_grain_type_lentil_maps_to_other(self):
        assert rx.extract_grain_type("Pasta de lentejas").value == "other"

    def test_grain_type_split_pea_maps_to_other(self):
        assert rx.extract_grain_type("Fusilli de pois cassés").value == "other"

    def test_grain_type_green_pea_german_maps_to_other(self):
        assert rx.extract_grain_type("Bio-Penne aus grünen Erbsen").value == "other"

    def test_grain_type_konjac_maps_to_other(self):
        assert rx.extract_grain_type("Konnyaku konjac noodles").value == "other"

    def test_grain_type_shirataki_maps_to_other(self):
        assert rx.extract_grain_type("Shirataki low-carb pasta").value == "other"

    def test_grain_type_legume_beats_wheat_keyword(self):
        """When both 'lentil flour' and 'wheat' appear, lentil should win
        because pulse pattern comes first in PATTERN order."""
        assert rx.extract_grain_type(
            "Lentil flour pasta with traces of wheat"
        ).value == "other"


class TestChocolateExtractors:
    def test_extract_cocoa_percentage_70(self):
        # Trek E [X, Y) convention: 70% → "70-85" (industry reading).
        out = rx.extract_all("Lindt Excellence 70% Dark", "", "", "chocolate")
        assert out["cocoa_percentage"].value == "70-85"

    def test_extract_cocoa_percentage_85(self):
        # 85% is canonically "85+" (matches Type C bucket boundary in llm_enricher)
        out = rx.extract_all("Hu Chocolate 85% Cacao", "", "", "chocolate")
        assert out["cocoa_percentage"].value == "85+"

    def test_extract_cocoa_percentage_75(self):
        out = rx.extract_all("Lindt 75% Cacao", "", "", "chocolate")
        assert out["cocoa_percentage"].value == "70-85"

    def test_extract_cocoa_percentage_above_85(self):
        out = rx.extract_all("Pure 90% dark chocolate", "", "", "chocolate")
        assert out["cocoa_percentage"].value == "85+"

    def test_extract_cocoa_percentage_low(self):
        out = rx.extract_all("Milk chocolate 25%", "", "", "chocolate")
        assert out["cocoa_percentage"].value == "<30"

    def test_extract_chocolate_type_dark(self):
        out = rx.extract_all("Côte d'Or Dark Chocolate", "", "", "chocolate")
        assert out["chocolate_type"].value == "dark"

    def test_extract_chocolate_type_milk_french(self):
        out = rx.extract_all("Lindt Lait Suisse", "", "", "chocolate")
        assert out["chocolate_type"].value == "milk"

    def test_extract_chocolate_type_white(self):
        out = rx.extract_all("Galak white chocolate", "", "", "chocolate")
        assert out["chocolate_type"].value == "white"

    def test_extract_chocolate_type_filled(self):
        out = rx.extract_all("Lindor truffle assortment", "", "", "chocolate")
        assert out["chocolate_type"].value == "filled"
