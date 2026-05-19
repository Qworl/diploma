"""Unit tests for src.pipeline.bayes.bucketize on synthetic networks."""
from __future__ import annotations

import pytest
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

from src.pipeline.bayes.bucketize import bucketize


def _net_with_numeric_bins() -> DiscreteBayesianNetwork:
    m = DiscreteBayesianNetwork()
    m.add_node("cocoa_percentage")
    cpd = TabularCPD(
        variable="cocoa_percentage",
        variable_card=5,
        values=[[0.2], [0.2], [0.2], [0.2], [0.2]],
        state_names={"cocoa_percentage": ["30-50", "50-70", "70-85", "85+", "other"]},
    )
    m.add_cpds(cpd)
    return m


def _net_with_bool() -> DiscreteBayesianNetwork:
    m = DiscreteBayesianNetwork()
    m.add_node("is_organic")
    cpd = TabularCPD(
        variable="is_organic",
        variable_card=2,
        values=[[0.5], [0.5]],
        state_names={"is_organic": ["False", "True"]},
    )
    m.add_cpds(cpd)
    return m


def _net_with_int_states() -> DiscreteBayesianNetwork:
    m = DiscreteBayesianNetwork()
    m.add_node("nova_group")
    cpd = TabularCPD(
        variable="nova_group",
        variable_card=4,
        values=[[0.25], [0.25], [0.25], [0.25]],
        state_names={"nova_group": ["0", "1", "3", "4"]},
    )
    m.add_cpds(cpd)
    return m


class TestNumericBinning:
    def test_value_in_low_bin(self):
        assert bucketize("cocoa_percentage", 40, _net_with_numeric_bins()) == "30-50"

    def test_value_at_bin_boundary_goes_to_upper_bin(self):
        # Convention: ranges are [lo, hi); 50 belongs to 50-70.
        assert bucketize("cocoa_percentage", 50, _net_with_numeric_bins()) == "50-70"

    def test_value_in_open_high_bin(self):
        assert bucketize("cocoa_percentage", 90, _net_with_numeric_bins()) == "85+"

    def test_value_in_70_85_bin_for_preset(self):
        # Demo preset cocoa_percentage=70 must reach the 70-85 state.
        assert bucketize("cocoa_percentage", 70, _net_with_numeric_bins()) == "70-85"

    def test_float_value(self):
        assert bucketize("cocoa_percentage", 72.5, _net_with_numeric_bins()) == "70-85"

    def test_string_numeric_value(self):
        assert bucketize("cocoa_percentage", "65", _net_with_numeric_bins()) == "50-70"

    def test_negative_value_returns_none(self):
        assert bucketize("cocoa_percentage", -5, _net_with_numeric_bins()) is None


class TestBoolNormalization:
    def test_python_true(self):
        assert bucketize("is_organic", True, _net_with_bool()) == "True"

    def test_python_false(self):
        assert bucketize("is_organic", False, _net_with_bool()) == "False"

    def test_str_true_capitalized(self):
        assert bucketize("is_organic", "True", _net_with_bool()) == "True"

    def test_str_true_lowercase(self):
        assert bucketize("is_organic", "true", _net_with_bool()) == "True"

    def test_int_one(self):
        assert bucketize("is_organic", 1, _net_with_bool()) == "True"

    def test_int_zero(self):
        assert bucketize("is_organic", 0, _net_with_bool()) == "False"

    def test_garbage_returns_none(self):
        assert bucketize("is_organic", "maybe", _net_with_bool()) is None


class TestIntStringStates:
    def test_int_in_supported_state(self):
        assert bucketize("nova_group", 3, _net_with_int_states()) == "3"

    def test_int_outside_states_returns_none(self):
        # 2 is not in {0,1,3,4} — gap in the trained model.
        assert bucketize("nova_group", 2, _net_with_int_states()) is None

    def test_string_match(self):
        assert bucketize("nova_group", "1", _net_with_int_states()) == "1"

    def test_integer_valued_float_matches_int_string_state(self):
        # Pandas often loads int columns as float64 (e.g. nova_group=4.0) when
        # NaNs are present, but training cast to int → str. bucketize must
        # accept 4.0 and map to state "4", not return None or "4.0".
        assert bucketize("nova_group", 4.0, _net_with_int_states()) == "4"

    def test_integer_valued_float_outside_states_returns_none(self):
        # 2.0 is integer-valued but state '2' doesn't exist in {0,1,3,4}.
        assert bucketize("nova_group", 2.0, _net_with_int_states()) is None

    def test_non_integer_float_returns_none_for_int_states(self):
        # 3.5 is not integer-valued and there are no range-bin states; no match.
        assert bucketize("nova_group", 3.5, _net_with_int_states()) is None


class TestUnknownNode:
    def test_attr_not_in_network_raises(self):
        m = _net_with_bool()
        with pytest.raises(KeyError):
            bucketize("nonexistent_attr", True, m)
