"""Tests for src.pipeline.bayes.validate on a hand-built network."""
from __future__ import annotations

import math

import pandas as pd
import pytest
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from src.pipeline.bayes.validate import (
    attribute_likelihood,
    _clean_evidence,
)


@pytest.fixture
def chocolate_like_net():
    """White chocolate cannot have cocoa>30%; dark almost always has cocoa>50%."""
    m = DiscreteBayesianNetwork([("chocolate_type", "cocoa_percentage")])
    m.add_cpds(
        TabularCPD(
            variable="chocolate_type",
            variable_card=2,
            values=[[0.5], [0.5]],
            state_names={"chocolate_type": ["white", "dark"]},
        ),
        TabularCPD(
            variable="cocoa_percentage",
            variable_card=3,
            # rows = states of cocoa_percentage; cols = states of chocolate_type
            values=[
                [0.95, 0.02],  # 0-30
                [0.04, 0.18],  # 30-70
                [0.01, 0.80],  # 70+
            ],
            evidence=["chocolate_type"],
            evidence_card=[2],
            state_names={
                "cocoa_percentage": ["0-30", "30-70", "70+"],
                "chocolate_type": ["white", "dark"],
            },
        ),
    )
    return m


def test_likelihood_high_for_consistent_pair(chocolate_like_net):
    inference = VariableElimination(chocolate_like_net)
    p = attribute_likelihood(
        attr="cocoa_percentage",
        value=80,
        evidence={"chocolate_type": "dark"},
        bayes_model=chocolate_like_net,
        inference=inference,
    )
    assert p == pytest.approx(0.80, abs=1e-6)


def test_likelihood_low_for_inconsistent_pair(chocolate_like_net):
    inference = VariableElimination(chocolate_like_net)
    p = attribute_likelihood(
        attr="cocoa_percentage",
        value=80,
        evidence={"chocolate_type": "white"},
        bayes_model=chocolate_like_net,
        inference=inference,
    )
    assert p == pytest.approx(0.01, abs=1e-6)


def test_likelihood_for_unknown_attr_returns_none(chocolate_like_net):
    inference = VariableElimination(chocolate_like_net)
    p = attribute_likelihood(
        attr="not_a_node",
        value="anything",
        evidence={},
        bayes_model=chocolate_like_net,
        inference=inference,
    )
    assert p is None


def test_likelihood_for_unbucketizable_value_returns_none(chocolate_like_net):
    inference = VariableElimination(chocolate_like_net)
    p = attribute_likelihood(
        attr="chocolate_type",
        value="неизвестный_тип",  # cyrillic not in support
        evidence={},
        bayes_model=chocolate_like_net,
        inference=inference,
    )
    assert p is None


def test_marginal_when_evidence_empty(chocolate_like_net):
    """With prior 0.5/0.5 on chocolate_type, P(cocoa=70+) = 0.5*0.01+0.5*0.80 = 0.405."""
    inference = VariableElimination(chocolate_like_net)
    p = attribute_likelihood(
        attr="cocoa_percentage",
        value=80,
        evidence={},
        bayes_model=chocolate_like_net,
        inference=inference,
    )
    assert p == pytest.approx(0.405, abs=1e-6)


def test_clean_evidence_drops_unknown_keys(chocolate_like_net):
    cleaned = _clean_evidence(
        {"chocolate_type": "dark", "not_a_node": "junk"},
        chocolate_like_net,
    )
    assert cleaned == {"chocolate_type": "dark"}


def test_clean_evidence_drops_unbucketizable_values(chocolate_like_net):
    cleaned = _clean_evidence(
        {"chocolate_type": "totallymadeup"},
        chocolate_like_net,
    )
    assert cleaned == {}


from src.pipeline.bayes.validate import top_contributors_pmi


def test_pmi_negative_when_evidence_contradicts(chocolate_like_net):
    """White⇒P(cocoa=70+)=0.01; without it, marginal=0.405.
    PMI = log(0.01/0.405) ≈ −3.70 (strongly negative — chocolate_type pushes
    down)."""
    inference = VariableElimination(chocolate_like_net)
    contribs = top_contributors_pmi(
        attr="cocoa_percentage",
        value=80,
        evidence={"chocolate_type": "white"},
        bayes_model=chocolate_like_net,
        inference=inference,
        k=2,
    )
    assert len(contribs) == 1
    assert contribs[0]["attr"] == "chocolate_type"
    assert contribs[0]["value"] == "white"
    assert contribs[0]["pmi"] == pytest.approx(math.log(0.01 / 0.405), abs=1e-4)


def test_pmi_positive_when_evidence_supports(chocolate_like_net):
    inference = VariableElimination(chocolate_like_net)
    contribs = top_contributors_pmi(
        attr="cocoa_percentage",
        value=80,
        evidence={"chocolate_type": "dark"},
        bayes_model=chocolate_like_net,
        inference=inference,
        k=2,
    )
    # log(0.80/0.405) ≈ +0.68 (positive — dark supports cocoa>=70).
    assert contribs[0]["pmi"] == pytest.approx(math.log(0.80 / 0.405), abs=1e-4)


def test_pmi_returns_top_k_most_negative(chocolate_like_net):
    """Contradicting evidence dominates supporting evidence in the ranking."""
    inference = VariableElimination(chocolate_like_net)
    contribs = top_contributors_pmi(
        attr="cocoa_percentage",
        value=80,
        evidence={"chocolate_type": "white"},
        bayes_model=chocolate_like_net,
        inference=inference,
        k=1,
    )
    assert len(contribs) == 1
    assert contribs[0]["pmi"] < 0


def test_pmi_empty_evidence_returns_empty(chocolate_like_net):
    inference = VariableElimination(chocolate_like_net)
    contribs = top_contributors_pmi(
        attr="cocoa_percentage",
        value=80,
        evidence={},
        bayes_model=chocolate_like_net,
        inference=inference,
        k=2,
    )
    assert contribs == []


from src.pipeline.bayes.validate import shapley_attribution


def test_shapley_efficiency_axiom_single_evidence(chocolate_like_net):
    """For n=1, Shapley = log P(v|e) - log P(v) for the single evidence var."""
    inference = VariableElimination(chocolate_like_net)
    res = shapley_attribution(
        attr="cocoa_percentage",
        value=80,
        evidence={"chocolate_type": "white"},
        bayes_model=chocolate_like_net,
        inference=inference,
        monte_carlo_samples=None,  # exact
    )
    # Single-evidence Shapley = full diff.
    assert "attribution" in res
    assert len(res["attribution"]) == 1
    diff = math.log(0.01) - math.log(0.405)
    assert res["attribution"][0]["shapley"] == pytest.approx(diff, abs=1e-6)
    assert res["sum_shapley"] == pytest.approx(diff, abs=1e-6)
    assert res["log_likelihood_diff"] == pytest.approx(diff, abs=1e-6)
    # Efficiency axiom: sum == log diff.
    assert abs(res["sum_shapley"] - res["log_likelihood_diff"]) < 1e-6


def test_shapley_sampled_runs_without_error(chocolate_like_net):
    inference = VariableElimination(chocolate_like_net)
    res = shapley_attribution(
        attr="cocoa_percentage",
        value=80,
        evidence={"chocolate_type": "white"},
        bayes_model=chocolate_like_net,
        inference=inference,
        monte_carlo_samples=20,
    )
    assert len(res["attribution"]) == 1
    assert isinstance(res["sum_shapley"], float)


from src.pipeline.bayes.validate import brand_status, calibrate_thresholds


@pytest.fixture
def net_with_brand():
    m = DiscreteBayesianNetwork()
    m.add_node("brand")
    m.add_cpds(
        TabularCPD(
            variable="brand",
            variable_card=3,
            values=[[0.4], [0.4], [0.2]],
            state_names={"brand": ["barilla", "lindt", "other"]},
        )
    )
    return m


def test_brand_status_known(net_with_brand):
    assert brand_status("barilla", net_with_brand) == "known"


def test_brand_status_ood(net_with_brand):
    assert brand_status("totally-new-brand", net_with_brand) == "ood"


def test_brand_status_no_brand_node():
    m = DiscreteBayesianNetwork()
    m.add_node("x")
    m.add_cpds(TabularCPD(variable="x", variable_card=2,
                          values=[[0.5], [0.5]],
                          state_names={"x": ["a", "b"]}))
    assert brand_status("anything", m) == "n/a"


def test_calibrate_thresholds_returns_per_attr_dict(chocolate_like_net):
    inference = VariableElimination(chocolate_like_net)
    df = pd.DataFrame({
        "chocolate_type": ["white", "white", "dark", "dark", "dark"],
        "cocoa_percentage": [20, 25, 75, 80, 90],
    })
    thresholds = calibrate_thresholds(
        chocolate_like_net, df, inference, q=0.5
    )
    # q=0.5 → median P. brand is excluded; only chocolate_type and cocoa_percentage.
    assert "chocolate_type" in thresholds
    assert "cocoa_percentage" in thresholds
    assert thresholds["cocoa_percentage"] > 0
    assert thresholds["chocolate_type"] > 0
