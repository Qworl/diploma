"""Map values (int/float/bool/str) to state-names of a trained DiscreteBayesianNetwork.

Trained Bayes networks (`models/{cat}_stratified_bayesian.pkl`) store node states
as strings. Numeric attributes (cocoa_percentage, fat_content, ...) are stored
as range labels like '30-50', '70-85', '85+'. Inference and validation must
project raw input values to those state names — otherwise lookups fail and the
validator produces false flags.
"""
from __future__ import annotations

import re

_RANGE_RE = re.compile(r"^(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)$")
_OPEN_HIGH_RE = re.compile(r"^(\d+(?:\.\d+)?)\+$")
_OPEN_LOW_RE = re.compile(r"^<(\d+(?:\.\d+)?)$")


def _bin_matches(state: str, v: float) -> bool:
    """Check whether numeric value v falls into the bin labelled `state`."""
    m = _RANGE_RE.match(state)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return lo <= v < hi
    m = _OPEN_HIGH_RE.match(state)
    if m:
        return v >= float(m.group(1))
    m = _OPEN_LOW_RE.match(state)
    if m:
        return v < float(m.group(1))
    return False


def bucketize(attr: str, value, bayes_model) -> str | None:
    """Project value to the state-name used for attr in bayes_model.

    Returns the state-name string on success, or None when the value cannot be
    placed in any state (out-of-domain input). Raises KeyError if attr is not
    a node of the network.
    """
    try:
        cpd = bayes_model.get_cpds(attr)
    except (ValueError, KeyError) as exc:
        raise KeyError(f"{attr!r} is not a node of the supplied network") from exc
    if cpd is None:
        raise KeyError(f"{attr!r} is not a node of the supplied network")
    states = [str(s) for s in cpd.state_names[attr]]
    is_bool_domain = set(states) == {"True", "False"}

    # 1. Direct string match (categorical, "True"/"False", "0"/"1"/...).
    if str(value) in states:
        return str(value)

    # 1b. Integer-valued float (e.g. nova_group=4.0) against int-string states
    #     (e.g. '4'). Pandas often loads small-integer columns as float when
    #     missing values are present, but training quantized to int → str.
    if isinstance(value, float) and not isinstance(value, bool):
        if value.is_integer():
            int_str = str(int(value))
            if int_str in states:
                return int_str

    # 2. Bool / int-zero-one normalization.
    if isinstance(value, bool):
        for variant in (str(value), str(value).lower(), str(int(value))):
            if variant in states:
                return variant
        return None

    # 3. String bool-aliases ("true"/"false", case-insensitive) for bool domains.
    if is_bool_domain and isinstance(value, str):
        low = value.strip().lower()
        if low == "true":
            return "True"
        if low == "false":
            return "False"
        return None

    # 4. Int 0/1 against bool domain → False/True.
    if is_bool_domain and isinstance(value, int):
        if value == 1:
            return "True"
        if value == 0:
            return "False"
        return None

    # 5. Numeric binning into "lo-hi" / "lo+" / "<hi" labels.
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v < 0:
        return None
    for state in states:
        if _bin_matches(state, v):
            return state
    return None
