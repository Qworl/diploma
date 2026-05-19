"""Curate OFF API JSON for inclusion in blind Opus audit prompts.

Excludes OFF-derived classifications (nutriscore_grade, nova_group,
ingredients_analysis_tags, etc.) — these are themselves target attributes
in our schema and would directly leak answers.

Keeps raw OFF fields (text, tags, numeric nutriments, packaging, image
URLs) — what a human expert would read on the OFF product page.

See spec §3.2 for the boundary rationale.
"""
from __future__ import annotations

from typing import Any

# Top-level OFF fields that ARE target attributes — must NOT enter the prompt.
DERIVED_BLACKLIST: frozenset[str] = frozenset({
    "nutriscore_grade",
    "nutriscore_score",
    "nutriscore_data",
    "nova_group",
    "nova_groups_tags",
    "ecoscore_grade",
    "ecoscore_score",
    "ecoscore_data",
    "ingredients_analysis_tags",
})

# Inside `nutriments`, only numeric measurement columns are kept.
# Anything ending in _grade / _score / _label is OFF-derived classification.
_NUTRIMENT_DERIVED_SUFFIXES = ("_grade", "_score", "_label")
_NUTRIMENT_DERIVED_KEYS = frozenset({
    "nutriscore_grade",
    "nutriscore_score",
    "nova_group",
})


def _filter_nutriments(nut: dict[str, Any]) -> dict[str, Any]:
    """Keep only numeric measurement columns; drop derived classifications."""
    out = {}
    for k, v in nut.items():
        if k in _NUTRIMENT_DERIVED_KEYS:
            continue
        if any(k.endswith(suf) for suf in _NUTRIMENT_DERIVED_SUFFIXES):
            continue
        out[k] = v
    return out


def curate_prompt_fields(off_product: dict[str, Any]) -> dict[str, Any]:
    """Return a dict of fields safe to include in a blind audit prompt.

    Drops `DERIVED_BLACKLIST` keys. Filters nutriments to numeric only.
    Keeps text, tags, packaging, and image URLs. Sorts categories_tags
    for deterministic prompts. Does not mutate input.
    """
    out: dict[str, Any] = {}
    for k, v in off_product.items():
        if k in DERIVED_BLACKLIST:
            continue
        if k == "nutriments" and isinstance(v, dict):
            filtered = _filter_nutriments(v)
            if filtered:  # omit if empty
                out[k] = filtered
            continue
        if k == "categories_tags" and isinstance(v, list):
            out[k] = sorted(v)
            continue
        out[k] = v
    return out
