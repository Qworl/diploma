"""Canonical attribute definitions for the manual-labeling UI.

Reads schemas from src.pipeline.schemas.<domain> and normalises the
shape for UI rendering: every attribute exposes `type`, `values`,
`nullable`, `description`.
"""
from __future__ import annotations
import importlib
from copy import deepcopy

from src.pipeline.schemas.pasta import PASTA_SCHEMA


# Maps a UI/CLI domain key to the (module, attr-name) for the schema dict.
_DOMAIN_SCHEMAS: dict[str, tuple[str, str]] = {
    "pasta": ("src.pipeline.schemas.pasta", "PASTA_SCHEMA"),
    "chocolate": ("src.pipeline.schemas.chocolate", "CHOCOLATE_SCHEMA"),
    "cheeses": ("src.pipeline.schemas.cheeses", "CHEESES_SCHEMA"),
    "beverages": ("src.pipeline.schemas.beverages", "BEVERAGE_SCHEMA"),
    "cereals": ("src.pipeline.schemas.cereals", "CEREALS_SCHEMA"),
    "cosmetics": ("src.pipeline.schemas.cosmetics", "COSMETICS_SCHEMA"),
}


def _normalise(schema: dict) -> dict:
    out: dict[str, dict] = {}
    for name, spec in schema.items():
        s = deepcopy(spec)
        if s["type"] == "bool":
            s.setdefault("values", ["True", "False"])
        s.setdefault("nullable", False)
        s.setdefault("description", "")
        out[name] = s
    return out


def load_domain_attrs(domain: str) -> dict[str, dict]:
    """Return canonical attribute spec for a domain, keyed by attribute name."""
    if domain not in _DOMAIN_SCHEMAS:
        raise KeyError(
            f"Unknown domain: {domain!r}. Known: {sorted(_DOMAIN_SCHEMAS)}"
        )
    mod_path, attr_name = _DOMAIN_SCHEMAS[domain]
    mod = importlib.import_module(mod_path)
    return _normalise(getattr(mod, attr_name))


def load_pasta_attrs() -> dict[str, dict]:
    """Return canonical pasta attribute spec keyed by attribute name."""
    return _normalise(PASTA_SCHEMA)
