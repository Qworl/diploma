"""Build LLM prompts from product input + schema + few-shot examples."""

import json

from src.pipeline.schemas import EXAMPLES_BY_SCHEMA, INPUT_FIELDS

# Extra fields rendered after INPUT_FIELDS when present (off_grounded mode).
# Critical for derived attrs: nutri_score_grade, protein_class, fat_class need numeric nutriments.
_EXTRA_RENDER_FIELDS = (
    "nutriments", "labels_tags", "allergens_tags",
    "manufacturing_places", "countries_tags",
)


def _render_product_block(product: dict) -> str:
    lines = [
        f"- {field}: {product.get(field, 'N/A')}"
        for field in INPUT_FIELDS
        if product.get(field)
    ]
    for field in _EXTRA_RENDER_FIELDS:
        v = product.get(field)
        if not v:
            continue
        if field == "nutriments" and isinstance(v, dict):
            nuts = ", ".join(f"{k}={val}" for k, val in sorted(v.items()))
            lines.append(f"- nutriments (per 100g): {nuts}")
        else:
            lines.append(f"- {field}: {v}")
    return "\n".join(lines)


def build_prompt(product: dict, schema: dict, *, include_examples: bool = True) -> str:
    """Build extraction prompt for a single product.

    include_examples=True (default) prepends 3 few-shot input/output pairs from
    EXAMPLES_BY_SCHEMA when available for this schema. Set False for ablation/benchmark.
    """
    product_text = _render_product_block(product)

    attrs_desc = []
    for attr, spec in schema.items():
        if spec["type"] == "enum":
            vals = ", ".join(f'"{v}"' for v in spec["values"])
            nullable = " or null" if spec.get("nullable") else ""
            attrs_desc.append(f'  "{attr}": one of [{vals}]{nullable} — {spec["description"]}')
        elif spec["type"] == "bool":
            attrs_desc.append(f'  "{attr}": true or false — {spec["description"]}')
        elif spec["type"] == "int":
            nullable = " or null" if spec.get("nullable") else ""
            attrs_desc.append(f'  "{attr}": integer{nullable} — {spec["description"]}')

    attrs_block = "\n".join(attrs_desc)

    derivation_block = ""
    schema_attrs = set(schema.keys())
    rules = []

    # Auto-derive bucket descriptions from TYPE_C_RULES — single source of truth.
    from src.pipeline.off_labels.rules import TYPE_C_RULES

    def _format_buckets(buckets):
        """[(15.0, 'low'), (22.0, 'medium'), ...] → '<15 → \"low\", 15-22 → \"medium\", ...'."""
        parts = []
        prev = 0.0
        for thr, label in buckets:
            if thr == float("inf"):
                parts.append(f'≥{prev:g}g → "{label}"')
            elif prev == 0.0:
                parts.append(f'<{thr:g}g → "{label}"')
            else:
                parts.append(f'{prev:g}–{thr:g}g → "{label}"')
            prev = thr
        return ", ".join(parts)

    if "nutri_score_grade" in schema_attrs:
        rules.append(
            "- nutri_score_grade: official OFF Nutri-Score A–E. "
            "If nutriments include energy, sugars, saturated-fat, salt, fiber, proteins — "
            "compute the standard Nutri-Score (negative points: energy/sat-fat/sugars/salt; "
            "positive points: fiber/proteins/fruits-veg) and map to grade. "
            "Return null ONLY if essential nutriments are missing."
        )
    if "protein_class" in schema_attrs and "protein_class" in TYPE_C_RULES:
        # Schema override may apply (e.g. PET_FOOD); use schema buckets if present
        sch_buckets = schema["protein_class"].get("buckets") if isinstance(schema.get("protein_class"), dict) else None
        buckets = sch_buckets or TYPE_C_RULES["protein_class"]["buckets"]
        rules.append(
            f"- protein_class from proteins_100g: {_format_buckets(buckets)}. "
            "If proteins_100g present, DO NOT return null."
        )
    if "fat_class" in schema_attrs and "fat_class" in TYPE_C_RULES:
        buckets = TYPE_C_RULES["fat_class"]["buckets"]
        rules.append(
            f"- fat_class from fat_100g: {_format_buckets(buckets)}. "
            "If fat_100g present, DO NOT return null."
        )
    if "sugar_class" in schema_attrs and "sugar_class" in TYPE_C_RULES:
        buckets = TYPE_C_RULES["sugar_class"]["buckets"]
        rules.append(
            f"- sugar_class from sugars_100g: {_format_buckets(buckets)}. "
            "If sugars_100g present, DO NOT return null."
        )
    if "alcohol_class" in schema_attrs and "alcohol_class" in TYPE_C_RULES:
        buckets = TYPE_C_RULES["alcohol_class"]["buckets"]
        rules.append(
            f"- alcohol_class from alcohol_100g: {_format_buckets(buckets)}. "
            "If alcohol_100g present, DO NOT return null."
        )
    if "cocoa_percentage" in schema_attrs and "cocoa_percentage" in TYPE_C_RULES:
        buckets = TYPE_C_RULES["cocoa_percentage"]["buckets"]
        # cocoa buckets are percentages, format slightly differently
        bk_str = ", ".join(
            f'≥{(prev or 0):g}% → "{label}"' if thr == float("inf")
            else f'<{thr:g}% → "{label}"'
            for prev, (thr, label) in zip([0] + [b[0] for b in buckets[:-1]], buckets)
        )
        rules.append(
            f"- cocoa_percentage: extract numeric % from product_name/ingredients_text "
            f'(patterns: "70%", "dark chocolate 70", "en:dark-chocolates-70-percent-cocoa"). '
            f"Then bucket: {bk_str}."
        )
    if rules:
        derivation_block = "Derivation rules (apply strictly when input is available):\n" + "\n".join(rules) + "\n\n"

    examples_block = ""
    if include_examples:
        examples = EXAMPLES_BY_SCHEMA.get(id(schema), [])
        if examples:
            rendered = []
            for ex_product, ex_output in examples:
                ex_input = _render_product_block(ex_product)
                ex_json = json.dumps(ex_output, ensure_ascii=False)
                rendered.append(f"Product:\n{ex_input}\n\nOutput: {ex_json}")
            examples_block = (
                "Examples:\n\n" + "\n\n---\n\n".join(rendered) + "\n\n---\n\n"
            )

    return f"""Extract product attributes from the following product information.
The product may be in any language (English, French, German, Spanish, etc.).

{examples_block}Product:
{product_text}

Extract these attributes as JSON:
{{
{attrs_block}
}}

{derivation_block}Respond ONLY with a valid JSON object, no explanation."""
