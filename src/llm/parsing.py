"""Parse JSON responses from LLM, validate against schema."""

import json


def _parse_with_status(raw: str, schema: dict) -> tuple[dict, bool]:
    """Parse + validate LLM response. Returns (parsed_dict, json_parsed_ok).

    json_parsed_ok=False means we couldn't locate valid JSON in the response —
    distinct from "JSON parsed but no fields validated" (caller may want to retry
    only on the former).
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end])
            except json.JSONDecodeError:
                return {}, False
        else:
            return {}, False

    if not isinstance(data, dict):
        return {}, False

    result = {}
    for attr, spec in schema.items():
        if attr not in data:
            continue
        value = data[attr]

        if value is None and spec.get("nullable"):
            result[attr] = None
            continue

        if spec["type"] == "enum":
            if value in spec["values"]:
                result[attr] = value
        elif spec["type"] == "bool":
            if isinstance(value, bool):
                result[attr] = value
        elif spec["type"] == "int":
            valid = spec.get("values")
            v_int = None
            if isinstance(value, bool):
                pass  # bool is subclass of int — reject
            elif isinstance(value, int):
                v_int = value
            elif isinstance(value, str) and value.isdigit():
                v_int = int(value)
            if v_int is not None and (valid is None or v_int in valid):
                result[attr] = v_int

    return result, True


def parse_llm_response(raw: str, schema: dict) -> dict:
    """Parse and validate LLM response against schema. Returns {} on parse failure."""
    parsed, _ = _parse_with_status(raw, schema)
    return parsed
