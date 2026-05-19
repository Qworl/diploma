"""OFF/OBF labels layer of the cascade.

Public API:
- apply_off_labels(row, schema): full OFF-tag-driven labeling
- apply_partner_type_f(row, schema): regex labelers on partner fields only
- PARTNER_FIELDS: set of input field names allowed for partner-side inference
"""

from src.pipeline.off_labels.apply import (
    apply_off_labels,
    apply_partner_type_f,
    PARTNER_FIELDS,
)

__all__ = ["apply_off_labels", "apply_partner_type_f", "PARTNER_FIELDS"]
