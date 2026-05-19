"""Status-derivation for manual annotation workflow.

Server-side fallback invoked by `save_row` when the client does not supply
an explicit status. Pure function — no I/O, no globals. Client mirror lives
in datasets/manual_label/app.py (function `deriveStatus`); keep the truth
table in sync if either side changes.

Note: this function does NOT derive `mode`. Mode is set independently by
the actor responsible for the audit: `blind` for human-typed cells
without pre-fill, `prefill` for cells where silver was pre-filled in the
input, `llm` for cells audited by an LLM (e.g., Opus 4.6 via
src/manual_label/llm_audit.py). Status is orthogonal to mode.
"""
from __future__ import annotations


def derive_status(silver: str, manual: str, prev: str) -> str:
    """Derive the new status for a (silver, manual) pair given prev status.

    `unsure` is sticky — never auto-cleared. All other inputs map to one of
    empty / confirmed / override / manual_only based on equality. `auto` is
    not an output of this function; it is set only by pre-fill at migration
    time.
    """
    if prev == "unsure":
        return "unsure"
    s = (silver or "").strip()
    m = (manual or "").strip()
    if m == "":
        return "empty"
    if s == "":
        return "manual_only"
    if m == s:
        return "confirmed"
    return "override"
