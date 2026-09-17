"""Development-only, source-scoped domain precision experiment.

Replaces one generic hint rather than accumulating historical examples. Never
rewrites sources, timings, validators, or provider settings.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from subforge.core.translate.guidance import target_language_style_rules

_GENERIC = (
    "Choose technical meanings from the object, operation, and local domain rather than the "
    "most literal dictionary sense; reject a rendering that is grammatically possible but "
    "physically or professionally implausible in context."
)
_HINTS = (
    (r"\b(?:mpg|miles per gallon|fuel economy)\b", "Distinguish fuel economy (distance per fuel) from consumption (fuel per distance); preserve the direction of comparisons and never invert units."),
    (r"\bturning (?:circle|radius|diameter)\b", "Distinguish turning-circle diameter from radius; do not substitute one measurement for the other or invent a conversion."),
    (r"\b(?:console|sidewall|shifter|pedal|slider)\b", "Name the actual component from local function: console is not necessarily an armrest box, sidewall height is not material thickness, and a slider is not a shift paddle."),
    (r"\b(?:bridge|mast|cable.stay|tender|bidding|contractor)\b", "Distinguish bridge tower, stay cable and suspension main cable. Preserve tendering roles: a firm approached to build is invited, not necessarily willing or selected; an only competitor is not a best competitor."),
)


def domain_precision_rules(target_language: str, source_texts: Iterable[str]) -> str:
    sources = tuple(source_texts)
    baseline = target_language_style_rules(target_language, sources)
    if not baseline:
        return baseline
    source = " ".join(sources)
    selected = [hint for pattern, hint in _HINTS if re.search(pattern, source, re.I)]
    if not selected:
        return baseline
    return baseline.replace(_GENERIC, " ".join(selected))
