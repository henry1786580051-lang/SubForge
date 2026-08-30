"""Chinese boundary defects that become misleading across a visible pause."""

from __future__ import annotations

import re
from dataclasses import dataclass

from subforge.core.translate.quality.boundary_features import ChineseBoundaryFeatures
from subforge.core.translate.quality.boundary_registry import boundary_rule_for_message


@dataclass(frozen=True, slots=True)
class BoundarySignalMatch:
    rule_id: str
    message: str


def _match(message: str) -> BoundarySignalMatch:
    rule = boundary_rule_for_message(message)
    if rule is None:
        raise RuntimeError(f"Unregistered Chinese boundary signal: {message}")
    return BoundarySignalMatch(rule_id=rule.rule_id, message=rule.legacy_message)


def detect_visible_pause_boundary(
    features: ChineseBoundaryFeatures,
    *,
    separated_gap_ms: int,
) -> BoundarySignalMatch | None:
    """Return the first registered display defect, preserving legacy precedence."""
    if features.gap_ms < separated_gap_ms:
        return None
    if re.search(r"\d(?:\.\d+)?$", features.display_left_compact) and re.match(
        r"^(?:百|千|万|亿|兆|美元|欧元|英镑|日元|元|米|公里|平方|吨|人次|%)",
        features.display_right_compact,
    ):
        return _match("number and unit are separated by a visible pause")
    if re.search(
        r"(?:约为|约合|高达|达到|增至|降至|获得了|会让|将让|能够让|"
        r"正处于|正处在|大幅|显著|明显|迅速|持续|不断|强劲的)$",
        features.display_left_compact,
    ):
        return _match("unfinished predicate or modifier crosses a visible pause")
    return None
