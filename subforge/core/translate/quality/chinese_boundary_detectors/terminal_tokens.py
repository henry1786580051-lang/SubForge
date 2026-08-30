"""Chinese terminal-token boundary defects."""

from __future__ import annotations

import re

from subforge.core.translate.quality.boundary_features import ChineseBoundaryFeatures

from .visible_pause import BoundarySignalMatch, _match

_SHORT_DEMONSTRATIVE_TAIL = re.compile(r"(?:这个|这些|那种)$")
_STRANDED_PARTICLES = frozenset({"了", "的", "得"})
_DUPLICATED_CONNECTOR_TAIL = re.compile(r"(所以|因为|不过|但是|而且|并且)$")


def detect_terminal_token_boundary(
    features: ChineseBoundaryFeatures,
) -> BoundarySignalMatch | None:
    """Return final terminal-token signals at legacy precedence."""
    if _SHORT_DEMONSTRATIVE_TAIL.search(features.left) and len(
        re.sub(r"\s+", "", features.left)
    ) <= 10:
        return _match("possible demonstrative split")

    if features.right in _STRANDED_PARTICLES:
        return _match("particle stranded at next subtitle start")

    if features.right.startswith("时"):
        return _match("unfinished Chinese grammatical structure")

    left_connector = _DUPLICATED_CONNECTOR_TAIL.search(features.left)
    if left_connector and features.right.startswith(left_connector.group(1)):
        return _match("duplicated boundary connective")

    return None
