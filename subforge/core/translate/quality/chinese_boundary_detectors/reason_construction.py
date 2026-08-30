"""Chinese unfinished reason-construction boundary defects."""

from __future__ import annotations

import re

from subforge.core.translate.quality.boundary_features import ChineseBoundaryFeatures

from .visible_pause import BoundarySignalMatch, _match

_UNFINISHED_REASON_TAIL = re.compile(r"之所以.+(?:部分|主要|根本|唯一)?原因$")


def detect_reason_construction_boundary(
    features: ChineseBoundaryFeatures,
) -> BoundarySignalMatch | None:
    """Return the unfinished reason signal at its legacy precedence position."""
    if _UNFINISHED_REASON_TAIL.search(features.left):
        return _match("unfinished Chinese reason construction")

    return None
