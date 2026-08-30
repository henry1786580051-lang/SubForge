"""Chinese generic unfinished-predicate boundary defects."""

from __future__ import annotations

import re

from subforge.core.translate.quality.boundary_features import ChineseBoundaryFeatures

from .visible_pause import BoundarySignalMatch, _match

_NOMINAL_ATTEMPT_TAIL = re.compile(r"(?:一|这|那)(?:次|场|项|个).{0,20}尝试$")
_UNFINISHED_PREDICATE_TAIL = re.compile(
    r"(?:大力|尽力|希望(?:能|能够|可以|将|会)?|预计(?:能|能够|可以|将|会)?|"
    r"有望|力求|试图|尝试|旨在|由)$"
)


def detect_unfinished_predicate_boundary(
    features: ChineseBoundaryFeatures,
) -> BoundarySignalMatch | None:
    """Return the generic unfinished-predicate signal at legacy precedence."""
    nominal_attempt = _NOMINAL_ATTEMPT_TAIL.search(features.left)
    if not nominal_attempt and _UNFINISHED_PREDICATE_TAIL.search(features.left):
        return _match("unfinished Chinese predicate or governing word")

    return None
