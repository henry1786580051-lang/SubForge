"""Chinese numeric-range and numeric-complement boundary defects."""

from __future__ import annotations

import re

from subforge.core.translate.quality.boundary_features import ChineseBoundaryFeatures

from .visible_pause import BoundarySignalMatch, _match

_NUMERIC_RANGE_TAIL = re.compile(
    r"(?:在|从|介于).{0,12}(?:零|一|二|三|四|五|六|七|八|九|十|\d+)"
    r"(?:本|个|条|项|年|岁|人|次|种)?$"
)
_NUMERIC_RANGE_CONTINUATION = re.compile(
    r"^(?:到|至|和|与)(?:零|一|二|三|四|五|六|七|八|九|十|\d+)"
)
_NUMERIC_COMPLEMENT_TAIL = re.compile(
    r"(?:增长|扩大|增加|提升|上升|升高|减少|下降|降低|达到)(?:到|至)$"
)
_NUMERIC_COMPLEMENT_VALUE = re.compile(
    r"(?:\d+(?:\.\d+)?|零|一|二|两|三|四|五|六|七|八|九|十)"
)


def detect_numeric_completion_boundary(
    features: ChineseBoundaryFeatures,
) -> BoundarySignalMatch | None:
    """Return the first numeric-completion signal in legacy precedence order."""
    if _NUMERIC_RANGE_TAIL.search(
        features.left
    ) and _NUMERIC_RANGE_CONTINUATION.match(features.right):
        return _match("numeric range is split")

    if _NUMERIC_COMPLEMENT_TAIL.search(
        features.left
    ) and _NUMERIC_COMPLEMENT_VALUE.search(features.right):
        return _match("numeric complement is stranded")

    return None
