"""Chinese semantic-frame and nominal-attachment boundary defects."""

from __future__ import annotations

import re

from subforge.core.translate.quality.boundary_features import ChineseBoundaryFeatures

from .visible_pause import BoundarySignalMatch, _match

_INCOMPLETE_SEMANTIC_FRAME = re.compile(
    r"(?:重点|关键|问题|原因|意义|独特之处).{0,8}(?:是|在于)$"
)
_REPORTING_FRAME_TAIL = re.compile(
    r"(?:我|我们|你|你们|他们|她们|它们|这|那).{0,6}"
    r"(?:看到|发现|认为|觉得|表明|说明)$"
)
_STRANDED_NOMINAL_MODIFIER = re.compile(r"(?:真正|实际|核心|主要|完整|严重)的$")


def detect_semantic_attachment_boundary(
    features: ChineseBoundaryFeatures,
) -> BoundarySignalMatch | None:
    """Return the first semantic-attachment signal in legacy precedence order."""
    if _INCOMPLETE_SEMANTIC_FRAME.search(features.left):
        return _match("semantic frame is incomplete")

    if _REPORTING_FRAME_TAIL.search(features.left):
        return _match("possible reporting frame")

    if _STRANDED_NOMINAL_MODIFIER.search(features.left):
        return _match("nominal modifier is stranded")

    return None
