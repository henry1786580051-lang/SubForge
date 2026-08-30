"""Chinese unfinished grammatical-frame boundary defects."""

from __future__ import annotations

import re

from subforge.core.translate.quality.boundary_features import ChineseBoundaryFeatures

from .visible_pause import BoundarySignalMatch, _match

_SPECIAL_UNFINISHED_TAIL = re.compile(
    r"(?:尤其是对|尤其是关于|我更想表达的重点|我无法确定我是否认为|既)$"
)
_REPORTING_FRAME_TAIL = re.compile(
    r"(?:我|我们)(?:仍|也|还)?(?:觉得|认为|相信|猜|希望)$"
)
_LOCATIVE_REPORTING_FRAME_TAIL = re.compile(
    r"(?:觉得|认为|看到|发现|问)在.{1,24}(?:之间|之中)$"
)
_PRONOUN_BA_TAIL = re.compile(
    r"(?:我|你|他|她|它|我们|你们|他们|她们|它们)(?:还|又|也|就|刚)?把$"
)
_COPULAR_RESULT_TAIL = re.compile(
    r"(?:^|[\s，,；;。.!?])(?:这|那|它|车辆|这辆车|那辆车)"
    r"(?:其实|基本|大概|完全|确实)?就是$"
)


def detect_unfinished_frame_boundary(
    features: ChineseBoundaryFeatures,
) -> BoundarySignalMatch | None:
    """Return the first unfinished-frame signal in legacy precedence order."""
    if _SPECIAL_UNFINISHED_TAIL.search(features.left):
        return _match("unfinished Chinese grammatical structure")

    if _REPORTING_FRAME_TAIL.search(features.left):
        return _match("unfinished Chinese grammatical structure")

    if _LOCATIVE_REPORTING_FRAME_TAIL.search(features.left):
        return _match("unfinished Chinese locative frame")

    if _PRONOUN_BA_TAIL.search(features.left):
        return _match("unfinished Chinese grammatical structure")

    if _COPULAR_RESULT_TAIL.search(features.left):
        return _match("copular frame is separated from its result")

    return None
