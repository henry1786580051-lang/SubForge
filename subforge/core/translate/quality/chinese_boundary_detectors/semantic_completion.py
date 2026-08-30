"""Chinese semantic and complement completion boundary defects."""

from __future__ import annotations

import re

from subforge.core.translate.quality.boundary_features import ChineseBoundaryFeatures

from .visible_pause import BoundarySignalMatch, _match

_SEMANTIC_FRAME_TAIL = re.compile(r"(?:想指出的更多是|真正想说的|真正想表达的)$")
_VAGUE_FILLER_FRAME = re.compile(
    r"(?:但|不过|所以|而且)?(?:我|我们)(?:只是)?觉得(?:吧|呢)?"
)
_NEGATED_SEMANTIC_FRAME_TAIL = re.compile(
    r"(?:真正想说的|真正想表达的)(?:重点)?并非如此$"
)
_ADJECTIVE_COMPLEMENT_TAIL = re.compile(
    r"(?:他|她|它|我|我们|他们|她们|它们).{0,6}(?:很|非常|确实)?适合$"
)
_ARTICLE_LENGTH_TAIL = re.compile(r"(?:这|那)篇.{0,8}(?:千|万|\d)字$")


def detect_semantic_completion_boundary(
    features: ChineseBoundaryFeatures,
) -> BoundarySignalMatch | None:
    """Return semantic-completion signals at legacy precedence."""
    if _SEMANTIC_FRAME_TAIL.search(features.left):
        return _match("semantic frame is incomplete")

    if _VAGUE_FILLER_FRAME.fullmatch(features.left):
        return _match("vague filler-only frame")

    if _NEGATED_SEMANTIC_FRAME_TAIL.search(features.left):
        return _match("semantic frame is incomplete")

    if _ADJECTIVE_COMPLEMENT_TAIL.search(features.left):
        return _match("adjective complement is missing")

    if _ARTICLE_LENGTH_TAIL.search(features.left):
        return _match("classifier phrase is stranded")

    return None
