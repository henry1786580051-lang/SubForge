"""Chinese temporal and locative attachment boundary defects."""

from __future__ import annotations

import re

from subforge.core.translate.quality.boundary_features import ChineseBoundaryFeatures

from .visible_pause import BoundarySignalMatch, _match

_STANDALONE_TEMPORAL_FRAGMENT = re.compile(
    r"(?:每(?:年|月|周|天|日|小时|分钟)|届时|当时|随后|此前|之后|以前|以后)"
)
_LOCATIVE_FRAME_TAIL = re.compile(
    r"(?:这|那|其|它).{0,18}(?:设计|体系|结构|流程|计划)(?:中|里|内)$"
)
_LOCATIVE_PHRASE_TAIL = re.compile(
    r"(?:在|从|基于|依托).{1,24}(?:基础上|前提下|条件下|背景下)$"
)
_LOCATIVE_PREDICATE_START = re.compile(
    r"^(?:进一步|继续|再|进而|从而|将|会|要|可以|能够|推动|发展|"
    r"建设|实现|提高|提升|扩大|增加|减少|改善)"
)
_DISTANCE_MODIFIER_TAIL = re.compile(
    r"(?:大约|约|近)?\d+(?:\.\d+)?(?:米|公里|千米|英里)(?:以|之)?外$"
)
_DISTANCE_HEAD_START = re.compile(r"^(?:这|那|其|新|该|当地|场地|地点|区域)")


def detect_temporal_locative_attachment_boundary(
    features: ChineseBoundaryFeatures,
) -> BoundarySignalMatch | None:
    """Return temporal and locative attachment signals at legacy precedence."""
    if _STANDALONE_TEMPORAL_FRAGMENT.fullmatch(features.right):
        return _match("standalone Chinese temporal fragment")

    if _LOCATIVE_FRAME_TAIL.search(features.left):
        return _match("locative frame is separated from its complement")

    if _LOCATIVE_PHRASE_TAIL.search(features.left) and _LOCATIVE_PREDICATE_START.match(
        features.right
    ):
        return _match("locative phrase is separated from its predicate")

    if _DISTANCE_MODIFIER_TAIL.search(features.left) and _DISTANCE_HEAD_START.match(
        features.right
    ):
        return _match("distance modifier is separated from its noun")

    return None
