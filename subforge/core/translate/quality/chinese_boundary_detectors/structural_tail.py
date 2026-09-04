"""Chinese structural-tail boundary defects."""

from __future__ import annotations

import re

from subforge.core.translate.quality.boundary_features import ChineseBoundaryFeatures

from .visible_pause import BoundarySignalMatch, _match

_NEGATED_COMPARISON_TAIL = re.compile(r"(?:并|还|也|却)?不(?:太|那么|算|够|怎么|完全)?$")
_COMPARISON_COMPLEMENT_START = re.compile(r"^(?:像|如|及|比|和|与|跟|同).{0,16}(?:一样|相比|那么)")
_SOFT_TAIL = re.compile(
    r"(?:的|是|把|被|让|给|和|与|对|向|从|比|像|后|前|大约|差不多|与其|为了|花)$"
)
_STRUCTURAL_TAIL = re.compile(
    r"(?:作为|没有|不会|不能|可以|应该|能够|正在|已经|只是|其实|确实|"
    r"相当|非常|远远|更|最|几乎|变得|如今|现在|目前|当时|后来|最终|实际上|像是|就像|就是|"
    r"我是说|我的意思是|来说|例如|比如|唯一|投入|专注于|致力于|同时|"
    r"成为|变成|属于|包括)$"
)
_COMPLETE_PERSPECTIVE_FRAME = re.compile(
    r"(?:从|在)(?:很多|许多|某些|多个|一些)方面来说$"
)
_COMPLETE_NOMINAL_SUPERLATIVE = re.compile(
    r"(?:世界|全球|全国|亚洲|欧洲|当地|行业)(?:上|范围内|业内)?之最$"
)
_STANDALONE_DANG = re.compile(r"(?:^|[\s，,])当$")
_ELLIPSIS_END = re.compile(r"…+[）)】\]\"'’”]*$")


def detect_structural_tail_boundary(
    features: ChineseBoundaryFeatures,
) -> BoundarySignalMatch | None:
    """Return structural-tail signals at legacy precedence."""
    if features.canonical_left in {
        "而在过去",
        "在过去",
        "与此同时",
        "如今",
        "现在",
        "目前",
    } and features.canonical_right.startswith(features.canonical_left):
        return _match("possible duplicated boundary phrase")

    if _NEGATED_COMPARISON_TAIL.search(features.left) and _COMPARISON_COMPLEMENT_START.match(
        features.right
    ):
        return _match("negated comparison is split from its complement")

    if _SOFT_TAIL.search(features.left) and not features.left_has_terminal_punctuation:
        return _match("possible function-word split")

    complete_perspective_frame = _COMPLETE_PERSPECTIVE_FRAME.search(features.left)
    complete_nominal_superlative = _COMPLETE_NOMINAL_SUPERLATIVE.search(features.left)
    left_is_closed = bool(
        features.left_has_terminal_punctuation
        and not _ELLIPSIS_END.search(features.raw_left)
    )
    if (
        not left_is_closed
        and _STRUCTURAL_TAIL.search(features.left)
        and not complete_perspective_frame
        and not complete_nominal_superlative
    ):
        return _match("unfinished Chinese grammatical structure")

    if not left_is_closed and _STANDALONE_DANG.search(features.left):
        return _match("unfinished Chinese grammatical structure")

    return None
