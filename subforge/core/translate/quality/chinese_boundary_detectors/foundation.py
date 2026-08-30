"""Foundational Chinese boundary defects at the start of the legacy chain."""

from __future__ import annotations

import re

from subforge.core.translate.quality.boundary_features import ChineseBoundaryFeatures

from .visible_pause import BoundarySignalMatch, _match

_STANDALONE_CONNECTIVES = frozenset(
    {
        "但",
        "但是",
        "不过",
        "而且",
        "并且",
        "所以",
        "因为",
        "如果",
        "尽管",
        "除非",
        "以及",
        "或者",
        "总之",
        "就是",
        "但与此同时",
        "与此同时",
    }
)

_DEMONSTRATIVE_SUBJECT = re.compile(
    r"(?:而|但|不过)?(?:这|那|它|此|这项|那项)(?:工程|工作|做法)?"
)
_SENTENCE_ADVERB_TAIL = re.compile(
    r"(?:基本上|大体上|本质上|原则上|理论上|总体上|整体上|从根本上)$"
)
_SUBJECT_ADVERB_TAIL = re.compile(
    r"(?:我|我们|你|你们|他|她|它|他们|她们|它们)"
    r"(?:现在|目前|如今|基本上|实际上|最终|可能|也许|大概|永远|始终|"
    r"一直|仍然|依然|已经|正在|还|也|都|从来|绝不)$"
)
_COORDINATED_SUBJECT_TAIL = re.compile(
    r"(?:^|[\s，,；;])"
    r"(?:我|我们|你|你们|他|她|它|他们|她们|它们)"
    r"(?:和|与|及)"
    r"(?P<tail>[㐀-鿿A-Za-z·]{1,24})$"
)
_PREDICATE_IN_COORDINATED_TAIL = re.compile(
    r"(?:开始|继续|完成|进行|正在|已经|曾经|将要|会|能|可以|能够|"
    r"需要|应该|认为|觉得|拥有|缺少|没有|搭建|建造|制作|使用|选择|"
    r"发现|看到|前往|抵达|工作|生活|阅读|写作|讨论|介绍)"
)


def detect_foundation_boundary(
    features: ChineseBoundaryFeatures,
) -> BoundarySignalMatch | None:
    """Return the first foundational signal in exact legacy precedence order."""
    if features.left in _STANDALONE_CONNECTIVES or features.right in _STANDALONE_CONNECTIVES:
        return _match("standalone connective")

    if _DEMONSTRATIVE_SUBJECT.fullmatch(features.left):
        return _match("demonstrative subject is stranded")

    if not features.left_has_terminal_punctuation and _SENTENCE_ADVERB_TAIL.search(
        features.compact_left
    ):
        return _match("sentence adverb is separated from its predicate")

    if not features.left_has_terminal_punctuation and _SUBJECT_ADVERB_TAIL.search(
        features.compact_left
    ):
        return _match("subject and sentence adverb are separated from their predicate")

    coordinated_subject = (
        _COORDINATED_SUBJECT_TAIL.search(features.raw_left)
        if not features.left_has_terminal_punctuation
        else None
    )
    if coordinated_subject and not _PREDICATE_IN_COORDINATED_TAIL.search(
        coordinated_subject.group("tail")
    ):
        return _match("coordinated subject is separated from its predicate")

    return None
