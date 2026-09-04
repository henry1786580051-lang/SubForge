"""Chinese clause-attachment boundary defects."""

from __future__ import annotations

import re

from subforge.core.translate.quality.boundary_features import ChineseBoundaryFeatures

from .visible_pause import BoundarySignalMatch, _match

_ERA_LOCATIVE_FRAME = re.compile(r"^而(?:是|要)?在.*时代$")
_REPORTING_PERFECTIVE_TAIL = re.compile(r"(?:看到|看见|发现)了$")
_COORDINATED_SUBJECT_TAIL = re.compile(r"(?:阅读|写作|书写|书籍|文字|信息)$")
_COORDINATED_SUBJECT_START = re.compile(r"^和(?:阅读|写作|书写|书籍|文字|信息)")
_PREDICATE_FRAGMENT_START = re.compile(r"^(?:而|从而|进而)?(?:成为|变成|进入|转化为)")
_COMPLETED_PREDICATE_FRAGMENT = re.compile(
    r"(?:会|将|能够|可以|开始|逐渐|最终)(?:成为|变成|进入|转化为).*$"
)
_COMPLETED_CHOICE = re.compile(r"(?:做出|作出|拥有|面临).{0,8}选择$")
_COMPLETED_PASSIVE_USE = re.compile(r"(?:被|仍在|正在|可供|用于).{0,6}使用$")
_TRANSITIVE_PREDICATE_TAIL = re.compile(r"(?:获取|接触|保存|传播|选择|使用|评价)$")
_TRANSITIVE_OBJECT_START = re.compile(
    r"^(?:这|那|这些|那些|我们|你们|他们|它们|任何|所有)"
)
_DESTINATION_PREDICATE_TAIL = re.compile(
    r"(?:来到|前往|抵达|赶到|回到|驶入|进入|走进|飞往|迁往|移至|搬到)$"
)
_DESTINATION_START = re.compile(
    r"^(?!(?:但|不过|而且|所以|因为|如果|尽管|然后|接着|随后))"
    r"(?:[A-Z0-9]|[\u3400-\u9fff])"
)
_EXISTENTIAL_PREDICATE_TAIL = re.compile(
    r"(?:我|我们|你|你们|他|她|它|他们|她们|它们|这边|这里)?"
    r"(?:还|也)?(?:有|配有)$"
)
_EXISTENTIAL_OBJECT_START = re.compile(
    r"^(?!(?:但|不过|而且|所以|因为|如果|尽管|然后|接着|随后|是|有|会|能|可以))"
    r"(?:[A-Z0-9]|[\u3400-\u9fff])"
)
_COMPLETE_EXISTENTIAL_IDIOM = re.compile(r"(?:应有尽有|一应俱全)$")


def detect_clause_attachment_boundary(
    features: ChineseBoundaryFeatures,
) -> BoundarySignalMatch | None:
    """Return clause-attachment signals at legacy precedence."""
    if _ERA_LOCATIVE_FRAME.match(features.left):
        return _match("unfinished Chinese locative frame")

    if _REPORTING_PERFECTIVE_TAIL.search(features.left):
        return _match("possible reporting frame")

    if _COORDINATED_SUBJECT_TAIL.search(features.left) and _COORDINATED_SUBJECT_START.match(
        features.right
    ):
        return _match("coordinated subject may be stranded")

    if _PREDICATE_FRAGMENT_START.match(
        features.right
    ) and not _COMPLETED_PREDICATE_FRAGMENT.search(features.left):
        return _match("predicate fragment starts at next subtitle")

    if (
        not features.left_has_terminal_punctuation
        and _DESTINATION_PREDICATE_TAIL.search(features.left)
        and _DESTINATION_START.match(features.right)
    ):
        return _match("motion predicate is separated from its destination")

    if (
        not features.left_has_terminal_punctuation
        and _EXISTENTIAL_PREDICATE_TAIL.search(features.left)
        and not _COMPLETE_EXISTENTIAL_IDIOM.search(features.left)
        and _EXISTENTIAL_OBJECT_START.match(features.right)
    ):
        return _match("existential predicate is separated from its object")

    completed_choice = _COMPLETED_CHOICE.search(features.left)
    completed_passive_use = _COMPLETED_PASSIVE_USE.search(features.left)
    if (
        not completed_choice
        and not completed_passive_use
        and _TRANSITIVE_PREDICATE_TAIL.search(features.left)
        and _TRANSITIVE_OBJECT_START.match(features.right)
    ):
        return _match("transitive predicate is split from its object")

    return None
