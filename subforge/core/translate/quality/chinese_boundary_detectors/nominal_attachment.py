"""Chinese head-noun and complement attachment boundary defects."""

from __future__ import annotations

import re

from subforge.core.translate.quality.boundary_features import ChineseBoundaryFeatures

from .visible_pause import BoundarySignalMatch, _match

_RELATIVE_CLAUSE_TAIL = re.compile(
    r"(?:看到|看见|提到|说到|展示|使用|拍摄|讨论|介绍|选择|购买|建造|完成)的$"
)
_RELATIVE_HEAD_NOUN = re.compile(
    r"^.{1,30}的(?:那个|这个)?(?:照片|图片|画面|视频|项目|产品|车辆|建筑|方案)$"
)
_DEMONSTRATIVE_RELATIVE_TAIL = re.compile(r"的(?:那个|这个|那一个|这一个)$")
_COMPARATIVE_OBJECT_TAIL = re.compile(r"(?:给|为).{1,24}更好的$")
_COMPARISON_FRAME_TAIL = re.compile(r"(?:布置|安排|设计|做|弄|看|听|感觉|显得).{0,12}像$")
_VEHICLE_MODIFIER_TAIL = re.compile(r"(?:一辆|一台|一款|一部).{1,16}的$")
_MODEL_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9-]*(?:\s+[A-Za-z][A-Za-z0-9-]*)?")
_REPORTING_PREDICATE = re.compile(
    r"^(?:(?:而且|但|不过|所以|然后)\s*)?"
    r"(?:我|我们)(?:(?:当时|那时|现在|真的|确实|一直|曾经|还|也)\s*){0,4}"
    r"(?:觉得|认为|相信|以为)$"
)
_STRANDED_CLASSIFIER = re.compile(r"(?:是|成了|算是)(?:一)?个$")
_DEMONSTRATIVE_MODIFIER = re.compile(r"有一个.{1,16}的$")
_SIMILARITY_HEAD = re.compile(r"^(?:像|类似)")
_CONTEXTUAL_COUNT_CLASSIFIER = re.compile(
    r"(?:已有|现有|建成|建设|兴建|规划|提出|获批|批准|公布|宣布|"
    r"取消|完成|交付|售出|卖出|生产|制造)(?:了|的|出)?"
    r"[^\d一二两三四五六七八九十百千万亿]{0,8}"
    r"(?:\d+|[一二两三四五六七八九十百千万亿]+)"
    r"(?:座|栋|个|项|套|台|处|家)$"
)


def detect_nominal_attachment_boundary(
    features: ChineseBoundaryFeatures,
) -> BoundarySignalMatch | None:
    """Return the first nominal/complement signal in legacy precedence order."""
    if _RELATIVE_CLAUSE_TAIL.search(features.left) and _RELATIVE_HEAD_NOUN.match(
        features.right
    ):
        return _match("relative clause is separated from its head noun")

    if _DEMONSTRATIVE_RELATIVE_TAIL.search(features.right):
        return _match("demonstrative relative clause lacks its head noun")

    if _COMPARATIVE_OBJECT_TAIL.search(features.left):
        return _match("comparative object is omitted after a governing verb")

    if _COMPARISON_FRAME_TAIL.search(features.left):
        return _match("comparison frame is separated from its object")

    if _VEHICLE_MODIFIER_TAIL.search(features.left) and _MODEL_NAME.match(features.right):
        return _match("vehicle modifier is separated from its model name")

    if _REPORTING_PREDICATE.search(features.left):
        return _match("reporting predicate is separated from its complement")

    if _STRANDED_CLASSIFIER.search(features.left):
        return _match("classifier phrase is stranded")

    if _DEMONSTRATIVE_MODIFIER.search(features.left) and _SIMILARITY_HEAD.match(
        features.right
    ):
        return _match("demonstrative modifier is separated from its head noun")

    if _CONTEXTUAL_COUNT_CLASSIFIER.search(features.compact_left):
        return _match("count classifier lacks its contextual head noun")

    return None
