"""Chinese subject ownership and coordinated-modifier boundary defects."""

from __future__ import annotations

import re

from subforge.core.translate.quality.boundary_features import ChineseBoundaryFeatures

from .visible_pause import BoundarySignalMatch, _match

_MATERIAL_SUBJECT = re.compile(
    r"(?:[\u3400-\u9fffA-Za-z]{1,16})"
    r"(?:人们|人士|选民|学生|读者|教师|家长|孩子|公司|学校|政府|机构|团队|群体|国家|社会)$"
)
_PREDICATE_START = re.compile(
    r"^(?:也|还(?!是)|又|却|则|都|只|仍|已经|正在|通常|往往|几乎|开始|继续|"
    r"可以|能够|应该|需要|认为|觉得|选择|拒绝|喜欢|愿意|拥有|缺少|没有|不)"
)
_MATERIAL_IS_OBJECT = re.compile(
    r"(?:教|让|给|问|帮助|要求|期待|提醒|告诉|采访|面向|针对)"
    r"[\u3400-\u9fffA-Za-z]{0,4}$"
)
_PEOPLE_SUBJECT = re.compile(r"(?:有|存在)(?:那么|这样|这些|那些|一些|几|很多|许多)?人$")
_PEOPLE_PREDICATE_START = re.compile(
    r"^(?:也|还|又|却|则|都|只|仍|正在|开始|继续|选择|认为|觉得|希望|"
    r"试图|愿意|会|能|可以|能够)"
)
_COORDINATED_MODIFIER_START = re.compile(
    r"^[\u3400-\u9fffA-Za-z]{1,12}[、，,]"
    r"[\u3400-\u9fffA-Za-z]{1,16}(?:[、，,]|的信息|的内容|的表达|的语言)"
)
_COPULAR_NOUN_SUBJECT_TAIL = re.compile(
    r"(?:的)?(?:调校|设计|布局|表现|体验|感觉|方式|方案|版本|结构|系统)$"
)
_COPULAR_PREDICATE_START = re.compile(
    r"^是(?:我|你|他|她|它|我们|你们|他们|最|目前|迄今)"
)
_POSSESSIVE_PRONOUN_TAIL = re.compile(r"(?:我们|你们|他们|她们|它们)$")
_POSSESSIVE_MODIFIER_START = re.compile(
    r"^(?:开头|一开始|刚才|之前|此前|前面|过去|当时).{0,12}"
    r"(?:的|那个|那些|话题|内容)"
)
_STYLE_MODIFIER_TAIL = re.compile(r"[\u3400-\u9fff]{1,8}式$")
_STYLE_HEAD_START = re.compile(
    r"^(?:座舱|设计|布局|结构|造型|内饰|外观|车身|座椅|方案|风格|图案)"
)


def detect_subject_attachment_boundary(
    features: ChineseBoundaryFeatures,
) -> BoundarySignalMatch | None:
    """Return the first subject/modifier signal in legacy precedence order."""
    material_subject = _MATERIAL_SUBJECT.search(features.left)
    predicate_start = _PREDICATE_START.match(features.right)
    material_is_object = _MATERIAL_IS_OBJECT.search(features.left)
    if material_subject and predicate_start and not material_is_object:
        return _match("material subject may be stranded")

    if _PEOPLE_SUBJECT.search(features.left) and _PEOPLE_PREDICATE_START.match(
        features.right
    ):
        return _match("material subject may be stranded")

    if _COPULAR_NOUN_SUBJECT_TAIL.search(features.left) and _COPULAR_PREDICATE_START.match(
        features.right
    ):
        return _match("nominal subject is separated from its copular predicate")

    if _POSSESSIVE_PRONOUN_TAIL.search(features.left) and _POSSESSIVE_MODIFIER_START.match(
        features.right
    ):
        return _match("possessive pronoun is separated from its head phrase")

    if _STYLE_MODIFIER_TAIL.search(features.left) and _STYLE_HEAD_START.match(features.right):
        return _match("style modifier is separated from its head noun")

    if (
        not features.left_has_terminal_punctuation
        and features.left.endswith("的")
        and _COORDINATED_MODIFIER_START.match(features.right)
    ):
        return _match("coordinated modifier may be stranded")

    return None
