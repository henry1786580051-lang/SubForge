"""Chinese subject and nominal-completion boundary defects."""

from __future__ import annotations

import re

from subforge.core.translate.quality.boundary_features import ChineseBoundaryFeatures

from .visible_pause import BoundarySignalMatch, _match

_TOPIC_FILLER_TAIL = re.compile(r"(?:所以|不过|但是|但)?(?:我|我们)觉得(?:确实)?如此$")
_TOPIC_START = re.compile(
    r"^(?:阅读|写作|书写|教育|政治|技术|社会|信息|这|那|它|他|她|他们|人们)"
)
_MATERIAL_SUBJECT_TAIL = re.compile(r"[\u3400-\u9fff]{2,10}们$")
_MATERIAL_PREDICATE_START = re.compile(
    r"^(?:也|还|又|却|都|只|仍|曾|将|会|能|可以|能够|应该|需要|"
    r"使用|利用|认为|觉得|选择|拒绝|拥有|发挥|开始|继续)"
)
_CLASSIFIER_TAIL = re.compile(
    r"(?:是|不是|有|没有|成了|成为|需要|构成|呈现为).{0,4}"
    r"(?:一|这|那)(?:个|种|幅|段|场|项|件|条|位|名|辆|台|套)$"
)
_INDEPENDENTLY_NOMINALIZED_PERSON = re.compile(r"(?:第一|最后|下一|前一|后一)(?:个|位|名)$")


def detect_subject_nominal_completion_boundary(
    features: ChineseBoundaryFeatures,
) -> BoundarySignalMatch | None:
    """Return subject and nominal-completion signals at legacy precedence."""
    if _TOPIC_FILLER_TAIL.search(features.left) and _TOPIC_START.match(features.right):
        return _match("unfinished Chinese grammatical structure")

    if _MATERIAL_SUBJECT_TAIL.search(features.left) and _MATERIAL_PREDICATE_START.match(
        features.right
    ):
        return _match("material subject may be stranded")

    classifier_tail = _CLASSIFIER_TAIL.search(features.left)
    independently_nominalized_person = _INDEPENDENTLY_NOMINALIZED_PERSON.search(
        features.left
    )
    if classifier_tail and not independently_nominalized_person:
        return _match("classifier phrase is stranded")

    return None
