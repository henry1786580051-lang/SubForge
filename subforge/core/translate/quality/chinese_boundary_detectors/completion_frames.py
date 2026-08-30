"""Chinese predicate, classifier, and example completion boundary defects."""

from __future__ import annotations

import re

from subforge.core.translate.quality.boundary_features import ChineseBoundaryFeatures

from .visible_pause import BoundarySignalMatch, _match

_PERCENTAGE_USE_CASE_TAIL = re.compile(r"\d+%.+(?:转向机|转向齿条)$")
_RESULTATIVE_PREDICATE_TAIL = re.compile(
    r"(?:让|使(?!用)).{0,8}(?:这|那)?(?:辆|台)?(?:车|车辆|东西)$"
)
_COMPLETE_RESULTATIVE_NOUN_PHRASE = re.compile(
    r"(?:让|使).{0,8}的(?:车|车辆|东西)$"
)
_CLASSIFIER_TAIL = re.compile(
    r"(?:了|着|给|为|与|跟|和|有)(?:一|这|那)"
    r"(?:个|位|名|只|件|辆|台|种|套|条|款|部)$"
)
_COMPARISON_EXAMPLE_TAIL = re.compile(
    r"(?:出现|预测|提到|说到|例如|比如|包括|有).{0,8}像$"
)


def detect_completion_frame_boundary(
    features: ChineseBoundaryFeatures,
) -> BoundarySignalMatch | None:
    """Return the first completion-frame signal in legacy precedence order."""
    if _PERCENTAGE_USE_CASE_TAIL.search(features.left):
        return _match("percentage use-case predicate is stranded")

    if _RESULTATIVE_PREDICATE_TAIL.search(
        features.left
    ) and not _COMPLETE_RESULTATIVE_NOUN_PHRASE.search(features.left):
        return _match("resultative predicate is stranded")

    if _CLASSIFIER_TAIL.search(features.left):
        return _match("classifier phrase is stranded")

    if _COMPARISON_EXAMPLE_TAIL.search(features.left):
        return _match("comparison example is stranded")

    return None
