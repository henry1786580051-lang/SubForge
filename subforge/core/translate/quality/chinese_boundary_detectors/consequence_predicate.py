"""Chinese missing consequence-predicate boundary defects."""

from __future__ import annotations

import re

from subforge.core.translate.quality.boundary_features import ChineseBoundaryFeatures

from .visible_pause import BoundarySignalMatch, _match

_CONSEQUENCE_WITHOUT_PREDICATE = re.compile(
    r"(?:以至于|从而|因此).{0,24}"
    r"(?:\d+|[一二两三四五六七八九十几多]+)(?:项|个|座|次)?"
    r"(?:诺贝尔奖|奖项|奖|荣誉|成果|结果)$"
)
_COMPLETE_CONSEQUENCE_PREDICATE = re.compile(
    r"(?:颁发|授予|获得|赢得|斩获|催生|产生|取得|带来)了?"
    r".{0,12}(?:诺贝尔奖|奖项|奖|荣誉|成果|结果)$"
)


def detect_consequence_predicate_boundary(
    features: ChineseBoundaryFeatures,
) -> BoundarySignalMatch | None:
    """Return the missing consequence-predicate signal at legacy precedence."""
    if _CONSEQUENCE_WITHOUT_PREDICATE.search(
        features.left
    ) and not _COMPLETE_CONSEQUENCE_PREDICATE.search(features.left):
        return _match("consequence predicate is missing")

    return None
