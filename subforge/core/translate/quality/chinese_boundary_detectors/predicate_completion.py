"""Chinese predicate-completion boundary defects."""

from __future__ import annotations

import re

from subforge.core.translate.quality.boundary_features import ChineseBoundaryFeatures

from .visible_pause import BoundarySignalMatch, _match

_ASPECT_PREDICATE_TAIL = re.compile(r"(?:开|驾驶|挂|拿|带|穿|戴|拖|拉|推)着$")
_ASPECT_COMPLEMENT_START = re.compile(
    r"^(?:一|这|那|某|每|任何|几|多|辆|台|个|件|块|张|把|在|向|往|从)"
)
_QUANTIFIED_OBJECT_PREDICATE_TAIL = re.compile(r"(?:接待|迎接|容纳)$")
_QUANTIFIED_OBJECT_START = re.compile(
    r"^(?:约|近|超过|多达|至少|至多|\d|[一二两三四五六七八九十百千万亿])"
)


def detect_predicate_completion_boundary(
    features: ChineseBoundaryFeatures,
) -> BoundarySignalMatch | None:
    """Return predicate-completion signals at legacy precedence."""
    if _ASPECT_PREDICATE_TAIL.search(features.left) and _ASPECT_COMPLEMENT_START.match(
        features.right
    ):
        return _match("aspect predicate is separated from its complement")

    if _QUANTIFIED_OBJECT_PREDICATE_TAIL.search(
        features.left
    ) and _QUANTIFIED_OBJECT_START.match(features.right):
        return _match("transitive predicate is split from its quantified object")

    return None
