"""Chinese stranded-connective and copular-bridge boundary defects."""

from __future__ import annotations

import re

from subforge.core.translate.quality.boundary_features import ChineseBoundaryFeatures

from .visible_pause import BoundarySignalMatch, _match

_CONNECTIVE_TAIL = re.compile(
    r"(?:但|但是|不过|而且|并且|所以|因为|如果|尽管|除非|和|与|及|以及|或者|总之)$"
)
_COPULAR_BRIDGE_TAIL = re.compile(r"在于$")
_COPULAR_TOPIC_TAIL = re.compile(r"(?:重点|关键|原因|问题|事实|观点|想法)$")
_COPULAR_RESULT_START = re.compile(r"^是(?:要|在|让|把|我们|你们|他们|她们|它们|这|那)")


def detect_discourse_bridge_boundary(
    features: ChineseBoundaryFeatures,
) -> BoundarySignalMatch | None:
    """Return the first connective/copular signal in legacy precedence order."""
    if _CONNECTIVE_TAIL.search(features.left):
        return _match("connective stranded at previous subtitle end")

    if _COPULAR_BRIDGE_TAIL.search(features.left):
        return _match("possible copular bridge")

    if _COPULAR_TOPIC_TAIL.search(features.left) and _COPULAR_RESULT_START.match(
        features.right
    ):
        return _match("possible copular bridge")

    return None
