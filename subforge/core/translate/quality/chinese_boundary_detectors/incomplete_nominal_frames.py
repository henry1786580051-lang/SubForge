"""Chinese incomplete nominal and determiner boundary frames."""

from __future__ import annotations

import re

from subforge.core.translate.quality.boundary_features import ChineseBoundaryFeatures

from .visible_pause import BoundarySignalMatch, _match

_COMPARATIVE_NOUN_MODIFIER_TAIL = re.compile(r"(?:大多数|许多|一些|所有)?其他$")
_COMPLETED_INDEFINITE_CHOICE = re.compile(
    r"(?:执着于|拘泥于|局限于|限定于|偏向于|选择)(?:某)?一种$"
)
_INDEFINITE_TAIL = re.compile(r"(?:任何|某种|一种|某些)$")
_INCOMPLETE_COPULAR_TAIL = re.compile(r"(?:并不是|不只是|有意思的是)$")
_INCOMPLETE_LEXICAL_TAIL = re.compile(
    r"(?:进一步|从而|进而|任何价值|这种新的|这类新的|我写|我们写|正在写)$"
)


def detect_incomplete_nominal_frame_boundary(
    features: ChineseBoundaryFeatures,
) -> BoundarySignalMatch | None:
    """Return incomplete nominal-frame signals at legacy precedence."""
    if _COMPARATIVE_NOUN_MODIFIER_TAIL.search(features.left):
        return _match("comparative noun modifier is stranded")

    completed_indefinite_choice = _COMPLETED_INDEFINITE_CHOICE.search(features.left)
    if not completed_indefinite_choice and _INDEFINITE_TAIL.search(features.left):
        return _match("unfinished Chinese grammatical structure")

    if _INCOMPLETE_COPULAR_TAIL.search(features.left):
        return _match("unfinished Chinese grammatical structure")

    if _INCOMPLETE_LEXICAL_TAIL.search(features.left):
        return _match("unfinished Chinese grammatical structure")

    return None
