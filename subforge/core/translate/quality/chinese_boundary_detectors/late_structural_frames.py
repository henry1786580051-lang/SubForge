"""Late Chinese structural-frame boundary defects."""

from __future__ import annotations

import re

from subforge.core.translate.quality.boundary_features import ChineseBoundaryFeatures

from .visible_pause import BoundarySignalMatch, _match

_LITERAL_FUNDAMENTAL_CALQUE = re.compile(
    r"^(?:.{0,12})?来说(?:太|很|非常)?根本(?:了|的)?(?:\s|所以|因此|而|但|$)"
)
_PURCHASE_CLASSIFIER_TAIL = re.compile(
    r"(?:买(?:到)?|选(?:择)?|找(?:到)?|换(?:成)?)"
    r"(?:一|这|那)(?:个|辆|台|种|套|位|名|条|款|部|件)$"
)
_LOCATIVE_SUBJECT_TAIL = re.compile(r"(?:身上|当中|之中|方面)$")
_COMPLETE_LOCATIVE_SUBJECT = re.compile(
    r"(?:(?:在|落在|发生在|位于|处于).{0,12}(?:身上|当中|之中|方面)|"
    r"(?:体现|反映|呈现)(?:在)?.{0,32}(?:身上|当中|之中|方面)|"
    r"(?:有|分为|包括|包含|涉及).{0,4}方面|"
    r"(?:另一个|这一|某一|各个)方面|"
    r"(?:是|属于).{0,20}(?:关键|重要)(?:的)?方面)$"
)


def detect_late_structural_frame_boundary(
    features: ChineseBoundaryFeatures,
) -> BoundarySignalMatch | None:
    """Return late structural-frame signals at legacy precedence."""
    if _LITERAL_FUNDAMENTAL_CALQUE.search(features.right):
        return _match("literal fundamental calque")

    if _PURCHASE_CLASSIFIER_TAIL.search(features.left):
        return _match("unfinished Chinese grammatical structure")

    if _LOCATIVE_SUBJECT_TAIL.search(features.left) and not _COMPLETE_LOCATIVE_SUBJECT.search(
        features.left
    ):
        return _match("unfinished Chinese locative subject")

    return None
