"""Chinese adverb, comparison, and pronoun boundary defects."""

from __future__ import annotations

import re

from subforge.core.translate.quality.boundary_features import ChineseBoundaryFeatures

from .visible_pause import BoundarySignalMatch, _match

_INCOMPLETE_ZAI_TAIL = re.compile(r"(?:已经|正在|仍然|仍|还|也|都|确实)在$")
_ADVERBIAL_TAIL = re.compile(r"(?:仍然|依然|仍旧)$")
_DEGREE_TAIL = re.compile(r"一路$")
_DEGREE_START = re.compile(r"^(?:升|涨|高|攀|上|到|达到)")
_COMPARISON_TAIL = re.compile(r"(?:像|如|不及|不如)[^。！？]*$")
_COMPARISON_RESULT_START = re.compile(r"^(?:那样|一样)")
_PRONOUN_TAIL = re.compile(r"(?:我|你|他|她|它|我们|你们|他们)$")
_STANDALONE_SUBJECTS = frozenset(
    {"我", "你", "他", "她", "它", "我们", "你们", "他们", "她们", "它们"}
)


def detect_adverb_pronoun_attachment_boundary(
    features: ChineseBoundaryFeatures,
) -> BoundarySignalMatch | None:
    """Return adverb, comparison, and pronoun signals at legacy precedence."""
    if _INCOMPLETE_ZAI_TAIL.search(features.left):
        return _match("unfinished Chinese grammatical structure")

    if _ADVERBIAL_TAIL.search(features.left):
        return _match("unfinished Chinese adverbial predicate")

    if _DEGREE_TAIL.search(features.left) and _DEGREE_START.match(features.right):
        return _match("unfinished Chinese degree phrase")

    if _COMPARISON_TAIL.search(features.left) and _COMPARISON_RESULT_START.match(
        features.right
    ):
        return _match("comparison phrase is stranded")

    if _PRONOUN_TAIL.search(features.left) and len(re.sub(r"\s+", "", features.left)) >= 5:
        # This remains a context-audit candidate rather than a hard rejection.
        return _match("possible pronoun boundary")

    if features.left in _STANDALONE_SUBJECTS:
        return _match("standalone subject is separated from its predicate")

    return None
