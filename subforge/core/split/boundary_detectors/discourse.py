"""Pure English spoken-discourse boundary detectors."""

from __future__ import annotations

import re

from subforge.core.split.boundary_features import EnglishBoundaryFeatures

_DISCOURSE_ONLY_TOKENS = {
    "and",
    "but",
    "guess",
    "i",
    "is",
    "it's",
    "mean",
    "so",
    "think",
    "well",
    "yeah",
    "you",
    "know",
}


def filler_demonstrative_noun(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\bthat\s+(?:really|very)\s+[a-z][a-z'’-]*[,]?$",
            features.semantic_left,
            re.IGNORECASE,
        )
        and re.match(
            r"^like[,]?\s+.+\s+one\b",
            features.semantic_right,
            re.IGNORECASE,
        )
    )


def standalone_bridge(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.fullmatch(
            r"(?:and|but|so)(?:\s+so)?(?:,?\s+(?:you\s+know|i\s+mean))?[,]?",
            features.left,
            re.IGNORECASE,
        )
    )


def filler_only_frame(features: EnglishBoundaryFeatures) -> bool:
    tokens = set(features.left_tokens)
    return bool(
        len(features.left_tokens) >= 3
        and tokens <= _DISCOURSE_ONLY_TOKENS
        and ({"think", "guess", "know", "mean"} & tokens)
    )


def sentence_opening_filler(features: EnglishBoundaryFeatures) -> bool:
    return bool(re.search(r"[.!?,;]\s+i\s+mean,?$", features.left, re.IGNORECASE))


def incomplete_predicate_before_filler(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\b(?:am|is|are|was|were|be|been|being|have|has|had|do|does|did|"
            r"can|could|may|might|must|shall|should|will|would|not),?\s+"
            r"(?:you\s+know|i\s+mean),?$",
            features.left,
            re.IGNORECASE,
        )
    )


def predicate_after_modifier_strong(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\b(?:it|this|that)(?:['’]s|\s+is)\s+actually,?\s+"
            r"(?:you\s+know|i\s+mean),?$",
            features.left,
            re.IGNORECASE,
        )
    )


def predicate_after_modifier(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\b(?:it|this|that)(?:['’]s|\s+is)\s+actually,?$",
            features.left,
            re.IGNORECASE,
        )
    )


def sentence_opening_opinion(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"[.!?]\s+(?:i|we|you)\s+(?:think|guess|believe),?$",
            features.left,
            re.IGNORECASE,
        )
    )


def parenthetical_opinion(
    features: EnglishBoundaryFeatures,
    *,
    head_is_incomplete_predicate: bool,
) -> bool:
    return head_is_incomplete_predicate and bool(
        re.search(
            r"\b(?:which|that|what)\s+(?:i|we|you|they|he|she)\s+"
            r"(?:think|guess|believe),?$",
            features.semantic_left,
            re.IGNORECASE,
        )
    )


def standalone_opinion(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.fullmatch(
            r"(?:i|we|you)\s+(?:think|guess|believe),?",
            features.left,
            re.IGNORECASE,
        )
        and re.match(
            r"^(?:\d|this|that|the|a|an|it|there|[a-z]+ing\b)",
            features.right,
            re.IGNORECASE,
        )
    )


def transitive_predicate_before_filler(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\b(?:i|we|you|they)\s+(?:(?:can|could|do|did)\s+)?"
            r"(?:see|saw|notice|noticed|find|found),?\s+"
            r"(?:you\s+know|i\s+mean),?$",
            features.left,
            re.IGNORECASE,
        )
        and re.match(
            r"^(?:a|an|the|some|many|several|changes?)\b",
            features.right,
            re.IGNORECASE,
        )
    )


def frame_following_clause(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(r"\bat\s+the\s+same\s+time,?$", features.left, re.IGNORECASE)
        and re.match(
            r"^(?:i|you|he|she|it|we|they)(?:['’](?:m|re|s|ve|d|ll))?\b",
            features.right,
            flags=re.IGNORECASE,
        )
    )
