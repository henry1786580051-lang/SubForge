"""Pure English coordination-boundary detectors."""

from __future__ import annotations

import re

from subforge.core.split.boundary_features import TERMINAL_RE, EnglishBoundaryFeatures


def relative_clause_subject(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(r"\b(?:that|who|which)\s+[A-Z][A-Za-z'’-]*$", features.left)
        and re.match(
            r"^and\s+(?:I|you|he|she|it|we|they)\s+"
            r"(?:am|are|can|could|did|do|does|had|has|have|is|may|might|must|"
            r"shall|should|was|were|will|would|[a-z][a-z'’-]*(?:ed|s)?)\b",
            features.right,
        )
    )


def paired_contrast(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\b(?:on|upon)\s+(?:the\s+)?(?:one|other)\s+hand$",
            features.semantic_left,
            re.IGNORECASE,
        )
        and re.match(r"^(?:and|or|but)\b", features.semantic_right, re.IGNORECASE)
    )


def directional_names(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        features.tail in {"east", "north", "south", "west"}
        and features.head in {"and", "or"}
        and len(features.right_tokens) > 1
        and features.right_tokens[1] in {"east", "north", "south", "west"}
    )


def short_predicate_continuation(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        features.head == "and"
        and len(features.left_tokens) <= 3
        and features.right[:1].islower()
        and not TERMINAL_RE.search(features.left)
        and re.match(
            r"^(?:i|you|he|she|it|we|they)\b",
            features.left,
            re.IGNORECASE,
        )
        and re.match(r"^and\s+[a-z]+", features.right, re.IGNORECASE)
    )


def automotive_intake_exhaust(features: EnglishBoundaryFeatures) -> bool:
    return features.tail == "intake" and features.right_tokens[:2] == (
        "and",
        "exhaust",
    )


def reported_subject_member(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\b(?:note|mention|observe|report|say|show)\b.*\bthat\b.*"
            r"\band\s+[a-z][a-z'’-]*$",
            features.semantic_left,
            re.IGNORECASE,
        )
        and re.match(
            r"^[a-z][a-z'’-]*s\s+and\s+[a-z][a-z'’-]*\s+"
            r"(?:as|are|were|become|became|becomes|have|has|had)\b",
            features.semantic_right,
            re.IGNORECASE,
        )
    )


def subject_predicate(features: EnglishBoundaryFeatures) -> bool:
    return features.head in {
        "become",
        "becomes",
        "became",
        "is",
        "are",
        "was",
        "were",
        "have",
        "has",
        "had",
    } and bool(
        re.search(
            r"\b[a-z][a-z'’-]*(?:\s+[a-z][a-z'’-]*){0,3}\s+and\s+"
            r"[a-z][a-z'’-]*$",
            features.left,
            flags=re.IGNORECASE,
        )
    )


def noun_subject_shared_predicate_simple(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.match(
            r"^and\s+(?!(?:i|you|he|she|it|we|they)\b)"
            r"[a-z][a-z'’-]*(?:\s+[a-z][a-z'’-]*){0,2}\s+"
            r"(?:as|are|were|become|became|becomes|have|has|had)\b",
            features.semantic_right,
            re.IGNORECASE,
        )
    )


def noun_subject_shared_predicate_compound(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.match(
            r"^and\s+(?!(?:i|you|he|she|it|we|they)\b)"
            r"(?:[a-z][a-z'’-]*\s+){1,5}and\s+"
            r"[a-z][a-z'’-]*\s+(?:as|are|were|become|became|becomes|have|has|had)\b",
            features.semantic_right,
            re.IGNORECASE,
        )
    )


def final_noun_progressive_predicate(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.match(
            r"^and\s+[a-z][a-z'’-]*s\s+"
            r"(?:becoming|entering|leaving|moving|spreading|turning)\b",
            features.semantic_right,
            re.IGNORECASE,
        )
        and re.search(
            r"\b[a-z][a-z'’-]*s$",
            features.semantic_left,
            re.IGNORECASE,
        )
    )


def omitted_subject_predicate(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.match(
            r"^and(?:,?\s+(?:you\s+know|i\s+mean))?\s+"
            r"(?:wanted|decided|chose|hoped|tried|worked|wrote|said|thought)\b",
            features.semantic_right,
            re.IGNORECASE,
        )
    )


def noun_list_progressive_predicate(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.match(
            r"^[a-z][a-z'’-]*ing\b",
            features.semantic_right,
            re.IGNORECASE,
        )
        and re.search(
            r"\b[a-z][a-z'’-]*(?:,\s*|\s+and\s+)"
            r"[a-z][a-z'’-]*(?:,\s*and\s+[a-z][a-z'’-]*)?[,]?$",
            features.semantic_left,
            re.IGNORECASE,
        )
    )


def what_clause_predicate(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.match(r"^(?:and\s+)?what\b", features.left, re.IGNORECASE)
        and features.head == "and"
        and len(features.right_tokens) >= 2
        and features.right_tokens[1] in {
            "makes",
            "sets",
            "keeps",
            "gives",
            "gets",
            "drives",
        }
    )
