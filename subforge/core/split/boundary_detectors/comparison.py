"""Pure English comparison-boundary detectors."""

from __future__ import annotations

import re

from subforge.core.split.boundary_features import EnglishBoundaryFeatures


def clause_after_than(features: EnglishBoundaryFeatures) -> bool:
    return features.tail == "than"


def noun_phrase_before_than(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(r"\b(?:less|more)\b", features.semantic_left, re.IGNORECASE)
        and re.match(
            r"^[a-z][a-z'’-]*\s+than\b",
            features.semantic_right,
            re.IGNORECASE,
        )
    )


def scalar_predicate(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\b(?:could|may|might|should|would)?\s*(?:have\s+)?"
            r"(?:cost|last|measure|take|weigh)\s*$",
            features.semantic_left,
            re.IGNORECASE,
        )
        and re.match(r"^(?:less|more)\b", features.semantic_right, re.IGNORECASE)
    )


def frame_object(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\b(?:arrange|arranged|behave|behaved|feel|felt|look|looked|"
            r"seem|seemed|sound|sounded|work|worked)(?:\s+\S+){0,6}\s+like$",
            features.semantic_left,
            re.IGNORECASE,
        )
    )


def repeated_degree(features: EnglishBoundaryFeatures) -> bool:
    return features.tail == "more" and features.right_tokens[:2] == ("and", "more")


def same_as(features: EnglishBoundaryFeatures) -> bool:
    return features.tail == "same" and features.head == "as"


def frame_counterpart(features: EnglishBoundaryFeatures) -> bool:
    return features.head in {"as", "than"} and bool(
        re.search(
            r"\b(?:same|similar|different)\b[^.!?]*\b(?:life|work)$",
            features.semantic_left,
            re.IGNORECASE,
        )
    )


def negated_complement(features: EnglishBoundaryFeatures) -> bool:
    return features.head in {"as", "than"} and bool(
        re.search(
            r"\b(?:isn['’]t|is\s+not|aren['’]t|are\s+not|wasn['’]t|was\s+not|"
            r"weren['’]t|were\s+not)\s+(?:quite|nearly|almost|as|so)$",
            features.left,
            flags=re.IGNORECASE,
        )
    )


def dynamic_complement(features: EnglishBoundaryFeatures) -> bool:
    return features.tail == "like" and features.head in {
        "this",
        "that",
        "it",
        "these",
        "those",
    }


def example(features: EnglishBoundaryFeatures) -> bool:
    return features.tail == "like" and bool(
        re.match(r"^[A-Z][A-Za-z'’-]+\b", features.right)
    )


def auxiliary_after_already(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        features.tail == "already"
        and features.head in {"is", "are", "was", "were"}
        and re.search(
            r"\bthan\b[^.!?]*\balready$",
            features.left,
            re.IGNORECASE,
        )
    )
