"""Pure English numeric-boundary detectors."""

from __future__ import annotations

import re

from subforge.core.split.boundary_features import EnglishBoundaryFeatures

_MONTHS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}


def measurement_comparative(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\b(?:\d[\d,.]*|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
            r"(?:centimetres?|centimeters?|feet|foot|inches?|kilometres?|kilometers?|"
            r"metres?|meters?|miles?|percent|%)$",
            features.semantic_left,
            re.IGNORECASE,
        )
        and re.match(
            r"^(?:closer|farther|further|higher|lower|greater|less|more|nearer|"
            r"shorter|taller|wider)\b",
            features.semantic_right,
            re.IGNORECASE,
        )
    )


def approximate_magnitude(features: EnglishBoundaryFeatures) -> bool:
    return features.tail in {"about", "around", "approximately", "roughly"} and bool(
        re.match(r"^\d[\d,.]*\b", features.semantic_right)
    )


def value_unit_or_noun(features: EnglishBoundaryFeatures) -> bool:
    if standalone_magnitude_named_population(features):
        return True
    completed_calendar_year = bool(
        re.search(r"\b(?:19|20)\d{2},$", features.semantic_left)
    )
    return bool(
        re.fullmatch(r"\d[\d,.]*", features.tail)
        and features.right[:1].islower()
        and not completed_calendar_year
    )


def standalone_magnitude_named_population(features: EnglishBoundaryFeatures) -> bool:
    """Keep a bare quantity with its capitalized plural population noun.

    Restrict the left side to a quantity fragment: a completed clause such as
    ``We raised a million`` can safely precede a new subject. Do not broadly
    treat every occurrence of ``million`` as an unfinished noun phrase.
    """
    return bool(
        features.tail in {"hundred", "thousand", "million", "billion", "trillion"}
        and re.match(r"^[A-Z][a-z]{2,}s\b", features.right)
        and features.head not in {"this", "thanks", "yours", "ours", "theirs", "hers"}
        and features.eligible
        and re.fullmatch(
            r"(?:(?:almost|nearly|about|around|approximately|roughly|over|under|"
            r"more than|less than|at least|at most)\s+)?"
            r"(?:a|one|two|three|four|five|six|seven|eight|nine|ten|\d[\d,.]*)\s+"
            r"(?:hundred|thousand|million|billion|trillion)",
            features.left,
            re.IGNORECASE,
        )
    )


def calendar_month_year(features: EnglishBoundaryFeatures) -> bool:
    return features.tail in _MONTHS and bool(
        re.match(r"^(?:19|20)\d{2}\b", features.semantic_right)
    )


def compound_modifier(features: EnglishBoundaryFeatures) -> bool:
    # ASR and source optimization may spell the same modifier as ``six speed``
    # or ``six-speed``. Keep both on the existing numeric dependency path;
    # sentence dashes (en/em dash) are not compound-word separators.
    return bool(
        re.search(
            r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)"
            r"(?:\s+|[ \t]*[-‐‑][ \t]*)"
            r"(?:channel|cylinder|door|inch|liter|litre|seat|speaker|speed)$",
            features.semantic_left,
            re.IGNORECASE,
        )
        and re.match(r"^[A-Za-z][A-Za-z0-9&.+/-]*\b", features.right)
    )


def multiplier_or_unit(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(r"\b\d[\d,.]*$", features.semantic_left)
        and features.head in {"percent", "percentage", "times"}
    )


def mixed_measurement(features: EnglishBoundaryFeatures) -> bool:
    return features.tail in {
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
    } and bool(
        re.match(
            r"^and\s+a\s+half\s+(?:foot|feet|inch|inches)\b",
            features.right,
            re.IGNORECASE,
        )
    )


def range_conjunction(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\bbetween\s+(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|\d+)$",
            features.left,
            re.IGNORECASE,
        )
        and re.match(
            r"^and\s+(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b",
            features.right,
            re.IGNORECASE,
        )
    )


def model_year_vehicle_name(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(r"\bthe\s+(?:19|20)?\d{2}$", features.semantic_left, re.IGNORECASE)
        and re.match(r"^[A-Za-z]+-?\d+[A-Za-z0-9-]*\b", features.right)
    )


def article_model_year_vehicle_name(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(r"\b(?:a|an|the)\s+(?:19|20)\d{2}$", features.left, re.IGNORECASE)
        and (
            re.match(r"^[A-Z][A-Za-z0-9-]+\b", features.right)
            or re.match(
                r"^(?:acura|audi|bmw|cadillac|chevrolet|dodge|ford|gmc|honda|hyundai|"
                r"jeep|kia|lexus|lincoln|mazda|mercedes|nissan|porsche|ram|subaru|"
                r"tesla|toyota|volkswagen|volvo)\b",
                features.right,
                flags=re.IGNORECASE,
            )
        )
    )
