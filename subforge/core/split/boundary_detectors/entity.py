"""Pure English entity-boundary detectors."""

from __future__ import annotations

import re

from subforge.core.split.boundary_features import EnglishBoundaryFeatures

_VEHICLE_BRANDS = {
    "acura",
    "audi",
    "bmw",
    "cadillac",
    "chevrolet",
    "dodge",
    "ford",
    "gmc",
    "honda",
    "hyundai",
    "jeep",
    "kia",
    "lexus",
    "lincoln",
    "mazda",
    "mercedes",
    "nissan",
    "porsche",
    "ram",
    "subaru",
    "tesla",
    "toyota",
    "volkswagen",
    "volvo",
}
_US_STATE_NAMES = {
    "alabama",
    "alaska",
    "arizona",
    "arkansas",
    "california",
    "colorado",
    "connecticut",
    "delaware",
    "florida",
    "georgia",
    "hawaii",
    "idaho",
    "illinois",
    "indiana",
    "iowa",
    "kansas",
    "kentucky",
    "louisiana",
    "maine",
    "maryland",
    "massachusetts",
    "michigan",
    "minnesota",
    "mississippi",
    "missouri",
    "montana",
    "nebraska",
    "nevada",
    "ohio",
    "oklahoma",
    "oregon",
    "pennsylvania",
    "tennessee",
    "texas",
    "utah",
    "vermont",
    "virginia",
    "washington",
    "wisconsin",
    "wyoming",
}
_NON_NAME_CONTINUATIONS = {
    "all",
    "and",
    "but",
    "hey",
    "just",
    "next",
    "no",
    "now",
    "okay",
    "so",
    "then",
    "well",
    "yes",
}


def powertrain_vehicle_name(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\b(?:a|an|the)\s+(?:turbocharged|supercharged)$",
            features.semantic_left,
            re.IGNORECASE,
        )
        and re.match(
            r"^[A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*)?\b",
            features.right,
        )
    )


def vehicle_brand_model(features: EnglishBoundaryFeatures) -> bool:
    return features.tail in _VEHICLE_BRANDS and features.head in {
        "car",
        "cars",
        "crossover",
        "electric",
        "hybrid",
        "model",
        "models",
        "sedan",
        "suv",
        "truck",
        "vehicle",
        "vehicles",
    }


def alphanumeric_model_alternative(features: EnglishBoundaryFeatures) -> bool:
    return features.head == "or" and bool(re.fullmatch(r"[a-z]+\d+", features.tail))


def proper_name(features: EnglishBoundaryFeatures) -> bool:
    left_name = re.search(r"(?:^|\s)([A-Z][A-Za-z'’-]{1,})$", features.left)
    right_name = re.match(r"^([A-Z][A-Za-z'’-]{1,})\b", features.right)
    return bool(
        left_name and right_name and right_name.group(1).lower() not in _NON_NAME_CONTINUATIONS
    )


def attributive_proper_name(features: EnglishBoundaryFeatures) -> bool:
    """Keep an attributive place or organization name with its lowercase head noun."""
    return bool(
        features.right[:1].islower()
        and re.search(
            r"\b(?:a|an|the)\s+(?:[A-Z][A-Za-z'’-]*\s+){0,2}"
            r"[A-Z][A-Za-z'’-]*$",
            features.left,
        )
        and re.match(
            r"^(?!(?:and|as|but|for|if|or|so|than|that|then|when|while|with)\b)"
            r"[a-z][a-z'’-]*\b",
            features.right,
        )
    )


def city_state(features: EnglishBoundaryFeatures) -> bool:
    """Match a capitalized city tail followed by a known single-token state."""
    return bool(
        re.search(r"\b([A-Z][A-Za-z'’-]+),\s*$", features.left) and features.head in _US_STATE_NAMES
    )


def vehicle_trim_model(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.fullmatch(
            r"(?:rt|srt|amg|rs|m)\d{1,3}",
            features.tail,
            flags=re.IGNORECASE,
        )
        and re.match(r"^[A-Z][A-Za-z0-9-]+\b", features.right)
    )
