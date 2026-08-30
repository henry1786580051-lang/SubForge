from dataclasses import FrozenInstanceError

import pytest

from subforge.core.split.boundary_features import extract_english_boundary_features


def test_boundary_features_normalize_literal_and_semantic_tails_once() -> None:
    features = extract_english_boundary_features(
        "I think that, you know,",
        "I mean, this matters",
    )

    assert features.left_tokens[-3:] == ("that", "you", "know")
    assert features.tail == "know"
    assert features.semantic_left == "I think that"
    assert features.semantic_tail == "that"
    assert features.semantic_right == "this matters"


def test_boundary_features_preserve_ellipsis_and_capitalized_of_exceptions() -> None:
    ellipsis = extract_english_boundary_features("I think it's...", "still useful")
    dependent_of = extract_english_boundary_features("A large share", "Of the total capacity")
    terminal = extract_english_boundary_features("This is complete.", "A new sentence")

    assert ellipsis.eligible is True
    assert dependent_of.eligible is True
    assert dependent_of.capitalized_dependent_of is True
    assert terminal.eligible is False


def test_boundary_features_are_immutable() -> None:
    features = extract_english_boundary_features("This", "works")

    with pytest.raises(FrozenInstanceError):
        features.tail = "that"  # type: ignore[misc]
