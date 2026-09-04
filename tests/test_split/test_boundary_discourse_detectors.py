import pytest

from subforge.core.split.boundary_detectors import discourse
from subforge.core.split.boundary_features import extract_english_boundary_features


@pytest.mark.parametrize(
    ("detector_name", "left", "right"),
    [
        ("filler_demonstrative_noun", "that really small", "like the tiny one"),
        ("standalone_bridge", "and so, you know,", "this works"),
        ("filler_only_frame", "well I think", "this works"),
        ("sentence_opening_filler", "Fine. I mean,", "this works"),
        ("incomplete_predicate_before_filler", "It is, you know,", "still useful"),
        (
            "predicate_after_modifier_strong",
            "This is actually, you know,",
            "still useful",
        ),
        ("predicate_after_modifier", "This is actually,", "still useful"),
        ("sentence_opening_opinion", "Fine. I think,", "this works"),
        ("standalone_opinion", "I think,", "this works"),
        (
            "transitive_predicate_before_filler",
            "I can see, you know,",
            "the issue",
        ),
        ("frame_following_clause", "At the same time,", "we can continue"),
        ("frame_following_clause", "The estimate rose. Initially,", "Cost was lower"),
        (
            "frame_following_clause",
            "It was cheaper. So, as you'll learn,",
            "the final route changed",
        ),
        ("frame_following_clause", "Officials expected more. In 2018,", "Doug Ford became premier"),
    ],
)
def test_discourse_detector_activation(detector_name: str, left: str, right: str) -> None:
    features = extract_english_boundary_features(left, right)

    assert getattr(discourse, detector_name)(features) is True


def test_parenthetical_opinion_uses_explicit_head_classification() -> None:
    features = extract_english_boundary_features("which I think,", "is useful")

    assert discourse.parenthetical_opinion(
        features,
        head_is_incomplete_predicate=True,
    )
    assert not discourse.parenthetical_opinion(
        features,
        head_is_incomplete_predicate=False,
    )


def test_filler_only_frame_rejects_content_words() -> None:
    features = extract_english_boundary_features("well I think cars", "are useful")

    assert discourse.filler_only_frame(features) is False
