import pytest

from subforge.core.split.boundary_detectors import comparison
from subforge.core.split.boundary_features import extract_english_boundary_features


@pytest.mark.parametrize(
    ("detector_name", "left", "right"),
    [
        ("clause_after_than", "It costs more than", "we expected"),
        ("noun_phrase_before_than", "It offers more comfort", "now than before"),
        ("scalar_predicate", "It could have cost", "less to build"),
        ("frame_object", "It worked like", "a charm"),
        ("repeated_degree", "It gets more", "and more useful"),
        ("same_as", "It feels the same", "as before"),
        ("frame_counterpart", "It creates a different work", "than before"),
        ("negated_complement", "It is not quite", "as useful"),
        ("dynamic_complement", "It looks like", "this"),
        ("example", "It looks like", "Toyota designed it"),
        ("auxiliary_after_already", "It is better than before already", "is clear"),
    ],
)
def test_comparison_detector_activation(detector_name: str, left: str, right: str) -> None:
    features = extract_english_boundary_features(left, right)

    assert getattr(comparison, detector_name)(features) is True


def test_comparison_detectors_do_not_generalize_from_unrelated_like() -> None:
    features = extract_english_boundary_features("I like", "lower prices")

    assert comparison.frame_object(features) is False
    assert comparison.dynamic_complement(features) is False
    assert comparison.example(features) is False
