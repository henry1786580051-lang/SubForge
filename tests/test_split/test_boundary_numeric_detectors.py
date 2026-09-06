import pytest

from subforge.core.split.boundary_detectors import numeric
from subforge.core.split.boundary_features import extract_english_boundary_features


@pytest.mark.parametrize(
    ("detector_name", "left", "right"),
    [
        ("measurement_comparative", "It is 10 feet", "higher than before"),
        ("approximate_magnitude", "It serves about", "500 people"),
        ("value_unit_or_noun", "It holds 500", "people at once"),
        ("calendar_month_year", "It opened in January", "2026"),
        ("compound_modifier", "It has a six speed", "manual transmission"),
        ("multiplier_or_unit", "It is 50", "percent larger"),
        ("mixed_measurement", "It is five", "and a half feet long"),
        ("range_conjunction", "It takes between two", "and three hours"),
        ("model_year_vehicle_name", "This is the 2026", "QX65 model"),
        ("article_model_year_vehicle_name", "This is a 2026", "Toyota Camry"),
    ],
)
def test_numeric_detector_activation(detector_name: str, left: str, right: str) -> None:
    features = extract_english_boundary_features(left, right)

    assert getattr(numeric, detector_name)(features) is True


def test_numeric_detector_exclusions_remain_narrow() -> None:
    calendar_year = extract_english_boundary_features("It opened in 2026,", "people arrived")
    bare_year = extract_english_boundary_features("It opened in 2026", "Toyota expanded")

    assert numeric.value_unit_or_noun(calendar_year) is False
    assert numeric.article_model_year_vehicle_name(bare_year) is False


@pytest.mark.parametrize("separator", [" ", "-", "‐", "‑", " - "])
@pytest.mark.parametrize("modifier,head", [
    ("six{sep}speed", "manual gearbox"),
    ("4{sep}channel", "audio interface"),
    ("two{sep}seat", "aircraft"),
    ("19{sep}inch", "display"),
])
def test_numeric_modifier_spelling_preserves_dependency(separator, modifier, head):
    from subforge.core.split.boundary import assess_english_boundary

    left = "It has a " + modifier.format(sep=separator)
    assessment = assess_english_boundary(left, head)
    assert any(
        c.rule_id == "split.boundary.english.numeric.compound_modifier"
        for c in assessment.contributions
    )
    assert assessment.unstable


@pytest.mark.parametrize("left,right", [
    ("It has a six-speed.", "Manual operation is optional."),
    ("It has a six-speed manual", "and works well."),
    ("It has speed", "control."),
    ("It has six—speed", "is irrelevant."),
    ("It has six–speed", "is irrelevant."),
    ("Model X6-speed", "controller"),
    ("", "manual"),
    ("It has a six-speed", ""),
])
def test_numeric_modifier_does_not_join_sentences_or_identifiers(left, right):
    assert not numeric.compound_modifier(extract_english_boundary_features(left, right))
