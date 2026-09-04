import pytest

from subforge.core.split.boundary_detectors import entity
from subforge.core.split.boundary_features import extract_english_boundary_features


@pytest.mark.parametrize(
    ("detector_name", "left", "right"),
    [
        ("powertrain_vehicle_name", "This is the supercharged", "Ford F-150"),
        ("vehicle_brand_model", "It is a Toyota", "SUV"),
        ("alphanumeric_model_alternative", "Choose the qx65", "or qx80"),
        ("proper_name", "The president is Donald", "Trump"),
        ("attributive_proper_name", "proposed by a New York", "consulting firm"),
        ("city_state", "We arrived in Ypsilanti,", "Michigan"),
        ("vehicle_trim_model", "This is the RT392", "Durango"),
    ],
)
def test_entity_detector_activation(detector_name: str, left: str, right: str) -> None:
    features = extract_english_boundary_features(left, right)

    assert getattr(entity, detector_name)(features) is True


def test_entity_detectors_preserve_case_and_continuation_exclusions() -> None:
    lowercase_name = extract_english_boundary_features("The president is donald", "trump")
    discourse_start = extract_english_boundary_features("I Like", "And this works")
    lowercase_city = extract_english_boundary_features("We arrived in ypsilanti,", "michigan")

    assert entity.proper_name(lowercase_name) is False
    assert entity.proper_name(discourse_start) is False
    assert entity.city_state(lowercase_city) is False
    assert (
        entity.attributive_proper_name(
            extract_english_boundary_features("We arrived in New York", "and checked in")
        )
        is False
    )
