import pytest

from scripts.evaluate_web_terminology import cue_indices


def test_live_probe_bounds_deduplicates_and_keeps_source_order():
    assert cue_indices("3-5,1,4", 8) == [0, 2, 3, 4]


@pytest.mark.parametrize("value", ["0", "8-2", "1-65", "2-3-4", "", "9"])
def test_live_probe_rejects_invalid_or_excessive_sample(value):
    with pytest.raises(ValueError):
        cue_indices(value, 65 if value == "1-65" else 8)
