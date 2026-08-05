import pytest

from scripts.prepare_diarization_benchmark import _safe_uri, _validated_turns


def test_safe_uri_normalizes_dataset_path():
    assert _safe_uri("folder/Meeting One.wav", "fallback") == "Meeting_One"


def test_validated_turns_accepts_aligned_arrays():
    turns = _validated_turns(
        [0.0, 1.2],
        [1.0, 2.0],
        ["A", "B"],
        duration_seconds=2.0,
    )

    assert [(turn.start_ms, turn.end_ms, turn.speaker_id) for turn in turns] == [
        (0, 1_000, "A"),
        (1_200, 2_000, "B"),
    ]


def test_validated_turns_rejects_timestamp_beyond_audio():
    with pytest.raises(ValueError, match="beyond"):
        _validated_turns(
            [0.0],
            [3.0],
            ["A"],
            duration_seconds=2.0,
        )
