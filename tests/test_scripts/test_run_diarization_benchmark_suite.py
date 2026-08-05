import pytest

from scripts.run_diarization_benchmark_suite import _aggregate


def test_aggregate_uses_reference_speaker_time_for_der():
    reports = [
        {
            "dataset": "vox",
            "mode": "known",
            "runtime_seconds": 2.0,
            "audio_duration_seconds": 10.0,
            "execution_device": "mps",
            "reference_speakers": 2,
            "detected_speakers": 2,
            "diagnostics": {"short_islands_1500ms": 1},
            "accuracy": {
                "reference_speaker_time_seconds": 10.0,
                "jer": 0.2,
                "der_components": {
                    "missed detection": 1.0,
                    "false alarm": 0.5,
                    "confusion": 0.5,
                },
                "boundary_250ms": {"f1": 0.7},
                "boundary_500ms": {"f1": 0.8},
            },
        }
    ]

    result = _aggregate(reports)["vox:known"]

    assert result["strict_regular_der"] == pytest.approx(0.2)
    assert result["real_time_factor"] == pytest.approx(0.2)
    assert result["speaker_count_exact"] == 1
