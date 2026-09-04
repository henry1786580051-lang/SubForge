import sys

import pytest

from scripts import benchmark_diarization


@pytest.mark.parametrize(
    ("options", "minimum", "fixed"),
    [
        (["--auto-speakers"], 1, None),
        (["--auto-speakers", "--min-speakers", "2"], 2, None),
        (["--num-speakers", "5"], 1, 5),
    ],
)
def test_benchmark_count_defaults_match_application(monkeypatch, options, minimum, fixed):
    received = {}

    def _run(args):
        received.update(vars(args))
        return {}

    monkeypatch.setattr(benchmark_diarization, "_run", _run)
    monkeypatch.setattr(sys, "argv", ["benchmark", "run", "audio.wav", "result.rttm", *options])

    assert benchmark_diarization.main() == 0
    assert received["min_speakers"] == minimum
    assert received["max_speakers"] == 10
    assert received["num_speakers"] == fixed
