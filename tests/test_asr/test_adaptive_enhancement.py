from pathlib import Path

import numpy as np
import soundfile as sf

from subforge.core.asr.adaptive_enhancement import (
    EnhancementCandidate,
    build_calibration_reel,
    choose_candidate,
    select_calibration_windows,
)
from subforge.core.asr.speaker_diarization import SpeakerTurn


def _candidate(
    attenuation_db,
    score,
    words,
    speaker_1,
    speaker_2,
    repeated=0.0,
):
    return EnhancementCandidate(
        attenuation_db=attenuation_db,
        score=score,
        word_count=words,
        repeated_ratio=repeated,
        speaker_word_rates={"Speaker 1": speaker_1, "Speaker 2": speaker_2},
    )


def test_select_calibration_windows_covers_both_speakers_without_large_overlap():
    turns = [
        SpeakerTurn(1_000, 6_000, "Speaker 1"),
        SpeakerTurn(30_000, 36_000, "Speaker 1"),
        SpeakerTurn(60_000, 67_000, "Speaker 2"),
        SpeakerTurn(90_000, 96_000, "Speaker 2"),
    ]

    windows = select_calibration_windows(
        turns,
        audio_duration_ms=120_000,
        window_ms=10_000,
        windows_per_speaker=1,
    )

    assert len(windows) == 2
    assert windows[0].start_ms <= 30_000 < windows[0].end_ms
    assert windows[1].start_ms <= 60_000 < windows[1].end_ms
    assert windows[0].end_ms <= windows[1].start_ms


def test_build_calibration_reel_remaps_turns(tmp_path: Path):
    audio_path = tmp_path / "source.wav"
    output_path = tmp_path / "reel.wav"
    sf.write(audio_path, np.zeros(16_000 * 40, dtype=np.float32), 16_000)
    turns = [
        SpeakerTurn(2_000, 8_000, "Speaker 1"),
        SpeakerTurn(24_000, 30_000, "Speaker 2"),
    ]

    mapped = build_calibration_reel(str(audio_path), turns, str(output_path))

    assert output_path.is_file()
    assert {turn.speaker_id for turn in mapped} == {"Speaker 1", "Speaker 2"}
    assert all(turn.end_ms > turn.start_ms >= 0 for turn in mapped)
    assert sf.info(output_path).duration < sf.info(audio_path).duration


def test_choose_candidate_requires_clear_gain_without_losing_quiet_speaker():
    baseline = _candidate(None, 5.0, 100, 2.0, 1.0)
    loses_quiet_speaker = _candidate(6.0, 5.8, 105, 2.4, 0.8)
    safe_gain = _candidate(12.0, 5.4, 104, 2.2, 1.02)

    selected = choose_candidate([baseline, loses_quiet_speaker, safe_gain])

    assert selected.attenuation_db == 12.0


def test_choose_candidate_prefers_original_for_small_or_suspicious_gain():
    baseline = _candidate(None, 5.0, 100, 2.0, 1.0)
    marginal = _candidate(6.0, 5.05, 103, 2.03, 1.0)
    too_many_words = _candidate(12.0, 7.0, 130, 2.8, 1.4)

    selected = choose_candidate([baseline, marginal, too_many_words])

    assert selected.attenuation_db is None
