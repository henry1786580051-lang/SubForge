from types import SimpleNamespace

import pytest

from subforge.core.asr import non_speech_review as review
from subforge.core.asr.asr_data import ASRData, ASRDataSeg


def data(text="Thank you."):
    parts = text.split()
    middle = [
        ASRDataSeg(t, 15000 + i * 300, 15250 + i * 300, timestamp_granularity="word")
        for i, t in enumerate(parts)
    ]
    result = ASRData(
        [
            ASRDataSeg("Previous", 0, 300, timestamp_granularity="word"),
            ASRDataSeg("speech", 320, 600, timestamp_granularity="word"),
            *middle,
            ASRDataSeg("Next", 30000, 30300, timestamp_granularity="word"),
            ASRDataSeg("sentence", 30320, 30600, timestamp_granularity="word"),
        ]
    )
    result.media_duration_ms = 31000
    return result


def acoustic(regular=638, strict=0, independent=0, surrounding=0):
    return {
        name: {"word_overlap_ms": value, "context_overlap_ms": max(value, surrounding)}
        for name, value in [("regular", regular), ("strict", strict), ("independent", independent)]
    }


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(review, "_acoustic_evidence", lambda *a: acoustic())
    monkeypatch.setattr(
        review,
        "_probe",
        lambda *a: {
            "no_speech_prob": 0.783,
            "alignment_scores": [0.1, 0.2],
            "alignment_complete": True,
            "decoded_text": "Thank you.",
        },
    )
    records = []
    monkeypatch.setattr(review, "_save_evidence", lambda r: records.extend(r))
    return SimpleNamespace(audio_path="original.wav"), SimpleNamespace(cancel_event=None), records


def run(a, setup):
    ctx, cfg, _ = setup
    return review.review_non_speech(a, "original.wav", cfg, lambda *a: None, None, ctx)


def test_music_false_positive_is_not_saved_by_repeated_text(setup):
    result = run(data(), setup)
    assert [s.text for s in result.segments] == ["Previous", "speech", "Next", "sentence"]
    assert result.confirmed_non_speech_ranges
    assert not result.coverage_issues
    assert setup[2][0]["status"] == "remove"


@pytest.mark.parametrize("phrase", ["Thank you.", "Yes", "Okay"])
@pytest.mark.parametrize("detector", ["strict", "independent"])
def test_real_short_reply_is_kept_without_decoding(monkeypatch, setup, phrase, detector):
    monkeypatch.setattr(review, "_acoustic_evidence", lambda *a: acoustic(**{detector: 100}))
    monkeypatch.setattr(
        review, "_probe", lambda *a: pytest.fail("real reply should not be decoded")
    )
    original = data(phrase)
    before = [s.text for s in original.segments]
    result = run(original, setup)
    assert [s.text for s in result.segments] == before
    assert not result.coverage_issues


@pytest.mark.parametrize(
    "probe",
    [
        {"no_speech_prob": 0.9, "alignment_scores": [0.8, 0.2], "alignment_complete": True},
        {"no_speech_prob": 0.3, "alignment_scores": [0.1, 0.2], "alignment_complete": True},
        {"no_speech_prob": None, "alignment_scores": [0.1, 0.2], "alignment_complete": True},
        {"no_speech_prob": 0.9, "alignment_scores": [0.1], "alignment_complete": False},
    ],
)
def test_conflicting_or_missing_evidence_requires_review(monkeypatch, setup, probe):
    monkeypatch.setattr(review, "_probe", lambda *a: probe)
    result = run(data(), setup)
    assert len(result.segments) == 6
    assert result.coverage_issues[0]["reason"] == "suspected_non_speech_text"
    assert not result.confirmed_non_speech_ranges


def test_music_with_nearby_voice_is_not_automatically_deleted(monkeypatch, setup):
    monkeypatch.setattr(review, "_acoustic_evidence", lambda *a: acoustic(surrounding=300))
    assert run(data(), setup).coverage_issues


def test_short_response_inside_conversation_is_not_a_candidate(monkeypatch, setup):
    original = data()
    original.segments[-2].start_time = 16000
    monkeypatch.setattr(review, "_acoustic_evidence", lambda *a: pytest.fail("not isolated"))
    assert len(run(original, setup).segments) == 6


def test_no_blacklist_required_for_removal(setup):
    assert len(run(data("Random phrase"), setup).segments) == 4


def test_unavailable_detector_keeps_content_and_reports(monkeypatch, setup):
    monkeypatch.setattr(
        review, "_acoustic_evidence", lambda *a: (_ for _ in ()).throw(RuntimeError("offline"))
    )
    result = run(data(), setup)
    assert len(result.segments) == 6
    assert result.coverage_issues


def test_budget_keeps_uncertain_words(monkeypatch, setup):
    monkeypatch.setattr(review, "MAX_REVIEWS", 0)
    monkeypatch.setattr(review, "_probe", lambda *a: pytest.fail("budget"))
    assert run(data(), setup).coverage_issues


def test_cancellation_during_probe_is_propagated(monkeypatch, setup):
    import threading

    event = threading.Event()
    setup[1].cancel_event = event

    def probe(*a):
        event.set()
        raise RuntimeError("stopped")

    monkeypatch.setattr(review, "_probe", probe)
    with pytest.raises(RuntimeError, match="cancelled"):
        run(data(), setup)


def test_whisperx_short_groups_wait_for_acoustic_review():
    original = data()
    original.filter_hallucinations(
        speech_segments=[],
        strict_speech_segments=[],
        corroborating_speech_segments=[],
        media_duration_ms=31000,
        defer_short_groups=True,
    )
    assert len(original.segments) == 6


def test_final_gap_audit_does_not_reinsert_confirmed_music(monkeypatch):
    import numpy as np

    from subforge.core.asr import final_coverage, whisperx_asr

    original = data()
    original.segments = [s for s in original.segments if not 15000 <= s.start_time < 16000]
    original.confirmed_non_speech_ranges = [{"start": 10, "end": 21}]
    monkeypatch.setattr(
        whisperx_asr, "_detect_speech_in_mlx_gaps", lambda *a, **k: [(11000, 20000)]
    )
    # Observe evidence passed to gap finding: exclude only certified intervals.
    captured = []

    def critical(aligned, speech, duration):
        captured.extend(speech)
        return []

    monkeypatch.setattr(whisperx_asr, "_critical_aligned_speech_gaps", critical)
    final_coverage._audit(original, np.zeros(31 * 16000))
    assert not captured
    monkeypatch.setattr(
        whisperx_asr, "_detect_speech_in_mlx_gaps", lambda *a, **k: [(11000, 25000)]
    )
    final_coverage._audit(original, np.zeros(31 * 16000))
    assert captured and all(a >= 21000 for a, b in captured)
