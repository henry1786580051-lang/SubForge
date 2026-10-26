import copy
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from subforge.core.asr import repetition_review as review
from subforge.core.asr.asr_data import ASRData, ASRDataSeg


def words(items):
    return ASRData([ASRDataSeg(t, a, b, timestamp_granularity="word") for t, a, b in items])


def state(segments):
    return copy.deepcopy([s.__dict__ for s in segments])


def loop(phrase="Are we recording now?", repeats=20):
    tokens = phrase.split()
    middle = [
        (text, 10000 + i * 1500 + j * 120, 10100 + i * 1500 + j * 120)
        for i in range(repeats)
        for j, text in enumerate(tokens)
    ]
    return words([("Second.", 8000, 8500), *middle, ("Well.", 42000, 42400)])


class Context:
    audio_path = "original.wav"

    def samples(self):
        return np.zeros(50 * 16000, dtype=np.float32)


@pytest.fixture
def setup(monkeypatch):
    context = Context()
    config = SimpleNamespace(cancel_event=None)
    records, windows = [], []
    monkeypatch.setattr(review, "_save_evidence", lambda r: records.extend(r))
    monkeypatch.setattr(review, "_speech_ranges", lambda *a: {"strict": [], "independent": []})

    def decode(ctx, left, right, *_):
        assert ctx is context
        windows.append((left, right))
        return words([])

    monkeypatch.setattr(review, "_decode", decode)
    return SimpleNamespace(context=context, config=config, records=records, windows=windows)


def run(data, setup):
    return review.review_repetitions(
        data, "original.wav", setup.config, lambda *a: None, None, setup.context
    )


@pytest.mark.parametrize("phrase", ["Are we recording now?", "Keep this rolling", "the"])
def test_repetition_split_across_word_cues_is_reviewed_without_a_blacklist(phrase):
    data = loop(phrase)
    suspects = review.candidates(data)
    assert len(suspects) == 1
    assert suspects[0].repeats == 20
    assert suspects[0].first == 1
    assert suspects[0].stop == len(data.segments) - 1


@pytest.mark.parametrize("repeats", [1, 2, 3])
def test_ordinary_emphasis_does_not_trigger_redecoding(repeats, setup):
    data = loop(repeats=repeats)
    before = state(data.segments)
    assert state(run(data, setup).segments) == before
    assert not setup.windows


def test_phrase_repeated_at_distant_times_is_not_a_loop(setup):
    data = loop()
    for i, s in enumerate(data.segments[1:-1]):
        s.start_time += i // 4 * 20000
        s.end_time += i // 4 * 20000
    assert not review.candidates(data)


def test_rejected_loop_is_removed_only_after_two_original_context_decodes(setup):
    data = loop()
    edges = state([data.segments[i] for i in [0, -1]])
    result = run(data, setup)
    assert state(result.segments) == edges
    assert len(set(setup.windows)) == 2
    assert not result.coverage_issues
    assert setup.records[0]["status"] == "repaired"
    assert result.confirmed_non_speech_ranges


def test_small_alignment_jitter_does_not_change_audio_contexts(setup):
    first = loop()
    for segment in first.segments[1:-1]:
        segment.start_time += 20
        segment.end_time += 20
    run(first, setup)
    initial = list(setup.windows)
    setup.windows.clear()
    # Compare two alignments away from the second boundary, not two documents
    # whose windows legitimately round to different seconds.
    shifted = loop()
    for segment in shifted.segments[1:-1]:
        segment.start_time += 40
        segment.end_time += 40
    run(shifted, setup)
    assert initial == setup.windows


def test_real_short_voice_is_recovered_without_rewriting_neighbors(monkeypatch, setup):
    data = loop()
    before, after = state([data.segments[0], data.segments[-1]])
    voiced = {"strict": [(13120, 13248)], "independent": [(13120, 13312)]}
    monkeypatch.setattr(review, "_speech_ranges", lambda *a: voiced)
    replies = [words([("Oh", 12560, 12900), ("yeah.", 12900, 13200)]),
               words([("Oh", 12560, 12920), ("yeah.", 12920, 13200)])]
    monkeypatch.setattr(review, "_decode", lambda *a: replies.pop(0))
    result = run(data, setup)
    assert [s.text for s in result.segments] == ["Second.", "Oh", "yeah.", "Well."]
    assert state([result.segments[0], result.segments[-1]]) == [before, after]
    assert not any(r["start"] <= 13.2 <= r["end"] for r in result.confirmed_non_speech_ranges)
    assert not result.coverage_issues


def test_genuine_repeated_speech_is_preserved_when_decodes_and_acoustics_agree(monkeypatch, setup):
    data = loop()
    middle = data.segments[1:-1]
    speech = [(s.start_time, s.end_time) for s in middle]
    monkeypatch.setattr(review, "_speech_ranges", lambda *a: {"strict": speech, "independent": speech})
    monkeypatch.setattr(review, "_decode", lambda *a: ASRData(copy.deepcopy(middle)))
    before = state(data.segments)
    result = run(data, setup)
    assert state(result.segments) == before
    assert not result.coverage_issues
    assert setup.records[0]["status"] == "confirmed_speech"


@pytest.mark.parametrize("failure", ["speech", "disagreement", "unsupported", "boundary"])
def test_uncertain_repair_keeps_document_and_flags_blocking_review(monkeypatch, setup, failure):
    data = loop()
    before = state(data.segments)
    if failure == "speech":
        monkeypatch.setattr(review, "_speech_ranges", lambda *a: {"strict": [(15000, 16000)], "independent": []})
    else:
        replies = [words([("Oh", 14000, 14300)]), words([("yeah", 14000, 14300)])]
        if failure == "unsupported":
            replies = [words([("Oh", 14000, 14300)]) for _ in range(2)]
        if failure == "boundary":
            replies = [words([("Oh", 9900, 10100)]) for _ in range(2)]
        monkeypatch.setattr(review, "_decode", lambda *a: replies.pop(0))
    result = run(data, setup)
    assert state(result.segments) == before
    assert result.coverage_issues[0]["reason"] == "suspected_repetition"
    assert not result.confirmed_non_speech_ranges


@pytest.mark.parametrize("budget", ["MAX_REVIEWS", "MAX_DECODE_SECONDS", "MAX_LOOP_MS"])
def test_budget_exhaustion_retains_and_flags_suspect_loop(monkeypatch, setup, budget):
    monkeypatch.setattr(review, budget, 0)
    result = run(loop(), setup)
    assert len(result.segments) == 82
    assert result.coverage_issues
    assert not setup.windows


def test_review_always_uses_original_signal_even_when_timing_used_enhanced(monkeypatch, setup):
    setup.context.audio_path = "enhanced.wav"
    from subforge.core.asr import audio_analysis

    def original(path):
        assert path == "original.wav"
        setup.context.audio_path = path
        return setup.context

    monkeypatch.setattr(audio_analysis, "AudioAnalysisContext", original)
    assert len(run(loop(), setup).segments) == 2


def test_unavailable_detector_does_not_silently_accept_or_delete_loop(monkeypatch, setup):
    def unavailable(*a):
        raise RuntimeError("offline")

    monkeypatch.setattr(review, "_speech_ranges", unavailable)
    result = run(loop(), setup)
    assert len(result.segments) == 82
    assert result.coverage_issues
    assert not setup.windows


def test_cancellation_during_decode_propagates_without_mutation(monkeypatch, setup):
    data = loop()
    before = state(data.segments)
    setup.config.cancel_event = threading.Event()

    def cancel(*a):
        setup.config.cancel_event.set()
        raise RuntimeError("interrupted")

    monkeypatch.setattr(review, "_decode", cancel)
    with pytest.raises(RuntimeError, match="cancelled"):
        run(data, setup)
    assert state(data.segments) == before


def test_repetition_failure_message_identifies_problem_and_time_range():
    from subforge.core.asr.speech_gap_repair import coverage_issue_message

    message = coverage_issue_message([{"start": 575.97, "end": 605.14, "reason": "suspected_repetition"}])
    assert "Repeated subtitles" in message
    assert "00:09:35.970" in message


def test_recorded_roush_loop_preserves_real_reply_and_splice_edges(monkeypatch, setup):
    fixture = json.loads((Path(__file__).parents[1] / "fixtures/asr/roush_repetition.json").read_text())
    data = words(fixture["words"])
    setup.context.samples = lambda: range(630 * 16000)
    monkeypatch.setattr(review, "_speech_ranges", lambda *a: fixture["speech"])
    decodes = [words(items) for items in fixture["original_decodes"]]
    monkeypatch.setattr(review, "_decode", lambda *a: decodes.pop(0))
    edges = state([data.segments[0], data.segments[-1]])
    result = run(data, setup)
    assert [s.text for s in result.segments] == ["second.", "Oh", "yeah.", "well"]
    assert state([result.segments[0], result.segments[-1]]) == edges
    assert setup.records[0]["repeats"] == 20
    assert not result.coverage_issues
    assert not review.candidates(result)
    assert all(not (r["start"] < 579.2 and r["end"] > 578.56) for r in result.confirmed_non_speech_ranges)


@pytest.mark.parametrize("empty", [False, True])
def test_local_decode_bypasses_cache_offsets_words_and_cleans_audio(tmp_path, empty):
    from pydub import AudioSegment

    context = SimpleNamespace(audio_segment=lambda: AudioSegment.silent(duration=2000, frame_rate=16000))
    local = words([] if empty else [("real", 50, 200)])
    exported = []
    asr = SimpleNamespace(use_cache=True, uses_mlx=False)

    def factory(path, config):
        exported.append(Path(path))
        assert Path(path).is_file()
        assert len(AudioSegment.from_wav(path)) == 400
        return asr

    def run(**kwargs):
        assert not asr.use_cache
        return local

    asr.run = run
    result = review._decode(context, 100, 500, SimpleNamespace(cancel_event=None), factory)
    assert not any(p.exists() for p in exported)
    assert [(s.text, s.start_time, s.end_time) for s in result.segments] == ([] if empty else [("real", 150, 300)])
