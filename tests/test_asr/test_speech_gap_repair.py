from copy import deepcopy

import pytest

from subforge.core.asr.asr_data import ASRData, ASRDataSeg
from subforge.core.asr.chunk_merger import ChunkMerger
from subforge.core.asr.speech_gap_repair import (
    anchored_candidate,
    confirmation_window,
    corroborates,
    coverage_issue_message,
    insert_anchored_gap,
    timed_words,
)
from subforge.core.asr.whisperx_asr import (
    MLX_GAP_RECOVERY_MIN_GAP_SECONDS,
    MLX_SHORT_GAP_MAX_SECONDS,
    _alignment_word_coverage,
    _critical_aligned_speech_gaps,
    _recover_short_mlx_speech_gaps,
)


def word(text, start, end):
    return {"word": " " + text, "start": start, "end": end}


def source():
    return {
        "segments": [
            {
                "text": "The engineer said.",
                "start": 8.0,
                "end": 10.0,
                "words": [word("The", 8, 9), word("engineer", 9, 9.5), word("said.", 9.5, 10)],
            },
            {
                "text": "The next phase follows.",
                "start": 14.2,
                "end": 18.0,
                "words": [
                    word("The", 14.2, 14.6),
                    word("next", 15, 16),
                    word("phase", 16, 17),
                    word("follows.", 17, 18),
                ],
            },
        ]
    }


def missing_words():
    return [
        word(t, 10.2 + i * 0.8, 11 + i * 0.8)
        for i, t in enumerate("We can finish the whole project.".split())
    ]


def test_gap_ranges_meet_without_a_blind_interval():
    assert MLX_SHORT_GAP_MAX_SECONDS == MLX_GAP_RECOVERY_MIN_GAP_SECONDS


@pytest.mark.parametrize("duration", [3.5, 3.51, 4.26, 5.99, 6.0])
def test_final_audit_catches_dense_speech_in_previously_unchecked_gaps(duration):
    result = {"segments": [{"words": [word("before", 0, 2), word("after", 2 + duration, 12)]}]}
    gaps = _critical_aligned_speech_gaps(result, [(2000, round((2 + duration) * 1000))], 12)
    assert len(gaps) == 1
    assert gaps[0].speech_seconds == pytest.approx(duration)


def test_final_audit_ignores_music_with_only_brief_vad_spikes():
    result = source()
    assert not _critical_aligned_speech_gaps(result, [(10500, 10900)], 20)


def test_long_gap_audit_keeps_existing_sensitivity():
    result = {"segments": [{"words": [word("before", 0, 2), word("after", 9, 12)]}]}
    gaps = _critical_aligned_speech_gaps(result, [(2000, 7100)], 12)
    assert len(gaps) == 1
    assert gaps[0].speech_seconds == pytest.approx(5.1)


def test_anchored_recovery_preserves_tail_and_all_original_text():
    original = source()
    snapshot = deepcopy(original)
    calls = []

    def decode(clip):
        offset, limit = clip[0] / 100, (clip[-1] + 1) / 100
        calls.append((offset, limit))
        if len(calls) == 2:
            return {"segments": []}
        words = [
            *original["segments"][0]["words"],
            *missing_words(),
            word("The", 15.2, 15.5),
            word("next", 15.5, 16),
            word("phase", 16, 17),
            word("follows.", 17, 18),
        ]
        return {
            "segments": [
                {
                    "text": "local speech",
                    "avg_logprob": -0.1,
                    "start": 0,
                    "end": limit - offset,
                    "words": [
                        {**w, "start": w["start"] - offset, "end": w["end"] - offset}
                        for w in words
                        if offset <= w["start"] and w["end"] <= limit
                    ],
                }
            ]
        }

    result = _recover_short_mlx_speech_gaps(
        original, list(range(3000)), 100, [(10100, 14200)], decode
    )
    assert original == snapshot
    assert len(calls) == 3
    assert calls[2] == pytest.approx(confirmation_window(original, 10, 14.2, 30))
    assert result["segments"][1]["text"] == "We can finish the whole project."
    assert result["segments"][1]["end"] == pytest.approx(15)
    assert result["segments"][2]["start"] == pytest.approx(15.01)
    assert result["segments"][0] == original["segments"][0]
    assert result["segments"][2]["text"] == original["segments"][1]["text"]
    assert not result["coverage_issues"]


@pytest.mark.parametrize(
    "budget,candidates,reason",
    [(0, 10, "decode_budget"), (180, 0, "candidate_budget"), (180, 10, "context_disagreement")],
)
def test_unresolved_dense_speech_is_reported_even_when_budget_exhausted(budget, candidates, reason):
    original = source()
    result = _recover_short_mlx_speech_gaps(
        original,
        list(range(3000)),
        100,
        [(10100, 14200)],
        lambda clip: {"segments": []},
        max_candidates=candidates,
        max_decode_seconds=budget,
    )
    assert result["segments"] == original["segments"]
    assert result["coverage_issues"][0]["reason"] == reason
    assert "00:00:10.000 - 00:00:14.200" in coverage_issue_message(result["coverage_issues"])


def test_consensus_rejects_lexical_or_acoustic_disagreement():
    words = missing_words()
    assert corroborates(words, deepcopy(words))
    changed = deepcopy(words)
    changed[-1]["word"] = " mistake."
    assert not corroborates(words, changed)
    shifted = [{**w, "start": w["start"] + 1, "end": w["end"] + 1} for w in words]
    assert not corroborates(words, shifted)


def test_anchor_does_not_accept_unrelated_context():
    original = source()
    assert not anchored_candidate(missing_words(), _alignment_word_coverage(original), 10, 14.2)


def test_insertion_refuses_to_overwrite_an_existing_utterance():
    original = source()
    original["segments"][0]["end"] = 12
    assert insert_anchored_gap(original, missing_words(), 10, 14.2) is None


def test_timed_words_reject_nonfinite_or_low_confidence_times():
    result = {
        "segments": [
            {"words": [word("bad", float("nan"), 2), word("bad", 2, 1), word("good", 1, 2)]}
        ]
    }
    assert timed_words(result, 10, lambda s: True) == [word("good", 11, 12)]
    assert timed_words(result, 10, lambda s: False) == []


def test_chunk_merge_keeps_coverage_warning_with_absolute_time():
    first = ASRData([ASRDataSeg("before", 0, 1000)])
    second = ASRData([ASRDataSeg("after", 0, 1000)])
    second.coverage_issues = [{"start": 2, "end": 6, "reason": "decode_budget"}]
    result = ChunkMerger().merge_chunks([first, second], [0, 10000])
    assert result.coverage_issues[0]["start"] == 12
    assert result.coverage_issues[0]["end"] == 16
    assert second.coverage_issues[0]["start"] == 2
