import json
import sys
import threading
import wave
from types import SimpleNamespace

import pytest

from subforge.core.asr.whisperx_asr import (
    WhisperXASR,
    _candidate_confirmed_language_bridges,
    _candidate_foreign_language_ranges,
    _classify_foreign_unresolved_ranges,
    _confirm_foreign_language_range,
    _critical_aligned_speech_gaps,
    _detect_speech_in_mlx_gaps,
    _expand_language_ranges_over_repetition,
    _find_sparse_mlx_segments,
    _find_speech_backed_mlx_gaps,
    _install_offline_sentence_tokenizer,
    _is_severe_repetition_hallucination,
    _language_ranges_from_foreign_audit,
    _LanguageProbe,
    _LanguageRange,
    _mlx_model_repo,
    _normalize_align_device,
    _parse_mlx_preview_lines,
    _prepare_mlx_model_path,
    _prepare_spoken_alignment,
    _recover_aligned_gaps_from_native_words,
    _recover_mlx_sparse_segments,
    _recover_mlx_speech_gaps,
    _refine_words_with_char_alignments,
    _restore_display_alignment,
    _script_language_evidence,
    _segments_for_alignment,
    _select_foreign_language_ranges,
    _SpeechBackedGap,
    _spoken_token,
    _stabilize_confirmed_language_ranges,
    _subtract_language_ranges,
    _time_range_is_covered_by_ranges,
    _usable_mlx_recovery_segment,
    default_mlx_model,
    install_whisperx_runtime_stubs,
)


def test_mlx_fixed_language_audit_separates_foreign_speech_from_real_omissions():
    ranges = [
        {"start": 10.0, "end": 20.0, "speech_seconds": 9.5},
        {"start": 30.0, "end": 40.0, "speech_seconds": 9.5},
    ]
    probes = iter(
        [
            {"language": "ja", "text": "日本語の発話です"},
            {"language": "en", "text": "This English speech is still missing"},
        ]
    )

    unresolved, foreign = _classify_foreign_unresolved_ranges(
        ranges,
        list(range(1000)),
        10,
        "en",
        lambda _clip: next(probes),
    )

    assert unresolved == [ranges[1]]
    assert foreign[0]["start"] == 10.0
    assert foreign[0]["detected_language"] == "ja"
    assert _time_range_is_covered_by_ranges(11.0, 19.0, foreign)
    assert not _time_range_is_covered_by_ranges(18.5, 30.0, foreign)


def test_mlx_foreign_audit_prefers_explicit_japanese_script_over_korean_label():
    ranges = [{"start": 10.0, "end": 14.0, "speech_seconds": 3.5}]

    unresolved, foreign = _classify_foreign_unresolved_ranges(
        ranges,
        list(range(200)),
        10,
        "en",
        lambda _clip: {"language": "ko", "text": "これは日本語です"},
    )
    converted = _language_ranges_from_foreign_audit(foreign, "en")

    assert unresolved == []
    assert foreign[0]["detected_language"] == "ja"
    assert foreign[0]["model_language"] == "ko"
    assert converted == [
        _LanguageRange(
            start=10.0,
            end=14.0,
            language="ja",
            confidence=1.0,
            support_seconds=3.5,
            probe_count=1,
            dominance=1.0,
            decoded_language="ko",
            script_language="ja",
            confirmation_agreement=True,
        )
    ]


def test_mlx_native_word_timestamps_fill_only_forced_alignment_holes():
    aligned = {
        "segments": [
            {
                "text": "Before",
                "start": 0.0,
                "end": 2.0,
                "words": [{"word": "Before", "start": 0.0, "end": 2.0}],
            },
            {
                "text": "After",
                "start": 10.0,
                "end": 12.0,
                "words": [{"word": "After", "start": 10.0, "end": 12.0}],
            },
        ]
    }
    native = {
        "segments": [
            {
                "text": "Before recovered words After",
                "start": 0.0,
                "end": 12.0,
                "words": [
                    {"word": " Before", "start": 0.0, "end": 2.0},
                    {"word": " recovered", "start": 4.0, "end": 5.0},
                    {"word": " words", "start": 5.1, "end": 6.0},
                    {"word": " After", "start": 10.0, "end": 12.0},
                ],
            }
        ]
    }
    gaps = _find_speech_backed_mlx_gaps(
        aligned["segments"],
        [(2_100, 9_900)],
        12.0,
    )

    recovered = _recover_aligned_gaps_from_native_words(aligned, native, gaps)

    assert recovered["native_word_gap_recovery"] == 1
    fallback = next(
        segment for segment in recovered["segments"] if segment.get("native_word_fallback")
    )
    assert fallback["text"] == "recovered words"
    assert [word["timing_source"] for word in fallback["words"]] == ["native", "native"]


def test_mlx_sparse_audit_finds_segment_envelope_hiding_missing_speech():
    segments = [
        {"text": "No boundaries? Taking away agency from you.", "start": 10.0, "end": 40.0},
        {
            "text": "This normally paced segment has enough words for its duration.",
            "start": 40.0,
            "end": 48.0,
        },
    ]

    candidates = _find_sparse_mlx_segments(segments)

    assert len(candidates) == 1
    assert candidates[0].index == 0
    assert candidates[0].start == 10.0
    assert candidates[0].end == 40.0


def test_mlx_sparse_recovery_replaces_only_when_local_decode_adds_content():
    result = {
        "segments": [
            {"text": "No boundaries? Taking away agency from you.", "start": 10.0, "end": 40.0},
            {"text": "After", "start": 40.5, "end": 42.0},
        ]
    }
    recovered = _recover_mlx_sparse_segments(
        result,
        list(range(500)),
        10,
        [(10_100, 39_900)],
        lambda _clip: {
            "segments": [
                {
                    "text": (
                        "No boundaries? I do not think having no boundaries is ever good. "
                        "People want a feeling of agency and control over their fate. "
                        "Even useful limits can feel wrong if they take agency away from you."
                    ),
                    "start": 0.0,
                    "end": 29.8,
                    "avg_logprob": -0.15,
                    "no_speech_prob": 0.02,
                    "compression_ratio": 1.2,
                }
            ]
        },
    )

    assert [item["text"] for item in recovered["segments"]] == [
        (
            "No boundaries? I do not think having no boundaries is ever good. "
            "People want a feeling of agency and control over their fate. "
            "Even useful limits can feel wrong if they take agency away from you."
        ),
        "After",
    ]
    assert recovered["sparse_segment_recovery"][0]["original_units"] == 7
    assert recovered["sparse_segment_recovery"][0]["recovered_units"] > 30


def test_mlx_sparse_recovery_reports_unresolved_dense_speech():
    result = {
        "segments": [
            {"text": "No boundaries? Taking away agency from you.", "start": 10.0, "end": 40.0}
        ]
    }

    recovered = _recover_mlx_sparse_segments(
        result,
        list(range(500)),
        10,
        [(10_100, 39_900)],
        lambda _clip: {"segments": []},
    )

    assert recovered["segments"] == result["segments"]
    assert recovered["unresolved_sparse_segments"][0]["speech_seconds"] == pytest.approx(29.8)


def test_final_alignment_audit_uses_words_instead_of_segment_envelopes():
    aligned = {
        "segments": [
            {
                "text": "No boundaries? Taking away agency from you.",
                "start": 10.0,
                "end": 40.0,
                "words": [
                    {"word": "No boundaries?", "start": 10.0, "end": 12.0},
                    {"word": "taking", "start": 38.0, "end": 38.5},
                    {"word": "away", "start": 38.5, "end": 39.0},
                    {"word": "agency", "start": 39.0, "end": 39.5},
                    {"word": "from you", "start": 39.5, "end": 40.0},
                ],
            }
        ]
    }

    gaps = _critical_aligned_speech_gaps(
        aligned,
        [(12_100, 37_900)],
        50.0,
    )

    assert len(gaps) == 1
    assert gaps[0].start == pytest.approx(12.0)
    assert gaps[0].end == pytest.approx(38.0)


def test_final_alignment_audit_catches_vad_activity_between_decoder_segments():
    aligned = {
        "segments": [
            {
                "text": "Before",
                "start": 0.0,
                "end": 3.0,
                "words": [{"word": "Before", "start": 0.0, "end": 3.0}],
            },
            {
                "text": "After",
                "start": 20.0,
                "end": 24.0,
                "words": [{"word": "After", "start": 20.0, "end": 24.0}],
            },
        ]
    }

    gaps = _critical_aligned_speech_gaps(aligned, [(3200, 19_800)], 24.0)

    assert len(gaps) == 1
    assert gaps[0].start == pytest.approx(3.0)
    assert gaps[0].end == pytest.approx(20.0)


def test_mlx_sparse_recovery_uses_context_and_trims_words_to_candidate():
    result = {
        "segments": [
            {"text": "No boundaries?", "start": 10.0, "end": 40.0},
            {"text": "taking away agency", "start": 40.0, "end": 43.0},
        ]
    }
    clips = []

    def transcribe_clip(clip):
        clips.append(clip)
        return {
            "segments": [
                {
                    "text": "outside No boundaries? I want agency outside",
                    "start": 0.0,
                    "end": 35.0,
                    "avg_logprob": -0.1,
                    "no_speech_prob": 0.01,
                    "compression_ratio": 1.1,
                    "words": [
                        {"word": " outside", "start": 0.0, "end": 1.0},
                        {"word": " No", "start": 3.9, "end": 4.3},
                        {"word": " boundaries?", "start": 4.3, "end": 4.8},
                        {"word": " I", "start": 8.0, "end": 8.2},
                        {"word": " want", "start": 8.2, "end": 8.6},
                        {"word": " meaningful", "start": 10.0, "end": 10.8},
                        {"word": " choice", "start": 11.0, "end": 11.6},
                        {"word": " and", "start": 13.0, "end": 13.3},
                        {"word": " personal", "start": 14.0, "end": 14.8},
                        {"word": " control", "start": 16.0, "end": 16.8},
                        {"word": " over", "start": 18.0, "end": 18.4},
                        {"word": " my", "start": 20.0, "end": 20.3},
                        {"word": " fate", "start": 22.0, "end": 22.5},
                        {"word": " and", "start": 24.0, "end": 24.3},
                        {"word": " some", "start": 26.0, "end": 26.4},
                        {"word": " useful", "start": 28.0, "end": 28.5},
                        {"word": " limits", "start": 30.0, "end": 30.6},
                        {"word": " outside", "start": 32.2, "end": 33.0},
                    ],
                }
            ]
        }

    recovered = _recover_mlx_sparse_segments(
        result,
        list(range(500)),
        10,
        [(10_100, 39_900)],
        transcribe_clip,
    )

    assert len(clips) == 1
    assert len(clips[0]) == 340
    assert recovered["segments"][0]["text"].startswith("No boundaries?")
    assert "outside" not in recovered["segments"][0]["text"]
    assert recovered["segments"][-1]["text"] == "taking away agency"


def test_mlx_sparse_recovery_expands_context_until_decode_adds_enough_content():
    result = {
        "segments": [
            {"text": "Short fragment", "start": 10.0, "end": 40.0},
            {"text": "After", "start": 40.5, "end": 42.0},
        ]
    }
    clip_lengths = []

    def transcribe_clip(clip):
        clip_lengths.append(len(clip))
        if len(clip_lengths) < 3:
            return {"segments": []}
        words = [
            {"word": f" word{index}", "start": 11.0 + index, "end": 11.5 + index}
            for index in range(15)
        ]
        return {
            "segments": [
                {
                    "text": "".join(word["word"] for word in words).strip(),
                    "start": 11.0,
                    "end": 25.5,
                    "avg_logprob": -0.1,
                    "no_speech_prob": 0.01,
                    "compression_ratio": 1.1,
                    "words": words,
                }
            ]
        }

    recovered = _recover_mlx_sparse_segments(
        result,
        list(range(1000)),
        10,
        [(10_100, 39_900)],
        transcribe_clip,
    )

    assert clip_lengths == [340, 550, 850]
    assert recovered.get("unresolved_sparse_segments") is None
    assert recovered["segments"][0]["text"].startswith("word0 word1")
    assert recovered["sparse_segment_recovery"][0]["recovered_units"] == 15


def test_mlx_gap_audit_finds_only_vad_confirmed_speech_holes():
    segments = [
        {"text": "Question", "start": 0.0, "end": 3.0},
        {"text": "Answer tail", "start": 20.0, "end": 24.0},
    ]

    gaps = _find_speech_backed_mlx_gaps(
        segments,
        [(3200, 19_800)],
        24.0,
    )

    assert len(gaps) == 1
    assert gaps[0].start == pytest.approx(3.0)
    assert gaps[0].end == pytest.approx(20.0)
    assert gaps[0].speech_seconds == pytest.approx(16.6)
    assert gaps[0].speech_ratio == pytest.approx(16.6 / 17.0)
    assert gaps[0].is_internal is True


def test_mlx_gap_audit_ignores_long_real_silence():
    segments = [
        {"text": "Before", "start": 0.0, "end": 2.0},
        {"text": "After", "start": 18.0, "end": 20.0},
    ]

    assert _find_speech_backed_mlx_gaps(segments, [(0, 2000), (18_000, 20_000)], 20.0) == []


def test_mlx_gap_audit_does_not_treat_empty_decoder_segment_as_coverage():
    segments = [
        {"text": "Before", "start": 0.0, "end": 3.0},
        {"text": "", "start": 3.0, "end": 20.0},
        {"text": "After", "start": 20.0, "end": 24.0},
    ]

    gaps = _find_speech_backed_mlx_gaps(
        segments,
        [(3200, 19_800)],
        24.0,
    )

    assert len(gaps) == 1
    assert gaps[0].start == pytest.approx(3.0)
    assert gaps[0].end == pytest.approx(20.0)


def test_mlx_gap_recovery_preserves_originals_and_rejects_low_confidence():
    audio = list(range(240))
    result = {
        "segments": [
            {"text": "Question", "start": 0.0, "end": 3.0},
            {"text": "less.", "start": 20.0, "end": 21.0},
        ],
        "language": "en",
    }
    clips = []

    def transcribe_clip(clip):
        clips.append(clip)
        return {
            "segments": [
                {
                    "text": "I was raised in a family of readers.",
                    "start": 0.0,
                    "end": 8.0,
                    "avg_logprob": -0.15,
                    "no_speech_prob": 0.03,
                    "compression_ratio": 1.2,
                },
                {
                    "text": "hallucinated repetition",
                    "start": 8.0,
                    "end": 12.0,
                    "avg_logprob": -1.4,
                    "no_speech_prob": 0.05,
                    "compression_ratio": 1.1,
                },
            ]
        }

    recovered = _recover_mlx_speech_gaps(
        result,
        audio,
        10,
        [(3200, 19_800)],
        transcribe_clip,
    )

    assert len(clips) == 1
    assert [item["text"] for item in recovered["segments"]] == [
        "Question",
        "I was raised in a family of readers.",
        "less.",
    ]
    assert recovered["segments"][0] == result["segments"][0]
    assert recovered["segments"][-1] == result["segments"][-1]
    assert recovered["segments"][1]["recovered_speech_gap"] is True
    assert recovered["speech_gap_recovery"][0]["segments"] == 1


def test_mlx_gap_recovery_uses_context_and_keeps_only_words_inside_gap():
    result = {
        "segments": [
            {"text": "Before context", "start": 0.0, "end": 3.0},
            {"text": "After context", "start": 20.0, "end": 24.0},
        ]
    }
    clips = []

    def transcribe_clip(clip):
        clips.append(clip)
        return {
            "segments": [
                {
                    "text": "before recovered speech after",
                    "start": 0.5,
                    "end": 20.0,
                    "avg_logprob": -0.1,
                    "no_speech_prob": 0.02,
                    "compression_ratio": 1.1,
                    "words": [
                        {"word": " before", "start": 0.5, "end": 1.0},
                        {"word": " recovered", "start": 2.5, "end": 3.5},
                        {"word": " speech", "start": 8.0, "end": 9.0},
                        {"word": " after", "start": 19.5, "end": 20.0},
                    ],
                }
            ]
        }

    recovered = _recover_mlx_speech_gaps(
        result,
        list(range(300)),
        10,
        [(3_200, 19_800)],
        transcribe_clip,
    )

    assert len(clips) == 1
    assert len(clips[0]) == 210
    inserted = recovered["segments"][1]
    assert inserted["text"] == "recovered speech"
    assert inserted["start"] == pytest.approx(3.5)
    assert inserted["end"] == pytest.approx(10.0)
    assert [word["word"].strip() for word in inserted["words"]] == [
        "recovered",
        "speech",
    ]


def test_mlx_gap_recovery_expands_context_after_isolated_decode_is_empty():
    result = {
        "segments": [
            {"text": "Before", "start": 0.0, "end": 3.0},
            {"text": "After", "start": 20.0, "end": 24.0},
        ]
    }
    clip_lengths = []

    def transcribe_clip(clip):
        clip_lengths.append(len(clip))
        if len(clip_lengths) == 1:
            return {"segments": []}
        return {
            "segments": [
                {
                    "text": "context recovered speech context",
                    "start": 1.0,
                    "end": 22.0,
                    "avg_logprob": -0.1,
                    "no_speech_prob": 0.02,
                    "compression_ratio": 1.1,
                    "words": [
                        {"word": " context", "start": 1.0, "end": 2.0},
                        {"word": " recovered", "start": 5.5, "end": 6.0},
                        {"word": " speech", "start": 8.0, "end": 9.0},
                        {"word": " context", "start": 21.0, "end": 22.0},
                    ],
                }
            ]
        }

    recovered = _recover_mlx_speech_gaps(
        result,
        list(range(300)),
        10,
        [(3_200, 19_800)],
        transcribe_clip,
    )

    assert clip_lengths == [210, 300]
    assert recovered.get("unresolved_speech_gaps") is None
    assert recovered["segments"][1]["text"] == "recovered speech"


def test_mlx_gap_vad_scans_only_uncovered_audio(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "subforge.core.asr.ten_vad.is_available",
        lambda: True,
    )

    def run_vad(samples, **kwargs):
        calls.append((list(samples), kwargs))
        return [(100, 900)]

    monkeypatch.setattr(
        "subforge.core.asr.ten_vad.run_vad_inference",
        run_vad,
    )

    speech = _detect_speech_in_mlx_gaps(
        list(range(300)),
        10,
        [(4.0, 10.0), (20.0, 25.0)],
    )

    assert [len(samples) for samples, _kwargs in calls] == [60, 50]
    assert speech == [(4100, 4900), (20_100, 20_900)]


def test_mlx_gap_vad_falls_back_to_silero(monkeypatch):
    monkeypatch.setattr(
        "subforge.core.asr.ten_vad.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "subforge.core.asr.ten_vad.run_vad_inference",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("native failure")),
    )
    monkeypatch.setattr(
        "subforge.core.asr.silero_vad.run_vad_inference",
        lambda *_args, **_kwargs: [(200, 800)],
    )

    speech = _detect_speech_in_mlx_gaps(
        list(range(100)),
        10,
        [(2.0, 8.0)],
    )

    assert speech == [(2200, 2800)]


def test_mlx_gap_recovery_reports_unresolved_confirmed_speech():
    result = {
        "segments": [
            {"text": "Before", "start": 0.0, "end": 3.0},
            {"text": "After", "start": 20.0, "end": 24.0},
        ]
    }

    recovered = _recover_mlx_speech_gaps(
        result,
        list(range(240)),
        10,
        [(3200, 19_800)],
        lambda _clip: {"segments": []},
    )

    assert recovered["segments"] == result["segments"]
    assert recovered["unresolved_speech_gaps"] == [
        {
            "start": 3.0,
            "end": 20.0,
            "speech_seconds": pytest.approx(16.6),
            "speech_ratio": pytest.approx(16.6 / 17.0),
            "is_internal": True,
        }
    ]


def test_mlx_gap_recovery_accepts_confident_speech_under_music():
    result = {
        "segments": [
            {"text": "Before", "start": 0.0, "end": 3.0},
            {"text": "After", "start": 20.0, "end": 24.0},
        ]
    }

    recovered = _recover_mlx_speech_gaps(
        result,
        list(range(240)),
        10,
        [(3200, 19_800)],
        lambda _clip: {
            "segments": [
                {
                    "text": "Thanks for watching this week's episode.",
                    "start": 0.0,
                    "end": 8.0,
                    "avg_logprob": -0.12,
                    "no_speech_prob": 0.71,
                    "compression_ratio": 1.44,
                }
            ]
        },
    )

    assert recovered.get("unresolved_speech_gaps") is None
    assert recovered["segments"][1]["text"] == "Thanks for watching this week's episode."


def test_mlx_recovery_quality_gate_counts_cjk_without_spaces():
    assert _usable_mlx_recovery_segment(
        {
            "text": "建物はこういう風に揺れます",
            "start": 0.0,
            "end": 4.5,
            "avg_logprob": -0.10,
            "no_speech_prob": 0.57,
            "compression_ratio": 1.2,
        },
        speech_ratio=0.90,
    )


def test_mlx_gap_recovery_stitches_decode_boundary_to_existing_text():
    result = {
        "segments": [
            {"text": "Question", "start": 0.0, "end": 3.0},
            {"text": "less. And it was", "start": 20.0, "end": 24.0},
        ]
    }

    recovered = _recover_mlx_speech_gaps(
        result,
        list(range(240)),
        10,
        [(3200, 19_800)],
        lambda _clip: {
            "segments": [
                {
                    "text": "I was steadily reading less and less.",
                    "start": 0.0,
                    "end": 17.0,
                    "avg_logprob": -0.12,
                    "no_speech_prob": 0.05,
                    "compression_ratio": 1.2,
                }
            ]
        },
    )

    assert [segment["text"] for segment in recovered["segments"]] == [
        "Question",
        "I was steadily reading less and",
        "less. And it was",
    ]


def test_mlx_gap_recovery_removes_false_period_before_lowercase_continuation():
    result = {
        "segments": [
            {"text": "Before", "start": 0.0, "end": 3.0},
            {"text": "of the episode.", "start": 20.0, "end": 24.0},
        ]
    }

    recovered = _recover_mlx_speech_gaps(
        result,
        list(range(240)),
        10,
        [(3200, 19_800)],
        lambda _clip: {
            "segments": [
                {
                    "text": "We want to know what you thought.",
                    "start": 0.0,
                    "end": 17.0,
                    "avg_logprob": -0.12,
                    "no_speech_prob": 0.05,
                    "compression_ratio": 1.2,
                }
            ]
        },
    )

    assert recovered["segments"][1]["text"] == "We want to know what you thought"


def test_mlx_verbose_lines_are_parsed_for_live_preview():
    segments = _parse_mlx_preview_lines(
        [
            "Loading model...",
            "[00:01.250 --> 00:03.500] Hello everyone.",
            "[01:02:03.004 --> 01:02:05.600] Long recording segment",
            "[00:09.000 --> 00:09.000] ",
        ]
    )

    assert [(item.text, item.start_time, item.end_time) for item in segments] == [
        ("Hello everyone.", 1250, 3500),
        ("Long recording segment", 3_723_004, 3_725_600),
    ]
    assert all(item.timestamp_granularity == "sentence" for item in segments)
    assert all(item.timing_source == "native" for item in segments)


def test_whisperx_selects_only_confident_supported_language_switches():
    probes = [
        _LanguageProbe(0.0, 4.0, "en", 0.99, 0.99),
        _LanguageProbe(10.0, 13.0, "es", 0.96, 0.02),
        _LanguageProbe(13.4, 17.0, "es", 0.91, 0.04),
        _LanguageProbe(30.0, 34.0, "es", 0.70, 0.20),
        _LanguageProbe(40.0, 44.0, "pt", 0.92, 0.30),
        _LanguageProbe(50.0, 51.5, "fr", 0.85, 0.05),
    ]

    ranges = _select_foreign_language_ranges(probes, "en")

    assert [(item.start, item.end, item.language) for item in ranges] == [(10.0, 17.0, "es")]


def test_whisperx_language_support_requires_sustained_probe_cores():
    probes = [
        _LanguageProbe(10.0, 12.5, "es", 0.91, 0.04),
        _LanguageProbe(12.5, 15.0, "es", 0.92, 0.03),
    ]

    ranges = _select_foreign_language_ranges(probes, "en")

    assert ranges == []


def test_whisperx_language_support_counts_voiced_time_not_window_width():
    probes = [
        _LanguageProbe(10.0, 14.0, "es", 0.96, 0.02, speech_seconds=1.0),
        _LanguageProbe(14.0, 18.0, "es", 0.95, 0.03, speech_seconds=1.2),
    ]

    candidates = _candidate_foreign_language_ranges(probes, "en")

    assert len(candidates) == 1
    assert candidates[0].support_seconds == pytest.approx(2.2)
    assert candidates[0].needs_confirmation


def test_whisperx_language_support_does_not_accumulate_across_distant_events():
    probes = [
        _LanguageProbe(10.0, 14.0, "ko", 0.99, 0.01),
        _LanguageProbe(110.0, 114.0, "ko", 0.99, 0.01),
    ]

    assert _select_foreign_language_ranges(probes, "en") == []
    assert len(_candidate_foreign_language_ranges(probes, "en")) == 2


def test_whisperx_cjk_confirmation_uses_event_text_not_window_winner():
    candidate = _LanguageRange(
        10.0,
        20.0,
        "ko",
        0.99,
        support_seconds=8.0,
        probe_count=2,
        dominance=0.67,
        needs_confirmation=True,
    )

    confirmed = _confirm_foreign_language_range(
        candidate,
        "en",
        "ko",
        "掘り進めた後には土が出てきますので",
    )

    assert confirmed is not None
    assert confirmed.language == "ja"
    assert confirmed.script_language == "ja"
    assert not confirmed.confirmation_agreement


def test_whisperx_expands_foreign_range_over_neighboring_decoder_loop():
    ranges = [
        _LanguageRange(
            34.0,
            38.0,
            "ja",
            0.98,
            support_seconds=3.4,
            script_language="ja",
        )
    ]
    repeated = "when you come back " + " ".join(["young"] * 80)

    expanded, source_ranges = _expand_language_ranges_over_repetition(
        ranges,
        [{"start": 32.0, "end": 56.0, "text": repeated}],
    )

    assert _is_severe_repetition_hallucination(repeated)
    assert source_ranges == [(32.0, 56.0)]
    assert [(item.start, item.end, item.language) for item in expanded] == [(32.0, 56.0, "ja")]


def test_whisperx_short_cjk_event_rejects_ambiguous_han_only_text():
    candidate = _LanguageRange(
        10.0,
        14.0,
        "ko",
        0.82,
        support_seconds=3.0,
        probe_count=1,
        needs_confirmation=True,
    )

    assert _confirm_foreign_language_range(candidate, "en", "ko", "地下空間") is None
    assert _script_language_evidence("日本語の発話です") == "ja"
    assert _script_language_evidence("한국어 발화입니다") == "ko"
    assert _script_language_evidence("地下空間") == ""


def test_whisperx_short_kana_event_survives_conflicting_cjk_label():
    candidate = _LanguageRange(
        10.0,
        14.0,
        "ko",
        0.82,
        support_seconds=3.0,
        probe_count=1,
        needs_confirmation=True,
    )

    confirmed = _confirm_foreign_language_range(candidate, "en", "ko", "これは")

    assert confirmed is not None
    assert confirmed.language == "ja"


def test_whisperx_suppresses_weak_minority_cjk_labels():
    ranges = [
        _LanguageRange(
            10.0,
            20.0,
            "ja",
            0.98,
            support_seconds=9.0,
            probe_count=3,
            confirmation_agreement=True,
        ),
        _LanguageRange(
            30.0,
            34.0,
            "ko",
            0.99,
            support_seconds=3.0,
            probe_count=1,
            confirmation_agreement=True,
        ),
    ]

    retained, rejected = _stabilize_confirmed_language_ranges(ranges)

    assert [item.language for item in retained] == ["ja"]
    assert [item.language for item in rejected] == ["ko"]


def test_whisperx_bridges_only_short_vad_backed_same_cjk_language_gaps():
    ranges = [
        _LanguageRange(10.0, 14.0, "ja", 0.96),
        _LanguageRange(20.0, 24.0, "ja", 0.95),
        _LanguageRange(50.0, 54.0, "ja", 0.97),
        _LanguageRange(58.0, 62.0, "ko", 0.98),
    ]

    bridges = _candidate_confirmed_language_bridges(
        ranges,
        [(14_100, 19_900), (24_100, 49_900), (54_100, 57_900)],
    )

    assert len(bridges) == 1
    left, right, speech_seconds, speech_ratio = bridges[0]
    assert (left.end, right.start, left.language) == (14.0, 20.0, "ja")
    assert speech_seconds == pytest.approx(5.8)
    assert speech_ratio == pytest.approx(5.8 / 6.0)


def test_whisperx_subtracts_only_foreign_parts_from_broad_primary_segment():
    ranges = [
        _LanguageRange(10.0, 15.0, "es", 0.95),
        _LanguageRange(18.0, 20.0, "es", 0.92),
    ]

    assert _subtract_language_ranges(5.0, 25.0, ranges) == [
        (5.0, 10.0),
        (15.0, 18.0),
        (20.0, 25.0),
    ]


def test_whisperx_multilingual_alignment_groups_languages_and_restores_time_order(
    monkeypatch,
):
    asr = WhisperXASR.__new__(WhisperXASR)
    calls = []

    def fake_align(result, _audio, language, _callback, _module):
        calls.append((language, [item["text"] for item in result["segments"]]))
        return {
            "segments": [
                {
                    **item,
                    "words": [
                        {
                            "word": item["text"],
                            "start": item["start"],
                            "end": item["end"],
                        }
                    ],
                }
                for item in result["segments"]
            ],
            "align_model": f"align-{language}",
        }

    monkeypatch.setattr(asr, "_align_result", fake_align)
    aligned = asr._align_multilingual_result(
        {
            "segments": [
                {"text": "Hello", "start": 0.0, "end": 1.0, "language": "en"},
                {"text": "Hola", "start": 1.1, "end": 2.0, "language": "es"},
                {"text": "Again", "start": 2.1, "end": 3.0, "language": "en"},
            ]
        },
        object(),
        "en",
        lambda *_args: None,
        object(),
    )

    assert calls == [("en", ["Hello", "Again"]), ("es", ["Hola"])]
    assert [item["text"] for item in aligned["segments"]] == ["Hello", "Hola", "Again"]
    assert aligned["languages"] == ["en", "es"]
    assert aligned["align_models"] == {"en": "align-en", "es": "align-es"}


def test_auto_language_missing_alignment_models_can_continue_with_native_timing(
    monkeypatch, tmp_path
):
    asr = WhisperXASR.__new__(WhisperXASR)
    asr.language = None
    asr.model_dir = str(tmp_path)
    requests = []
    asr.missing_alignment_model_callback = lambda models: requests.append(models) or "continue"
    monkeypatch.setattr(
        "subforge.core.asr.whisperx_asr.is_alignment_model_ready",
        lambda *_args: False,
    )
    ranges = [_LanguageRange(10.0, 15.0, "es", 0.94)]

    retained, skipped = asr._resolve_missing_alignment_models("en", ranges)

    assert retained == ranges
    assert skipped == {"en", "es"}
    assert {item["language"] for item in requests[0]} == {"en", "es"}
    assert all(item["model_id"].startswith("whisperx-align-") for item in requests[0])


def test_explicit_source_language_does_not_request_missing_alignment_models(tmp_path):
    asr = WhisperXASR.__new__(WhisperXASR)
    asr.language = "en"
    asr.model_dir = str(tmp_path)
    asr.missing_alignment_model_callback = lambda _models: pytest.fail(
        "Explicit source language must not enter the auto-language prompt"
    )
    ranges = [_LanguageRange(10.0, 15.0, "es", 0.94)]

    retained, skipped = asr._resolve_missing_alignment_models("en", ranges)

    assert retained == ranges
    assert skipped == set()


def test_fixed_language_supplement_requests_only_foreign_alignment_models(monkeypatch, tmp_path):
    asr = WhisperXASR.__new__(WhisperXASR)
    asr.language = "en"
    asr.detect_additional_languages = True
    asr.model_dir = str(tmp_path)
    requests = []
    asr.missing_alignment_model_callback = lambda models: requests.append(models) or "continue"
    monkeypatch.setattr(
        "subforge.core.asr.whisperx_asr.is_alignment_model_ready",
        lambda *_args: False,
    )
    ranges = [_LanguageRange(10.0, 15.0, "ja", 0.97)]

    retained, skipped = asr._resolve_missing_alignment_models("en", ranges)

    assert retained == ranges
    assert skipped == {"ja"}
    assert [item["language"] for item in requests[0]] == ["ja"]
    assert requests[0][0]["model_id"] == "whisperx-align-ja"


def test_skipped_single_language_alignment_preserves_sentence_timestamps(monkeypatch):
    asr = WhisperXASR.__new__(WhisperXASR)
    monkeypatch.setattr(
        asr,
        "_align_result",
        lambda *_args, **_kwargs: pytest.fail("Skipped language must not load an aligner"),
    )

    aligned = asr._align_multilingual_result(
        {"segments": [{"text": "Hello", "start": 1.0, "end": 2.2, "language": "en"}]},
        object(),
        "en",
        lambda *_args: None,
        object(),
        {"en"},
    )
    asr.need_word_time_stamp = True
    segments = asr._make_segments(aligned)

    assert [(item.text, item.start_time, item.end_time) for item in segments] == [
        ("Hello", 1000, 2200)
    ]
    assert segments[0].timestamp_granularity == "sentence"
    assert segments[0].timing_source == "native"


def test_whisperx_uses_offline_sentence_tokenizer_when_punkt_is_missing():
    module = SimpleNamespace(
        nltk_load=lambda _resource: (_ for _ in ()).throw(LookupError("missing"))
    )

    _install_offline_sentence_tokenizer(module)
    tokenizer = module.nltk_load("tokenizers/punkt_tab/english.pickle")

    assert list(tokenizer.span_tokenize("First sentence. Second one!")) == [
        (0, 15),
        (16, 27),
    ]


def test_whisperx_refines_word_edges_from_matching_character_timestamps():
    words = [{"word": "Today,", "start": 1.0, "end": 1.3}]
    chars = [
        {"char": "T", "start": 1.02, "end": 1.08},
        {"char": "o", "start": 1.08, "end": 1.14},
        {"char": "d", "start": 1.14, "end": 1.20},
        {"char": "a", "start": 1.20, "end": 1.27},
        {"char": "y", "start": 1.27, "end": 1.36},
        {"char": ",", "start": 1.36, "end": 1.40},
    ]

    refined = _refine_words_with_char_alignments(words, chars)

    assert refined[0]["start"] == 1.02
    assert refined[0]["end"] == 1.40
    assert words[0]["end"] == 1.3


def test_whisperx_keeps_word_timing_when_character_text_does_not_match():
    words = [{"word": "Lexus", "start": 2.0, "end": 2.5}]
    chars = [{"char": "B", "start": 2.1, "end": 2.2}]

    assert _refine_words_with_char_alignments(words, chars) == words


def test_whisperx_character_mismatch_does_not_block_following_word():
    words = [
        {"word": "2026", "start": 2.0, "end": 2.3},
        {"word": "Lexus", "start": 2.4, "end": 2.8},
    ]
    chars = [
        {"char": "?", "start": 2.0, "end": 2.3},
        {"char": " ", "start": 2.3, "end": 2.4},
        {"char": "L", "start": 2.42, "end": 2.48},
        {"char": "e", "start": 2.48, "end": 2.55},
        {"char": "x", "start": 2.55, "end": 2.63},
        {"char": "u", "start": 2.63, "end": 2.72},
        {"char": "s", "start": 2.72, "end": 2.84},
    ]

    refined = _refine_words_with_char_alignments(words, chars)

    assert refined[0] == words[0]
    assert refined[1]["start"] == 2.42
    assert refined[1]["end"] == 2.84


def test_whisperx_normalizes_english_numbers_models_units_and_symbols():
    assert _spoken_token("2026") == "twenty twenty six"
    assert _spoken_token("M4") == "M four"
    assert _spoken_token("543hp") == "five hundred forty three horsepower"
    assert _spoken_token("$79,995.") == ("seventy nine thousand nine hundred ninety five dollars.")
    assert _spoken_token("10%") == "ten percent"
    assert _spoken_token("0-60") == "zero to sixty"
    assert _spoken_token("AWD") == "A W D"
    assert _spoken_token("IS") == "IS"
    assert _spoken_token("is") == "is"


def test_whisperx_spoken_alignment_is_english_only():
    segments = [{"text": "2026 Lexus", "start": 1.0, "end": 2.0}]

    normalized, plans = _prepare_spoken_alignment(segments, "zh")

    assert normalized == segments
    assert plans is None


def test_whisperx_plain_english_uses_original_alignment_path():
    segments = [{"text": "Today we drive the new Lexus.", "start": 1.0, "end": 3.0}]

    normalized, plans = _prepare_spoken_alignment(segments, "en")

    assert normalized == segments
    assert plans is None


def test_whisperx_restores_display_tokens_from_spoken_alignment():
    segments = [
        {
            "text": "The 2026 M4 makes 543hp.",
            "start": 1.0,
            "end": 4.0,
        }
    ]
    normalized, plans = _prepare_spoken_alignment(segments, "en")
    assert normalized[0]["text"] == (
        "The twenty twenty six M four makes five hundred forty three horsepower."
    )
    assert plans is not None
    spoken = normalized[0]["text"].split()
    aligned_words = [
        {
            "word": word,
            "start": 1.0 + index * 0.2,
            "end": 1.18 + index * 0.2,
            "score": 0.9,
        }
        for index, word in enumerate(spoken)
    ]

    restored = _restore_display_alignment({"segments": [{"words": aligned_words}]}, plans)

    assert restored is not None
    words = restored["segments"][0]["words"]
    assert [word["word"] for word in words] == ["The", "2026", "M4", "makes", "543hp."]
    assert words[1]["start"] == 1.2
    assert words[1]["end"] == 1.78
    assert words[2]["start"] == 1.8
    assert words[2]["end"] == pytest.approx(2.18)
    assert restored["segments"][0]["text"] == segments[0]["text"]


def test_whisperx_rejects_incomplete_spoken_mapping():
    segments = [{"text": "2026 Lexus", "start": 1.0, "end": 2.0}]
    _, plans = _prepare_spoken_alignment(segments, "en")

    assert plans is not None
    assert (
        _restore_display_alignment(
            {"segments": [{"words": [{"word": "twenty", "start": 1.0, "end": 1.2}]}]},
            plans,
        )
        is None
    )


def test_whisperx_maps_standard_model_name_to_mlx_repo():
    assert _mlx_model_repo("large-v2") == "mlx-community/whisper-large-v2-mlx"
    assert _mlx_model_repo("mlx-community/custom-model") == "mlx-community/custom-model"


def test_whisperx_defaults_to_local_mlx_model_when_available(monkeypatch, tmp_path):
    local_model = tmp_path / "whisper-large-v3-fp16"
    local_model.mkdir()
    monkeypatch.setattr(
        "subforge.core.asr.whisperx_asr._find_local_mlx_model",
        lambda: local_model,
    )

    assert default_mlx_model() == str(local_model)
    assert _mlx_model_repo("") == str(local_model)


def test_whisperx_defaults_to_remote_mlx_repo_without_local_model(monkeypatch):
    monkeypatch.setattr(
        "subforge.core.asr.whisperx_asr._find_local_mlx_model",
        lambda: None,
    )

    assert default_mlx_model() == "large-v3"
    assert _mlx_model_repo("") == "mlx-community/whisper-large-v3-mlx"


def test_whisperx_prepares_model_safetensors_alias(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "multilingual.tiktoken").write_text("tokens", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")

    prepared = _prepare_mlx_model_path(str(model_dir), tmp_path)

    prepared_path = tmp_path / "mlx_model"
    assert prepared == str(prepared_path)
    assert (prepared_path / "weights.safetensors").exists()
    assert (prepared_path / "config.json").exists()


def test_whisperx_runs_mlx_decode_through_worker_protocol(monkeypatch, tmp_path):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    model_path = tmp_path / "model"
    model_path.mkdir()
    observed = {}

    class CompletedProcess:
        returncode = 0
        poll_count = 0

        def poll(self):
            self.poll_count += 1
            if self.poll_count == 1:
                return None
            return 0

        @staticmethod
        def terminate():
            raise AssertionError("completed worker must not be terminated")

    def fake_popen(command, *, env, stdout, **_kwargs):
        observed["command"] = command
        request = json.loads(open(env["SUBFORGE_MLX_WHISPER_REQUEST"], encoding="utf-8").read())
        observed["request"] = request
        with open(env["SUBFORGE_MLX_WHISPER_OUTPUT"], "w", encoding="utf-8") as output:
            json.dump(
                {
                    "ok": True,
                    "data": {
                        "language": "en",
                        "segments": [{"text": "hello", "start": 0.0, "end": 1.0}],
                    },
                },
                output,
            )
        stdout.write("[00:00.100 --> 00:01.000] hello\n")
        stdout.flush()
        return CompletedProcess()

    monkeypatch.setattr("subforge.core.asr.whisperx_asr.subprocess.Popen", fake_popen)
    monkeypatch.setattr("subforge.core.asr.whisperx_asr.time.sleep", lambda _seconds: None)
    asr = WhisperXASR.__new__(WhisperXASR)
    asr.language = "en"
    asr.cancel_event = None
    previews = []
    asr.segment_callback = lambda data: previews.append(data)

    result = asr._transcribe_mlx_in_worker(
        str(audio_path),
        str(model_path),
        lambda *_args: None,
    )

    assert result["segments"][0]["text"] == "hello"
    assert observed["request"] == {
        "audio": str(audio_path),
        "model": str(model_path),
        "language": "en",
        "word_timestamps": True,
    }
    assert "subforge.core.asr.mlx_worker" in observed["command"]
    assert [[segment.text for segment in data.segments] for data in previews] == [["hello"]]


def test_whisperx_classifies_final_audit_ranges_through_worker(monkeypatch, tmp_path):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    model_path = tmp_path / "model"
    model_path.mkdir()
    observed = {}

    class CompletedProcess:
        returncode = 0
        poll_count = 0

        def poll(self):
            self.poll_count += 1
            if self.poll_count == 1:
                return None
            return 0

        @staticmethod
        def terminate():
            raise AssertionError("completed worker must not be terminated")

    def fake_popen(_command, *, env, **_kwargs):
        request = json.loads(open(env["SUBFORGE_MLX_WHISPER_REQUEST"], encoding="utf-8").read())
        observed["request"] = request
        with open(env["SUBFORGE_MLX_WHISPER_OUTPUT"], "w", encoding="utf-8") as output:
            json.dump(
                {
                    "ok": True,
                    "data": {
                        "foreign_language_speech_ranges": [
                            {
                                **request["ranges"][0],
                                "detected_language": "ja",
                                "detected_units": 8,
                            }
                        ]
                    },
                },
                output,
            )
        return CompletedProcess()

    monkeypatch.setattr("subforge.core.asr.whisperx_asr.subprocess.Popen", fake_popen)
    monkeypatch.setattr("subforge.core.asr.whisperx_asr.time.sleep", lambda _seconds: None)
    asr = WhisperXASR.__new__(WhisperXASR)
    asr.language = "en"
    asr.cancel_event = None
    gap = _SpeechBackedGap(10.0, 20.0, 9.5, 0.95, True)

    result = asr._classify_mlx_ranges_in_worker(
        str(audio_path),
        str(model_path),
        [gap],
        lambda *_args: None,
    )

    assert result[0]["detected_language"] == "ja"
    assert observed["request"] == {
        "operation": "classify_ranges",
        "audio": str(audio_path),
        "model": str(model_path),
        "primary_language": "en",
        "ranges": [
            {
                "start": 10.0,
                "end": 20.0,
                "speech_seconds": 9.5,
                "speech_ratio": 0.95,
                "is_internal": True,
            }
        ],
    }


def test_whisperx_alignment_uses_cpu_for_mps_device():
    assert _normalize_align_device("mps") == "cpu"


def test_whisperx_alignment_falls_back_when_torch_cuda_is_unavailable(monkeypatch):
    torch = SimpleNamespace(
        version=SimpleNamespace(cuda=None),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert _normalize_align_device("cuda") == "cpu"


def test_whisperx_reuses_managed_faster_whisper_model(tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "faster-whisper-large-v3"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}" * 100, encoding="utf-8")
    (model_dir / "model.bin").write_bytes(b"x" * (1024 * 1024))
    (model_dir / "tokenizer.json").write_text("{}" * 1024, encoding="utf-8")
    audio_path = tmp_path / "audio.wav"
    with wave.open(str(audio_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\0\0" * 1_600)
    monkeypatch.setattr("subforge.core.asr.whisperx_asr.platform.system", lambda: "Windows")
    monkeypatch.setattr(
        "subforge.core.asr.whisperx_asr.resolve_faster_whisper_runtime",
        lambda *_args: ("cpu", "int8"),
    )

    asr = WhisperXASR(
        str(audio_path),
        whisper_model="large-v3",
        model_dir=str(tmp_path / "models"),
        device="cuda",
    )

    assert asr.whisper_model == str(model_dir)
    assert asr.transcribe_device == "cpu"
    assert asr.align_device == "cpu"


def test_whisperx_builds_alignment_segments_from_mlx_result():
    segments = _segments_for_alignment(
        {
            "segments": [
                {"text": " hello ", "start": 0.09, "end": 0.42},
                {"text": "", "start": 0.5, "end": 0.8},
                {"text": "bad", "start": None, "end": 1.0},
                {"text": "world", "start": 1, "end": 1.3},
            ]
        }
    )

    assert segments == [
        {"text": "hello", "start": 0.09, "end": 0.42},
        {"text": "world", "start": 1.0, "end": 1.3},
    ]


def test_whisperx_make_segments_uses_forced_aligned_word_segments():
    asr = WhisperXASR.__new__(WhisperXASR)
    asr.need_word_time_stamp = True

    segments = asr._make_segments(
        {
            "word_segments": [
                {"word": "Today,", "start": 9.964, "end": 10.48, "score": 0.91},
                {"word": "we", "start": 10.50, "end": 10.62, "score": 0.98},
                {"word": "drive", "start": 10.70, "end": 11.12, "score": 0.95},
                {"word": "bad", "start": None, "end": 12.0},
            ],
            "segments": [
                {"text": "Today, we drive", "start": 9.9, "end": 11.2},
            ],
        }
    )

    assert [(seg.text, seg.start_time, seg.end_time) for seg in segments] == [
        ("Today,", 9964, 10480),
        ("we", 10500, 10620),
        ("drive", 10700, 11120),
    ]


def test_whisperx_make_segments_keeps_unaligned_words_from_aligned_segments():
    asr = WhisperXASR.__new__(WhisperXASR)
    asr.need_word_time_stamp = True

    segments = asr._make_segments(
        {
            "segments": [
                {
                    "text": "newly refreshed 2026 Lexus IS 350 F Sport Design.",
                    "start": 11.693,
                    "end": 17.779,
                    "words": [
                        {"word": "newly", "start": 11.693, "end": 11.973},
                        {"word": "refreshed", "start": 12.053, "end": 12.674},
                        {"word": "2026"},
                        {"word": "Lexus", "start": 13.995, "end": 14.396},
                        {"word": "IS", "start": 14.716, "end": 14.976},
                        {"word": "350"},
                        {"word": "F", "start": 15.777, "end": 16.057},
                        {"word": "Sport", "start": 16.678, "end": 16.998},
                        {"word": "Design.", "start": 17.199, "end": 17.779},
                    ],
                }
            ],
            "word_segments": [
                {"word": "newly", "start": 11.693, "end": 11.973},
                {"word": "refreshed", "start": 12.053, "end": 12.674},
                {"word": "2026"},
                {"word": "Lexus", "start": 13.995, "end": 14.396},
            ],
        }
    )

    assert [seg.text for seg in segments] == [
        "newly",
        "refreshed",
        "2026",
        "Lexus",
        "IS",
        "350",
        "F",
        "Sport",
        "Design.",
    ]
    assert (segments[2].start_time, segments[2].end_time) == (12674, 13995)
    assert (segments[5].start_time, segments[5].end_time) == (14976, 15777)


def test_whisperx_make_segments_can_return_sentence_segments_when_requested():
    asr = WhisperXASR.__new__(WhisperXASR)
    asr.need_word_time_stamp = False

    segments = asr._make_segments(
        {
            "segments": [
                {"text": "Today, we drive.", "start": 9.964, "end": 12.15},
            ],
        }
    )

    assert [(seg.text, seg.start_time, seg.end_time) for seg in segments] == [
        ("Today, we drive.", 9964, 12150),
    ]
    assert segments[0].timestamp_granularity == "sentence"
    assert segments[0].timing_source == "native"


def test_whisperx_make_segments_ignores_words_when_word_timestamps_disabled():
    asr = WhisperXASR.__new__(WhisperXASR)
    asr.need_word_time_stamp = False

    segments = asr._make_segments(
        {
            "word_segments": [
                {"word": "Today,", "start": 9.964, "end": 10.48},
            ],
            "segments": [
                {"text": "Today, we drive.", "start": 9.964, "end": 12.15},
            ],
        }
    )

    assert [(seg.text, seg.start_time, seg.end_time) for seg in segments] == [
        ("Today, we drive.", 9964, 12150),
    ]


def test_whisperx_word_segments_keep_forced_alignment_metadata():
    asr = WhisperXASR.__new__(WhisperXASR)
    asr.need_word_time_stamp = True

    segments = asr._make_segments(
        {
            "segments": [
                {
                    "text": "Today,",
                    "start": 1.0,
                    "end": 1.4,
                    "words": [{"word": "Today,", "start": 1.02, "end": 1.38, "score": 0.93}],
                }
            ]
        }
    )

    assert len(segments) == 1
    assert segments[0].timestamp_granularity == "word"
    assert segments[0].timing_source == "forced_alignment"
    assert segments[0].words[0].alignment_score == pytest.approx(0.93)


def test_whisperx_word_segments_keep_detected_source_language():
    asr = WhisperXASR.__new__(WhisperXASR)
    asr.need_word_time_stamp = True

    segments = asr._make_segments(
        {
            "language": "en",
            "segments": [
                {
                    "language": "ja",
                    "text": "こんにちは",
                    "start": 1.0,
                    "end": 1.8,
                    "words": [{"word": "こんにちは", "start": 1.02, "end": 1.78, "score": 0.96}],
                }
            ],
        }
    )

    assert segments[0].language_code == "ja"
    assert segments[0].words[0].language_code == "ja"


def test_whisperx_replaces_default_english_alignment_model_for_korean():
    asr = WhisperXASR.__new__(WhisperXASR)
    asr.align_model = "WAV2VEC2_ASR_LARGE_LV60K_960H"

    assert asr._resolve_align_model_name("ko") == "kresnik/wav2vec2-large-xlsr-korean"


def test_whisperx_keeps_default_english_alignment_model_for_english():
    asr = WhisperXASR.__new__(WhisperXASR)
    asr.align_model = "WAV2VEC2_ASR_LARGE_LV60K_960H"

    assert asr._resolve_align_model_name("en") == "WAV2VEC2_ASR_LARGE_LV60K_960H"


def test_whisperx_maps_local_torchaudio_weight_to_pipeline_identifier(tmp_path):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    asr = WhisperXASR(
        str(audio_path),
        align_model=str(tmp_path / "wav2vec2_fairseq_large_lv60k_asr_ls960.pth"),
    )

    assert asr.align_model == "WAV2VEC2_ASR_LARGE_LV60K_960H"


def test_whisperx_keeps_explicit_non_english_alignment_model():
    asr = WhisperXASR.__new__(WhisperXASR)
    asr.align_model = "example/korean-alignment-model"

    assert asr._resolve_align_model_name("ko") == "example/korean-alignment-model"


@pytest.mark.parametrize(("uses_mlx", "method_name"), [(True, "mlx"), (False, "standard")])
def test_whisperx_routes_to_platform_backend(monkeypatch, uses_mlx, method_name):
    asr = WhisperXASR.__new__(WhisperXASR)
    asr.uses_mlx = uses_mlx
    calls = []
    monkeypatch.setattr(asr, "_run_mlx", lambda **_kwargs: calls.append("mlx") or {"ok": True})
    monkeypatch.setattr(
        asr,
        "_run_standard",
        lambda **_kwargs: calls.append("standard") or {"ok": True},
    )

    assert asr._run() == {"ok": True}
    assert calls == [method_name]


def test_whisperx_checks_cancellation_between_native_stages():
    asr = WhisperXASR.__new__(WhisperXASR)
    asr.cancel_event = threading.Event()
    asr.cancel_event.set()

    with pytest.raises(RuntimeError, match="cancelled during model loading"):
        asr._raise_if_cancelled("model loading")


def test_whisperx_runtime_diarization_stub_exposes_vad_segment(monkeypatch):
    monkeypatch.delitem(sys.modules, "whisperx.diarize", raising=False)

    install_whisperx_runtime_stubs()

    segment = sys.modules["whisperx.diarize"].Segment(1, 2, "speaker")
    assert (segment.start, segment.end, segment.speaker) == (1, 2, "speaker")
