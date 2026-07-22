from types import SimpleNamespace

import pytest

from subforge.core.asr.whisperx_asr import (
    WhisperXASR,
    _install_offline_sentence_tokenizer,
    _mlx_model_repo,
    _normalize_align_device,
    _prepare_mlx_model_path,
    _prepare_spoken_alignment,
    _refine_words_with_char_alignments,
    _restore_display_alignment,
    _segments_for_alignment,
    _spoken_token,
    default_mlx_model,
)


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


def test_whisperx_alignment_uses_cpu_for_mps_device():
    assert _normalize_align_device("mps") == "cpu"


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
                    "words": [
                        {"word": "Today,", "start": 1.02, "end": 1.38, "score": 0.93}
                    ],
                }
            ]
        }
    )

    assert len(segments) == 1
    assert segments[0].timestamp_granularity == "word"
    assert segments[0].timing_source == "forced_alignment"
    assert segments[0].words[0].alignment_score == pytest.approx(0.93)


def test_whisperx_ignores_default_english_alignment_model_for_korean():
    asr = WhisperXASR.__new__(WhisperXASR)
    asr.align_model = "WAV2VEC2_ASR_LARGE_LV60K_960H"

    assert asr._resolve_align_model_name("ko") is None


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
