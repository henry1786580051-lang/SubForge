from subforge.core.asr.whisperx_asr import (
    DEFAULT_LOCAL_MLX_MODEL,
    WhisperXASR,
    _prepare_mlx_model_path,
    _mlx_model_repo,
    _normalize_align_device,
    _segments_for_alignment,
    default_mlx_model,
)


def test_whisperx_maps_standard_model_name_to_mlx_repo():
    assert _mlx_model_repo("large-v2") == "mlx-community/whisper-large-v2-mlx"
    assert _mlx_model_repo("mlx-community/custom-model") == "mlx-community/custom-model"


def test_whisperx_defaults_to_local_mlx_model_when_available():
    assert default_mlx_model() == DEFAULT_LOCAL_MLX_MODEL
    assert _mlx_model_repo("") == DEFAULT_LOCAL_MLX_MODEL


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
