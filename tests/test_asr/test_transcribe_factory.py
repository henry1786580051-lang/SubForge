import importlib

from subforge.core.asr.asr_data import ASRData, ASRDataSeg
from subforge.core.entities import (
    FasterWhisperModelEnum,
    TranscribeConfig,
    TranscribeModelEnum,
    WhisperModelEnum,
)

transcribe_module = importlib.import_module("subforge.core.asr.transcribe")
whisper_cpp_module = importlib.import_module("subforge.core.asr.whisper_cpp")


class DummyChunkedASR:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class DummyWhisperCppASR:
    def __init__(self, audio_path, **kwargs):
        self.audio_path = audio_path
        self.kwargs = kwargs


class DummyWhisperXASR:
    def __init__(self, audio_path, **kwargs):
        self.audio_path = audio_path
        self.kwargs = kwargs


class DummyWordTimestampASR:
    def run(self, callback=None):
        return ASRData(
            [
                ASRDataSeg("This", 1_100, 1_300),
                ASRDataSeg("ends", 1_350, 1_700),
                ASRDataSeg("Next", 2_300, 2_600),
                ASRDataSeg("line", 2_650, 2_900),
            ]
        )


def _whisper_cpp_config() -> TranscribeConfig:
    return TranscribeConfig(
        transcribe_model=TranscribeModelEnum.WHISPER_CPP,
        transcribe_language="en",
        whisper_model=WhisperModelEnum.LARGE_V2,
    )


def test_create_asr_instance_whisper_cpp(monkeypatch):
    monkeypatch.setattr(transcribe_module, "ChunkedASR", DummyChunkedASR)
    config = _whisper_cpp_config()
    config.whisper_cpp_path = "/tmp/whisper-cli"

    def on_segment(_asr_data):
        pass

    asr = transcribe_module._create_asr_instance("audio.wav", config, on_segment=on_segment)

    assert asr.kwargs["asr_class"] is transcribe_module.WhisperCppASR
    assert asr.kwargs["asr_kwargs"]["whisper_model"] == "large-v2"
    assert asr.kwargs["asr_kwargs"]["whisper_cpp_path"] == "/tmp/whisper-cli"
    assert asr.kwargs["asr_kwargs"]["use_cache"] is False
    assert asr.kwargs["asr_kwargs"]["use_vad"] is False
    assert asr.kwargs["asr_kwargs"]["segment_callback"] is on_segment
    assert asr.kwargs["chunk_concurrency"] == 1
    assert asr.kwargs["chunk_length"] == 60 * 60


def test_whisper_cpp_keeps_full_audio_when_word_timestamps_enabled():
    config = _whisper_cpp_config()
    config.need_word_time_stamp = True

    kwargs = transcribe_module._build_whisper_cpp_kwargs(config, use_vad=True)

    assert kwargs["need_word_time_stamp"] is True
    assert kwargs["use_vad"] is False


def test_whisper_cpp_receives_instance_scoped_model_directory():
    config = _whisper_cpp_config()
    config.faster_whisper_model_dir = "/tmp/custom-models"

    kwargs = transcribe_module._build_whisper_cpp_kwargs(config)

    assert kwargs["model_dir"] == "/tmp/custom-models"


def test_whisper_cpp_disables_internal_vad_for_sentence_level_runs():
    config = _whisper_cpp_config()
    config.need_word_time_stamp = False

    kwargs = transcribe_module._build_whisper_cpp_kwargs(config, use_vad=True)

    assert kwargs["need_word_time_stamp"] is False
    assert kwargs["use_vad"] is False


def test_whisper_cpp_skips_outer_vad_to_avoid_reloading_model_per_segment():
    assert transcribe_module._should_use_outer_vad(_whisper_cpp_config()) is False

    whisperx_config = TranscribeConfig(transcribe_model=TranscribeModelEnum.WHISPERX)
    assert transcribe_module._should_use_outer_vad(whisperx_config) is False

    config = TranscribeConfig(transcribe_model=TranscribeModelEnum.FASTER_WHISPER)
    assert transcribe_module._should_use_outer_vad(config) is True


def test_transcribe_refines_word_edges_without_changing_text(monkeypatch):
    silero = importlib.import_module("subforge.core.asr.silero_vad")
    monkeypatch.setattr(
        transcribe_module,
        "_create_asr_instance",
        lambda *_args, **_kwargs: DummyWordTimestampASR(),
    )
    monkeypatch.setattr(silero, "is_available", lambda: True)
    monkeypatch.setattr(
        silero,
        "detect_speech_segments",
        lambda *_args, **_kwargs: [(1_000, 2_000), (2_200, 3_000)],
    )
    config = TranscribeConfig(
        transcribe_model=TranscribeModelEnum.WHISPERX,
        need_word_time_stamp=True,
        enable_audio_enhancement=False,
    )

    result = transcribe_module.transcribe("audio.wav", config)

    assert [segment.text for segment in result.segments] == [
        "This",
        "ends",
        "Next",
        "line",
    ]
    assert result.segments[1].end_time == 2_030
    assert result.segments[2].start_time == 2_170


def test_transcribe_keeps_word_results_when_vad_refinement_fails(monkeypatch):
    silero = importlib.import_module("subforge.core.asr.silero_vad")
    monkeypatch.setattr(
        transcribe_module,
        "_create_asr_instance",
        lambda *_args, **_kwargs: DummyWordTimestampASR(),
    )
    monkeypatch.setattr(silero, "is_available", lambda: True)

    def _fail_vad(*_args, **_kwargs):
        raise RuntimeError("VAD unavailable")

    monkeypatch.setattr(silero, "detect_speech_segments", _fail_vad)
    config = TranscribeConfig(
        transcribe_model=TranscribeModelEnum.WHISPERX,
        need_word_time_stamp=True,
        enable_audio_enhancement=False,
    )

    result = transcribe_module.transcribe("audio.wav", config)

    assert [segment.text for segment in result.segments] == [
        "This",
        "ends",
        "Next",
        "line",
    ]
    assert result.segments[1].end_time == 1_700


def test_create_asr_instance_whisperx_uses_forced_alignment_backend(monkeypatch):
    monkeypatch.setattr(transcribe_module, "ChunkedASR", DummyChunkedASR)
    config = TranscribeConfig(
        transcribe_model=TranscribeModelEnum.WHISPERX,
        transcribe_language="en",
        faster_whisper_model=FasterWhisperModelEnum.LARGE_V2,
        faster_whisper_model_dir="/tmp/models",
        faster_whisper_device="auto",
        faster_whisper_compute_type="default",
        whisperx_align_model="WAV2VEC2_ASR_LARGE_LV60K_960H",
        whisperx_batch_size=2,
    )

    asr = transcribe_module._create_asr_instance("audio.wav", config)

    assert asr.kwargs["asr_class"] is transcribe_module.WhisperXASR
    assert asr.kwargs["asr_kwargs"]["whisper_model"] == "large-v2"
    assert asr.kwargs["asr_kwargs"]["model_dir"] == "/tmp/models"
    assert asr.kwargs["asr_kwargs"]["align_model"] == "WAV2VEC2_ASR_LARGE_LV60K_960H"
    assert asr.kwargs["asr_kwargs"]["batch_size"] == 2
    assert asr.kwargs["asr_kwargs"]["use_cache"] is False
    assert asr.kwargs["chunk_length"] == 60 * 60


def test_create_single_asr_whisperx_does_not_require_whisper_cpp(monkeypatch):
    monkeypatch.setattr(transcribe_module, "WhisperXASR", DummyWhisperXASR)
    config = TranscribeConfig(
        transcribe_model=TranscribeModelEnum.WHISPERX,
        transcribe_language="en",
        faster_whisper_model=FasterWhisperModelEnum.LARGE_V2,
    )

    asr = transcribe_module._create_single_asr("audio.wav", config)

    assert asr.audio_path == "audio.wav"
    assert asr.kwargs["whisper_model"] == "large-v2"
    assert asr.kwargs["use_cache"] is False


def test_create_single_asr_whisperx_prefers_explicit_mlx_model_path(monkeypatch):
    monkeypatch.setattr(transcribe_module, "WhisperXASR", DummyWhisperXASR)
    config = TranscribeConfig(
        transcribe_model=TranscribeModelEnum.WHISPERX,
        transcribe_language="en",
        faster_whisper_model=FasterWhisperModelEnum.LARGE_V2,
        whisperx_model="/Users/guwenhan/Desktop/YouTube/model/whisper-large-v3-fp16",
    )

    asr = transcribe_module._create_single_asr("audio.wav", config)

    assert asr.kwargs["whisper_model"] == "/Users/guwenhan/Desktop/YouTube/model/whisper-large-v3-fp16"


def test_create_single_asr_whisper_cpp(monkeypatch):
    monkeypatch.setattr(transcribe_module, "WhisperCppASR", DummyWhisperCppASR)

    asr = transcribe_module._create_single_asr("audio.wav", _whisper_cpp_config())

    assert asr.audio_path == "audio.wav"
    assert asr.kwargs["whisper_model"] == "large-v2"
    assert asr.kwargs["use_cache"] is False


def test_detect_whisper_executable_checks_user_bin(monkeypatch, tmp_path):
    whisper_cpp_module = importlib.import_module("subforge.core.asr.whisper_cpp")
    exe = tmp_path / "whisper-cli"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    exe.chmod(exe.stat().st_mode | 0o111)

    monkeypatch.setattr(whisper_cpp_module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(whisper_cpp_module, "BIN_PATH", tmp_path)
    monkeypatch.setattr(whisper_cpp_module, "BUNDLED_BIN_PATH", tmp_path / "missing")
    monkeypatch.setattr(whisper_cpp_module, "MODEL_PATH", tmp_path / "models")
    monkeypatch.setattr(
        whisper_cpp_module,
        "_whisper_executable_search_dirs",
        lambda: [tmp_path],
    )

    assert whisper_cpp_module.detect_whisper_executable() == str(exe)


def test_detect_whisper_executable_error_explains_binary_requirement(monkeypatch, tmp_path):
    whisper_cpp_module = importlib.import_module("subforge.core.asr.whisper_cpp")

    monkeypatch.setattr(whisper_cpp_module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(whisper_cpp_module, "BIN_PATH", tmp_path / "bin")
    monkeypatch.setattr(whisper_cpp_module, "BUNDLED_BIN_PATH", tmp_path / "bundled")
    monkeypatch.setattr(whisper_cpp_module, "MODEL_PATH", tmp_path / "models")
    monkeypatch.setattr(
        whisper_cpp_module,
        "_whisper_executable_search_dirs",
        lambda: [tmp_path / "bin", tmp_path / "bundled", tmp_path / "models"],
    )

    try:
        whisper_cpp_module.detect_whisper_executable()
    except RuntimeError as exc:
        assert "executable not found" in str(exc)
        assert "model file alone is not enough" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


def test_cli_validator_accepts_whisper_cli_binary(monkeypatch):
    validators = importlib.import_module("subforge.cli.validators")

    monkeypatch.setattr(
        validators.shutil,
        "which",
        lambda name: "/opt/homebrew/bin/whisper-cli" if name == "whisper-cli" else None,
    )

    assert validators.validate_whisper_cpp() is True


def test_whisper_cpp_json_tokens_are_grouped_by_sentence_timestamps():
    data = {
        "transcription": [
            {
                "tokens": [
                    {"text": "[_BEG_]", "offsets": {"from": 0, "to": 0}},
                    {"text": " this", "offsets": {"from": 0, "to": 490}},
                    {"text": " is", "offsets": {"from": 500, "to": 700}},
                    {"text": " a", "offsets": {"from": 710, "to": 820}},
                    {"text": " car", "offsets": {"from": 830, "to": 1200}},
                    {"text": " that", "offsets": {"from": 1210, "to": 1500}},
                    {"text": " I", "offsets": {"from": 1510, "to": 1700}},
                    {"text": "'ve", "offsets": {"from": 1700, "to": 1800}},
                    {"text": " always", "offsets": {"from": 1810, "to": 2200}},
                    {"text": " wanted", "offsets": {"from": 2210, "to": 2700}},
                    {"text": " to", "offsets": {"from": 2710, "to": 2900}},
                    {"text": " drive", "offsets": {"from": 2910, "to": 3400}},
                    {"text": ".", "offsets": {"from": 3400, "to": 3450}},
                    {"text": " It", "offsets": {"from": 3900, "to": 4100}},
                    {"text": "'s", "offsets": {"from": 4100, "to": 4200}},
                    {"text": " always", "offsets": {"from": 4210, "to": 4600}},
                    {"text": " been", "offsets": {"from": 4610, "to": 4900}},
                    {"text": " interesting", "offsets": {"from": 4910, "to": 5600}},
                    {"text": ".", "offsets": {"from": 5600, "to": 5650}},
                    {"text": "[_TT_282]", "offsets": {"from": 5650, "to": 5650}},
                ]
            }
        ]
    }

    import json

    segments = whisper_cpp_module._segments_from_whisper_json(
        json.dumps(data),
        need_word_time_stamp=False,
    )

    assert [seg.text for seg in segments] == [
        "this is a car that I've always wanted to drive.",
        "It's always been interesting.",
    ]
    assert segments[0].start_time == 0
    assert segments[0].end_time == 3450
    assert segments[1].start_time == 3900


def test_whisper_cpp_json_tokens_can_return_word_timestamps():
    data = {
        "transcription": [
            {
                "tokens": [
                    {"text": " Ac", "offsets": {"from": 1000, "to": 1200}},
                    {"text": "ura", "offsets": {"from": 1200, "to": 1350}},
                    {"text": " TL", "offsets": {"from": 1400, "to": 1600}},
                    {"text": ".", "offsets": {"from": 1600, "to": 1650}},
                ]
            }
        ]
    }

    import json

    segments = whisper_cpp_module._segments_from_whisper_json(
        json.dumps(data),
        need_word_time_stamp=True,
    )

    assert [(seg.text, seg.start_time, seg.end_time) for seg in segments] == [
        ("Acura", 1000, 1350),
        ("TL.", 1400, 1650),
    ]


def test_whisper_cpp_caps_unreasonable_word_token_duration():
    data = {
        "transcription": [
            {
                "tokens": [
                    {"text": " Yeah", "offsets": {"from": 90, "to": 7400}},
                    {"text": ".", "offsets": {"from": 7400, "to": 7420}},
                    {"text": " That", "offsets": {"from": 7420, "to": 7570}},
                    {"text": " is", "offsets": {"from": 7670, "to": 7780}},
                    {"text": " zesty", "offsets": {"from": 7810, "to": 8190}},
                    {"text": ".", "offsets": {"from": 8190, "to": 8200}},
                ]
            }
        ]
    }

    import json

    segments = whisper_cpp_module._segments_from_whisper_json(
        json.dumps(data),
        need_word_time_stamp=True,
    )

    assert [(seg.text, seg.start_time, seg.end_time) for seg in segments] == [
        ("Yeah.", 6520, 7420),
        ("That", 7420, 7570),
        ("is", 7670, 7780),
        ("zesty.", 7810, 8200),
    ]


def test_whisper_cpp_cli_segment_line_parses_for_live_preview():
    segment = whisper_cpp_module.WhisperCppASR._parse_cli_segment_line(
        "[00:00:07.250 --> 00:00:09.040]   and this is a car that I've always wanted to drive."
    )

    assert segment is not None
    assert segment.start_time == 7250
    assert segment.end_time == 9040
    assert segment.text == "and this is a car that I've always wanted to drive."


def test_whisper_cpp_command_disables_vad_for_word_timestamps(tmp_path):
    asr = whisper_cpp_module.WhisperCppASR.__new__(whisper_cpp_module.WhisperCppASR)
    asr.whisper_cpp_path = tmp_path / "whisper-cli"
    asr.model_path = tmp_path / "ggml-large-v2.bin"
    asr.language = "en"
    asr.n_threads = 4
    asr.use_vad = True
    asr.need_word_time_stamp = True
    asr.vad_model_path = str(tmp_path / "silero-vad.bin")

    cmd = asr._build_command(tmp_path / "audio.wav", tmp_path / "out.srt", False)

    assert "--vad" not in cmd


def test_whisper_cpp_command_disables_vad_for_sentence_timestamps(tmp_path):
    asr = whisper_cpp_module.WhisperCppASR.__new__(whisper_cpp_module.WhisperCppASR)
    asr.whisper_cpp_path = tmp_path / "whisper-cli"
    asr.model_path = tmp_path / "ggml-large-v2.bin"
    asr.language = "en"
    asr.n_threads = 4
    asr.use_vad = True
    asr.need_word_time_stamp = False
    asr.vad_model_path = str(tmp_path / "silero-vad.bin")

    cmd = asr._build_command(tmp_path / "audio.wav", tmp_path / "out.srt", False)

    assert "--vad" not in cmd
    assert "--vad-threshold" not in cmd
    assert "--vad-min-silence-duration-ms" not in cmd
