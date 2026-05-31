import importlib

from subforge.core.entities import TranscribeConfig, TranscribeModelEnum, WhisperModelEnum

transcribe_module = importlib.import_module("subforge.core.asr.transcribe")


class DummyChunkedASR:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class DummyWhisperCppASR:
    def __init__(self, audio_path, **kwargs):
        self.audio_path = audio_path
        self.kwargs = kwargs


def _whisper_cpp_config() -> TranscribeConfig:
    return TranscribeConfig(
        transcribe_model=TranscribeModelEnum.WHISPER_CPP,
        transcribe_language="en",
        whisper_model=WhisperModelEnum.LARGE_V2,
    )


def test_create_asr_instance_whisper_cpp_does_not_require_removed_jianying_enum(monkeypatch):
    monkeypatch.setattr(transcribe_module, "ChunkedASR", DummyChunkedASR)
    config = _whisper_cpp_config()
    config.whisper_cpp_path = "/tmp/whisper-cli"

    asr = transcribe_module._create_asr_instance("audio.wav", config)

    assert asr.kwargs["asr_class"] is transcribe_module.WhisperCppASR
    assert asr.kwargs["asr_kwargs"]["whisper_model"] == "large-v2"
    assert asr.kwargs["asr_kwargs"]["whisper_cpp_path"] == "/tmp/whisper-cli"
    assert asr.kwargs["asr_kwargs"]["use_cache"] is False


def test_create_single_asr_whisper_cpp_does_not_require_removed_jianying_enum(monkeypatch):
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
