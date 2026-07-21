import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from subforge.core.asr.faster_whisper import (
    FasterWhisperASR,
    is_faster_whisper_model_dir,
    resolve_faster_whisper_runtime,
)


def _write_model(path: Path, *, complete: bool = True) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{" + " " * 128 + "}", encoding="utf-8")
    (path / "model.bin").write_bytes(b"x" * (1024 * 1024))
    if complete:
        (path / "tokenizer.json").write_bytes(b"x" * 1024)


def test_faster_whisper_model_validation_requires_ctranslate2_files(tmp_path: Path):
    incomplete = tmp_path / "incomplete"
    _write_model(incomplete, complete=False)
    complete = tmp_path / "complete"
    _write_model(complete)

    assert not is_faster_whisper_model_dir(incomplete)
    assert is_faster_whisper_model_dir(complete)


def test_cuda_request_falls_back_to_cpu_when_runtime_is_missing(monkeypatch):
    monkeypatch.setattr(
        "subforge.core.asr.faster_whisper.is_faster_whisper_cuda_available",
        lambda: False,
    )

    assert resolve_faster_whisper_runtime("auto", "default") == ("cpu", "int8")
    assert resolve_faster_whisper_runtime("cuda", "float16") == ("cpu", "int8")


def test_faster_whisper_does_not_accept_whisper_cpp_bin(tmp_path: Path):
    (tmp_path / "ggml-small.bin").write_bytes(b"lmgg" + b"x" * 1024)

    with pytest.raises(RuntimeError, match="GGML .bin"):
        FasterWhisperASR._resolve_model_path("small", str(tmp_path))


def test_faster_whisper_runs_packaged_python_runtime(tmp_path: Path, monkeypatch):
    model = tmp_path / "faster-whisper-small"
    _write_model(model)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    captured = {}

    class FakeWhisperModel:
        def __init__(self, model_path, **kwargs):
            captured["model_path"] = model_path
            captured["init"] = kwargs

        def transcribe(self, audio_path, **kwargs):
            captured["audio_path"] = audio_path
            captured["transcribe"] = kwargs
            return iter([SimpleNamespace(start=0.25, end=1.5, text=" hello ", words=[])]), SimpleNamespace(duration=2.0)

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisperModel))
    asr = FasterWhisperASR(
        str(audio),
        whisper_model="small",
        model_dir=str(tmp_path),
        device="cpu",
        compute_type="int8",
        language="en",
    )

    result = asr.run()

    assert captured["model_path"] == str(model)
    assert captured["init"]["local_files_only"] is True
    assert captured["transcribe"]["language"] == "en"
    assert result.segments[0].text == "hello"
    assert result.segments[0].start_time == 250
    assert result.segments[0].end_time == 1500
