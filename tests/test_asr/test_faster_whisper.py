import sys
import threading
import wave
from types import SimpleNamespace

import pytest

from subforge.core.asr import faster_whisper as module


def _write_model(model_dir):
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}" * 100, encoding="utf-8")
    (model_dir / "model.bin").write_bytes(b"x" * (1024 * 1024))
    (model_dir / "tokenizer.json").write_text("{}" * 1024, encoding="utf-8")


def _write_audio(path):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\0\0" * 1_600)


def test_model_directory_requires_complete_ctranslate2_snapshot(tmp_path):
    model_dir = tmp_path / "faster-whisper-base"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}" * 100, encoding="utf-8")

    assert module.is_faster_whisper_model_dir(model_dir) is False

    (model_dir / "model.bin").write_bytes(b"x" * (1024 * 1024))
    (model_dir / "tokenizer.json").write_text("{}" * 1024, encoding="utf-8")

    assert module.is_faster_whisper_model_dir(model_dir) is True


@pytest.mark.parametrize(
    ("cuda_available", "requested", "compute", "expected"),
    [
        (True, "auto", "default", ("cuda", "float16")),
        (False, "auto", "default", ("cpu", "int8")),
        (False, "cuda", "float16", ("cpu", "int8")),
        (True, "cpu", "default", ("cpu", "int8")),
    ],
)
def test_runtime_resolution_falls_back_safely(
    monkeypatch, cuda_available, requested, compute, expected
):
    monkeypatch.setattr(
        module, "is_faster_whisper_cuda_available", lambda: cuda_available
    )

    assert module.resolve_faster_whisper_runtime(requested, compute) == expected


def test_direct_runtime_keeps_segment_timestamps_when_word_mode_requested(
    tmp_path, monkeypatch
):
    model_dir = tmp_path / "faster-whisper-base"
    _write_model(model_dir)
    audio_path = tmp_path / "audio.wav"
    _write_audio(audio_path)
    calls = {}

    class FakeWhisperModel:
        def __init__(self, path, **kwargs):
            calls["path"] = path
            calls["init"] = kwargs

        def transcribe(self, path, **kwargs):
            calls["audio"] = path
            calls["transcribe"] = kwargs
            segments = [
                SimpleNamespace(start=0.1, end=1.4, text=" Hello world "),
                SimpleNamespace(start=1.5, end=2.2, text=" Second sentence. "),
            ]
            return iter(segments), SimpleNamespace(duration=2.2)

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel=FakeWhisperModel),
    )
    monkeypatch.setattr(module, "is_faster_whisper_cuda_available", lambda: False)

    asr = module.FasterWhisperASR(
        str(audio_path),
        whisper_model=str(model_dir),
        device="auto",
        need_word_time_stamp=True,
    )
    result = asr.run()

    assert [(seg.start_time, seg.end_time, seg.text) for seg in result.segments] == [
        (100, 1400, "Hello world"),
        (1500, 2200, "Second sentence."),
    ]
    assert result.granularity == "sentence"
    assert result.timing_source == "native"
    assert not result.is_word_timestamp()
    assert calls["transcribe"]["word_timestamps"] is False
    assert calls["init"]["local_files_only"] is True


def test_packaged_worker_is_terminated_when_task_is_cancelled(tmp_path, monkeypatch):
    model_dir = tmp_path / "faster-whisper-base"
    _write_model(model_dir)
    audio_path = tmp_path / "audio.wav"
    _write_audio(audio_path)
    cancelled = threading.Event()
    cancelled.set()
    process_state = {"terminated": False}

    class FakeProcess:
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            process_state["terminated"] = True
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(module, "is_faster_whisper_cuda_available", lambda: True)
    asr = module.FasterWhisperASR(
        str(audio_path),
        whisper_model=str(model_dir),
        device="cuda",
        cancel_event=cancelled,
    )
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())

    with pytest.raises(RuntimeError, match="cancelled"):
        asr._run_in_packaged_worker()

    assert process_state["terminated"] is True
