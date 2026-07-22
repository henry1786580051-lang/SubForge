import wave

import numpy as np

from subforge.core.asr import ten_vad
from subforge.core.asr.audio_analysis import AudioAnalysisContext


def _write_wav(path, duration_seconds: float = 1.0) -> None:
    samples = np.zeros(round(16000 * duration_seconds), dtype=np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(samples.tobytes())


def test_audio_analysis_reuses_decode_energy_and_ten_frames(tmp_path, monkeypatch):
    audio_path = tmp_path / "audio.wav"
    _write_wav(audio_path)
    calls = {"frames": 0}

    monkeypatch.setattr(ten_vad, "is_available", lambda: True)

    def fake_flags(samples, threshold):
        calls["frames"] += 1
        assert threshold == 0.5
        return [0] * 10 + [1] * 40 + [0] * 13

    monkeypatch.setattr(ten_vad, "infer_vad_flags", fake_flags)
    context = AudioAnalysisContext(str(audio_path))

    first = context.speech_segments(
        threshold=0.5,
        min_speech_ms=160,
        min_silence_ms=180,
        speech_pad_ms=0,
    )
    second = context.speech_segments(
        threshold=0.5,
        min_speech_ms=200,
        min_silence_ms=350,
        speech_pad_ms=120,
    )

    assert first
    assert second
    assert calls["frames"] == 1
    assert context.audio_segment() is context.audio_segment()
    assert context.energy_windows() is context.energy_windows()


def test_audio_analysis_caches_identical_speech_profile(tmp_path, monkeypatch):
    audio_path = tmp_path / "audio.wav"
    _write_wav(audio_path)
    calls = {"frames": 0}
    monkeypatch.setattr(ten_vad, "is_available", lambda: True)
    monkeypatch.setattr(
        ten_vad,
        "infer_vad_flags",
        lambda *_args: calls.__setitem__("frames", calls["frames"] + 1) or [0] * 63,
    )
    context = AudioAnalysisContext(str(audio_path))

    first = context.speech_segments()
    second = context.speech_segments()

    assert first is second
    assert calls["frames"] == 1
